# Earth Guardian — Architecture

Earth Guardian is a temporal satellite-imagery change monitoring platform.
This document describes the architecture as it exists today and marks clearly
what is implemented and what is planned.

**What the system does today:** it accepts a pair of co-located optical images
of the same place at two dates and produces a pixel-level **binary** change
mask, quantified in pixels and percentage, with per-region statistics and a
confidence value. Nothing else is implemented.

---

## 1. Layering

```
              INPUT SOURCE     examples/catalogue.json  (logical example IDs)
                               FUTURE: imagery provider (location + date)
                        │
                        │  applications resolve inputs by ID, never by path
                        ▼
                        APPLICATIONS
            predict.py (CLI)      app.py (Streamlit)
                        │
                        │  ask for a domain by name, never for a model
                        ▼
        ┌───────────────────────────────────────────┐
        │  CORE CONTRACTS            src/core/       │
        │    registry   name -> engine (lazy)        │
        │    engine     ChangeEngineProtocol         │
        │    types      ChangeResult, Layer, Region, │
        │               Quantities, Provenance,      │
        │               GeoRef, InputSpec            │
        └───────────────────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────────────────┐
        │  DOMAIN ENGINES         src/domains/        │
        │                                             │
        │   built_environment      IMPLEMENTED        │
        │     data/levir.py    LEVIR-CD dataset       │
        │     engine.py        Siamese U-Net engine   │
        │     model_card.py    claims and envelope    │
        │                                             │
        │   environmental          FUTURE             │
        │   disaster               FUTURE             │
        └───────────────────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────────────────┐
        │  COMMON ANALYSIS        src/common/, src/eval│
        │    preprocessing   normalisation contract    │
        │    model_loader    checkpoint -> model       │
        │    metrics         P/R/F1/IoU, PR sweep, AP  │
        │    viz             figures                   │
        └───────────────────────────────────────────┘

        src/config.py   every project path, in one place
```

**Dependency rule:** applications depend on `core`; domains depend on `core` and
`common`; `core` depends on nothing. `common` and `core` contain no
domain-specific logic and no dataset knowledge.

### Inputs — current and future

| | |
|---|---|
| **Current input** | A **local image pair**, selected by logical ID from `examples/catalogue.json`, or uploaded directly. |
| **Future input** | An **imagery provider**: choose a location and two dates, and the provider returns the pair. **Not implemented.** |

The applications call `src/common/examples.py` and receive resolved image paths.
They do not know that a LEVIR-CD tile directory exists. When an imagery provider
is added it fills the same slot and returns the same thing — a before/after pair
— so neither application nor engine has to change.

The catalogue records only facts that exist: file paths, whether a ground-truth
mask is available, and the measured ground-truth change fraction. These images
carry **no location or acquisition-date metadata**, so none is recorded.

### Product V1

**Product V1 is paired-image structural change analysis.** The user supplies two
images of the same area at different dates — from the bundled example catalogue
or their own upload — and receives a pixel-level change analysis.

```
    User
      -> Before + After imagery        (catalogue example or upload)
      -> Built Environment Engine      (registry.get("built_environment"))
      -> ChangeResult                  (schema v1)
      -> Visual analysis               (overlay, mask, probability, error map,
                                        summary, region details, model card)
```

Screens: **Home** (`ui/home.py`), **Workspace** (`ui/analysis.py`, input and
validation) and **Results** (`ui/results.py`). `app.py` is only a router.
Inference runs once, when Analyze is pressed; every later interaction reads the
stored `ChangeResult`.

**Future — not implemented:** choosing a location and dates and having imagery
acquired for you, and the Environmental and Disaster monitors. The UI names
those as coming later and provides no functionality for them.

### Regions and georeferencing (Stage 3A)

**What a region is.** A region is a connected component of the binary change
mask — a contiguous patch of pixels scored above the decision threshold. It is
an area of detected change. It is **not** an identified building, and it carries
**no direction**: the model cannot say whether the change was construction or
demolition.

**Region confidence** is the **mean predicted change probability over that
region's pixels**, on the same 0–1 scale as the threshold. It summarises how
strongly the model scored the patch. It is *not* a calibrated probability that
the region is a true change, and it has not been validated per region — only the
pixel-level metrics have. Without a probability map the field stays absent
rather than being invented.

**Minimum region area.** Components smaller than `config.DEFAULT_MIN_AREA_PX`
(32 px) are discarded as noise and removed from the mask, so the reported
changed-pixel count always agrees with the regions listed. The default is
defined once in `src/config.py` and read by the engine, the CLI and the UI.

**When m² is available.** Never for PNG/JPEG: those formats carry no
georeferencing, so there is no ground scale to convert pixels with — any figure
would be invented. For GeoTIFF, `gsd_m` is set **only** when:

* a projected coordinate system is declared, and
* the linear unit is absent or states metre (EPSG 9001), and
* pixel width and height are positive and equal to within 0.1 %.

A geographic CRS (degrees, e.g. EPSG:4326) never yields metres — converting
degrees to metres depends on latitude, which would be an assumption rather than
a measurement. `area_m2` is produced only by `Quantities.from_mask()` and
`extract_regions()`, and only from a `GeoRef` that passed those checks.

**GeoTIFF requirements.** Both images must have identical dimensions, the same
CRS, and the same geotransform. Different CRSs or geotransforms are **errors**,
not warnings: Earth Guardian does **no reprojection, resampling or
registration**, so it will not pretend two rasters are aligned when they are
not. Metadata is read with Pillow's TIFF tags — no rasterio or GDAL dependency.

**Model limitation.** The trained model is 3-channel RGB. RGB and RGBA GeoTIFFs
are supported and single-band follows the existing policy; **products with more
than four bands are rejected** rather than having arbitrary bands selected as a
fake RGB composite.

### Application flow

```
    example ID (or upload)
        -> src/common/examples.py         resolve to a before/after pair
        -> registry.get("built_environment")
        -> engine.analyze(before, after)
        -> ChangeResult                   the representation both apps render
        -> presentation (metrics, images) / export (ChangeResult.to_dict())
```

---

## 2. Domains

| Domain | Status | Task | Data |
|---|---|---|---|
| **`built_environment`** | **Implemented, operational** | Pixel-level binary structural (building) change | LEVIR-CD |
| `environmental` | **Future — not implemented** | Vegetation, land-cover, water change | To be decided |
| `disaster` | **Future — not implemented** | Flood, burn scar, landslide, damage | To be decided |

The future domains are deliberately absent from the codebase rather than
stubbed. The registry lists only what exists.

Each domain owns its dataset, model configuration, engine and model card,
because different domains will need different imagery modalities, band counts
and metrics. `InputSpec.in_channels` exists so a future multispectral or SAR
engine can declare its own requirement; the current engine is and remains
3-channel RGB.

---

## 3. The result contract

```
ChangeResult
  schema_version   "1.0"
  layers[]         name, mask, score_map, threshold, mean_confidence
  regions[]        id, area_px, bbox, centroid, confidence
  quantities       changed_pixels, total_pixels, changed_percentage, area_m2
  provenance       model, version, task, dataset, threshold, weights_hash,
                   evaluation_protocol, operating_envelope
  georef           crs, transform, gsd_m   (optional)
```

`layers` is a list, not a single mask, because a semantic or multi-class engine
will return several. The Built Environment engine returns exactly one layer,
`structural_change`.

**Area rule, enforced in code:** `area_m2` is produced only by
`Quantities.from_mask()`, and only when a `GeoRef` with a real ground sample
distance is supplied. LEVIR-CD PNGs carry no georeferencing, so today it is
always `null`. There is no code path that fabricates it.

---

## 4. The pipeline in use today

```
   BEFORE image                AFTER image        (any size, co-located)
        |                           |
        +-------------+-------------+
                      |
              [ VALIDATION ]        same size? 3 channels?
                      |
              [ PREPROCESS ]        src/common/preprocessing.py
                      |             ImageNet mean/std, identical for both dates
              [ TILING ]            256x256 tiles, 64 px overlap
                      |
        +=============v=============+
        |  Siamese U-Net            |  shared ResNet-34 encoder, both dates
        |  fuse: conv1x1(           |
        |    cat[|a-b|, a+b] )      |
        |  U-Net decoder            |
        |  -> 1 logit per pixel     |
        +=============v=============+
                      |
              [ STITCHING ]         Hann-weighted blend -> no tile seams
                      |
              [ THRESHOLD ]         selected on VALIDATION, applied unchanged
                      |
              [ POST-PROCESS ]      min-area filter, connected components
                      |
              [ QUANTIFY ]          pixels, percentage, regions, confidence
                      |
                 ChangeResult
```

---

## 5. Module map

| Module | Responsibility |
|---|---|
| `src/config.py` | All project paths: data, tiles, checkpoints, outputs, results, figures |
| `src/core/types.py` | Result contract: ChangeResult, Layer, Region, Quantities, Provenance, GeoRef, InputSpec |
| `src/core/engine.py` | `ChangeEngineProtocol`, `EngineMetadata` |
| `src/core/registry.py` | Engine name -> implementation, resolved lazily |
| `src/common/preprocessing.py` | Normalisation constants and tensor conversion |
| `src/common/model_loader.py` | Checkpoint reading, architecture resolution, SHA-256 |
| `src/common/examples.py` | Example catalogue loader — the current input source |
| `src/common/visualization.py` | Error map, F1 and ground-truth comparison for applications |
| `src/common/regions.py` | Connected-component regions, min-area filtering, region confidence |
| `src/common/georef.py` | GeoTIFF CRS / geotransform / pixel size, read from TIFF tags |
| `examples/catalogue.json` | Bundled example definitions (logical ID → file paths) |
| `scripts/build_example_catalogue.py` | Regenerates the catalogue from prepared data |
| `src/domains/built_environment/data/levir.py` | LEVIR-CD dataset and augmentation |
| `src/domains/built_environment/engine.py` | Inference engine: tiling, stitching, post-processing, result |
| `src/domains/built_environment/model_card.py` | What this model may claim; operating envelope |
| `src/models/siamese_unet.py` | Architecture only (shared, not domain-specific) |
| `src/train/` | Loss and training loop |
| `src/eval/metrics.py` | Domain-agnostic metrics |
| `src/eval/evaluate.py` | Validation-threshold selection, then test evaluation |
| `src/eval/evaluate_baseline.py` | Stage 0 frozen-ViT baseline, same protocol |
| `src/viz/` | Presentation figures |
| `predict.py`, `app.py` | CLI and Streamlit, both through the registry |
| `baseline/` | Stage 0 proof of concept, preserved unchanged |

Compatibility shims (`src/data/levir.py`, `src/inference/engine.py`) re-export
from the new locations so older imports keep working. They will be removed once
nothing depends on them.

---

## 6. Design rationale

**Why a Siamese encoder with shared weights?**
Both dates must land in the same feature space for a difference to be
meaningful. Two independent encoders could learn unrelated representations,
making `|f_a − f_b|` arbitrary. Sharing also halves the parameters and makes the
model symmetric.

**Why fuse with `conv1x1(cat[|a−b|, a+b])`?**
The absolute difference is the change signal. The sum supplies scene context, so
the network can distinguish a genuine structural change from a global brightness
shift. The 1×1 conv learns how much of each to use, per channel.

**Why U-Net?**
Change detection is pixel-level segmentation. Skip connections carry the
high-resolution detail the encoder's downsampling discards.

**Why ResNet-34 and not ViT-B/16?**
Plain ViT-B/16 at 224 has a single fixed 14×14 token scale and no feature
pyramid, so its finest output cell is ~73 m on the ground at 0.5 m GSD — coarser
than the buildings being detected. A ResNet gives a 5-level pyramid down to
stride 2. The frozen ViT remains in `baseline/` as the Stage 0 control, and is
scored against the same test set.

ResNet-34 specifically, chosen by profiling on the actual GPU
(RTX 5060 Laptop, 7.93 GB), batch 32, AMP:

| Encoder | Params | Peak VRAM | Throughput |
|---|---|---|---|
| ResNet-18 | 15.0 M | 2.78 GB (35%) | 206 img/s |
| **ResNet-34** | **25.1 M** | **3.31 GB (42%)** | **160 img/s** |
| ResNet-50 | 43.7 M | 6.18 GB (78%) | 99 img/s |

**Why select the threshold on validation?**
A threshold picked on test is a tuned parameter, and the resulting score is
optimistic. Selecting on validation and transferring to test unchanged is the
honest protocol; `src/eval/evaluate.py` additionally reports what test-optimal
*would* have been, so the size of that gap is visible rather than hidden.

**Task symmetry.** The current task is binary: the label marks that pixels
differ, not in which direction. The task is therefore symmetric under swapping
the two dates, which is why date-swap is a valid training augmentation.
Construction-vs-demolition directionality is a future capability; adding it
requires removing that augmentation first, because direction would then be part
of the label.

---

## 7. Where this sits in the product roadmap

```
  [ FUTURE ]  location search -> date selection -> imagery provider
                                                        │
  [ FUTURE ]                                   registration / alignment
                                                        │
  [ BUILT  ]  ==========================================v=================
              |  CORE CONTRACTS + DOMAIN ENGINE + COMMON ANALYSIS        |
              ==========================================================
                                                        │
  [ BUILT  ]                              ChangeResult (mask, regions)
  [ BUILT  ]                              quantification in pixels and %
  [ FUTURE ]                              quantification in m2 (needs GeoRef)
                                                        │
  [ BUILT  ]                              visualisation / CLI / Streamlit
  [ FUTURE ]                              timeline, multi-date monitoring
  [ FUTURE ]                              additional domains
```

The engine is a pure function of `(before, after) -> ChangeResult`. Nothing
above or below it is baked in, so a future imagery-provider layer can be added
without touching model, training or evaluation code.
