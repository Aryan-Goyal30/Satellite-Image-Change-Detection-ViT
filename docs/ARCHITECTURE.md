# Earth Guardian — Architecture

Earth Guardian is a temporal satellite-imagery change monitoring platform.
This document describes the architecture as it exists today and marks clearly
what is implemented and what is planned.

**What the system does today:** given a pair of co-located images of the same
place at two dates, a domain engine produces a pixel-level **binary** change
mask, quantified in pixels and percentage, with per-region statistics and a
confidence value — for either of two domains, **Built Environment** (RGB,
structural change, optionally direction-classified) or **Environment**
(six-band Sentinel-2, forest loss). A third domain, **Disaster**, is named in
the product as planned but has no model and no code.

---

## 1. Layering

```
              INPUT SOURCE     examples/catalogue.json  (logical example IDs)
                               FUTURE: imagery provider (location + date)
                        │
                        │  applications resolve inputs by ID, never by path
                        ▼
                        APPLICATIONS
            predict.py (CLI)      app.py (Streamlit: Home / Analysis / Results)
                        │
                        │  ask for a domain by name, never for a model
                        ▼
        ┌───────────────────────────────────────────┐
        │  CORE CONTRACTS            src/core/       │
        │    registry   name -> engine (lazy)        │
        │    engine     ChangeEngineProtocol         │
        │    types      ChangeResult (schema 1.2),   │
        │               Layer, Region, Quantities,   │
        │               Provenance, GeoRef, InputSpec│
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
        │     direction/       optional: Construction/│
        │                      Demolition/Uncertain   │
        │                      (ResNet-18, opt-in)    │
        │                                             │
        │   environment            IMPLEMENTED        │
        │     data/            Sentinel-2, TMF, STAC  │
        │     engine.py        6-band Siamese U-Net   │
        │     model_card.py    claims and envelope    │
        │                                             │
        │   disaster               NOT IMPLEMENTED    │
        └───────────────────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────────────────┐
        │  COMMON                 src/common/          │
        │    pair_input       domain-aware input       │
        │                      dispatch -> PreparedPair │
        │    tiling            shared sliding-window    │
        │                      inference                │
        │    preprocessing    normalisation contract    │
        │    model_loader     checkpoint -> model        │
        │    regions          connected components       │
        │    georef           GeoTIFF CRS / scale         │
        │    visualization    overlay, error map, figures  │
        └───────────────────────────────────────────┘

        src/config.py   every project path, in one place
```

**Dependency rule:** applications depend on `core`; domains depend on `core`
and `common`; `core` depends on nothing. `common` and `core` contain no
domain-specific logic and no dataset knowledge — `pair_input.py` and
`tiling.py` are DISPATCH and shared arithmetic respectively, not domain code.

### Inputs — current and future

| | |
|---|---|
| **Current input** | A **local image pair**, selected by logical ID from `examples/catalogue.json`, or uploaded directly. |
| **Future input** | An **imagery provider**: choose a location and two dates, and the provider returns the pair. **Not implemented.** |

The applications call `src/common/examples.py` and receive resolved image
paths; they do not know that a LEVIR-CD tile directory or a Sentinel-2 archive
exists. When an imagery provider is added it fills the same slot and returns
the same thing — a before/after pair — so neither application nor engine has
to change.

The catalogue records only facts that exist: file paths, whether a
ground-truth mask is available, and (for Environment entries) the declared
band list, GSD and CRS. Built Environment entries carry **no location or
acquisition-date metadata**, so none is recorded for them.

### Domain-aware input preparation — `PreparedPair`

Engines do not share an input contract: the Built Environment detector reads
3-channel RGB; the Environment detector reads six Sentinel-2 bands as surface
reflectance and refuses anything else. `src/common/pair_input.py` resolves
that difference once, for both applications:

```python
pair = pair_input.prepare(engine.domain, before, after, engine.metadata.input_spec)
if not pair.ok:
    ...report pair.errors...
out = engine.analyze(pair.before, pair.after, georef=pair.georef)
```

It **dispatches only** — no normalisation, no band arithmetic, no model
knowledge lives here. Each adapter (`src.common.image_input:prepare_rgb_pair`
for Built Environment, `src.domains.environment.inputs:prepare_application_pair`
for Environment) is resolved lazily by import path, for the same reason the
engine registry is lazy: naming a domain must not import that domain's
dependencies.

`PreparedPair` carries two, deliberately separate, pairs of arrays:

```python
@dataclass
class PreparedPair:
    domain: str
    before: Any = None              # whatever THIS domain's engine accepts —
    after: Any = None               #   3-channel RGB, or 6-band reflectance
    preview_before: Optional[Any] = None   # uint8 RGB, display/export ONLY
    preview_after: Optional[Any] = None    # — never what the model reads
    georef: Optional[Any] = None
```

The UI reads only `preview_before`/`preview_after`; it never touches
`before`/`after`. That is a structural guarantee, not a convention — the
Streamlit payload stores only the display arrays, so a six-band array is never
even reachable from a rendering code path.

A pair that cannot be analysed comes back with `ok = False` and user-facing
error text — a value, not an exception — because both callers (a CLI that must
print and exit, a UI that must render a message) need the same text.

### Product V1

**Product V1 is paired-image change analysis across two domains.** The user
picks a monitor, supplies two images of the same area at different dates —
from the bundled example catalogue or their own upload — and receives a
pixel-level change analysis.

```
    User
      -> choose a monitor              Home (registry-driven: a monitor
                                        appears only if an engine is
                                        registered for it)
      -> Before + After imagery        (catalogue example or upload,
                                        validated per-domain by PreparedPair)
      -> registry.get(domain)
      -> engine.analyze(...)
      -> ChangeResult                  (schema v1.2)
      -> Results                       (verdict, visualization, metrics,
                                        regions, direction when applicable,
                                        ground truth, JSON export)
```

Screens: **Home** (`ui/home.py`), **Analysis** (`ui/analysis.py`, input and
validation) and **Results** (`ui/results.py`). `app.py` is only a router.
Inference runs once, when Analyze is pressed; every later interaction reads
the stored `ChangeResult`.

The UI is **capability-driven, never domain-name-driven**: it checks
`engine.supports_direction`, `provenance.score_calibrated`,
`georef.has_scale`, `provenance.dataset_version`, and the result's own
`warnings`, and never branches on `if domain == "..."`. `ui/domains.py` holds
only presentation copy (what a card says, what an upload hint reads) and
contains no band names, no normalisation logic and no model behaviour — those
are read from each engine's own declared `InputSpec` at render time, so the
requirement a user reads cannot drift from the one the model enforces.
`ui/panels/direction.py` holds every piece of direction-specific UI logic in
one place, gated by a single function, `direction_panel.applies(cr)`.

**Future — not implemented:** choosing a location and dates and having
imagery acquired for you; the Disaster monitor.

### Regions and georeferencing

**What a region is.** A region is a connected component of a layer's binary
change mask — a contiguous patch of pixels scored above the decision
threshold. It is an area of detected change. It is **not** an identified
building or an identified clearing.

**Region confidence** is the **mean predicted change probability over that
region's pixels**, on the same 0–1 scale as the threshold. It summarises how
strongly the model scored the patch. It is *not* a calibrated probability that
the region is a true change, and it has not been validated per region — only
the pixel-level metrics have.

**Region direction (schema 1.2, Built Environment only, optional).**
`Region.direction` is one of `"construction"`, `"demolition"` or
`"uncertain"`, with `direction_score` the classifier's score for the predicted
class. Both stay `None` unless direction classification was explicitly
requested, and both belong exclusively to the **direction classifier** — a
different, frozen model from the detector. `direction_score` is not a
calibrated probability; `"uncertain"` is an abstention below the operating
threshold, not a trained ground-truth class.

**Minimum region area.** Components smaller than `config.DEFAULT_MIN_AREA_PX`
(32 px) are discarded as noise and removed from the mask, so the reported
changed-pixel count always agrees with the regions listed. The default is
defined once in `src/config.py` and read by every engine, the CLI and the UI.

**When m² is available.** Never for PNG/JPEG or an unreferenced `.npy` array:
those carry no georeferencing, so there is no ground scale to convert pixels
with — any figure would be invented. For GeoTIFF, `gsd_m` is set **only**
when:

* a projected coordinate system is declared, and
* the linear unit is absent or states metre (EPSG 9001), and
* pixel width and height are positive and equal to within 0.1 %.

A geographic CRS (degrees, e.g. EPSG:4326) never yields metres — converting
degrees to metres depends on latitude, which would be an assumption rather
than a measurement. `area_m2` is produced only by `Quantities.from_mask()`
and `extract_regions()`, and only from a `GeoRef` that passed those checks —
this rule applies identically to both domains; neither has a code path that
fabricates it.

**GeoTIFF requirements.** Both images must have identical dimensions, the same
CRS, and the same geotransform. Different CRSs or geotransforms are **errors**,
not warnings: Earth Guardian performs **no reprojection, resampling or
registration**, so it will not pretend two rasters are aligned when they are
not. Metadata is read with Pillow's TIFF tags — no rasterio or GDAL dependency
anywhere in the project.

**Model input limitation.** The Built Environment model is 3-channel RGB; RGB
and RGBA GeoTIFFs are supported, and products with more than four bands are
rejected rather than having arbitrary bands selected as a fake RGB composite.
The Environment model requires exactly the six declared Sentinel-2 bands as
surface reflectance; anything else is refused, never padded or substituted.

### Application flow

```
    example ID (or upload)
        -> src/common/examples.py         resolve to a before/after pair
        -> src/common/pair_input.py       domain-aware preparation -> PreparedPair
        -> registry.get(domain)
        -> engine.analyze(pair.before, pair.after, georef=pair.georef)
        -> ChangeResult                   the representation both apps render
        -> presentation (metrics, images) / export (ChangeResult.to_dict())
```

---

## 2. Domains

| Domain | Status | Task | Data |
|---|---|---|---|
| **`built_environment`** | **Implemented, operational** | Pixel-level binary structural (building) change; optional Construction/Demolition/Uncertain direction per region | LEVIR-CD; direction classifier trained on S2Looking |
| **`environment`** | **Implemented, V1 research model** | Pixel-level binary forest-loss change | JRC TMF + Sentinel-2, frozen dataset v24 |
| `disaster` | **Not implemented** | Flood, burn scar, landslide, damage | To be decided |

`disaster` is deliberately absent from the codebase rather than stubbed — the
registry lists only what exists (`src/core/registry.py`).

Each domain owns its dataset, model configuration, engine and model card,
because different domains need different imagery modalities, band counts and
metrics. `InputSpec.in_channels` / `.bands` exist for exactly this: Built
Environment declares 3-channel RGB, Environment declares 6-channel
`B02,B03,B04,B08,B11,B12` — a future domain declares its own.

---

## 3. The result contract

```
ChangeResult
  schema_version   "1.2"
  layers[]         name, mask, score_map, threshold, mean_confidence
  regions[]        id, area_px, bbox, centroid, confidence,
                   direction, direction_score        (1.2, optional, Built Environment only)
  quantities       changed_pixels, total_pixels, changed_percentage, area_m2
  provenance       model, version, task, dataset, threshold, weights_hash,
                   evaluation_protocol, operating_envelope,
                   dataset_version, score_calibrated  (1.2, optional)
  georef           crs, transform, gsd_m   (optional)
```

`layers` is a list, not a single mask, because a semantic or multi-class
engine will return several; each current engine returns exactly one
(`structural_change` or `forest_loss`).

**Schema 1.2 additions are all optional and additive.** `Region.direction` /
`direction_score` and `Provenance.dataset_version` / `score_calibrated` are
omitted from `to_dict()` when unset, so a consumer of an engine that declares
none of them sees exactly the payload it saw under schema 1.0 — that is why
the earlier additions did not force a schema bump; 1.2 exists because
`direction` is a genuinely new capability, not a bookkeeping change.
`score_calibrated` states whether a layer's scores may be read as
probabilities: `False` means the threshold is an operating point on a
validation split and the 0–1 scores are **not** calibrated likelihoods; `None`
means the engine has made no claim either way. The Built Environment engine
currently makes no claim; the Environment engine explicitly declares `false`.

**Area rule, enforced in code:** `area_m2` is produced only by
`Quantities.from_mask()`, and only when a `GeoRef` with a real ground sample
distance is supplied. There is no code path that fabricates it.

---

## 4. The pipeline in use today

### Built Environment

```
   BEFORE image                AFTER image        (any size, co-located)
        |                           |
        +-------------+-------------+
                      |
              [ VALIDATION ]        same size? 3 channels? (PreparedPair)
                      |
              [ PREPROCESS ]        src/common/preprocessing.py
                      |             ImageNet mean/std, identical for both dates
              [ TILING ]            256x256 tiles, 64 px overlap (src/common/tiling.py)
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
              [ THRESHOLD ]         0.625, selected on VALIDATION
                      |
              [ POST-PROCESS ]      min-area filter, connected components
                      |
              [ DIRECTION ]         optional: crop each region, run the
                      |             frozen ResNet-18 classifier
              [ QUANTIFY ]          pixels, percentage, regions, confidence
                      |
                 ChangeResult
```

### Environment

```
   BEFORE (6-band reflectance)   AFTER (6-band reflectance)   256x256, 10 m GSD
        |                           |
        +-------------+-------------+
                      |
              [ VALIDATION ]        exactly 6 bands, surface reflectance
                      |             (RGB rejected, never padded)
              [ PREPROCESS ]        per-band standardisation, v24 TRAIN
                      |             statistics only (not ImageNet stats)
              [ TILING ]            shared src/common/tiling.py
                      |
        +=============v=============+
        |  Siamese U-Net            |  shared ResNet-34 encoder, 6-channel
        |  fuse: conv1x1(           |  stem (conv1 widened 3->6)
        |    cat[|a-b|, a+b] )      |
        |  U-Net decoder            |
        |  -> 1 logit per pixel     |
        +=============v=============+
                      |
              [ THRESHOLD ]         0.91, selected on VALIDATION
                      |
              [ POST-PROCESS ]      min-area filter, connected components
                      |
              [ QUANTIFY ]          pixels, percentage, regions, confidence
                      |             (score_calibrated = False; no direction)
                 ChangeResult
```

---

## 5. Module map

| Module | Responsibility |
|---|---|
| `src/config.py` | All project paths: data, tiles, checkpoints, outputs, results, figures |
| `src/core/types.py` | Result contract: ChangeResult (schema 1.2), Layer, Region, Quantities, Provenance, GeoRef, InputSpec |
| `src/core/engine.py` | `ChangeEngineProtocol`, `EngineMetadata` |
| `src/core/registry.py` | Engine name -> implementation, resolved lazily |
| `src/common/pair_input.py` | Domain-aware input dispatch; `PreparedPair` (model input vs display preview) |
| `src/common/tiling.py` | Shared sliding-window inference, Hann blending — one implementation for both domains |
| `src/common/preprocessing.py` | Built Environment normalisation constants and tensor conversion |
| `src/common/model_loader.py` | Checkpoint reading, architecture resolution via `register_builder()`, SHA-256 |
| `src/common/examples.py` | Example catalogue loader — the current input source |
| `src/common/visualization.py` | Overlay, error map, F1 and ground-truth comparison for applications |
| `src/common/regions.py` | Connected-component regions, min-area filtering, region confidence |
| `src/common/georef.py` | GeoTIFF CRS / geotransform / pixel size, read from TIFF tags |
| `src/common/image_input.py` | Built Environment's RGB (+ GeoTIFF) input adapter |
| `examples/catalogue.json` | Bundled example definitions (logical ID → file paths, per-domain metadata) |
| `scripts/build_example_catalogue.py` | Regenerates the Built Environment catalogue entries from prepared data |
| `scripts/build_environment_examples.py` | Copies the 3 curated Environment demo examples from the frozen v24 test split |
| `src/domains/built_environment/data/levir.py` | LEVIR-CD dataset and augmentation |
| `src/domains/built_environment/data/s2looking.py` | Region-level construction/demolition target derivation — research only, not imported by the engine, CLI or UI |
| `src/domains/built_environment/direction/` | Optional direction classifier: model, dataset, crop extraction, model card |
| `src/domains/built_environment/engine.py` | Inference engine: tiling, stitching, post-processing, optional direction, result |
| `src/domains/built_environment/model_card.py` | What this engine may claim; operating envelope |
| `src/domains/environment/data/` | Sentinel-2 acquisition, TMF label handling, STAC querying, dataset assembly |
| `src/domains/environment/engine.py`, `inputs.py`, `normalization.py`, `region_metrics.py` | Six-band inference engine, input validation, per-domain normalisation, region metrics |
| `src/domains/environment/model_card.py` | What this engine may claim; operating envelope; frozen dataset identity |
| `src/models/siamese_unet.py` | Architecture only (shared, not domain-specific) |
| `src/train/` | Loss and training loop |
| `src/eval/metrics.py` | Domain-agnostic metrics |
| `src/eval/evaluate.py` | Built Environment: validation-threshold selection, then test evaluation |
| `src/eval/evaluate_baseline.py` | Stage 0 frozen-ViT baseline, same protocol |
| `src/viz/` | Presentation figures |
| `predict.py`, `app.py` | CLI and Streamlit, both through the registry and `pair_input` |
| `ui/domains.py` | Presentation copy per domain — no band names, no model logic, band lists read live from each engine's `InputSpec` |
| `ui/panels/direction.py` | All direction-specific UI logic, gated by one function |
| `baseline/` | Stage 0 proof of concept, preserved unchanged |

Two files are dead compatibility shims with **no remaining callers anywhere in
this repository** (verified by import search, not assumed):
`src/data/levir.py` re-exports `src/domains/built_environment/data/levir.py`,
and `src/inference/engine.py` re-exports
`src/domains/built_environment/engine.py`. Neither `predict.py`, `app.py`, any
domain engine, nor any test imports either shim. They are candidates for
removal.

---

## 6. Design rationale

**Why a Siamese encoder with shared weights?**
Both dates must land in the same feature space for a difference to be
meaningful. Two independent encoders could learn unrelated representations,
making `|f_a − f_b|` arbitrary. Sharing also halves the parameters and makes
the model symmetric. Both domains use this design.

**Why fuse with `conv1x1(cat[|a−b|, a+b])`?**
The absolute difference is the change signal. The sum supplies scene context,
so the network can distinguish a genuine change from a global brightness
shift. The 1×1 conv learns how much of each to use, per channel.

**Why U-Net?**
Change detection is pixel-level segmentation. Skip connections carry the
high-resolution detail the encoder's downsampling discards.

**Why ResNet-34 and not ViT-B/16?**
Plain ViT-B/16 at 224 has a single fixed 14×14 token scale and no feature
pyramid, so its finest output cell is ~73 m on the ground at 0.5 m GSD —
coarser than the buildings being detected. A ResNet gives a 5-level pyramid
down to stride 2. The frozen ViT remains in `baseline/` as the Stage 0
control, and is scored against the same test set. **This is not a claim that
CNNs beat ViTs in general** — a ViT could be trained for this task too; the
comparison that matters is trained vs. untrained, not architecture family.

ResNet-34 specifically, chosen by profiling on the actual GPU
(RTX 5060 Laptop, 7.93 GB), batch 32, AMP:

| Encoder | Params | Peak VRAM | Throughput |
|---|---|---|---|
| ResNet-18 | 15.0 M | 2.78 GB (35%) | 206 img/s |
| **ResNet-34** | **25.1 M** | **3.31 GB (42%)** | **160 img/s** |
| ResNet-50 | 43.7 M | 6.18 GB (78%) | 99 img/s |

**Why does the Environment encoder start from ImageNet weights at all?**
Transfer learning still helps the visible-band half of the stem (B02/B03/B04
map by wavelength to the pretrained B/G/R filters) and the rest of the
network (layer1–layer4). It supplies **no** semantic knowledge of NIR/SWIR
reflectance — B08/B11/B12 receive the achromatic mean of the RGB filters, a
generic edge detector with no spectral selectivity, and any SWIR-specific
structure had to be learned from the TMF data.

**Why select the threshold on validation?**
A threshold picked on test is a tuned parameter, and the resulting score is
optimistic. Selecting on validation and transferring to test unchanged is the
honest protocol; `src/eval/evaluate.py` additionally reports what test-optimal
*would* have been, so the size of that gap is visible rather than hidden. Both
domains follow this rule.

**Task symmetry, and why direction is a separate model.**
The Built Environment task is binary: the label marks that pixels differ, not
in which direction. The task is therefore symmetric under swapping the two
dates, which is why date-swap is a valid training augmentation for the primary
detector — and why that detector genuinely cannot say construction vs.
demolition on its own. Adding direction to the detector itself would require
removing that augmentation, since direction would then be part of the label.
Instead, direction is a **separate, frozen ResNet-18** that classifies
already-detected regions, trained on a dataset (S2Looking) that actually
labels direction. This keeps the certified detector simple and lets direction
be improved independently.

**Why is Environment's evaluation region-level poor even though pixel-level is
moderate?** An explicit follow-up experiment (E3) tested whether supervising
at TMF's native ~30 m spatial support — rather than the model's 10 m grid —
would close that gap. It did not (E3@10 F1 0.5324, E3@30 F1 0.5375, vs. the
frozen model's own E2@30 F1 0.5981): the hypothesis was refuted, so label
granularity alone does not explain the disagreement. The cause is documented
as open in `docs/ENVIRONMENT_BASELINE_MODEL_CARD.md`.

---

## 7. Where this sits in the product roadmap

```
  [ FUTURE ]  location search -> date selection -> imagery provider
                                                        │
  [ FUTURE ]                                   registration / alignment
                                                        │
  [ BUILT  ]  ==========================================v=================
              |  CORE CONTRACTS + TWO DOMAIN ENGINES + COMMON ANALYSIS      |
              |  Built Environment (+ optional direction) · Environment     |
              ==========================================================
                                                        │
  [ BUILT  ]                              ChangeResult (mask, regions,
                                           direction where applicable)
  [ BUILT  ]                              quantification in pixels and %
  [ BUILT  ]                              quantification in m² (requires
                                           valid GeoRef; not universal)
                                                        │
  [ BUILT  ]                              visualisation / CLI / Streamlit,
                                           capability-driven, domain-aware
  [ FUTURE ]                              timeline, multi-date monitoring
  [ FUTURE ]                              Disaster domain
  [ FUTURE ]                              calibrated Environment scores
  [ FUTURE ]                              direction for the Environment domain
```

Each engine is a pure function of `(before, after) -> ChangeResult`. Nothing
above or below it is baked in, so a future imagery-provider layer, a
calibration pass, or a third domain can be added without touching an existing
engine's model, training or evaluation code.
