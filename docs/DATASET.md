# Dataset

Three datasets back the current implementation: **LEVIR-CD** (Built
Environment detector), **S2Looking** (direction classifier), and **JRC
Tropical Moist Forest + Sentinel-2**, frozen as dataset **v24** (Environment
detector). None of the raw imagery is committed to this repository — `data/`
is entirely gitignored — and every dataset below is reproducible from the
scripts referenced in its section.

---

## LEVIR-CD

Building change detection, 637 pairs of 1024x1024 Google Earth images at
0.5 m/px, 20 regions in Texas USA, captured 2002-2018, with 31,333 annotated
change instances. Binary labels: 1 = change, 0 = no change.

Official pair-level split: **445 train / 64 val / 128 test**.

### Obtaining it

```bash
python scripts/download_levir.py      # ~2.5 GB, resumable
python scripts/prepare_tiles.py       # 1024^2 -> 256^2 tiles
```

`download_levir.py` pulls the official train/val/test archives from the
Hugging Face mirror `satellite-image-deep-learning/LEVIR-CD`, which preserves
the official split. Re-running skips completed files.

### Directory structure

After download:

```
data/levir_cd/
  raw/              train.zip val.zip test.zip
  train/  A/ B/ label/      445 scenes, 1024x1024 PNG
  val/    A/ B/ label/       64 scenes
  test/   A/ B/ label/      128 scenes
```

`A` = before image, `B` = after image, `label` = binary change mask.

After tiling:

```
data/levir_cd_tiles/
  train/  A/ B/ label/      *_r<row>_c<col>.png, 256x256
  val/    A/ B/ label/
  test/   A/ B/ label/
  stats.json
```

### Leakage

Tiling happens strictly **within** each official split. A scene's 16 tiles all
stay in that scene's split, so no tile can appear in both train and test. The
split boundary is the scene, and we never cross it.

### Licensing

LEVIR-CD imagery is derived from Google Earth and is subject to Google Earth
terms of use. It is used here for academic research only.

### Units

LEVIR-CD ships plain PNGs with **no georeferencing and no documented GSD**.
The published 0.5 m/px figure describes the source imagery but is not carried
in the files. The inference engine therefore reports pixel counts and
percentages, and leaves `area_m2` as `null` unless a real GSD is supplied by
the caller. We do not fabricate ground areas.

---

## S2Looking

Used only to train and evaluate the **optional direction classifier**
(Construction / Demolition / Uncertain) — the Built Environment detector and
`ChangeResult` schema do not depend on it.

**Source:** Shen et al., *S2Looking: A Satellite Side-Looking Dataset for
Building Change Detection*, Remote Sensing 13(24):5094, 2021
(https://doi.org/10.3390/rs13245094). Official distribution via
`https://github.com/S2Looking/Dataset`.

### Obtaining it

```bash
python scripts/download_s2looking.py    # official archive only, no mirror
python scripts/inspect_s2looking.py     # confirms on-disk layout, assumes nothing
```

### Licensing — ambiguous, stated plainly

**No dataset licence is declared by the authors.** The GitHub repository
carries no licence file, and the paper's Data Availability Statement only
points back at the same repository. The CC BY 4.0 notice in the paper applies
to the *article*, not to the data. Consequences adopted here, deliberately
conservative: imagery and label rasters are **never committed** (`data/` is
gitignored), and this repository asserts no rights over the raw dataset.
Derived, non-reversible artifacts (region-level labels, metadata, trained
weights) are treated as this project's own research output. Full detail:
[docs/STAGE_3B_S2LOOKING.md](STAGE_3B_S2LOOKING.md).

### Temporal ordering — read before using any construction/demolition result

S2Looking's two annotation maps (`label1`, `label2`) are documented by the
authors only as "newly built" and "demolished" respectively — **not** which of
`Image1`/`Image2` is the earlier acquisition. This project investigated the
question directly (evidence combined from documentary sources and on-disk
measurement — see [docs/S2LOOKING_TEMPORAL_SEMANTICS.md](S2LOOKING_TEMPORAL_SEMANTICS.md))
and **ratified**, as a project decision rather than an author-documented fact:

```
BEFORE = Image2        AFTER = Image1
CONSTRUCTION = label1  DEMOLITION = label2
```

This is carried as `documented_by_authors: false` in every downstream
artifact. If it is ever overturned, every construction/demolition label
produced by the direction classifier inverts.

### Region-level targets and split

Region-level construction/demolition ground truth is **derived by this
project**, not shipped by S2Looking — `scripts/prepare_s2looking_regions.py`
runs the product's own region extractor over the label rasters (there is no
second region implementation) and verifies the result with
`scripts/verify_target_consistency.py` and, visually,
`scripts/verify_s2looking_visually.py`.

| Split | Scenes with usable regions | Regions | Construction | Demolition |
|---|---|---|---|---|
| train | 3,468 / 3,500 | 26,743 | 17,016 | 9,727 |
| val | 497 / 500 | 4,001 | 2,730 | 1,271 |
| test | 990 / 1,000 | 7,617 | 4,726 | 2,891 |

MIXED regions (1,860 overall — genuine ground truth, but not a single
direction) are excluded from classifier training and evaluation. The split is
the dataset's official **scene-level** split; leakage was checked directly on
the derived crop metadata and confirmed zero scene-id overlap between
train/val/test.

---

## JRC Tropical Moist Forest + Sentinel-2 (frozen dataset v24)

Used to train and evaluate the **Environment** engine (forest-loss
detection).

### Label source: JRC Tropical Moist Forest

- **Citation:** Vancutsem, C. et al. *Long-term (1990–2019) monitoring of
  forest cover changes in the humid tropics.* Science Advances 7, eabe1603
  (2021).
- **Licence (as declared by JRC):** free of charge, without restriction of
  use; citation required.
- **Native resolution:** 30 m. Labels are resampled nearest-neighbour to the
  model's 10 m grid — supervision is therefore **30 m-quantised**, and
  predicted boundaries should be read as accurate to about one TMF pixel, not
  to 10 m.
- TMF "deforestation" means forest-cover loss recorded in that product,
  including conversion to plantation or water — **not necessarily illegal or
  permanent clearing.**

### Imagery source: Sentinel-2 L2A

Surface-reflectance observations, six bands used —
**B02, B03, B04, B08, B11, B12** — queried via STAC. No licence terms for
Sentinel-2 access are recorded in this repository's documentation; consult the
Copernicus data policy directly before any use beyond this project's academic
scope.

Each sample pairs one acquisition from the year **before** a TMF-recorded
deforestation year with one from the year **after**, both constrained to a
1 Aug – 15 Sep window, at 10 m ground sample distance.

### Composition (v24, frozen)

| | |
|---|---|
| Total samples | **295** (140 positive / 155 negative) |
| Split | **207 train / 44 validation / 44 test** |
| Geography | 249 Amazon, 46 Southeast Asia — **no African samples**: the JRC endpoint returns HTTP 500 for every Congo Basin longitude tried |
| Split unit | MGRS tile — sets are pairwise disjoint |
| Minimum spatial separation between splits | **100.58 km**, measured directly, not assumed |

### Lineage — how v24 was reached

The dataset was built and corrected in a documented sequence, each step
reproducible and preserved for audit rather than silently overwritten:

1. **Milestone 1** — `scripts/build_environment_sample.py`: one hand-inspectable
   sample, validating that TMF-labelled loss actually appears between the two
   Sentinel-2 dates.
2. **Milestone 2** — `scripts/build_environment_dataset.py`: the sampling and
   quality-gate protocol applied at scale, audited by
   `scripts/audit_environment_dataset.py`.
3. **v2.3 correction** — `scripts/rebuild_environment_dataset_v23.py`: fixes
   three defects found by the v2.2 audit (including a BOA-offset conversion
   fault); every value not affected by those defects is carried across
   byte-identically.
4. **v2.4 (current, frozen)** — `scripts/build_environment_dataset_v24.py`:
   two changes only — a class-aware geographic split optimiser (v2.3's
   size-only packing left the test split with too few positives to measure a
   segmentation model on) and a targeted Southeast Asia top-up, both under the
   same frozen sampling gates.

### Reproducing it

```bash
python scripts/download_levir.py 2>/dev/null || true   # unrelated; not needed here
python scripts/build_environment_sample.py     # Milestone 1 sanity check
python scripts/build_environment_dataset.py     # Milestone 2 base dataset
python scripts/rebuild_environment_dataset_v23.py
python scripts/build_environment_dataset_v24.py  # -> data/environment/dataset_v24/
```

Each step queries Sentinel-2/TMF over the network and is slow; the result is
never committed (`data/` is gitignored, and v24 alone is roughly 503 MB).

### Leakage and normalisation discipline

- Splits are geographically disjoint by MGRS tile, verified at 100.58 km
  minimum separation — not just assumed from tile adjacency.
- Per-band normalisation statistics are computed from the **v24 TRAIN split
  only**; the test split is never read before evaluation.
- The same threshold and the same TRAIN-derived normalisation are applied to
  both Amazon and Southeast Asia samples deliberately — per-region
  normalisation would conceal the geographic domain shift the split exists to
  expose.

### Documented caveats (carried into the model card, not hidden here)

- The loss is computed over all pixels, including those flagged invalid by
  Sentinel-2's scene classification layer (mean 0.27% of a sample, max
  4.98%) — they are not masked out.
- Eight samples with reversed inside-vs-outside dNBR are retained in the
  frozen dataset by design and were not excluded from training or scoring.
- This dataset version is used to support one training run (the frozen "E2"
  production model) plus a small number of controlled experiments — a
  spectral ablation (RGB / RGB+NIR / six-band) and a label-granularity
  experiment (E3, native ~30 m supervision) — not a hyperparameter search.

Full detail: [docs/ENVIRONMENT_BASELINE_MODEL_CARD.md](ENVIRONMENT_BASELINE_MODEL_CARD.md).

---

## Curated demo examples (tracked, not the full datasets)

`examples/catalogue.json` and `examples/environment/` ship a small, tracked
subset for the product's bundled demo — this is **not** a substitute for the
full datasets above, only enough to run the UI without a local `data/`
directory:

- Built Environment: LEVIR-CD **test**-split tiles (never train/val) plus 3
  untraced sample pairs with no recorded split membership.
- Environment: 3 samples copied from the frozen v24 **test** split, chosen on
  ground-truth extent, spatial coherence and image quality only — never on how
  the model scores them.
