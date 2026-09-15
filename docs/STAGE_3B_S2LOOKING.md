# Stage 3B-0/1 — S2Looking acquisition and region-level directionality targets

Data preparation only. Nothing in this stage trains a model, changes the LEVIR
building detector, alters `ChangeResult`, or adds construction/demolition
prediction to the product. The Stage 1 detector and its published metrics are
untouched.

The purpose is narrow: obtain S2Looking, **verify on disk** what its
annotations actually are, and derive region-level construction / demolition /
mixed ground truth that could support a future direction classifier.

> **STATUS: temporal direction ratified; targets regenerated. Classifier training
> still not started.**
> The dataset, the official split and the two separate annotation maps are all
> verified. The temporal direction of those maps is *not* stated by any official
> source (section 4.1), so the project has **ratified an explicit decision**:
> `BEFORE = Image2`, `AFTER = Image1`, `construction = label1`,
> `demolition = label2`. This is an assumption taken on the evidence, not a
> documented property of the dataset — see
> [S2LOOKING_TEMPORAL_SEMANTICS.md](S2LOOKING_TEMPORAL_SEMANTICS.md) section 8.

---

## 1. Source and provenance

Acquired from the official source only. No mirror was used.

| | |
|---|---|
| Official repository | https://github.com/S2Looking/Dataset |
| Official download | Google Drive folder `S2Looking-CD dataset` ([link](https://drive.google.com/drive/folders/1zzb6hif2hwWx4z8UIMLpMAInkhbmmrFY)) |
| Folder contents | exactly one file, `S2Looking.zip` |
| Drive file id | `1YAUrUp3QtZ6VMM1nSuqOjveIVO9ADOY1` |
| Archive size | 10,959,772,884 bytes (10.96 GB) |
| SHA-256 | `4100e5bfadd53556a3508cdec99bdf789a2289a303328119469437d6d1b0c6b8` |
| Access date | 2026-09-12 (UTC) |
| Archive entries | 25,027 |

Recorded machine-readably in `data/s2looking/metadata/provenance.json`.

**Paper.** Shen et al., *S2Looking: A Satellite Side-Looking Dataset for
Building Change Detection*, Remote Sensing 13(24):5094, 2021.
https://doi.org/10.3390/rs13245094

## 2. Licensing — ambiguous, stated plainly

**No dataset licence is declared by the authors.** The GitHub repository
carries no licence file (`license: null` via the GitHub API), and the paper's
Data Availability Statement says only:

> "Publicly available datasets were analyzed in this study. The datasets can be
> found here: https://github.com/S2Looking/."

The CC BY 4.0 notice in the paper applies to the **article**, not to the data.

Consequences adopted here, deliberately conservative:

- Imagery and label rasters are **never committed**. `data/` is gitignored, and
  `data/s2looking` was confirmed ignored by the `/data/` rule.
- Derived region labels and rendered verification figures are also written
  under `data/` and are **not** committed.
- **No redistribution right is claimed.** The repository ships the code that
  regenerates everything from the official download, not the data.

## 3. Verified on-disk structure

Read off the extracted archive, not assumed:

```
data/s2looking/S2Looking/{train,val,test}/
    Image1/<id>.png     before image   1024x1024 RGB uint8
    Image2/<id>.png     after image    1024x1024 RGB uint8
    label/<id>.png      combined change mask   (mode L, {0,255})
    label1/<id>.png     newly built  -> construction  (RGB, {0,255})
    label2/<id>.png     demolished   -> demolition    (RGB, {0,255})
```

All five subdirectories of a split share an identical filename set.
(`__MACOSX/` and `.DS_Store` are packaging artifacts and are ignored.)

## 4. Verified label semantics

The audit's claim — two separate pixel-precise maps for newly built and
demolished areas — **holds on the released files**, with one detail the paper
does not state:

- `label1` carries its annotation in the **blue channel only** (R=0, G=0, B=255)
- `label2` carries its annotation in the **red channel only** (R=255, G=0, B=0)
- values are strictly `{0, 255}`

This matters: collapsing RGB to channel 0 — the obvious default — returns an
**all-zero mask for every construction annotation**. `binarize()` therefore
treats *any* channel above the threshold as annotated, and
`tests/test_stage3b.py` pins both the blue-channel and red-channel cases so the
bug cannot return.

Independent cross-check: `label1 OR label2` reproduced the bundled combined
`label` mask in **120/120 sampled scenes** (40 per split), and is verified for
every scene during generation.

### 4.1 Temporal direction — CONTRADICTION FOUND

The audit assumed `label1 = newly built` **together with** `Image1 = before,
Image2 = after`. Measurement on the released files contradicts that
combination.

Buildings carry strong edges; bare ground does not. Measuring mean Sobel
gradient energy inside pure single-direction regions
(`scripts/check_s2looking_semantics.py`):

| Split | label1: more structure in Image2 | label2: more structure in Image2 |
|---|---|---|
| train | 14.4% | 61.6% |
| val | 10.0% | 65.2% |
| test | 13.2% | 64.4% |

Mean edge energy for label1 regions falls sharply from Image1 to Image2
(train 54.8 → 20.5; val 52.9 → 16.9; test 46.1 → 17.8), while label2 regions
rise (train 31.7 → 39.9; val 30.2 → 40.2; test 22.9 → 39.0).

So **label1 marks places where building structure disappears** between Image1
and Image2, and **label2 where it appears**. Visual inspection agrees:
`train/280`, `train/4599` and `test/746` show buildings in Image1 and bare
ground in Image2 yet are annotated in label1; `train/765` and `test/4100` show
bare ground in Image1 and buildings in Image2 and are annotated in label2.

Two interpretations fit the evidence equally well:

- **(A)** `Image1` is the **later** image and `Image2` the **earlier** one, with
  the paper's label meanings intact (label1 = newly built, label2 = demolished).
- **(B)** `Image1` is the earlier image and the label1/label2 meanings are the
  reverse of the paper's wording.

Both produce the same *pairing*; they differ only in naming. What matters for a
classifier is that the image ordering and the target must agree.

**Resolution.** Interpretation (A) has been ratified as an explicit project
decision: `BEFORE = Image2`, `AFTER = Image1`, `construction = label1`,
`demolition = label2`. The decision is expressed once, in four constants in
`src/domains/built_environment/data/s2looking.py`, pinned by
`tests/test_stage3b.py`, and stamped into every generated artifact as
`temporal_ordering` (including `documented_by_authors: false`). The earlier
inconsistency — pairing "construction" with label1 *while* treating `Image1` as
"before" — is thereby removed: the target names are unchanged, and it is the
image ordering that was corrected.

Caveat on the measurement: edge energy is a proxy and individual scenes are
confounded by haze and illumination (e.g. `train/2336` looks inconsistent on
its own). The conclusion rests on the aggregate over ~2,900 pure regions across
all three splits, where the separation is large and consistent in sign.

## 5. Verified official split

| Split | Scenes | Expected |
|---|---|---|
| train | 3,500 | 3,500 |
| val | 500 | 500 |
| test | 1,000 | 1,000 |

Scene ids are integers drawn from a single 1..5000 pool, 5,000 unique in total,
with **zero filename overlap** between any two splits (train∩val = train∩test =
val∩test = 0). The numeric ranges interleave but never collide.

**The scene is the atomic split unit.** Regions are never split randomly: every
region inherits the split of the scene it came from, so no scene contributes to
more than one split.

## 6. Region target definition

```
change_gt = label1 OR label2
    -> connected components, 8-connectivity
    -> discard components smaller than min_area_px
    -> per component: n_construction, n_demolition, overlap
    -> purity = max(n_c, n_d) / (n_c + n_d)

purity >= 0.95  ->  construction if n_c > n_d, else demolition
otherwise       ->  mixed
```

Region geometry uses the **same semantics as production**: component ordering,
deterministic ids and area filtering match `src/common/regions.extract_regions`,
and `test_geometry_matches_production_region_extraction` asserts byte-level
agreement so the two cannot drift.

**MIXED is ground truth, not model uncertainty.** It denotes a component in
which construction and demolition genuinely co-occur — a redevelopment site. No
"uncertain" target is generated anywhere. Uncertainty will only ever exist later
as *model abstention*, which is a different thing entirely.

| Parameter | Value | Source |
|---|---|---|
| `min_area_px` | 32 | `config.DEFAULT_MIN_AREA_PX` (shared with production) |
| `purity_threshold` | 0.95 | dataset-preparation parameter, fixed in advance |
| connectivity | 8 | matches production region extraction |
| `large_component_fraction` | 0.25 | flag-only, nothing is filtered by it |

The purity threshold is a **dataset-preparation parameter**, fixed before any
statistics were computed and recorded in the generated metadata. It is not
tuned against the test split — or against any split.

## 7. Reproduction

```bash
python scripts/download_s2looking.py          # official Drive source, resumable
python scripts/inspect_s2looking.py           # report actual structure + label values
python scripts/prepare_s2looking_regions.py   # region targets, stats, quality checks
python scripts/verify_s2looking_visually.py   # BEFORE | AFTER | GROUND TRUTH panels
python tests/test_stage3b.py                  # label-generation tests
```

Outputs, all under `data/` and therefore untracked:

```
data/s2looking/metadata/provenance.json
data/s2looking/metadata/region_labels.csv
data/s2looking/metadata/dataset_stats.json
data/s2looking/metadata/quality_checks.json
data/s2looking/metadata/verification/*.png
```

`region_labels.csv` columns: `scene_id, split, region_id, target, area_px,
bbox_x, bbox_y, bbox_w, bbox_h, centroid_x, centroid_y, construction_pixels,
demolition_pixels, overlap_pixels, purity`.

## 8. Dataset statistics

Generated by `scripts/prepare_s2looking_regions.py` into
`data/s2looking/metadata/dataset_stats.json`. All 5,000 scenes were processed.

Class names follow the ratified mapping (section 4.1): `construction = label1`
(building present only in the AFTER image, `Image1`) and `demolition = label2`
(building present only in the BEFORE image, `Image2`). The ordering is a
project decision, not an author-documented fact.

Overall: **5,000 scenes, 40,221 regions**

| Target | Regions | Share |
|---|---|---|
| construction (label1-dominant) | 24,472 | 60.844% |
| demolition (label2-dominant) | 13,889 | 34.532% |
| mixed | 1,860 | 4.624% |

Per split:

| Split | Scenes | Regions | constr. | demol. | mixed | c:d ratio |
|---|---|---|---|---|---|---|
| train | 3,500 | 28,074 | 17,016 | 9,727 | 1,331 | 1.749 |
| val | 500 | 4,177 | 2,730 | 1,271 | 176 | 2.148 |
| test | 1,000 | 7,970 | 4,726 | 2,891 | 353 | 1.635 |

Area in pixels (min / median / mean / max):

| Target | min | median | mean | max |
|---|---|---|---|---|
| construction | 33 | 844 | 1,712.41 | 187,966 |
| demolition | 32 | 655 | 1,279.73 | 133,265 |
| mixed | 105 | 1,791 | 3,694.95 | 76,952 |
| all | 32 | 803 | 1,654.68 | 187,966 |

Pixels: construction 46,629,585; demolition 21,971,311; overlap 2,048,068;
total changed 66,552,828.

Per scene: median 5 regions (mean 8.04, max 94); **2,570 of 5,000 scenes contain
both directions**; 7 scenes contain no annotation at all.

**Demolition representation gate: 34.532% vs the 5% engineering gate — passes
comfortably.** Class imbalance is mild (roughly 1.75:1), which is far healthier
than the audit anticipated.

### 8.1 Quality checks

From `data/s2looking/metadata/quality_checks.json`, over all 5,000 scenes:

| Check | Result |
|---|---|
| image dimension mismatch | 0 |
| label dimension mismatch | 0 |
| unexpected label values | 0 (strictly {0, 255}) |
| `label1 OR label2` differs from bundled `label` | 0 |
| unreadable / corrupt files | 0 |
| scenes with no annotation | 7 |
| scenes yielding no region | 7 (the same 7) |
| components larger than 25% of a scene | 0 |
| deterministic region ids | true |
| split integrity | ok: 5,000 unique ids, no cross-split duplicates |

The 7 empty scenes (`train/560, 594, 2225, 2930, 3915`, `test/1755, 4434`) were
inspected individually: `label1`, `label2` and `label` are all uniformly zero
and only 4,813 bytes, while both images decode normally. They are genuinely
unannotated scenes, not read failures.

### 8.2 Annotation overlap — investigated, benign

`label1` and `label2` overlap on 2,048,068 pixels (3.08% of changed pixels),
touching 1,818 regions (4.52%) across 1,263 scenes. The overlap is not diffuse
noise; it concentrates precisely where redevelopment occurs:

| Target | Regions | With overlap | Overlap px |
|---|---|---|---|
| construction | 24,472 | 14 (0.06%) | 10,233 |
| demolition | 13,889 | 10 (0.07%) | 3,358 |
| mixed | 1,860 | 1,794 (**96.45%**) | 2,034,477 |

Every one of the 341 regions whose overlap exceeds half the region area is
classified `mixed`. Overlap therefore behaves exactly as a redevelopment
signal — an old footprint demolished and a new one built on top — and the
purity rule already routes it to `mixed` rather than to a spurious direction.
No special handling is required.

## 9. Scope boundary

Not done in this stage, by instruction: no classifier training, no ResNet-18, no
checkpoint, no `Region.change_type`, no `ChangeResult` schema change, no UI
change, and no construction/demolition prediction in production. The Stage 3B
data-preparation module is verified to be fully decoupled from inference — the
engine loads without importing it.
