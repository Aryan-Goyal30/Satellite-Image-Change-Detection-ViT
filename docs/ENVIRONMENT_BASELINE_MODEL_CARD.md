# Environmental Change Baseline — Model Card

**Phase 4B-1B · six-band Siamese U-Net · the first environmental model experiment for Earth Guardian.**

This model establishes a baseline and nothing more. It is not integrated into the product, has not been compared against any alternative, and no ablation has been run.

## What it does

Given two Sentinel-2 Level-2A surface-reflectance observations of the same ground — one from the year before a JRC TMF deforestation year and one from the year after — it emits a per-pixel logit for *deforestation recorded by TMF between those two dates*.

- Bands: B02, B03, B04, B08, B11, B12
- Input: two 256×256×6 tensors, 10 m ground sample distance
- Output: one 256×256 logit map

## Architecture

- `environment-siamese-unet-6band`, resnet34 encoder, **25,152,209 parameters**
- Weight sharing: single encoder applied to both dates
- Fusion: `conv1x1(concat[|f_before - f_after|, f_before + f_after])`

### Six-band input stem

| Band | Initialised from |
|---|---|
| B02 | B |
| B03 | G |
| B04 | R |
| B08 | achromatic mean of RGB |
| B11 | achromatic mean of RGB |
| B12 | achromatic mean of RGB |

All filters are scaled by 0.5000 (= 3/6) to keep the stem response magnitude comparable to the pretrained three-channel case.

- Reused unchanged from ImageNet: bn1, layer1, layer2, layer3, layer4
- Replaced: conv1 (3ch -> 6ch, re-initialised)
- Randomly initialised: fuse blocks, decoder, head

**ImageNet pretraining supplies no semantic knowledge of NIR or SWIR reflectance.** The visible bands are mapped by wavelength, which is defensible; B08/B11/B12 receive the achromatic mean of the RGB filters — a generic edge detector with no spectral selectivity. Any SWIR-specific structure had to be learned from the TMF data.

## Training data

Frozen dataset **v24** — the official split is **207 train / 44 validation / 44
test** (see `docs/DATASET.md`). Of the 207 train records, this experiment
trains on the **201 marked `training_eligible`** in the manifest's own
`training_eligibility` field (92 positive, 109 negative); the other **6 are
flagged `small_event`** and are retained in the archive but excluded from the
optimisation pool by `EnvironmentChangeDataset(..., eligible_only=True)`.
`small_event` does not mean a small JRC-recorded event (their `event_size`
ranges from "small" to "very_large") — it means the event's positive-pixel
footprint *actually visible inside this sample's 256x256 crop* falls below the
dataset's own 200-pixel minimum (45-180 px, 0.07%-0.27% of the tile), too
little signal to train on. Validation uses all 44 records unfiltered
(`eligible_only=False`); the test split was not read during training.

Splits are geographic: the MGRS tile is the atomic unit, sets are pairwise disjoint, and the minimum separation between splits is 100.58 km.

## Preprocessing

Per-band standardisation using statistics from **v24 TRAIN split only (Phase 4B-1A characterization)** (`sha256 95e58cc37815f175…`). No ImageNet statistics are used; they describe 8-bit sRGB photographs and are meaningless for surface reflectance, least of all SWIR.

| Band | mean | std |
|---|---|---|
| B02 | 0.044130 | 0.025442 |
| B03 | 0.063217 | 0.027485 |
| B04 | 0.056493 | 0.044445 |
| B08 | 0.274392 | 0.062492 |
| B11 | 0.206111 | 0.084928 |
| B12 | 0.112363 | 0.073203 |

One global normalisation is applied to Amazon and SE Asia alike — deliberately, since per-region normalisation would conceal the domain shift the geographic evaluation exists to expose.

## Training configuration

- Loss: `0.5 * BCEWithLogits(pos_weight) + 0.5 * (1 - soft Dice)`
- `pos_weight` = **13.364** — TRAIN pixels only: 12,621,541 neg / 944,411 pos
- Optimiser: AdamW, lr 0.0003, weight decay 0.0001, CosineAnnealingLR(T_max=80)
- Batch size 8, seed 20260913, precision bfloat16
- Epochs run: 46 of 80 (early stopping)
- Augmentation: dihedral group: rot90 x {0,1,2,3} optionally followed by horizontal flip, applied identically to before, after, label, invalid. No date swap, no per-date geometry, no colour jitter.
- Hardware: NVIDIA GeForce RTX 5060 Laptop GPU; training time 3.1 min
- Checkpoint selection: highest validation best-F1 over the threshold sweep (best epoch 31)

## Results

Threshold **0.910**, selected by: argmax F1 over a 201-point sweep of the VALIDATION probability histogram; frozen before the test split is read.

| Split | Precision | Recall | F1 | IoU | AP | Pixel acc. |
|---|---|---|---|---|---|---|
| Validation | 0.4251 | 0.5974 | 0.4967 | 0.3304 | 0.4728 | 0.9555 |
| **Test** | 0.5505 | 0.6472 | 0.5950 | 0.4235 | 0.5670 | 0.9329 |

Test confusion: TP 142,024 · FP 115,968 · FN 77,405 · TN 2,548,187

Pixel accuracy is reported only to show it is uninformative: predicting 'no change' everywhere scores about 93% on this data.

### Region level

Criterion: greedy matching by descending predicted area; a prediction is a true positive when IoU with an unclaimed ground-truth region is >= 0.25; each ground-truth region matches at most once, on components of at least 32 px from the project's shared region protocol.

| Split | Region P | Region R | Region F1 | Predicted | Ground truth |
|---|---|---|---|---|---|
| Validation | 0.2267 | 0.1149 | 0.1525 | 75 | 148 |
| Test | 0.2500 | 0.0735 | 0.1136 | 80 | 272 |

### Geographic breakdown (test, identical threshold and normalisation)

| Region | Samples | Positive samples | Positive px | P | R | F1 | IoU | AP |
|---|---|---|---|---|---|---|---|---|
| Amazon | 37 | 18 | 216,018 | 0.5546 | 0.6508 | 0.5989 | 0.4274 | 0.5746 |
| SE Asia | 7 | 3 | 3,411 | 0.3191 | 0.4225 | 0.3636 | 0.2222 | 0.3051 |

## Limitations

- This is a baseline, not a validated detector. It has been trained once, on one dataset version, with no hyperparameter search.
- **Not a global forest-loss detector.** Training data covers the Amazon (4 TMF tiles) and SE Asia (2 tiles). Africa is absent entirely — the JRC endpoint returns HTTP 500 for every Congo Basin longitude — so nothing here says anything about African forest.
- **Outputs are not calibrated probabilities.** The sigmoid output is a score used with a validation-selected threshold; it has not been calibration-tested.
- **No claim of superiority** over any other model or over fewer bands. The RGB and RGB+NIR ablations that would support such a claim have not been run.
- Supervision is 30 m-quantised: TMF labels are resampled from 30 m to the 10 m grid, so boundaries are accurate to about one TMF pixel, not to 10 m.
- TMF 'deforestation' includes conversion to plantation or water. It means forest-cover loss, not necessarily illegal or permanent clearing.
- The loss is computed over all pixels including those flagged invalid by SCL (mean 0.27% of a sample, max 4.98%); they are not masked out.
- Eight samples with reversed inside-vs-outside dNBR are retained in the frozen dataset by design and were not excluded from training or scoring.

## Reproducibility

- Dataset: v24 (frozen)
- Seed: 20260913
- Checkpoint: `outputs\environment_baseline\environment_sixband_best.pt`
- Checkpoint sha256: `326e8a272c6c4536c3a07ccaabc3feb8ad6905ac0db7e7e67b88ceb98096cc45`
- Normalisation sha256: `95e58cc37815f1759e2661ec3fd010bffd0954dd80d0951a32be3ac326d3c9c1`
- Generated: 2026-09-14T06:58:46Z
