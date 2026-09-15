# Model card — region direction classifier (construction vs demolition)

Stage 3B-2 research component. **Not integrated into the product.** The frozen
LEVIR Siamese U-Net change detector, its checkpoint, its published metrics, the
production engine, the UI and the `ChangeResult` schema are all untouched.

| | |
|---|---|
| Task | classify an already-detected change region as **construction** or **demolition** |
| Unit of prediction | region (connected component), matching the product's existing unit |
| Architecture | ResNet-18, 6-channel early fusion, 2 logits |
| Parameters | **11,186,946** (ablations: 11,177,538) |
| Training data | S2Looking region targets (Stage 3B-1) |
| Headline metric | **test macro-F1 0.9608** |

## 1. Temporal-ordering provenance — read this first

The mapping below is a **ratified project decision, not a documented property of
S2Looking**:

```
BEFORE = Image2        AFTER = Image1
CONSTRUCTION = label1  DEMOLITION = label2
```

No official S2Looking source states which of `Image1`/`Image2` is the earlier
acquisition. The paper fixes only that `label1` = newly built and `label2` =
demolished. See [S2LOOKING_TEMPORAL_SEMANTICS.md](S2LOOKING_TEMPORAL_SEMANTICS.md).

`documented_by_authors: false` is carried verbatim in every artifact
(`direction_train_*.json`, checkpoints, crop manifest). **If this decision is
ever overturned, every construction/demolition label in this model inverts** —
the model would be exactly as accurate and exactly wrong. Anyone citing these
results must describe the ordering as an assumption.

## 2. Dataset and split

S2Looking, official **scene-level** split. Regions come from the Stage 3B-1
targets, which were produced by the production region extractor — there is no
second region implementation.

| Split | Scenes with usable regions | Regions | Construction | Demolition |
|---|---|---|---|---|
| train | 3,468 / 3,500 | 26,743 | 17,016 | 9,727 |
| val | 497 / 500 | 4,001 | 2,730 | 1,271 |
| test | 990 / 1,000 | 7,617 | 4,726 | 2,891 |

**MIXED regions are excluded** (1,860 overall) — they are genuine ground truth
but not a direction. Scenes short of the split totals contributed only mixed
regions, or no annotation at all.

**Leakage:** verified directly on the crop metadata — train∩val, train∩test and
val∩test share **zero** scene ids. A region can never appear in two splits
because it inherits its scene's split.

Class ratio differs by split (train 1.749:1, **val 2.148:1**, test 1.635:1),
which is why selection uses macro-F1 rather than accuracy.

## 3. Preprocessing

```
region bbox (x, y, w, h)                      from Stage 3B-1 targets
  -> expand 25% of w and h on each side       CONTEXT_FRACTION = 0.25
  -> square about the bbox centre             (no aspect distortion on resize)
  -> clamp into the scene, shifting not shrinking
  -> resize 128 x 128 bilinear
  -> stack: channels 0:3 BEFORE (Image2), 3:6 AFTER (Image1)
  -> ImageNet mean/std per date
```

Crops are pre-extracted once to memory-mapped uint8 arrays
(`data/s2looking/crops/`), because decoding two 1024x1024 PNGs per region inside
the training loop would re-decode each scene once per region.

**Augmentation:** dihedral only — random horizontal flip, vertical flip and
k*90° rotation, applied **identically to both dates**. The halves are never
swapped (that would invert the label) and never augmented independently (that
would destroy the pixel correspondence). No photometric jitter: applied to one
date it would simulate the very illumination difference the model must not rely
on, and applied jointly it adds nothing.

## 4. Architecture and first-convolution initialisation

ResNet-18 (ImageNet), `conv1` widened 3→6 channels, `fc` replaced with 2 logits.

```
W_new[:, 0:3] = W_new[:, 3:6] = W_pretrained * 0.5
```

Chosen because it makes `conv6([I, I]) == conv3(I)` exactly for an unchanged
pair: pretrained features start in the distribution they expect, no activation
drift into the following batch-norm, and the model begins *blind to change* and
must learn the difference. It is deterministic — a pure function of the
pretrained tensor, drawing no random numbers.

Verified (seeded probe, recorded in `direction_train_*.json`):

| Check | Result |
|---|---|
| halves identical | true |
| equals `W_pretrained / 2` | true (exact) |
| reproduces pretrained conv on unchanged pair | true |
| max **relative** difference | 9.09e-07 (float32 accumulation only) |
| identical across two builds | true |

Rejected alternatives: zeros in the extra channels (asymmetric — ignores AFTER
at init); antisymmetric ±W/2 (computes a difference but gives zero response for
unchanged input and discards ImageNet feature semantics); random init (adds an
RNG dependence and breaks the equality property).

Ablation models use the **unmodified** 3-channel pretrained `conv1`.

## 5. Training configuration

| | |
|---|---|
| Loss | cross-entropy, inverse-frequency class weights, mean-normalised → **construction 0.7274 / demolition 1.2726** |
| Optimizer | AdamW, weight decay 1e-4 |
| LR | 3e-4, 1-epoch linear warmup then cosine to ~0 |
| Batch size | 128 |
| Precision | AMP fp16 |
| Epochs | max 30, early stopping patience 7 |
| Seed | 1337 (python/numpy/torch, cuDNN deterministic) |
| DataLoader workers | **0** |
| Selection | **best validation macro-F1** |

Class imbalance is handled explicitly by the weighted loss, not absorbed into a
flattering accuracy figure.

`workers=0` is deliberate. With workers, Windows `spawn` pickles the dataset to
each worker every epoch, and numpy materialises a memmap when pickled — pushing
2.6 GB per worker down a pipe until the commit charge was exhausted. The dataset
now stores only the crop **path** and opens the memmap lazily per process
(`__getstate__` drops the handle), so the pickled dataset is ~214 KB; worker
count was left at 0 rather than raised to chase throughput.

## 6. Checkpoints

All under `checkpoints/` (gitignored). Every run ended by **legitimate early
stopping**, not truncation.

| Mode | Epochs | Best epoch | Val macro-F1 | Minutes | SHA-256 |
|---|---|---|---|---|---|
| **both** (primary) | 25/30 | 18 | 0.963586 | 8.16 | `571c731b86baa854a3b4b02a8c1f8995572b002c71446d215fe4a89fb8113f01` |
| before (ablation) | 21/30 | 14 | 0.923897 | 6.69 | `e6c158b1228f490d80dc97f0be45b37019b8ca2660506ddff30b87e8e4c0209d` |
| after (ablation) | 19/30 | 12 | 0.919347 | 5.68 | `b67b0abd53f199b0269d0f2e79cf27f5023624192532421db004426c1c77a9b9` |

Earlier invalid artifacts (a 1-epoch smoke checkpoint and two crash-truncated
ablations) are quarantined under
`checkpoints/INVALID_stage3b2_pre_fix/` with a README. They must never be used.

## 7. Results — primary model, official test split

7,617 regions, never used for training, checkpoint selection or threshold
selection.

| Metric | Value |
|---|---|
| **Macro-F1 (headline)** | **0.9608** |
| Balanced accuracy | 0.9601 |
| Accuracy | 0.9631 |
| ECE (15-bin, max-prob) | 0.0172 |
| Mean confidence | 0.9770 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| construction | 0.9682 | 0.9725 | 0.9703 | 4,726 |
| demolition | 0.9547 | 0.9478 | 0.9512 | 2,891 |

Confusion matrix (rows = true, cols = predicted, order construction/demolition):

```
                 pred constr   pred demol
true constr           4596          130
true demol             151         2740
```

Demolition — the minority class — is the weaker one, as expected, but only by
~1.9 F1 points; the weighted loss kept the gap small.

## 8. Ablations

| Model | Macro-F1 | Balanced acc | Accuracy | ECE |
|---|---|---|---|---|
| **both (primary, 6-ch)** | **0.9608** | 0.9601 | 0.9631 | 0.0172 |
| before-only (3-ch) | 0.9210 | 0.9236 | 0.9252 | 0.0249 |
| after-only (3-ch) | 0.9229 | 0.9222 | 0.9275 | 0.0216 |

Seeing both dates is worth **+3.8 macro-F1** over the better single-date model.
That the single-date ablations still reach ~0.92 is itself informative: a single
image carries substantial signal — a bare construction site and a finished
building look different — so the task is not purely comparative. The primary
model must remain the 6-channel one.

## 9. Abstention (UNCERTAIN)

**UNCERTAIN is an abstention, never a ground-truth class.** No region is
labelled "uncertain" anywhere in the data; the label space is exactly
{construction, demolition}.

Confidence = max softmax probability. The threshold is selected **on validation
only** as the largest τ still reaching 90% coverage there; the test split is
never used to choose it.

| | |
|---|---|
| τ (from validation) | **0.960** |
| Validation coverage at τ | 0.9053 |
| Test coverage at τ | 0.8948 (6,816 retained, 801 abstained) |
| Test accuracy on retained | **0.9868** |
| Test macro-F1 on retained | **0.9858** |

Test coverage lands slightly under the 90% target because τ was fixed on
validation — that gap is the honest cost of not tuning on test.

Accuracy-versus-coverage on test (primary):

| τ | Coverage | Accuracy | Macro-F1 |
|---|---|---|---|
| 0.500 | 1.0000 | 0.9631 | 0.9608 |
| 0.700 | 0.9737 | 0.9711 | 0.9692 |
| 0.800 | 0.9577 | 0.9774 | 0.9758 |
| 0.900 | 0.9308 | 0.9827 | 0.9814 |
| 0.950 | 0.9047 | 0.9864 | 0.9853 |
| 0.975 | 0.8702 | 0.9888 | 0.9880 |

## 10. Calibration

ECE 0.0172 (15 equal-width bins over [0.5, 1.0], max-probability confidence).
The model is mildly **over-confident** — mean confidence 0.9770 against accuracy
0.9631. If a confidence number is ever shown to users it should be described as
a model score, or calibrated first (e.g. temperature scaling on validation).

## 11. Limitations

1. **Ordering is an assumption** (§1). If overturned, every label inverts.
2. **Domain.** S2Looking is rural, globally distributed, off-nadir (mean |angle|
   ~9.9°, max 35.4°), 0.5–0.8 m. These numbers do **not** transfer to the urban
   near-nadir LEVIR imagery the product demo uses.
3. **Ground-truth regions, not detected regions.** The primary result classifies
   S2Looking GT regions. It is *not* an end-to-end number and must not be quoted
   as "the product can tell construction from demolition".
4. **No end-to-end evaluation was run.** Chaining the frozen LEVIR detector into
   S2Looking would confound detector domain shift with classifier quality; if
   ever run it must be labelled exploratory cross-domain.
5. **MIXED excluded.** Redevelopment sites where both occur in one component are
   real (1,860 regions) and this model has no answer for them.
6. **Mild over-confidence** (§10).
7. **Small regions are blurry.** A 32 px region upsamples from ~9×9 to 128×128.
8. **Not calibrated per region**, and region confidence from the detector is a
   separate, unrelated quantity.

## 12. Scope

Not done, by instruction: no UI integration, no `Region.change_type`, no
`ChangeResult` schema change, no production inference path, and no modification
to the detector or its checkpoint. Verified: the production engine imports zero
`s2looking`/`direction` modules.

## 13. Reproduction

```bash
python scripts/build_s2looking_crops.py
python scripts/train_direction_classifier.py --mode both    # then --mode before / --mode after
python scripts/evaluate_direction_classifier.py --all
python scripts/evaluate_direction_classifier.py --summary
python tests/test_stage3b2.py
```

Artifacts (all gitignored): `data/s2looking/crops/`,
`data/s2looking/metadata/direction_train_*.json`,
`direction_eval_*.json`, `direction_ablations.json`, `checkpoints/direction_*.pt`.
