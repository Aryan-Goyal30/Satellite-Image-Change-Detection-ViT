# Earth Guardian — Architecture

## 1. What is implemented now (midterm vertical slice)

```
   BEFORE image                AFTER image
   (RGB, any size)             (RGB, any size)
        |                           |
        +-------------+-------------+
                      |
              [ 1. VALIDATION ]         same size? 3 channels?
                      |
              [ 2. PREPROCESS ]         ImageNet normalisation
                      |
              [ 3. TILING ]             256x256 tiles, 64 px overlap
                      |
        +=============v=============+
        |   CHANGE DETECTION ENGINE |
        |                           |
        |   Siamese ResNet-34       |   shared weights, both dates
        |   encoder  (5 scales)     |
        |          |                |
        |   Fusion per scale:       |
        |   conv1x1(cat[|a-b|,a+b]) |
        |          |                |
        |   U-Net decoder           |
        |          |                |
        |   1 logit / pixel         |
        +=============v=============+
                      |
              [ 4. STITCHING ]          Hann-weighted blend -> no seams
                      |
              [ 5. THRESHOLD ]          tau selected on VALIDATION split
                      |
              [ 6. POST-PROCESS ]       min-area filter, connected components
                      |
        +-------------+-------------+
        |             |             |
   CHANGE MASK   QUANTIFICATION   REGIONS
   (PNG/array)   px, % of area    count, bbox, centroid, area_px
        |             |             |
        +-------------+-------------+
                      |
              [ 7. OUTPUT ]
                      |
        +-------------+-------------+-------------+
        |             |             |             |
    result.json   overlay.png   mask.png     prob.png
        |
        +--> CLI (predict.py)   +--> Streamlit UI (app.py)
             both call the SAME engine function
```

## 2. Where this sits in the long-term product

```
  [ FUTURE ]  User -> search location -> select dates -> imagery provider
                                                              |
                                                              v
  [ FUTURE ]                                    image validation / registration
                                                              |
  [ BUILT  ]  ================================================v===========
              |            CHANGE DETECTION ENGINE  (src/inference)      |
              |            + model            (src/models)               |
              |            + training         (src/train)                |
              |            + evaluation       (src/eval)                 |
              ===========================================================
                                                              |
  [ BUILT  ]                                          change mask
                                                              |
  [ BUILT  ]                                     quantification (px, %)
  [ FUTURE ]                                     quantification (m2, GeoJSON)
                                                              |
  [ BUILT  ]                                    visualization / report
                                                              |
  [ FUTURE ]                                    alerts, timeline, semantic types
```

The engine is deliberately a pure function of `(before, after) -> result`.
Nothing above or below it is baked in, so the future imagery-provider layer can
be added without touching model, training or evaluation code.

## 3. Module map

| Module | Responsibility |
|---|---|
| `scripts/download_levir.py` | fetch official LEVIR-CD train/val/test archives |
| `scripts/prepare_tiles.py` | 1024² scenes -> 256² tiles, **within each split** |
| `src/data/levir.py` | Dataset, normalisation, train-only augmentation |
| `src/models/siamese_unet.py` | architecture only — no training logic |
| `src/train/losses.py` | BCE(pos_weight) + soft Dice |
| `src/train/train.py` | training loop, AMP, checkpoint on best val F1 |
| `src/eval/metrics.py` | global TP/FP/FN/TN, P/R/F1/IoU, PR sweep, AP |
| `src/eval/evaluate.py` | val-threshold selection -> test evaluation |
| `src/eval/evaluate_baseline.py` | Stage 0 frozen-ViT baseline scored with the same tiles, metrics and protocol |
| `src/viz/compare_figures.py` | Stage 0 vs Stage 1 comparison figures |
| `src/inference/engine.py` | **product core**: tiling, stitching, post-proc, JSON |
| `src/viz/figures.py` | presentation figures |
| `predict.py` | CLI entry point |
| `app.py` | Streamlit UI (thin layer over the engine) |
| `baseline/` | **Stage 0** — the original frozen-ViT POC, preserved |

## 4. Why this architecture (viva answers)

**Why Siamese with shared weights?**
Both dates must land in the same feature space for a difference to be meaningful.
Two independent encoders could learn unrelated representations, making `|f_a−f_b|`
arbitrary. Sharing also halves the parameters and makes the model symmetric.

**Why fuse with `conv1x1(cat[|a−b|, a+b])`?**
The absolute difference is the change signal. The sum supplies scene context, so
the network can distinguish a genuine structural change from a global brightness
shift. The 1×1 conv lets the network learn how much of each to use, per channel.

**Why U-Net?**
Change detection is pixel-level segmentation. Skip connections carry the
high-resolution detail that the encoder's downsampling discards, which is what
lets the output localise a 12 m building rather than a 73 m block.

**Why a ResNet-34 encoder and not ViT-B/16?**
Plain ViT-B/16 at 224 has a single fixed 14×14 token scale and no feature
pyramid, so its finest output cell is ~73 m on the ground at 0.5 m GSD — coarser
than the buildings being detected. A ResNet gives a 5-level pyramid down to
stride 2 and is straightforward to justify. The frozen ViT remains in
`baseline/` as the Stage 0 unsupervised control.

**Why ResNet-34 specifically?** Chosen by profiling on the actual GPU
(RTX 5060 Laptop, 7.93 GB), batch 32, AMP:

| Encoder | Params | Peak VRAM | Throughput |
|---|---|---|---|
| ResNet-18 | 15.0 M | 2.78 GB (35%) | 206 img/s |
| **ResNet-34** | **25.1 M** | **3.31 GB (42%)** | **160 img/s** |
| ResNet-50 | 43.7 M | 6.18 GB (78%) | 99 img/s |

ResNet-34 costs little over ResNet-18 and leaves a safe VRAM margin on a laptop
GPU that also drives the display; ResNet-50 is too close to the limit. The full
50-epoch run took 42 minutes.

**Why threshold on validation?**
A threshold picked on test is a tuned parameter, and the resulting score is
optimistic. Selecting on val and transferring to test unchanged is the honest
protocol; `src/eval/evaluate.py` additionally reports what test-optimal *would*
have been, so the size of that gap is visible rather than hidden.
