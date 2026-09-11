# Dataset

## LEVIR-CD

Building change detection, 637 pairs of 1024x1024 Google Earth images at
0.5 m/px, 20 regions in Texas USA, captured 2002-2018, with 31,333 annotated
change instances. Binary labels: 1 = change, 0 = no change.

Official pair-level split: **445 train / 64 val / 128 test**.

## Obtaining it

```bash
python scripts/download_levir.py      # ~2.5 GB, resumable
python scripts/prepare_tiles.py       # 1024^2 -> 256^2 tiles
```

`download_levir.py` pulls the official train/val/test archives from the
Hugging Face mirror `satellite-image-deep-learning/LEVIR-CD`, which preserves
the official split. Re-running skips completed files.

## Directory structure

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

## Leakage

Tiling happens strictly **within** each official split. A scene's 16 tiles all
stay in that scene's split, so no tile can appear in both train and test. The
split boundary is the scene, and we never cross it.

## Licensing

LEVIR-CD imagery is derived from Google Earth and is subject to Google Earth
terms of use. It is used here for academic research only.

## Units

LEVIR-CD ships plain PNGs with **no georeferencing and no documented GSD**.
The published 0.5 m/px figure describes the source imagery but is not carried
in the files. The inference engine therefore reports pixel counts and
percentages, and leaves `area_m2` as `null` unless a real GSD is supplied by
the caller. We do not fabricate ground areas.
