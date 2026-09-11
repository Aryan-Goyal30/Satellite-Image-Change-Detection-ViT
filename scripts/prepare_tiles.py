"""Tile LEVIR-CD 1024x1024 scenes into 256x256 patches, split by split.

LEVIR-CD ships an OFFICIAL pair-level split (445 train / 64 val / 128 test).
We tile strictly *within* each split, so a tile can never move between splits.
That is what guarantees no train/test leakage: the split boundary is the scene,
and we never cross it.

Input  (after scripts/download_levir.py):
    data/levir_cd/{train,val,test}/{A,B,label}/*.png      1024x1024

Output:
    data/levir_cd_tiles/{train,val,test}/{A,B,label}/<scene>_r<i>_c<j>.png   256x256
    data/levir_cd_tiles/stats.json

Usage:  python scripts/prepare_tiles.py [--tile 256]
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "levir_cd")
DST = os.path.join(ROOT, "data", "levir_cd_tiles")
SPLITS = ("train", "val", "test")
SUBDIRS = ("A", "B", "label")


def resolve_split(split):
    """Locate a split's images and return (source_dir, filenames).

    Two layouts are supported, because LEVIR-CD mirrors differ:

      nested : <root>/<split>/{A,B,label}/*.png
      flat   : <root>/{A,B,label}/<split>_*.png     <-- the HF mirror we use

    In the FLAT layout all three splits share one directory and the official
    split is encoded in the filename prefix (train_*, val_*, test_*). That
    prefix is the official assignment shipped with the dataset, so selecting on
    it reproduces the official 445/64/128 split exactly - it does not invent one.
    """
    # nested layout
    for cand in (os.path.join(SRC, split), os.path.join(SRC, "LEVIR-CD", split)):
        if all(os.path.isdir(os.path.join(cand, s)) for s in SUBDIRS):
            names = sorted(f for f in os.listdir(os.path.join(cand, "A"))
                           if f.lower().endswith(".png"))
            if names:
                return cand, names

    # flat layout, split taken from the filename prefix
    if all(os.path.isdir(os.path.join(SRC, s)) for s in SUBDIRS):
        names = sorted(f for f in os.listdir(os.path.join(SRC, "A"))
                       if f.lower().endswith(".png")
                       and f.lower().startswith(split + "_"))
        if names:
            return SRC, names

    return None, []


EXPECTED_SCENES = {"train": 445, "val": 64, "test": 128}


def tile_split(split, tile):
    src, names = resolve_split(split)
    if src is None or not names:
        raise SystemExit(
            f"Could not find images for split '{split}' under {SRC}.\n"
            f"Expected either {SRC}\\{split}\\{{A,B,label}}\\*.png\n"
            f"or {SRC}\\{{A,B,label}}\\{split}_*.png\n"
            f"Run: python scripts/download_levir.py"
        )
    exp = EXPECTED_SCENES[split]
    if len(names) != exp:
        print(f"  !! WARNING: {split} has {len(names)} scenes, official split "
              f"has {exp}. Results will not be comparable to published numbers.")
    else:
        print(f"  {split}: {len(names)} scenes (matches official split)")
    for s in SUBDIRS:
        os.makedirs(os.path.join(DST, split, s), exist_ok=True)

    n_tiles = 0
    n_pos = 0           # tiles containing at least one change pixel
    change_px = 0
    total_px = 0

    for k, name in enumerate(names, 1):
        stem = os.path.splitext(name)[0]
        imgs = {}
        for s in SUBDIRS:
            p = os.path.join(src, s, name)
            if not os.path.exists(p):
                raise SystemExit(f"Missing {p} - dataset incomplete.")
            imgs[s] = Image.open(p)

        w, h = imgs["A"].size
        arrs = {"A": imgs["A"].convert("RGB"), "B": imgs["B"].convert("RGB"),
                "label": imgs["label"].convert("L")}

        for r in range(0, h - tile + 1, tile):
            for c in range(0, w - tile + 1, tile):
                box = (c, r, c + tile, r + tile)
                lab = arrs["label"].crop(box)
                lab_np = np.array(lab)
                # LEVIR labels are 0 / 255. Binarise defensively.
                lab_bin = (lab_np > 127).astype(np.uint8)

                out_name = f"{stem}_r{r // tile}_c{c // tile}.png"
                arrs["A"].crop(box).save(os.path.join(DST, split, "A", out_name), optimize=False)
                arrs["B"].crop(box).save(os.path.join(DST, split, "B", out_name), optimize=False)
                Image.fromarray(lab_bin * 255).save(
                    os.path.join(DST, split, "label", out_name), optimize=False)

                n_tiles += 1
                pos = int(lab_bin.sum())
                change_px += pos
                total_px += lab_bin.size
                if pos > 0:
                    n_pos += 1

        if k % 25 == 0 or k == len(names):
            print(f"  {split}: {k}/{len(names)} scenes -> {n_tiles} tiles", flush=True)

    return {
        "scenes": len(names),
        "tiles": n_tiles,
        "tiles_with_change": n_pos,
        "tiles_with_change_pct": round(100 * n_pos / max(n_tiles, 1), 2),
        "change_pixel_pct": round(100 * change_px / max(total_px, 1), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", type=int, default=256)
    args = ap.parse_args()

    # --- leakage check BEFORE doing any work -------------------------------
    # Assert the three splits are disjoint at the scene level. Tiles inherit
    # their scene's split, so disjoint scenes => disjoint tiles => no leakage.
    scene_sets = {s: set(resolve_split(s)[1]) for s in SPLITS}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            overlap = scene_sets[a] & scene_sets[b]
            if overlap:
                raise SystemExit(
                    f"LEAKAGE: {len(overlap)} scenes appear in both '{a}' and "
                    f"'{b}', e.g. {sorted(overlap)[:5]}. Refusing to continue.")
    print(f"leakage check: train/val/test scene sets are disjoint "
          f"({'/'.join(str(len(scene_sets[s])) for s in SPLITS)} scenes)\n")

    stats = {"tile_size": args.tile, "splits": {},
             "scene_counts": {s: len(scene_sets[s]) for s in SPLITS},
             "split_source": "official LEVIR-CD split (filename prefix)",
             "leakage_check": "passed - splits disjoint at scene level"}
    for split in SPLITS:
        print(f"[{split}]", flush=True)
        stats["splits"][split] = tile_split(split, args.tile)

    os.makedirs(DST, exist_ok=True)
    with open(os.path.join(DST, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print("\n=== Tiling summary ===")
    for split, s in stats["splits"].items():
        print(f"{split:>5}: {s['scenes']:>4} scenes -> {s['tiles']:>5} tiles | "
              f"{s['tiles_with_change_pct']:>5.1f}% tiles contain change | "
              f"{s['change_pixel_pct']:>5.2f}% of pixels are change")
    print(f"\nWrote {os.path.join(DST, 'stats.json')}")


if __name__ == "__main__":
    main()
