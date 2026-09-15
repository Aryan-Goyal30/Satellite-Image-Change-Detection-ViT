"""Pre-extract fixed-size region crops for the Stage 3B-2 direction classifier.

Reading two 1024x1024 PNGs per region inside the training loop would decode the
same scene once per region, so crops are extracted once into memory-mapped
uint8 arrays and the training loop just indexes them.

Ratified temporal ordering (a project decision, not an author-documented
property of S2Looking - see docs/S2LOOKING_TEMPORAL_SEMANTICS.md):

    BEFORE = Image2     AFTER = Image1
    channels 0:3 = BEFORE RGB, channels 3:6 = AFTER RGB

Region geometry comes from the Stage 3B-1 targets (region_labels.csv), which
were produced by the production region extractor, so no second region
implementation exists.

Crop geometry
-------------
1. take the region bbox (x, y, w, h)
2. expand by CONTEXT_FRACTION of w and h on EACH side -> 1.5x linear size at 0.25
3. square it off around the bbox centre, so the resize does not distort aspect
4. clamp into the scene, shifting rather than shrinking where possible
5. resize to CROP_SIZE x CROP_SIZE (bilinear)

MIXED regions are excluded: they are genuine ground truth but not a direction.

Outputs (under data/, never committed):
    data/s2looking/crops/<split>_x.npy     uint8 (N, 128, 128, 6)
    data/s2looking/crops/<split>_y.npy     int64 (N,)   0=construction 1=demolition
    data/s2looking/crops/<split>_meta.csv  scene_id, region_id, target, area_px
    data/s2looking/crops/manifest.json

Usage:  python scripts/build_s2looking_crops.py [--limit N]
"""
import argparse
import csv
import hashlib
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config  # noqa: E402
from src.domains.built_environment.data.s2looking import (  # noqa: E402
    AFTER_IMAGE_DIR, BEFORE_IMAGE_DIR, CONSTRUCTION, DEMOLITION,
    TEMPORAL_ORDERING)
# Canonical crop protocol. This script consumes the production module; the
# dependency never runs the other way (production must not import scripts/).
from src.domains.built_environment.direction.crops import (  # noqa: E402
    CONTEXT_FRACTION, CROP_SIZE, MIN_SIDE, crop_box, crop_image)

DATASET_DIR = os.path.join(config.S2LOOKING_RAW, "S2Looking")
CROPS_DIR = os.path.join(config.S2LOOKING_RAW, "crops")
CSV_PATH = os.path.join(config.S2LOOKING_META, "region_labels.csv")

SPLITS = ("train", "val", "test")
CLASSES = (CONSTRUCTION, DEMOLITION)          # index 0, 1
# CROP_SIZE / CONTEXT_FRACTION / MIN_SIDE and the crop geometry itself are
# imported from src/domains/built_environment/direction/crops.py, so training
# and inference cannot drift apart.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N regions per split (smoke run)")
    args = ap.parse_args()

    if not os.path.exists(CSV_PATH):
        print(f"NOT FOUND: {CSV_PATH}")
        return 1
    os.makedirs(CROPS_DIR, exist_ok=True)

    rows = [r for r in csv.DictReader(open(CSV_PATH, newline="", encoding="utf-8"))
            if r["target"] in CLASSES]

    manifest = {
        "crop_size": CROP_SIZE,
        "context_fraction": CONTEXT_FRACTION,
        "min_side_px": MIN_SIDE,
        "channels": "0:3 = BEFORE RGB, 3:6 = AFTER RGB",
        "classes": {CONSTRUCTION: 0, DEMOLITION: 1},
        "mixed_excluded": True,
        "temporal_ordering": TEMPORAL_ORDERING,
        "source_csv": os.path.relpath(CSV_PATH, config.ROOT),
        "splits": {},
    }

    for split in SPLITS:
        subset = [r for r in rows if r["split"] == split]
        # Deterministic order: scene id then region id, so a scene is opened once.
        subset.sort(key=lambda r: (int(r["scene_id"]), int(r["region_id"])))
        if args.limit:
            subset = subset[:args.limit]
        n = len(subset)
        print(f"[{split}] {n} regions", flush=True)

        x_path = os.path.join(CROPS_DIR, f"{split}_x.npy")
        y_path = os.path.join(CROPS_DIR, f"{split}_y.npy")
        X = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.uint8,
                                      shape=(n, CROP_SIZE, CROP_SIZE, 6))
        y = np.zeros(n, dtype=np.int64)

        meta_rows = []
        current_scene, before_img, after_img = None, None, None
        for i, r in enumerate(subset):
            if i and i % 2500 == 0:
                print(f"  {split}: {i}/{n}", flush=True)
            scene = r["scene_id"]
            if scene != current_scene:
                d = os.path.join(DATASET_DIR, split)
                if before_img is not None:
                    before_img.close(); after_img.close()
                before_img = Image.open(os.path.join(d, BEFORE_IMAGE_DIR, scene + ".png")).convert("RGB")
                after_img = Image.open(os.path.join(d, AFTER_IMAGE_DIR, scene + ".png")).convert("RGB")
                current_scene = scene
            box = crop_box(int(r["bbox_x"]), int(r["bbox_y"]),
                           int(r["bbox_w"]), int(r["bbox_h"]),
                           before_img.width, before_img.height)
            X[i, :, :, 0:3] = crop_image(before_img, box)
            X[i, :, :, 3:6] = crop_image(after_img, box)
            y[i] = CLASSES.index(r["target"])
            meta_rows.append({"scene_id": scene, "region_id": r["region_id"],
                              "target": r["target"], "area_px": r["area_px"]})
        if before_img is not None:
            before_img.close(); after_img.close()
        X.flush()
        del X
        np.save(y_path, y)

        with open(os.path.join(CROPS_DIR, f"{split}_meta.csv"), "w",
                  newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["scene_id", "region_id", "target", "area_px"])
            w.writeheader()
            w.writerows(meta_rows)

        counts = {c: int((y == i).sum()) for i, c in enumerate(CLASSES)}
        manifest["splits"][split] = {
            "regions": n,
            "scenes": len({m["scene_id"] for m in meta_rows}),
            "counts": counts,
            "x_bytes": os.path.getsize(x_path),
            "x_sha256": hashlib.sha256(open(x_path, "rb").read()).hexdigest()
            if os.path.getsize(x_path) < (2 << 30) else None,
        }
        print(f"  {split}: {counts}", flush=True)

    json.dump(manifest, open(os.path.join(CROPS_DIR, "manifest.json"), "w",
                             encoding="utf-8"), indent=2)
    print("\n[write]", os.path.join(CROPS_DIR, "manifest.json"))
    for split, m in manifest["splits"].items():
        print(f"  {split:5s} regions={m['regions']:6d} scenes={m['scenes']:5d} {m['counts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
