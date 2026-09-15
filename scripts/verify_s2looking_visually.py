"""Visual verification of the generated S2Looking region targets.

Stage 3B-1, step 8. Renders BEFORE | AFTER | GROUND TRUTH panels for
representative construction, demolition and mixed regions so a human can
confirm that a region labelled "construction" really does show a building
appearing, and one labelled "demolition" really does show one disappearing.

The ground-truth panel uses the dataset's own convention:
    construction (label1) -> blue
    demolition   (label2) -> red

Example selection is deterministic and is NOT used to tune labels: regions are
sorted by descending area and sampled at even intervals across the largest
candidates, which keeps them legible without hand-picking.

Figures are written under data/ and are therefore never committed.

Usage:  python scripts/verify_s2looking_visually.py [--per-class 5] [--test 3]
"""
import argparse
import csv
import os
import sys

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches          # noqa: E402
import matplotlib.pyplot as plt               # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config  # noqa: E402
from src.domains.built_environment.data.s2looking import (  # noqa: E402
    AFTER_IMAGE_DIR, BEFORE_IMAGE_DIR, CONSTRUCTION, CONSTRUCTION_LABEL_DIR,
    DEMOLITION, DEMOLITION_LABEL_DIR, MIXED, binarize)

DATASET_DIR = os.path.join(config.S2LOOKING_RAW, "S2Looking")
OUT_DIR = os.path.join(config.S2LOOKING_META, "verification")
CSV_PATH = os.path.join(config.S2LOOKING_META, "region_labels.csv")

#: candidates considered before even-interval sampling (largest regions first)
CANDIDATE_POOL = 150
CONTEXT = 0.6            # crop margin around the region bbox, as a fraction


def load_records():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("region_id", "area_px", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
                  "construction_pixels", "demolition_pixels", "overlap_pixels"):
            r[k] = int(r[k])
        r["purity"] = float(r["purity"])
        r["centroid_x"] = float(r["centroid_x"])
        r["centroid_y"] = float(r["centroid_y"])
    return rows


def pick(rows, target, splits, n):
    """Deterministic spread over the largest regions of one class."""
    pool = sorted((r for r in rows if r["target"] == target and r["split"] in splits),
                  key=lambda r: (-r["area_px"], r["split"], int(r["scene_id"]),
                                 r["region_id"]))[:CANDIDATE_POOL]
    if not pool:
        return []
    if len(pool) <= n:
        return pool
    step = len(pool) / n
    return [pool[int(i * step)] for i in range(n)]


def read_rgb(split, kind, scene):
    with Image.open(os.path.join(DATASET_DIR, split, kind, scene + ".png")) as im:
        return np.array(im.convert("RGB"))


def read_label(split, kind, scene):
    with Image.open(os.path.join(DATASET_DIR, split, kind, scene + ".png")) as im:
        return binarize(np.array(im))


def gt_rgb(l1, l2):
    """Colour-coded ground truth: blue = construction, red = demolition."""
    out = np.zeros(l1.shape + (3,), np.uint8)
    out[l1, 2] = 255
    out[l2, 0] = 255
    return out


def crop_box(rec, shape):
    h, w = shape
    x, y, bw, bh = rec["bbox_x"], rec["bbox_y"], rec["bbox_w"], rec["bbox_h"]
    mx, my = int(bw * CONTEXT) + 16, int(bh * CONTEXT) + 16
    return (max(0, x - mx), max(0, y - my),
            min(w, x + bw + mx), min(h, y + bh + my))


def render(rec, path):
    split, scene = rec["split"], rec["scene_id"]
    # Ratified ordering: BEFORE is Image2 and AFTER is Image1.
    before = read_rgb(split, BEFORE_IMAGE_DIR, scene)
    after = read_rgb(split, AFTER_IMAGE_DIR, scene)
    l1 = read_label(split, CONSTRUCTION_LABEL_DIR, scene)
    l2 = read_label(split, DEMOLITION_LABEL_DIR, scene)
    gt = gt_rgb(l1, l2)

    x0, y0, x1, y1 = crop_box(rec, l1.shape)
    panels = [("BEFORE", before), ("AFTER", after), ("GROUND TRUTH", gt)]

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.4))
    for col, (title, img) in enumerate(panels):
        ax = axes[0, col]
        ax.imshow(img)
        ax.add_patch(patches.Rectangle(
            (rec["bbox_x"], rec["bbox_y"]), rec["bbox_w"], rec["bbox_h"],
            fill=False, edgecolor="yellow", linewidth=1.6))
        ax.set_title(f"{title} (full scene)", fontsize=10)
        ax.axis("off")

        ax = axes[1, col]
        ax.imshow(img[y0:y1, x0:x1])
        ax.add_patch(patches.Rectangle(
            (rec["bbox_x"] - x0, rec["bbox_y"] - y0), rec["bbox_w"], rec["bbox_h"],
            fill=False, edgecolor="yellow", linewidth=1.6))
        ax.set_title(f"{title} (region)", fontsize=10)
        ax.axis("off")

    fig.suptitle(
        f"{rec['target'].upper()}   {split}/{scene}  region {rec['region_id']}   "
        f"area {rec['area_px']:,} px   purity {rec['purity']:.3f}   "
        f"construction px {rec['construction_pixels']:,} / "
        f"demolition px {rec['demolition_pixels']:,}\n"
        f"ground truth: blue = construction (label1), red = demolition (label2)   |   "
        f"BEFORE = {BEFORE_IMAGE_DIR}, AFTER = {AFTER_IMAGE_DIR} "
        f"(ratified project decision, not author-documented)",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=95)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=5)
    ap.add_argument("--test", type=int, default=3)
    args = ap.parse_args()

    if not os.path.exists(CSV_PATH):
        print(f"NOT FOUND: {CSV_PATH}\nRun scripts/prepare_s2looking_regions.py first.")
        return 1

    rows = load_records()
    os.makedirs(OUT_DIR, exist_ok=True)
    made = []

    for target in (CONSTRUCTION, DEMOLITION, MIXED):
        chosen = pick(rows, target, {"train", "val"}, args.per_class)
        print(f"{target:12s} train/val: requested {args.per_class}, got {len(chosen)}")
        for r in chosen:
            name = f"{target}_{r['split']}_{r['scene_id']}_r{r['region_id']}.png"
            render(r, os.path.join(OUT_DIR, name))
            made.append(name)

    for target in (CONSTRUCTION, DEMOLITION, MIXED):
        chosen = pick(rows, target, {"test"}, args.test)
        print(f"{target:12s} test     : requested {args.test}, got {len(chosen)}")
        for r in chosen:
            name = f"TEST_{target}_{r['scene_id']}_r{r['region_id']}.png"
            render(r, os.path.join(OUT_DIR, name))
            made.append(name)

    print(f"\n[write] {len(made)} figures -> {OUT_DIR}")
    for n in made:
        print("   ", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
