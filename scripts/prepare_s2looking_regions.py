"""Generate region-level construction/demolition targets from S2Looking.

Stage 3B-1 data preparation. This script reads the extracted dataset, derives
region-level ground truth, runs the label-quality checks and writes machine
readable metadata. It trains nothing, changes no production behaviour and is
not imported by the engine, the CLI or the UI.

Layout verified on the released archive:

    data/s2looking/S2Looking/{train,val,test}/
        Image1/<id>.png     before image, 1024x1024 RGB
        Image2/<id>.png     after image,  1024x1024 RGB
        label/<id>.png      combined change mask (grayscale 0/255)
        label1/<id>.png     newly built  -> construction (BLUE channel, 0/255)
        label2/<id>.png     demolished   -> demolition   (RED channel, 0/255)

Outputs (all under data/, never committed):

    data/s2looking/metadata/region_labels.csv
    data/s2looking/metadata/dataset_stats.json
    data/s2looking/metadata/quality_checks.json

Usage:  python scripts/prepare_s2looking_regions.py [--limit N]
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config  # noqa: E402
from src.domains.built_environment.data.s2looking import (  # noqa: E402
    AFTER_IMAGE_DIR, BEFORE_IMAGE_DIR, CONSTRUCTION, CONSTRUCTION_LABEL_DIR,
    DEFAULT_MIN_AREA_PX, DEMOLITION, DEMOLITION_LABEL_DIR, MIXED,
    PURITY_THRESHOLD, TEMPORAL_ORDERING, binarize, region_targets,
    split_integrity, summarize_regions)

DATASET_DIR = os.path.join(config.S2LOOKING_RAW, "S2Looking")
SPLITS = ("train", "val", "test")
EXPECTED_SCENES = {"train": 3500, "val": 500, "test": 1000}
KINDS = ("Image1", "Image2", "label", "label1", "label2")

CSV_FIELDS = [
    "scene_id", "split", "region_id", "target", "area_px",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h", "centroid_x", "centroid_y",
    "construction_pixels", "demolition_pixels", "overlap_pixels", "purity",
]

#: A component covering more than this fraction of the scene is reported for
#: inspection. It is not filtered out - only flagged.
LARGE_COMPONENT_FRACTION = 0.25


def scene_ids(split):
    d = os.path.join(DATASET_DIR, split, "label1")
    return sorted((f[:-4] for f in os.listdir(d) if f.endswith(".png")), key=int)


def read_mask(split, kind, scene):
    path = os.path.join(DATASET_DIR, split, kind, scene + ".png")
    with Image.open(path) as im:
        return binarize(np.array(im))


def image_size(split, kind, scene):
    """Lazy header read - no pixel decode."""
    path = os.path.join(DATASET_DIR, split, kind, scene + ".png")
    with Image.open(path) as im:
        return im.size


def raw_values(split, kind, scene):
    path = os.path.join(DATASET_DIR, split, kind, scene + ".png")
    with Image.open(path) as im:
        return np.unique(np.array(im)).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N scenes per split (smoke run)")
    args = ap.parse_args()

    if not os.path.isdir(DATASET_DIR):
        print(f"NOT FOUND: {DATASET_DIR}\nRun scripts/download_s2looking.py first.")
        return 1

    scenes_by_split = {s: scene_ids(s) for s in SPLITS}
    if args.limit:
        scenes_by_split = {s: v[:args.limit] for s, v in scenes_by_split.items()}

    issues = {
        "image_dimension_mismatch": [],
        "label_dimension_mismatch": [],
        "unexpected_label_values": [],
        "union_differs_from_combined_label": [],
        "unreadable": [],
        "empty_change_scenes": [],
        "scenes_without_regions": [],
        "large_components": [],
    }
    overlap_scenes = 0
    overlap_pixels_total = 0
    value_counter = Counter()
    records = []

    for split in SPLITS:
        ids = scenes_by_split[split]
        print(f"[{split}] {len(ids)} scenes", flush=True)
        for i, scene in enumerate(ids, 1):
            if i % 250 == 0:
                print(f"  {split}: {i}/{len(ids)}", flush=True)
            try:
                # Read through the ratified mapping, never a hard-coded folder:
                # construction = label1, demolition = label2, before = Image2.
                l1 = read_mask(split, CONSTRUCTION_LABEL_DIR, scene)
                l2 = read_mask(split, DEMOLITION_LABEL_DIR, scene)
                combined = read_mask(split, "label", scene)
                size1 = image_size(split, BEFORE_IMAGE_DIR, scene)
                size2 = image_size(split, AFTER_IMAGE_DIR, scene)
            except Exception as e:
                issues["unreadable"].append(f"{split}/{scene}: {type(e).__name__}: {e}")
                continue

            # --- dimension checks -------------------------------------------
            if size1 != size2:
                issues["image_dimension_mismatch"].append(
                    f"{split}/{scene}: before({BEFORE_IMAGE_DIR}) {size1} vs "
                    f"after({AFTER_IMAGE_DIR}) {size2}")
            if not (l1.shape == l2.shape == combined.shape
                    == (size1[1], size1[0])):
                issues["label_dimension_mismatch"].append(
                    f"{split}/{scene}: label1 {l1.shape} label2 {l2.shape} "
                    f"label {combined.shape} image {(size1[1], size1[0])}")
                continue

            # --- value checks (sampled: full read of every raster is costly) --
            if i <= 50:
                for kind in ("label1", "label2", "label"):
                    vals = raw_values(split, kind, scene)
                    value_counter.update(vals)
                    if set(vals) - {0, 255}:
                        issues["unexpected_label_values"].append(
                            f"{split}/{scene}/{kind}: {vals[:8]}")

            # --- cross-check against the bundled combined mask ---------------
            union = l1 | l2
            if not np.array_equal(union, combined):
                issues["union_differs_from_combined_label"].append(
                    f"{split}/{scene}: |l1|l2|={int(union.sum())} "
                    f"|label|={int(combined.sum())}")

            # --- overlap between the two annotation maps ---------------------
            overlap = int(np.count_nonzero(l1 & l2))
            if overlap:
                overlap_scenes += 1
                overlap_pixels_total += overlap

            if not union.any():
                issues["empty_change_scenes"].append(f"{split}/{scene}")

            # --- regions ------------------------------------------------------
            regions = region_targets(l1, l2)
            if not regions:
                issues["scenes_without_regions"].append(f"{split}/{scene}")
            area_limit = union.size * LARGE_COMPONENT_FRACTION
            for r in regions:
                if r["area_px"] > area_limit:
                    issues["large_components"].append(
                        f"{split}/{scene}/r{r['region_id']}: {r['area_px']} px "
                        f"({100 * r['area_px'] / union.size:.1f}% of scene)")
                records.append({"scene_id": scene, "split": split, **r})

    # ---------------------------------------------------------------- outputs
    os.makedirs(config.S2LOOKING_META, exist_ok=True)

    csv_path = os.path.join(config.S2LOOKING_META, "region_labels.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in records:
            w.writerow({k: r[k] for k in CSV_FIELDS})
    print(f"\n[write] {csv_path}  ({len(records)} regions)")

    stats = summarize_regions(records, scenes_by_split=scenes_by_split)
    stats["dataset"] = {
        "name": "S2Looking",
        "root": os.path.relpath(DATASET_DIR, config.ROOT),
        "label1_meaning": "newly built -> construction (blue channel, 0/255)",
        "label2_meaning": "demolished -> demolition (red channel, 0/255)",
        "scene_size": [1024, 1024],
        "expected_scenes": EXPECTED_SCENES,
        "limit_applied": args.limit,
        "temporal_ordering": TEMPORAL_ORDERING,
    }
    stats_path = os.path.join(config.S2LOOKING_META, "dataset_stats.json")
    json.dump(stats, open(stats_path, "w", encoding="utf-8"), indent=2)
    print(f"[write] {stats_path}")

    integrity = split_integrity(scenes_by_split)
    integrity["counts_match_official"] = (
        {s: len(v) for s, v in scenes_by_split.items()} == EXPECTED_SCENES
        if not args.limit else None)

    # Determinism: recompute a sample of scenes and require identical output.
    determinism_ok = True
    for split in SPLITS:
        for scene in scenes_by_split[split][:5]:
            l1 = read_mask(split, "label1", scene)
            l2 = read_mask(split, "label2", scene)
            if region_targets(l1, l2) != region_targets(l1, l2):
                determinism_ok = False

    quality = {
        "parameters": {
            "purity_threshold": PURITY_THRESHOLD,
            "min_area_px": DEFAULT_MIN_AREA_PX,
            "connectivity": 8,
            "large_component_fraction": LARGE_COMPONENT_FRACTION,
        },
        "temporal_ordering": TEMPORAL_ORDERING,
        "split_integrity": integrity,
        "deterministic_region_ids": determinism_ok,
        "label_value_histogram": {str(k): v for k, v in sorted(value_counter.items())},
        "annotation_overlap": {
            "scenes_with_overlap": overlap_scenes,
            "overlap_pixels_total": overlap_pixels_total,
        },
        "issue_counts": {k: len(v) for k, v in issues.items()},
        "issues": {k: v[:25] for k, v in issues.items()},
    }
    quality_path = os.path.join(config.S2LOOKING_META, "quality_checks.json")
    json.dump(quality, open(quality_path, "w", encoding="utf-8"), indent=2)
    print(f"[write] {quality_path}")

    # ---------------------------------------------------------------- summary
    o = stats["overall"]
    print("\n=== overall ===")
    print(f"  scenes      : {o['scenes']}")
    print(f"  regions     : {o['regions']}")
    print(f"  construction: {o[CONSTRUCTION]:6d}  ({o[CONSTRUCTION + '_pct']}%)")
    print(f"  demolition  : {o[DEMOLITION]:6d}  ({o[DEMOLITION + '_pct']}%)")
    print(f"  mixed       : {o[MIXED]:6d}  ({o[MIXED + '_pct']}%)")
    print("\n=== per split ===")
    for split, e in stats["per_split"].items():
        print(f"  {split:6s} scenes={e['scenes']:5d} regions={e['regions']:6d} "
              f"c={e[CONSTRUCTION]:6d} d={e[DEMOLITION]:5d} m={e[MIXED]:5d}")
    print("\n=== gate ===")
    print(f"  demolition {stats['gate']['demolition_pct']}% vs gate "
          f"{stats['gate']['gate_pct']}% -> "
          f"{'PASS' if stats['gate']['demolition_sufficiently_represented'] else 'BELOW GATE'}")
    print("\n=== issues ===")
    for k, v in quality["issue_counts"].items():
        print(f"  {k:35s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
