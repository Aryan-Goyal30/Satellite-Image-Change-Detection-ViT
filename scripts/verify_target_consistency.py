"""Verify generated targets are semantically consistent with the ratified ordering.

Stage 3B-1 gate check, run after regeneration. The project has ratified:

    BEFORE = Image2      AFTER = Image1
    CONSTRUCTION = label1    DEMOLITION = label2

If that mapping is coherent, then for the generated region targets:

    construction regions -> the building stands in the AFTER image
    demolition regions   -> the building stood in the BEFORE image

This script tests exactly that, on the regenerated region_labels.csv, using the
only building-presence feature that passed a known-answer control on LEVIR-CD
(gradient-orientation concentration; see docs/S2LOOKING_TEMPORAL_SEMANTICS.md).
Raw texture energy is deliberately NOT used - it failed that control.

Measures and reports only. Changes no labels and trains nothing.

Usage:  python scripts/verify_target_consistency.py [--per-class 250]
"""
import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config  # noqa: E402
from src.domains.built_environment.data.s2looking import (  # noqa: E402
    AFTER_IMAGE_DIR, BEFORE_IMAGE_DIR, CONSTRUCTION, CONSTRUCTION_LABEL_DIR,
    DEMOLITION, DEMOLITION_LABEL_DIR, TEMPORAL_ORDERING, binarize)

DATASET_DIR = os.path.join(config.S2LOOKING_RAW, "S2Looking")
CSV_PATH = os.path.join(config.S2LOOKING_META, "region_labels.csv")
OUT_PATH = os.path.join(config.S2LOOKING_META, "target_consistency.json")
MIN_AREA = 1000
PASS_FRACTION = 0.60


def gray(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), np.float32)


def orientation_concentration(g, comp, bbox):
    x, y, w, h = bbox
    sub = g[y:y + h, x:x + w]
    m = comp[y:y + h, x:x + w]
    gx = cv2.Sobel(sub, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(sub, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.arctan2(gy, gx)[m] + np.pi) % np.pi
    hist, _ = np.histogram(ang, bins=18, range=(0, np.pi), weights=mag[m])
    return float(hist.max() / (hist.sum() + 1e-6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=250)
    args = ap.parse_args()

    if not os.path.exists(CSV_PATH):
        print(f"NOT FOUND: {CSV_PATH}")
        return 1

    rows = [r for r in csv.DictReader(open(CSV_PATH, newline="", encoding="utf-8"))
            if int(r["area_px"]) >= MIN_AREA]
    # Deterministic selection: largest regions first, grouped by scene so each
    # scene is opened once.
    wanted = {}
    for target in (CONSTRUCTION, DEMOLITION):
        pool = sorted((r for r in rows if r["target"] == target),
                      key=lambda r: (-int(r["area_px"]), r["split"],
                                     int(r["scene_id"]), int(r["region_id"])))
        wanted[target] = pool[:args.per_class]

    by_scene = {}
    for target, recs in wanted.items():
        for r in recs:
            by_scene.setdefault((r["split"], r["scene_id"]), []).append((target, r))

    acc = {CONSTRUCTION: [], DEMOLITION: []}
    for (split, scene), recs in sorted(by_scene.items()):
        d = os.path.join(DATASET_DIR, split)
        l1 = binarize(np.array(Image.open(os.path.join(d, CONSTRUCTION_LABEL_DIR, scene + ".png"))))
        l2 = binarize(np.array(Image.open(os.path.join(d, DEMOLITION_LABEL_DIR, scene + ".png"))))
        g_before = gray(os.path.join(d, BEFORE_IMAGE_DIR, scene + ".png"))
        g_after = gray(os.path.join(d, AFTER_IMAGE_DIR, scene + ".png"))
        n, lab, st, _ = cv2.connectedComponentsWithStats(
            (l1 | l2).astype(np.uint8), connectivity=8)
        # Rebuild the same ordering region_targets used: descending area.
        kept = sorted(((int(st[i, cv2.CC_STAT_AREA]), i) for i in range(1, n)
                       if int(st[i, cv2.CC_STAT_AREA]) >= config.DEFAULT_MIN_AREA_PX),
                      key=lambda t: -t[0])
        index = {rid: label for rid, (_a, label) in enumerate(kept, 1)}
        for target, r in recs:
            label_value = index.get(int(r["region_id"]))
            if label_value is None:
                continue
            comp = lab == label_value
            ys, xs = np.where(comp)
            if not len(ys):
                continue
            bb = (int(xs.min()), int(ys.min()),
                  int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
            acc[target].append((orientation_concentration(g_before, comp, bb),
                                orientation_concentration(g_after, comp, bb)))

    result = {"ratified_ordering": TEMPORAL_ORDERING,
              "feature": "gradient-orientation concentration (passed LEVIR control)",
              "min_area_px": MIN_AREA, "pass_fraction": PASS_FRACTION,
              "classes": {}}
    ok = True
    print(f"BEFORE = {BEFORE_IMAGE_DIR}   AFTER = {AFTER_IMAGE_DIR}")
    print(f"expectation: construction -> building in AFTER; demolition -> building in BEFORE\n")
    for target, pairs in acc.items():
        if not pairs:
            result["classes"][target] = None
            ok = False
            continue
        a = np.array(pairs)
        frac_after = float((a[:, 1] > a[:, 0]).mean())
        expected_in = "after" if target == CONSTRUCTION else "before"
        passes = frac_after >= PASS_FRACTION if target == CONSTRUCTION \
            else (1 - frac_after) >= PASS_FRACTION
        ok &= passes
        result["classes"][target] = {
            "regions": len(pairs),
            "mean_before": round(float(a[:, 0].mean()), 4),
            "mean_after": round(float(a[:, 1].mean()), 4),
            "fraction_building_in_after": round(frac_after, 4),
            "expected_building_in": expected_in,
            "passes": bool(passes),
        }
        print(f"{target.upper():13s} n={len(pairs):4d}  "
              f"before={a[:,0].mean():.4f}  after={a[:,1].mean():.4f}  "
              f"frac(after>before)={frac_after:.3f}  "
              f"expected building in {expected_in.upper()}  "
              f"-> {'CONSISTENT' if passes else 'INCONSISTENT'}")

    result["consistent"] = bool(ok)
    os.makedirs(config.S2LOOKING_META, exist_ok=True)
    json.dump(result, open(OUT_PATH, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n[write] {OUT_PATH}")
    print("\nVERDICT:", "CONSISTENT with the ratified ordering" if ok
          else "INCONSISTENT - do not proceed")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
