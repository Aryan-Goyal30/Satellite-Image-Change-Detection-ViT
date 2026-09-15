"""E3 evaluation at BOTH spatial supports, with E2 as the control.

The experiment asks whether supervising at TMF's native ~30 m support resolves
the region/topology failure. Answering that needs four measurements, not two:

    E2 @ 10 m   the frozen baseline
    E2 @ 30 m   THE CONTROL - E2's own predictions pooled to 30 m
    E3 @ 10 m   E3's retained 10 m output, for compatibility with E2
    E3 @ 30 m   E3's primary objective

Without "E2 @ 30 m" the comparison would confound two different things: whether
30 m SUPERVISION helps, and whether 30 m EVALUATION is simply an easier scoring
support. Any honest verdict has to separate them.

Thresholds are selected on VALIDATION only, separately per (model, support),
using the same 201-point argmax-F1 sweep, and frozen before test is read.

Usage:  python scripts/evaluate_environment_e3.py
"""
import csv
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config                                              # noqa: E402
from src.common.regions import extract_regions                      # noqa: E402
from src.domains.environment import region_metrics as rmetrics      # noqa: E402
from src.domains.environment import support as sp                   # noqa: E402
from src.domains.environment.data import sentinel2 as s2            # noqa: E402
from src.domains.environment.data.loader import (                   # noqa: E402
    EnvironmentChangeDataset, load_band_statistics)
from src.domains.environment.model import BAND_SETS, build_model    # noqa: E402
from src.eval.metrics import ConfusionAccumulator, ThresholdSweep   # noqa: E402

DATASET_DIR = os.path.join(config.DATA_DIR, "environment", "dataset_v24")
STATS_PATH = os.path.join(config.DATA_DIR, "environment",
                          "characterization_v24_train_bands.json")
OUT_DIR = os.path.join(config.OUTPUTS, "environment_e3")
MODELS = {
    "E2": (os.path.join(config.OUTPUTS, "environment_baseline"),
           "environment_sixband_best.pt"),
    "E3": (OUT_DIR, "environment_e3_best.pt"),
}
THRESHOLD_RULE = ("argmax F1 over a 201-point sweep of the VALIDATION "
                  "probability histogram, selected separately per (model, "
                  "support) and frozen before any test data is read")
FIXED_MAX, FIXED_GAMMA = 0.30, 0.6


@torch.no_grad()
def infer(code, split, device):
    """10 m probability maps plus 30 m pooled probability maps."""
    directory, name = MODELS[code]
    payload = torch.load(os.path.join(directory, name), map_location=device,
                         weights_only=False)
    bands = tuple(payload.get("bands") or BAND_SETS["six_band"])
    stats = load_band_statistics(STATS_PATH, bands)
    model = build_model(bands=bands, encoder=payload.get("encoder", "resnet34"),
                        pretrained=False)
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    dataset = EnvironmentChangeDataset(DATASET_DIR, split, stats, bands=bands)
    amp = torch.bfloat16 if (device.type == "cuda"
                             and torch.cuda.is_bf16_supported()) else torch.float16
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    fine, coarse = {}, {}
    for batch in loader:
        with torch.autocast("cuda", dtype=amp, enabled=device.type == "cuda"):
            logits = model(batch["before"].to(device), batch["after"].to(device))
        logits = logits.float()
        p10 = torch.sigmoid(logits).cpu().numpy()
        p30 = torch.sigmoid(sp.pool_mean_torch(logits)).cpu().numpy()
        for i, sid in enumerate(batch["sample_id"]):
            fine[sid] = p10[i, 0]
            coarse[sid] = p30[i, 0]
    return fine, coarse, [r["sample_id"] for r in dataset.records], payload


def labels(split, ids):
    fine, coarse = {}, {}
    for sid in ids:
        label = np.load(os.path.join(DATASET_DIR, f"{sid}_label.npy")) > 0.5
        fine[sid] = label
        coarse[sid] = sp.label_30m(label) > 0.5
    return fine, coarse


def pixel_metrics(probs, truth, ids, threshold):
    acc = ConfusionAccumulator(threshold=threshold)
    sweep = ThresholdSweep()
    for sid in ids:
        p = torch.from_numpy(probs[sid])
        t = torch.from_numpy(truth[sid].astype(np.float32))
        acc.update(p, t)
        sweep.update(p, t)
    out = acc.compute()
    out["average_precision"] = sweep.average_precision()
    return out


def choose_threshold(probs, truth, ids):
    sweep = ThresholdSweep()
    for sid in ids:
        sweep.update(torch.from_numpy(probs[sid]),
                     torch.from_numpy(truth[sid].astype(np.float32)))
    return float(sweep.best_f1()["threshold"])


def components(mask, min_area):
    import cv2
    filtered, regions = extract_regions(mask.astype(np.uint8), min_area_px=min_area)
    if not regions:
        return [], []
    n, lab, _, _ = cv2.connectedComponentsWithStats(filtered.astype(np.uint8),
                                                    connectivity=8)
    masks = [lab == i for i in range(1, n)]
    return masks, [int(m.sum()) for m in masks]


def region_stats(probs, truth, ids, threshold, min_area):
    acc = rmetrics.RegionAccumulator(min_area_px=min_area)
    gt_sizes, pred_sizes = [], []
    for sid in ids:
        pred = probs[sid] >= threshold
        acc.update(pred, truth[sid])
        gt_sizes += components(truth[sid], min_area)[1]
        pred_sizes += components(pred, min_area)[1]
    out = acc.compute()
    out["gt_component_sizes_median"] = float(np.median(gt_sizes)) if gt_sizes else None
    out["pred_component_sizes_median"] = float(np.median(pred_sizes)) if pred_sizes else None
    out["gt_components_per_sample"] = round(len(gt_sizes) / len(ids), 2)
    out["pred_components_per_sample"] = round(len(pred_sizes) / len(ids), 2)
    out["pred_over_gt_component_ratio"] = (round(len(pred_sizes) / len(gt_sizes), 4)
                                           if gt_sizes else None)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    infers, payloads = {}, {}
    for code in MODELS:
        v10, v30, vids, payload = infer(code, "val", device)
        t10, t30, tids, _ = infer(code, "test", device)
        infers[code] = {"val": (v10, v30, vids), "test": (t10, t30, tids)}
        payloads[code] = payload

    vids = infers["E2"]["val"][2]
    tids = infers["E2"]["test"][2]
    vl10, vl30 = labels("val", vids)
    tl10, tl30 = labels("test", tids)

    # --- thresholds: validation only, one per (model, support) --------------
    thresholds = {}
    for code in MODELS:
        v10, v30, _ = infers[code]["val"]
        thresholds[f"{code}_10m"] = choose_threshold(v10, vl10, vids)
        thresholds[f"{code}_30m"] = choose_threshold(v30, vl30, vids)
    print("[thresholds] selected on validation only:",
          json.dumps({k: round(v, 3) for k, v in thresholds.items()}))

    validation, test, regions = {}, {}, {}
    for code in MODELS:
        v10, v30, _ = infers[code]["val"]
        t10, t30, _ = infers[code]["test"]
        for tag, vp, vt, tp_, tt, truth_v, truth_t, min_area in (
                ("10m", v10, vl10, t10, tl10, vl10, tl10, rmetrics.MIN_AREA_PX),
                ("30m", v30, vl30, t30, tl30, vl30, tl30, sp.MIN_AREA_PX_30M)):
            key = f"{code}_{tag}"
            threshold = thresholds[key]
            validation[key] = pixel_metrics(vp, truth_v, vids, threshold)
            test[key] = pixel_metrics(tp_, truth_t, tids, threshold)
            regions[key] = region_stats(tp_, truth_t, tids, threshold, min_area)

    # --- per-sample comparison ---------------------------------------------
    with open(os.path.join(DATASET_DIR, "manifest.json"), encoding="utf-8") as fh:
        records = {r["sample_id"]: r for r in json.load(fh)["samples"]}
    try:
        with open(os.path.join(config.OUTPUTS, "environment_failure_analysis",
                               "per_sample_errors.json"), encoding="utf-8") as fh:
            taxonomy = {r["sample_id"]: r["primary_category"] for r in json.load(fh)}
    except FileNotFoundError:
        taxonomy = {}

    def sample_iou(prob, truth, threshold):
        pred = prob >= threshold
        union = np.logical_or(pred, truth).sum()
        return float(np.logical_and(pred, truth).sum() / union) if union else 1.0

    def sample_recall(prob, truth, threshold):
        if not truth.any():
            return None
        pred = prob >= threshold
        return float(np.logical_and(pred, truth).sum() / truth.sum())

    rows = []
    for sid in tids:
        record = records[sid]
        e2_10, e2_30, _ = infers["E2"]["test"]
        e3_10, e3_30, _ = infers["E3"]["test"]
        row = {
            "sample_id": sid, "sample_type": record["sample_type"],
            "region": "SE Asia" if record["tmf_tile"] in ("N0_E110", "N10_E100") else "Amazon",
            "event_size": record.get("event_size"),
            "primary_category_e2": taxonomy.get(sid),
            "gt_px_10m": int(tl10[sid].sum()), "gt_px_30m": int(tl30[sid].sum()),
            "e2_px_10m": int((e2_10[sid] >= thresholds["E2_10m"]).sum()),
            "e3_px_10m": int((e3_10[sid] >= thresholds["E3_10m"]).sum()),
            "e3_px_30m": int((e3_30[sid] >= thresholds["E3_30m"]).sum()),
            "e2_px_30m": int((e2_30[sid] >= thresholds["E2_30m"]).sum()),
            "e2_iou_10m": round(sample_iou(e2_10[sid], tl10[sid], thresholds["E2_10m"]), 4),
            "e3_iou_10m": round(sample_iou(e3_10[sid], tl10[sid], thresholds["E3_10m"]), 4),
            "e2_iou_30m": round(sample_iou(e2_30[sid], tl30[sid], thresholds["E2_30m"]), 4),
            "e3_iou_30m": round(sample_iou(e3_30[sid], tl30[sid], thresholds["E3_30m"]), 4),
        }
        for tag, prob, truth, threshold in (
                ("e2_recall_10m", e2_10[sid], tl10[sid], thresholds["E2_10m"]),
                ("e3_recall_10m", e3_10[sid], tl10[sid], thresholds["E3_10m"]),
                ("e2_recall_30m", e2_30[sid], tl30[sid], thresholds["E2_30m"]),
                ("e3_recall_30m", e3_30[sid], tl30[sid], thresholds["E3_30m"])):
            value = sample_recall(prob, truth, threshold)
            row[tag] = round(value, 4) if value is not None else None
        row["area_ratio_e2_10m"] = (round(row["e2_px_10m"] / row["gt_px_10m"], 4)
                                    if row["gt_px_10m"] else None)
        row["area_ratio_e3_10m"] = (round(row["e3_px_10m"] / row["gt_px_10m"], 4)
                                    if row["gt_px_10m"] else None)
        row["area_ratio_e3_30m"] = (round(row["e3_px_30m"] / row["gt_px_30m"], 4)
                                    if row["gt_px_30m"] else None)
        row["area_ratio_e2_30m"] = (round(row["e2_px_30m"] / row["gt_px_30m"], 4)
                                    if row["gt_px_30m"] else None)
        row["delta_iou_10m"] = round(row["e3_iou_10m"] - row["e2_iou_10m"], 4)
        row["delta_iou_30m"] = round(row["e3_iou_30m"] - row["e2_iou_30m"], 4)
        rows.append(row)

    with open(os.path.join(OUT_DIR, "per_sample_comparison.csv"), "w", newline="",
              encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    positives = [r for r in rows if r["sample_type"] == "positive"]
    improved = [r for r in positives if r["delta_iou_30m"] > 0.10]
    worsened = [r for r in positives if r["delta_iou_30m"] < -0.10]
    unchanged = [r for r in positives if abs(r["delta_iou_30m"]) <= 0.10]

    area = {}
    for key in ("E2_10m", "E3_10m", "E2_30m", "E3_30m"):
        code, tag = key.split("_")
        gt = sum(r[f"gt_px_{tag}"] for r in rows)
        pred = sum(r[f"{code.lower()}_px_{tag}"] for r in rows)
        area[key] = {"gt_px": gt, "pred_px": pred,
                     "pred_over_gt": round(pred / max(gt, 1), 4)}

    payload = {
        "experiment": "phase4b-4-e3-native-30m-supervision",
        "dataset_version": "v24",
        "spatial_support": sp.DESCRIPTION,
        "threshold_rule": THRESHOLD_RULE, "thresholds": thresholds,
        "checkpoints": {c: payloads[c].get("epoch") for c in MODELS},
        "validation": validation, "test": test, "regions": regions,
        "area_analysis": area,
        "comparison_groups": {
            "e3_better_30m": [r["sample_id"] for r in improved],
            "e3_worse_30m": [r["sample_id"] for r in worsened],
            "unchanged_30m": [r["sample_id"] for r in unchanged],
        },
        "category_delta": {},
        "control_note": ("E2_30m is the control: E2's own predictions pooled to "
                         "30 m. Comparing E3_30m against E2_10m alone would "
                         "confound 30 m supervision with 30 m evaluation."),
        "comparability": {
            "apples_to_apples": ["E2_10m vs E3_10m", "E2_30m vs E3_30m"],
            "not_comparable": ["E2_10m vs E3_30m - different spatial support "
                               "and therefore a different physical target"],
        },
    }
    for category in sorted({r["primary_category_e2"] for r in positives
                            if r["primary_category_e2"]}):
        members = [r for r in positives if r["primary_category_e2"] == category]
        payload["category_delta"][category] = {
            "n": len(members),
            "mean_delta_iou_10m": round(float(np.mean([r["delta_iou_10m"] for r in members])), 4),
            "mean_delta_iou_30m": round(float(np.mean([r["delta_iou_30m"] for r in members])), 4),
            "mean_e2_iou_30m": round(float(np.mean([r["e2_iou_30m"] for r in members])), 4),
            "mean_e3_iou_30m": round(float(np.mean([r["e3_iou_30m"] for r in members])), 4),
        }

    with open(os.path.join(OUT_DIR, "test_evaluation.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    with open(os.path.join(OUT_DIR, "validation_evaluation.json"), "w", encoding="utf-8") as fh:
        json.dump({"thresholds": thresholds, "validation": validation}, fh, indent=2)
    with open(os.path.join(OUT_DIR, "topology_analysis.json"), "w", encoding="utf-8") as fh:
        json.dump(regions, fh, indent=2)
    with open(os.path.join(OUT_DIR, "area_analysis.json"), "w", encoding="utf-8") as fh:
        json.dump(area, fh, indent=2)

    header = "%-8s %9s %9s %9s %9s %9s %11s" % ("model", "P", "R", "F1", "IoU", "AP", "regionF1")
    print("\n" + header)
    print("-" * len(header))
    for key in ("E2_10m", "E3_10m", "E2_30m", "E3_30m"):
        m, r = test[key], regions[key]
        print("%-8s %9.4f %9.4f %9.4f %9.4f %9.4f %11.4f" % (
            key, m["precision"], m["recall"], m["f1"], m["iou"],
            m["average_precision"], r["region_f1"]))
    print("\n%-8s %10s %10s %10s %10s %12s" % (
        "model", "GT comp", "pred comp", "GT med", "pred med", "pred/GT"))
    for key in ("E2_10m", "E3_10m", "E2_30m", "E3_30m"):
        r = regions[key]
        print("%-8s %10d %10d %10.1f %10.1f %12.3f" % (
            key, r["ground_truth_regions"], r["predicted_regions"],
            r["gt_component_sizes_median"] or 0, r["pred_component_sizes_median"] or 0,
            r["pred_over_gt_component_ratio"] or 0))
    print("\narea pred/GT:", json.dumps({k: v["pred_over_gt"] for k, v in area.items()}))
    print("per-sample (positives, |dIoU30| > 0.10): improved %d, worsened %d, unchanged %d"
          % (len(improved), len(worsened), len(unchanged)))
    print(f"\n[artifacts] {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
