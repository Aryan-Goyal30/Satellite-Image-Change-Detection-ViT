"""Phase 4B-3 failure analysis of the frozen E2 six-band environmental model.

READ-ONLY. Nothing is trained, no threshold is re-selected, no dataset or
evaluation artifact is modified. Everything is written under
outputs/environment_failure_analysis/ with new filenames.

The predictions are regenerated from the frozen E2 checkpoint in eval mode at
the frozen threshold 0.910, and the script REFUSES TO CONTINUE unless the
aggregate metrics reproduce the recorded values exactly. Without that check the
analysis could silently describe a different model than the one reported.

Usage:  python scripts/analyze_environment_failures.py
"""
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config                                              # noqa: E402
from src.common.regions import extract_regions                      # noqa: E402
from src.domains.environment import region_metrics as rmetrics      # noqa: E402
from src.domains.environment.data import sentinel2 as s2            # noqa: E402
from src.domains.environment.data.loader import (                   # noqa: E402
    EnvironmentChangeDataset, load_band_statistics)
from src.domains.environment.model import build_model               # noqa: E402
from src.eval.metrics import ConfusionAccumulator, ThresholdSweep   # noqa: E402

DATASET_DIR = os.path.join(config.DATA_DIR, "environment", "dataset_v24")
STATS_PATH = os.path.join(config.DATA_DIR, "environment",
                          "characterization_v24_train_bands.json")
OUT_DIR = os.path.join(config.OUTPUTS, "environment_failure_analysis")
EXPERIMENTS = {
    "E2": (os.path.join(config.OUTPUTS, "environment_baseline"),
           "environment_sixband_best.pt"),
    "E0": (os.path.join(config.OUTPUTS, "environment_ablation", "E0_rgb"),
           "environment_best.pt"),
    "E1": (os.path.join(config.OUTPUTS, "environment_ablation", "E1_rgb_nir"),
           "environment_best.pt"),
}

#: The frozen operating point. Never re-selected here.
FROZEN = {"threshold": 0.910, "precision": 0.5505, "recall": 0.6472,
          "f1": 0.5950, "iou": 0.4235, "average_precision": 0.5670}

MIN_AREA_PX = rmetrics.MIN_AREA_PX      # 32, the project-wide component floor

# --------------------------------------------------------------- taxonomy
#: Operational thresholds. Fixed before the categories were counted, chosen from
#: the metric definitions rather than tuned to produce a tidy distribution.
T = {
    "strong_iou": 0.50,          # half the union overlapping is a strong detection
    "strong_recall": 0.70,       # or most of the event found
    "strong_recall_iou": 0.35,
    "miss_recall": 0.05,         # essentially nothing of the event recovered
    "small_event_px": 2000,      # ~3% of a 256x256 window
    "small_recall": 0.30,
    "fragment_components": 5,    # GT split into many pieces
    "fragment_match_rate": 0.20,
    "extent_over_precision": 0.40,
    "extent_under_recall": 0.50,
    "residual_cloud": 0.005,     # an order below the 0.05 accept gate
    "water": 0.10,
    "fp_pixels": MIN_AREA_PX,    # a negative predicting less than one region
    "weak_swir_delta": 0.02,     # |median dB12| inside GT
}

RULES = {
    "STRONG_DETECTION": f"IoU >= {T['strong_iou']} OR (recall >= {T['strong_recall']} AND IoU >= {T['strong_recall_iou']})",
    "COMPLETE_MISS": f"recall < {T['miss_recall']} (includes zero predicted pixels)",
    "SMALL_EVENT_MISS": f"GT positive pixels < {T['small_event_px']} AND recall < {T['small_recall']}",
    "FRAGMENTATION_MISS": f"GT components >= {T['fragment_components']} AND matched/GT components < {T['fragment_match_rate']} AND recall >= {T['small_recall']}",
    "EXTENT_ERROR": f"recall >= {T['extent_under_recall']} AND precision < {T['extent_over_precision']} (over-extent), or precision >= 0.6 AND recall < {T['extent_under_recall']} (under-extent)",
    "SPECTRAL_SUBTLETY": f"|median delta B12 inside GT| < {T['weak_swir_delta']} AND recall < {T['strong_recall']}",
    "CLOUD_HAZE_CONTAMINATION": f"residual cloud fraction > {T['residual_cloud']} on either date",
    "WATER_OR_WETLAND_FALSE_POSITIVE": f"negative sample with >= {T['fp_pixels']} predicted pixels AND water fraction > {T['water']}",
    "LAND_COVER_FALSE_POSITIVE": f"negative sample with >= {T['fp_pixels']} predicted pixels",
    "CLEAN_NEGATIVE": f"negative sample with < {T['fp_pixels']} predicted pixels",
    "UNKNOWN": "no rule above fired",
}
ORDER_POSITIVE = ["STRONG_DETECTION", "COMPLETE_MISS", "FRAGMENTATION_MISS",
                  "SMALL_EVENT_MISS", "EXTENT_ERROR", "SPECTRAL_SUBTLETY",
                  "CLOUD_HAZE_CONTAMINATION", "UNKNOWN"]


def components(mask):
    """8-connected components >= MIN_AREA_PX, via the shared region protocol."""
    import cv2
    filtered, regions = extract_regions(mask.astype(np.uint8),
                                        min_area_px=MIN_AREA_PX)
    if not regions:
        return [], filtered
    n, labels, _, _ = cv2.connectedComponentsWithStats(
        filtered.astype(np.uint8), connectivity=8)
    return [labels == i for i in range(1, n)], filtered


@torch.no_grad()
def predictions(code, device):
    """Probability maps for the whole test split from one frozen checkpoint.

    The frozen inference path is reproduced EXACTLY: batch size 8 under
    bfloat16 autocast, in dataset order. Running per-sample in fp32 instead
    shifts precision by -9e-4 and recall by +1.6e-3, because bfloat16 carries
    about three decimal digits and a handful of pixels sit close enough to the
    0.910 threshold to flip. Small, but it would mean analysing a slightly
    different model than the one whose numbers are being explained.
    """
    from torch.utils.data import DataLoader

    directory, name = EXPERIMENTS[code]
    payload = torch.load(os.path.join(directory, name), map_location=device,
                         weights_only=False)
    bands = tuple(payload.get("bands") or s2.BANDS)
    stats = load_band_statistics(STATS_PATH, bands)
    model = build_model(bands=bands, encoder=payload.get("encoder", "resnet34"),
                        pretrained=False)
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    dataset = EnvironmentChangeDataset(DATASET_DIR, "test", stats, bands=bands)
    with open(os.path.join(directory, "evaluation.json"), encoding="utf-8") as fh:
        threshold = json.load(fh)["threshold"]

    amp_dtype = torch.bfloat16 if (device.type == "cuda"
                                   and torch.cuda.is_bf16_supported()) else torch.float16
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    out = {}
    for batch in loader:
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            logits = model(batch["before"].to(device), batch["after"].to(device))
        probability = torch.sigmoid(logits.float()).cpu().numpy()
        for i, sid in enumerate(batch["sample_id"]):
            out[sid] = probability[i, 0]
    return out, threshold, dataset.records


def spectral_delta(record, gt):
    """Robust per-band BEFORE->AFTER change, inside and outside the label."""
    sid = record["sample_id"]
    before = s2.to_reflectance(np.load(os.path.join(DATASET_DIR, f"{sid}_before.npy")),
                               record["before_offset_applied"], record["before_baseline"])
    after = s2.to_reflectance(np.load(os.path.join(DATASET_DIR, f"{sid}_after.npy")),
                              record["after_offset_applied"], record["after_baseline"])
    invalid = np.load(os.path.join(DATASET_DIR, f"{sid}_invalid.npy")).astype(bool)
    usable = ~invalid
    inside, outside = usable & gt, usable & ~gt
    out = {}
    for index, band in enumerate(s2.BANDS):
        delta = after[:, :, index] - before[:, :, index]
        for tag, mask in (("in", inside), ("out", outside)):
            if mask.sum() >= 50:
                values = delta[mask]
                out[f"d{band}_{tag}_median"] = round(float(np.median(values)), 6)
                out[f"d{band}_{tag}_iqr"] = round(
                    float(np.percentile(values, 75) - np.percentile(values, 25)), 6)
            else:
                out[f"d{band}_{tag}_median"] = None
                out[f"d{band}_{tag}_iqr"] = None
    return out


def classify(row):
    """Deterministic primary/secondary category. Rules are in RULES."""
    reasons = []
    if row["sample_type"] == "negative":
        if row["pred_px"] < T["fp_pixels"]:
            return "CLEAN_NEGATIVE", None, reasons
        if (row["water_fraction"] or 0) > T["water"]:
            return "WATER_OR_WETLAND_FALSE_POSITIVE", "LAND_COVER_FALSE_POSITIVE", reasons
        if (row["residual_cloud_fraction"] or 0) > T["residual_cloud"]:
            return "CLOUD_HAZE_CONTAMINATION", "LAND_COVER_FALSE_POSITIVE", reasons
        return "LAND_COVER_FALSE_POSITIVE", None, reasons

    recall, iou, precision = row["recall"], row["iou"], row["precision"]
    hazy = (row["residual_cloud_fraction"] or 0) > T["residual_cloud"]
    weak_swir = (row.get("dB12_in_median") is not None
                 and abs(row["dB12_in_median"]) < T["weak_swir_delta"])
    match_rate = (row["region_matched"] / row["gt_components"]
                  if row["gt_components"] else 1.0)

    if iou >= T["strong_iou"] or (recall >= T["strong_recall"]
                                  and iou >= T["strong_recall_iou"]):
        return "STRONG_DETECTION", None, reasons
    if recall < T["miss_recall"]:
        secondary = ("CLOUD_HAZE_CONTAMINATION" if hazy else
                     "SPECTRAL_SUBTLETY" if weak_swir else None)
        return "COMPLETE_MISS", secondary, reasons
    if (row["gt_components"] >= T["fragment_components"]
            and match_rate < T["fragment_match_rate"]
            and recall >= T["small_recall"]):
        return "FRAGMENTATION_MISS", ("SPECTRAL_SUBTLETY" if weak_swir else None), reasons
    if row["true_px"] < T["small_event_px"] and recall < T["small_recall"]:
        return "SMALL_EVENT_MISS", ("SPECTRAL_SUBTLETY" if weak_swir else None), reasons
    if (recall >= T["extent_under_recall"] and precision < T["extent_over_precision"]):
        return "EXTENT_ERROR", "over-extent", reasons
    if precision >= 0.6 and recall < T["extent_under_recall"]:
        return "EXTENT_ERROR", "under-extent", reasons
    if weak_swir and recall < T["strong_recall"]:
        return "SPECTRAL_SUBTLETY", None, reasons
    if hazy:
        return "CLOUD_HAZE_CONTAMINATION", None, reasons
    return "UNKNOWN", None, reasons


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    probs, threshold, records = predictions("E2", device)
    if abs(threshold - FROZEN["threshold"]) > 1e-9:
        raise SystemExit(f"threshold {threshold} is not the frozen {FROZEN['threshold']}")
    by_id = {r["sample_id"]: r for r in records}

    # --- reproduce the frozen aggregate before analysing anything -----------
    acc = ConfusionAccumulator(threshold=threshold)
    sweep = ThresholdSweep()
    for sid, probability in probs.items():
        gt = np.load(os.path.join(DATASET_DIR, f"{sid}_label.npy")) > 0.5
        acc.update(torch.from_numpy(probability), torch.from_numpy(gt.astype(np.float32)))
        sweep.update(torch.from_numpy(probability), torch.from_numpy(gt.astype(np.float32)))
    reproduced = acc.compute()
    reproduced["average_precision"] = sweep.average_precision()
    drift = {k: round(reproduced[k] - FROZEN[k], 6)
             for k in ("precision", "recall", "f1", "iou", "average_precision")}
    print("[verify] reproduced vs frozen:", drift)
    if max(abs(v) for v in drift.values()) > 5e-4:
        raise SystemExit("regenerated predictions do NOT reproduce the frozen "
                         "metrics; analysis aborted")
    print("[verify] frozen operating point reproduced - analysis may proceed")

    # --- per-sample table ---------------------------------------------------
    rows = []
    for sid, probability in probs.items():
        record = by_id[sid]
        gt = np.load(os.path.join(DATASET_DIR, f"{sid}_label.npy")) > 0.5
        invalid = np.load(os.path.join(DATASET_DIR, f"{sid}_invalid.npy")).astype(bool)
        pred = probability >= threshold

        tp = int(np.logical_and(pred, gt).sum())
        fp = int(np.logical_and(pred, ~gt).sum())
        fn = int(np.logical_and(~pred, gt).sum())
        tn = int(np.logical_and(~pred, ~gt).sum())
        eps = 1e-9
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)

        gt_comp, _ = components(gt)
        pred_comp, _ = components(pred)
        region = rmetrics.region_confusion(pred, gt)

        row = {
            "sample_id": sid, "event_id": record.get("event_id"), "split": "test",
            "region": "SE Asia" if record["tmf_tile"] in ("N0_E110", "N10_E100") else "Amazon",
            "tmf_tile": record["tmf_tile"], "mgrs_tile": record["mgrs_tile"],
            "utm_zone": record["utm_zone"], "tmf_year": record["tmf_year"],
            "before_date": record["before_datetime"][:10],
            "after_date": record["after_datetime"][:10],
            "doy_difference": record["doy_difference"],
            "sample_type": record["sample_type"],
            "event_size": record.get("event_size"),
            "positive_fraction": record["positive_fraction"],
            "true_px": int(gt.sum()), "pred_px": int(pred.sum()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 6), "recall": round(recall, 6),
            "f1": round(2 * precision * recall / (precision + recall + eps), 6),
            "iou": round(tp / (tp + fp + fn + eps), 6),
            "gt_components": len(gt_comp), "pred_components": len(pred_comp),
            "region_matched": region["tp"], "region_fp": region["fp"],
            "region_fn": region["fn"],
            "region_precision": round(region["tp"] / max(region["n_pred"], 1), 6),
            "region_recall": round(region["tp"] / max(region["n_true"], 1), 6),
            "max_prob": round(float(probability.max()), 6),
            "mean_prob": round(float(probability.mean()), 6),
            "mean_prob_in_gt": (round(float(probability[gt].mean()), 6)
                                if gt.any() else None),
            "mean_prob_out_gt": round(float(probability[~gt].mean()), 6),
            "invalid_fraction": record["invalid_fraction"],
            "residual_cloud_fraction": record.get("residual_cloud_fraction"),
            "residual_cloud_before": record.get("residual_cloud_before"),
            "residual_cloud_after": record.get("residual_cloud_after"),
            "before_cloud_pct": record["before_cloud_pct"],
            "after_cloud_pct": record["after_cloud_pct"],
            "water_fraction": record.get("water_fraction"),
            "dnbr_reversed": record.get("dnbr_reversed"),
            "training_eligibility": record.get("training_eligibility"),
            "offset_overridden": record.get("offset_overridden"),
        }
        row["region_f1"] = round(
            2 * row["region_precision"] * row["region_recall"]
            / (row["region_precision"] + row["region_recall"] + eps), 6)
        row.update(spectral_delta(record, gt))
        rows.append(row)

    for row in rows:
        primary, secondary, _ = classify(row)
        row["primary_category"] = primary
        row["secondary_category"] = secondary

    rows.sort(key=lambda r: r["sample_id"])
    fields = list(rows[0].keys())
    with open(os.path.join(OUT_DIR, "per_sample_errors.csv"), "w",
              newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(OUT_DIR, "per_sample_errors.json"), "w",
              encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)

    # --- category statistics, by sample AND by pixel ------------------------
    total_fn = sum(r["fn"] for r in rows)
    total_fp = sum(r["fp"] for r in rows)
    positives = [r for r in rows if r["sample_type"] == "positive"]
    categories = {}
    for row in rows:
        bucket = categories.setdefault(row["primary_category"], [])
        bucket.append(row)
    stats = {}
    for name, bucket in sorted(categories.items()):
        pos = [r for r in bucket if r["sample_type"] == "positive"]
        stats[name] = {
            "samples": len(bucket),
            "pct_of_test": round(100 * len(bucket) / len(rows), 1),
            "positive_samples": len(pos),
            "pct_of_positive": round(100 * len(pos) / max(len(positives), 1), 1),
            "median_iou": round(float(np.median([r["iou"] for r in bucket])), 4),
            "mean_iou": round(float(np.mean([r["iou"] for r in bucket])), 4),
            "median_recall": round(float(np.median([r["recall"] for r in bucket])), 4),
            "mean_gt_positive_fraction": round(
                float(np.mean([r["positive_fraction"] for r in bucket])), 6),
            "mean_pred_fraction": round(
                float(np.mean([r["pred_px"] / 65536 for r in bucket])), 6),
            "fn_pixels": sum(r["fn"] for r in bucket),
            "pct_of_all_fn": round(100 * sum(r["fn"] for r in bucket) / max(total_fn, 1), 1),
            "fp_pixels": sum(r["fp"] for r in bucket),
            "pct_of_all_fp": round(100 * sum(r["fp"] for r in bucket) / max(total_fp, 1), 1),
        }

    # --- label granularity --------------------------------------------------
    gt_sizes, pred_sizes = [], []
    for sid, probability in probs.items():
        gt = np.load(os.path.join(DATASET_DIR, f"{sid}_label.npy")) > 0.5
        for component in components(gt)[0]:
            gt_sizes.append(int(component.sum()))
        for component in components(probability >= threshold)[0]:
            pred_sizes.append(int(component.sum()))

    def describe(values):
        if not values:
            return None
        a = np.asarray(values)
        return {"n": len(a), "min": int(a.min()), "p25": float(np.percentile(a, 25)),
                "median": float(np.median(a)), "p75": float(np.percentile(a, 75)),
                "p95": float(np.percentile(a, 95)), "max": int(a.max()),
                "mean": round(float(a.mean()), 1), "total": int(a.sum())}

    granularity = {
        "gt_component_sizes": describe(gt_sizes),
        "pred_component_sizes": describe(pred_sizes),
        "gt_components_per_sample": round(len(gt_sizes) / len(rows), 2),
        "pred_components_per_sample": round(len(pred_sizes) / len(rows), 2),
        "gt_components_per_positive_sample": round(
            sum(r["gt_components"] for r in positives) / max(len(positives), 1), 2),
        "pred_area_over_gt_area": round(
            sum(r["pred_px"] for r in rows) / max(sum(r["true_px"] for r in rows), 1), 4),
        "note": ("If predicted components are far larger and far fewer than GT "
                 "components while pixel overlap is reasonable, the region metric "
                 "is measuring label topology rather than detection quality."),
    }

    summary = {
        "phase": "4B-3 failure analysis (read-only)",
        "model": "E2 six-band", "dataset_version": "v24",
        "frozen_threshold": threshold,
        "frozen_metrics": FROZEN,
        "reproduced_metrics": {k: round(reproduced[k], 6) for k in FROZEN if k != "threshold"},
        "reproduction_drift": drift,
        "taxonomy_rules": RULES,
        "taxonomy_thresholds": T,
        "category_evaluation_order_positive": ORDER_POSITIVE,
        "category_statistics": stats,
        "totals": {"test_samples": len(rows), "positive": len(positives),
                   "negative": len(rows) - len(positives),
                   "fn_pixels": total_fn, "fp_pixels": total_fp,
                   "tp_pixels": sum(r["tp"] for r in rows)},
        "label_granularity": granularity,
    }
    with open(os.path.join(OUT_DIR, "failure_analysis.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n[table] {len(rows)} rows -> per_sample_errors.csv / .json")
    print("\n%-34s %6s %8s %8s %9s %9s %9s" % (
        "category", "n", "%pos", "medIoU", "medRecall", "%allFN", "%allFP"))
    print("-" * 92)
    for name, block in sorted(stats.items(), key=lambda kv: -kv[1]["samples"]):
        print("%-34s %6d %8.1f %8.4f %9.4f %9.1f %9.1f" % (
            name, block["samples"], block["pct_of_positive"], block["median_iou"],
            block["median_recall"], block["pct_of_all_fn"], block["pct_of_all_fp"]))
    print(f"\n[granularity] GT components/sample {granularity['gt_components_per_sample']}, "
          f"predicted {granularity['pred_components_per_sample']}, "
          f"pred/GT area {granularity['pred_area_over_gt_area']}")
    print(f"              GT component median {granularity['gt_component_sizes']['median']} px, "
          f"predicted median {granularity['pred_component_sizes']['median']} px")
    return 0


if __name__ == "__main__":
    sys.exit(main())
