"""Evaluate the Phase 4B-1B six-band environmental baseline on frozen v24.

Discipline this script enforces:

  * the decision threshold is chosen on VALIDATION ONLY, by maximising F1 over a
    201-point sweep of the validation probability histogram;
  * the threshold is then FROZEN and the TEST split is scored exactly once with
    it - no re-tuning, no per-region threshold, no second look;
  * the identical threshold and the identical TRAIN-derived normalisation are
    used for Amazon and for SE Asia, because the point of the geographic split
    is to EXPOSE domain shift, not to tune it away.

Usage:  python scripts/evaluate_environment_baseline.py
"""
import argparse
import datetime
import hashlib
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config                                              # noqa: E402
from src.domains.environment import region_metrics as rmetrics      # noqa: E402
from src.domains.environment.data import sentinel2 as s2            # noqa: E402
from src.domains.environment.data.loader import (                   # noqa: E402
    EnvironmentChangeDataset, load_band_statistics)
from src.domains.environment.model import BAND_SETS, build_model    # noqa: E402
from src.eval.metrics import ConfusionAccumulator, ThresholdSweep   # noqa: E402

DATASET_DIR = os.path.join(config.DATA_DIR, "environment", "dataset_v24")
STATS_PATH = os.path.join(config.DATA_DIR, "environment",
                          "characterization_v24_train_bands.json")
OUT_DIR = os.path.join(config.OUTPUTS, "environment_baseline")
CHECKPOINT = os.path.join(OUT_DIR, "environment_sixband_best.pt")

THRESHOLD_RULE = ("argmax F1 over a 201-point sweep of the VALIDATION "
                  "probability histogram; frozen before the test split is read")

FIXED_SCALE_MAX, FIXED_SCALE_GAMMA = 0.30, 0.6


@torch.no_grad()
def infer(model, dataset, device, amp_dtype, batch_size=8):
    """Probability maps for a whole split, in dataset order."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    probs, labels, meta = [], [], []
    model.eval()
    for batch in loader:
        before = batch["before"].to(device, non_blocking=True)
        after = batch["after"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            logits = model(before, after)
        probs.append(torch.sigmoid(logits.float()).cpu())
        labels.append(batch["label"])
        for i in range(len(batch["sample_id"])):
            meta.append({"sample_id": batch["sample_id"][i],
                         "region": batch["region"][i],
                         "sample_type": batch["sample_type"][i]})
    return torch.cat(probs), torch.cat(labels), meta


def pixel_metrics(probs, labels, threshold):
    acc = ConfusionAccumulator(threshold=threshold)
    acc.update(probs, labels)
    out = acc.compute()
    sweep = ThresholdSweep()
    sweep.update(probs, labels)
    out["average_precision"] = sweep.average_precision()
    return out


def region_metrics(probs, labels, threshold):
    acc = rmetrics.RegionAccumulator()
    for i in range(probs.shape[0]):
        acc.update((probs[i, 0].numpy() >= threshold),
                   (labels[i, 0].numpy() > 0.5))
    return acc.compute()


def subset(probs, labels, meta, predicate):
    idx = [i for i, m in enumerate(meta) if predicate(m)]
    if not idx:
        return None, None, []
    return probs[idx], labels[idx], [meta[i] for i in idx]


def composite(reflectance):
    rgb = [s2.BAND_INDEX[b] for b in ("B04", "B03", "B02")]
    scaled = np.stack([reflectance[:, :, i] for i in rgb], -1) / FIXED_SCALE_MAX
    return np.clip(scaled, 0, 1) ** FIXED_SCALE_GAMMA


def render_failures(cases, threshold, path):
    """BEFORE / AFTER / ground truth / prediction for chosen test samples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    with open(os.path.join(DATASET_DIR, "manifest.json"), encoding="utf-8") as fh:
        records = {r["sample_id"]: r for r in json.load(fh)["samples"]}
    cmap = ListedColormap(["#111111", "#ff2d2d"])
    fig, axes = plt.subplots(len(cases), 4, figsize=(14, 3.5 * len(cases)))
    if len(cases) == 1:
        axes = np.array([axes])

    for row, case in enumerate(cases):
        record = records[case["sample_id"]]
        sid = case["sample_id"]
        before = s2.to_reflectance(np.load(os.path.join(DATASET_DIR, f"{sid}_before.npy")),
                                   record["before_offset_applied"], record["before_baseline"])
        after = s2.to_reflectance(np.load(os.path.join(DATASET_DIR, f"{sid}_after.npy")),
                                  record["after_offset_applied"], record["after_baseline"])
        truth = np.load(os.path.join(DATASET_DIR, f"{sid}_label.npy"))
        pred = case["prob"] >= threshold

        axes[row, 0].imshow(composite(before))
        axes[row, 0].set_title(f"BEFORE {record['before_datetime'][:10]}", fontsize=9)
        axes[row, 1].imshow(composite(after))
        axes[row, 1].set_title(f"AFTER {record['after_datetime'][:10]}", fontsize=9)
        axes[row, 2].imshow(truth, cmap=cmap, vmin=0, vmax=1)
        axes[row, 2].set_title(f"TMF label  {100*truth.mean():.2f}%", fontsize=9)
        axes[row, 3].imshow(pred, cmap=cmap, vmin=0, vmax=1)
        axes[row, 3].set_title(f"prediction @ {threshold:.3f}  IoU {case['iou']:.3f}",
                               fontsize=9)
        axes[row, 0].set_ylabel(f"{sid}\n{record['mgrs_tile']} · {case['region']}\n"
                                f"{case['caption']}", fontsize=7.5, rotation=0,
                                ha="right", va="center", labelpad=72)
        for col in range(4):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    fig.suptitle("Phase 4B-1B six-band baseline - qualitative analysis on the frozen TEST split\n"
                 f"fixed reflectance scale 0-{FIXED_SCALE_MAX}, gamma {FIXED_SCALE_GAMMA}; "
                 f"threshold {threshold:.3f} selected on validation only", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def choose_cases(probs, labels, meta, threshold):
    """Representative test cases spanning the required failure categories."""
    with open(os.path.join(DATASET_DIR, "manifest.json"), encoding="utf-8") as fh:
        records = {r["sample_id"]: r for r in json.load(fh)["samples"]}
    rows = []
    for i, m in enumerate(meta):
        p = probs[i, 0].numpy()
        t = labels[i, 0].numpy() > 0.5
        pred = p >= threshold
        inter = np.logical_and(pred, t).sum()
        union = np.logical_or(pred, t).sum()
        rows.append({"sample_id": m["sample_id"], "region": m["region"],
                     "sample_type": m["sample_type"], "prob": p,
                     "iou": float(inter / union) if union else 1.0,
                     "recall": float(inter / t.sum()) if t.sum() else None,
                     "pred_px": int(pred.sum()), "true_px": int(t.sum()),
                     "event_size": records[m["sample_id"]].get("event_size")})

    chosen, used = [], set()

    def take(row, caption):
        if row is not None and row["sample_id"] not in used:
            used.add(row["sample_id"])
            chosen.append({**row, "caption": caption})

    positives = [r for r in rows if r["sample_type"] == "positive"]
    negatives = [r for r in rows if r["sample_type"] == "negative"]
    if positives:
        take(max(positives, key=lambda r: r["iou"]), "true positive (best IoU)")
        take(min(positives, key=lambda r: (r["recall"] if r["recall"] is not None else 1)),
             "missed change (lowest recall)")
        small = [r for r in positives if r["event_size"] == "small"]
        take(min(small, key=lambda r: r["iou"]) if small else None,
             "difficult small event")
        big = [r for r in positives if r["event_size"] in ("large", "very_large")]
        take(min(big, key=lambda r: r["iou"]) if big else None,
             "difficult large event")
    if negatives:
        take(max(negatives, key=lambda r: r["pred_px"]),
             "false positive (negative with most predicted pixels)")
    asia = [r for r in rows if r["region"] == "SE Asia"]
    take(max(asia, key=lambda r: r["true_px"]) if asia else None, "SE Asia example")
    amazon = [r for r in rows if r["region"] == "Amazon"]
    take(max(amazon, key=lambda r: r["true_px"]) if amazon else None, "Amazon example")
    return chosen, rows


def main():
    parser = argparse.ArgumentParser()
    global OUT_DIR, CHECKPOINT
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bands", default="six_band", choices=sorted(BAND_SETS))
    parser.add_argument("--dir", default=None,
                        help="experiment directory holding the checkpoint")
    args = parser.parse_args()

    OUT_DIR = args.dir or OUT_DIR
    candidates = [os.path.join(OUT_DIR, n) for n in
                  ("environment_best.pt", "environment_sixband_best.pt")]
    found = [c for c in candidates if os.path.exists(c)]
    if not found:
        raise SystemExit(f"no checkpoint in {OUT_DIR}; expected one of "
                         + ", ".join(os.path.basename(c) for c in candidates))
    CHECKPOINT = found[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (device.type == "cuda"
                                   and torch.cuda.is_bf16_supported()) else torch.float16
    payload = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    # Bands come from the checkpoint where recorded, so an experiment can never
    # be evaluated with a band set it was not trained on.
    bands = tuple(payload.get("bands") or BAND_SETS[args.bands])
    stats = load_band_statistics(STATS_PATH, bands)
    model = build_model(bands=bands, encoder=payload.get("encoder", "resnet34"),
                        pretrained=False)
    model.load_state_dict(payload["model"])
    model.to(device)
    digest = hashlib.sha256(open(CHECKPOINT, "rb").read()).hexdigest()
    print(f"[checkpoint] epoch {payload['epoch']}  val F1 {payload['val_f1']:.4f}")
    print(f"[checkpoint] sha256 {digest}")

    # ---- threshold selection: VALIDATION ONLY ------------------------------
    val_set = EnvironmentChangeDataset(DATASET_DIR, "val", stats, bands=bands)
    vprobs, vlabels, vmeta = infer(model, val_set, device, amp_dtype, args.batch_size)
    sweep = ThresholdSweep()
    sweep.update(vprobs, vlabels)
    peak = sweep.best_f1()
    threshold = float(peak["threshold"])
    print(f"[threshold] selected on VALIDATION: {threshold:.3f} "
          f"(val F1 {peak['f1']:.4f})  rule: {THRESHOLD_RULE}")

    val_pixel = pixel_metrics(vprobs, vlabels, threshold)
    val_region = region_metrics(vprobs, vlabels, threshold)

    # ---- test: read exactly once, with the frozen threshold -----------------
    test_set = EnvironmentChangeDataset(DATASET_DIR, "test", stats, bands=bands)
    tprobs, tlabels, tmeta = infer(model, test_set, device, amp_dtype, args.batch_size)
    test_pixel = pixel_metrics(tprobs, tlabels, threshold)
    test_region = region_metrics(tprobs, tlabels, threshold)

    regional = {}
    for region in ("Amazon", "SE Asia"):
        p, l, m = subset(tprobs, tlabels, tmeta, lambda x, r=region: x["region"] == r)
        if p is None:
            continue
        positives = sum(1 for x in m if x["sample_type"] == "positive")
        regional[region] = {
            "samples": len(m), "positive_samples": positives,
            "negative_samples": len(m) - positives,
            "positive_pixels": int((l > 0.5).sum()),
            "pixel": pixel_metrics(p, l, threshold),
            "region": region_metrics(p, l, threshold),
        }

    cases, rows = choose_cases(tprobs, tlabels, tmeta, threshold)
    figure = os.path.join(OUT_DIR, "test_qualitative.png")
    render_failures(cases, threshold, figure)

    with open(os.path.join(DATASET_DIR, "manifest.json"), encoding="utf-8") as fh:
        manifest_records = json.load(fh)["samples"]
    reversed_ids = {r["sample_id"]: r["split"] for r in manifest_records
                    if r.get("dnbr_reversed")}
    reversed_here = []
    for row in rows:
        if row["sample_id"] in reversed_ids:
            reversed_here.append({"sample_id": row["sample_id"], "split": "test",
                                  "iou": round(row["iou"], 4),
                                  "recall": (round(row["recall"], 4)
                                             if row["recall"] is not None else None),
                                  "sample_type": row["sample_type"]})

    results = {
        "experiment": f"phase4b-{len(bands)}band",
        "bands": list(bands),
        "dataset_version": "v24",
        "checkpoint": os.path.relpath(CHECKPOINT),
        "checkpoint_sha256": digest,
        "best_epoch": payload["epoch"],
        "threshold": threshold,
        "threshold_rule": THRESHOLD_RULE,
        "normalisation_sha256": stats["sha256"],
        "validation": {"pixel": val_pixel, "region": val_region,
                       "n": len(val_set)},
        "test": {"pixel": test_pixel, "region": test_region, "n": len(test_set)},
        "test_by_region": regional,
        "reversed_dnbr_in_test": reversed_here,
        "qualitative_cases": [{k: v for k, v in c.items() if k != "prob"}
                              for c in cases],
        "per_sample_test": [{k: v for k, v in r.items() if k != "prob"}
                            for r in rows],
        "figure": os.path.relpath(figure),
        "generated_utc": datetime.datetime.now(datetime.timezone.utc)
                         .strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    path = os.path.join(OUT_DIR, "evaluation.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print(f"\n[validation] F1 {val_pixel['f1']:.4f}  IoU {val_pixel['iou']:.4f}  "
          f"AP {val_pixel['average_precision']:.4f}")
    print(f"[test]       F1 {test_pixel['f1']:.4f}  IoU {test_pixel['iou']:.4f}  "
          f"AP {test_pixel['average_precision']:.4f}  "
          f"P {test_pixel['precision']:.4f}  R {test_pixel['recall']:.4f}")
    print(f"[test region] F1 {test_region['region_f1']:.4f} "
          f"(pred {test_region['predicted_regions']}, gt {test_region['ground_truth_regions']})")
    for region, block in regional.items():
        print(f"  {region:<8} n={block['samples']:<3} pos={block['positive_samples']:<3} "
              f"F1 {block['pixel']['f1']:.4f}  IoU {block['pixel']['iou']:.4f}  "
              f"AP {block['pixel']['average_precision']:.4f}")
    print(f"\n[artifacts] {path}\n            {figure}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
