"""Phase 4B-3: comparative failure analysis and the fixed qualitative grid.

READ-ONLY. Consumes the frozen checkpoints and the per-sample error table
produced by analyze_environment_failures.py. Writes only new files under
outputs/environment_failure_analysis/.

Three outputs:

  * E0 / E1 / E2 comparative failure analysis - where the extra bands help and
    where they hurt, and what those samples have in common;
  * the top false positives with their land-cover context;
  * a fixed qualitative grid covering the required categories, chosen by
    CATEGORY MEMBERSHIP from the error table rather than by picking whichever
    samples flatter the model.

Usage:  python scripts/environment_failure_figures.py
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config                                              # noqa: E402
from src.domains.environment.data import sentinel2 as s2            # noqa: E402
from src.domains.environment.data.loader import (                   # noqa: E402
    EnvironmentChangeDataset, load_band_statistics)
from src.domains.environment.model import build_model               # noqa: E402

DATASET_DIR = os.path.join(config.DATA_DIR, "environment", "dataset_v24")
STATS_PATH = os.path.join(config.DATA_DIR, "environment",
                          "characterization_v24_train_bands.json")
OUT_DIR = os.path.join(config.OUTPUTS, "environment_failure_analysis")
EXPERIMENTS = {
    "E0": (os.path.join(config.OUTPUTS, "environment_ablation", "E0_rgb"),
           "environment_best.pt"),
    "E1": (os.path.join(config.OUTPUTS, "environment_ablation", "E1_rgb_nir"),
           "environment_best.pt"),
    "E2": (os.path.join(config.OUTPUTS, "environment_baseline"),
           "environment_sixband_best.pt"),
}
#: An IoU change larger than this is treated as a substantial difference.
SUBSTANTIAL = 0.10
FIXED_MAX, FIXED_GAMMA = 0.30, 0.6


@torch.no_grad()
def predictions(code, device):
    """Reproduce each experiment's frozen inference path exactly.

    Batch size 8 under bfloat16 autocast, in dataset order - the same path the
    recorded evaluations used. Per-sample fp32 inference shifts borderline
    pixels around the threshold and would not reproduce the frozen metrics.
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
    return out, threshold


def iou_of(prob, gt, threshold):
    pred = prob >= threshold
    union = np.logical_or(pred, gt).sum()
    return float(np.logical_and(pred, gt).sum() / union) if union else 1.0


def composite(reflectance, bands=("B04", "B03", "B02")):
    idx = [s2.BAND_INDEX[b] for b in bands]
    scaled = np.stack([reflectance[:, :, i] for i in idx], -1) / FIXED_MAX
    return np.clip(scaled, 0, 1) ** FIXED_GAMMA


def load_pair(record):
    sid = record["sample_id"]
    before = s2.to_reflectance(np.load(os.path.join(DATASET_DIR, f"{sid}_before.npy")),
                               record["before_offset_applied"], record["before_baseline"])
    after = s2.to_reflectance(np.load(os.path.join(DATASET_DIR, f"{sid}_after.npy")),
                              record["after_offset_applied"], record["after_baseline"])
    return before, after


def render_grid(cases, records, probs, thresholds, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    binary = ListedColormap(["#111111", "#ff2d2d"])
    #  0 background, 1 TP green, 2 FP blue, 3 FN yellow
    error_cmap = ListedColormap(["#111111", "#2ecc71", "#3498db", "#f1c40f"])
    columns = 6
    fig, axes = plt.subplots(len(cases), columns,
                             figsize=(3.0 * columns, 3.15 * len(cases)))
    if len(cases) == 1:
        axes = np.array([axes])

    for row, (sid, caption) in enumerate(cases):
        record = records[sid]
        before, after = load_pair(record)
        gt = np.load(os.path.join(DATASET_DIR, f"{sid}_label.npy")) > 0.5
        pred = probs["E2"][sid] >= thresholds["E2"]
        error = np.zeros(gt.shape, np.uint8)
        error[np.logical_and(pred, gt)] = 1
        error[np.logical_and(pred, ~gt)] = 2
        error[np.logical_and(~pred, gt)] = 3

        panels = [
            (composite(before), f"BEFORE {record['before_datetime'][:10]}", None),
            (composite(after), f"AFTER {record['after_datetime'][:10]}", None),
            (composite(after, ("B12", "B08", "B04")), "AFTER SWIR/NIR/Red", None),
            (gt, f"TMF label {100*gt.mean():.2f}%", binary),
            (pred, f"E2 pred @ {thresholds['E2']:.3f}", binary),
            (error, f"TP/FP/FN  IoU {iou_of(probs['E2'][sid], gt, thresholds['E2']):.3f}",
             error_cmap),
        ]
        for col, (image, title, cmap) in enumerate(panels):
            ax = axes[row, col]
            if cmap is None:
                ax.imshow(image)
            else:
                ax.imshow(image, cmap=cmap, vmin=0,
                          vmax=3 if cmap is error_cmap else 1)
            ax.set_title(title, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 0].set_ylabel(f"{sid}\n{record['mgrs_tile']}\n{caption}",
                                fontsize=7, rotation=0, ha="right", va="center",
                                labelpad=70)

    fig.legend(handles=[Patch(color="#2ecc71", label="TP"),
                        Patch(color="#3498db", label="FP"),
                        Patch(color="#f1c40f", label="FN")],
               loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle("Phase 4B-3 failure analysis - frozen E2 six-band model on the frozen TEST split\n"
                 "fixed reflectance scale 0-0.30 gamma 0.6; threshold 0.910; "
                 "examples chosen by error-category membership", fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.965))
    fig.savefig(path, dpi=105, bbox_inches="tight")
    plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(OUT_DIR, "per_sample_errors.json"), encoding="utf-8") as fh:
        table = {r["sample_id"]: r for r in json.load(fh)}
    with open(os.path.join(DATASET_DIR, "manifest.json"), encoding="utf-8") as fh:
        records = {r["sample_id"]: r for r in json.load(fh)["samples"]
                   if r["split"] == "test"}

    probs, thresholds = {}, {}
    for code in EXPERIMENTS:
        probs[code], thresholds[code] = predictions(code, device)
        print(f"  {code} threshold {thresholds[code]:.3f}")

    # --- E0 / E1 / E2 comparative failure analysis --------------------------
    comparison = []
    for sid, row in table.items():
        gt = np.load(os.path.join(DATASET_DIR, f"{sid}_label.npy")) > 0.5
        ious = {c: iou_of(probs[c][sid], gt, thresholds[c]) for c in EXPERIMENTS}
        comparison.append({
            "sample_id": sid, "sample_type": row["sample_type"],
            "region": row["region"], "event_size": row["event_size"],
            "positive_fraction": row["positive_fraction"],
            "true_px": row["true_px"],
            "residual_cloud_fraction": row["residual_cloud_fraction"],
            "dB12_in_median": row.get("dB12_in_median"),
            "dB08_in_median": row.get("dB08_in_median"),
            "dB04_in_median": row.get("dB04_in_median"),
            "iou_E0": round(ious["E0"], 4), "iou_E1": round(ious["E1"], 4),
            "iou_E2": round(ious["E2"], 4),
            "E2_minus_E0": round(ious["E2"] - ious["E0"], 4),
            "E2_minus_E1": round(ious["E2"] - ious["E1"], 4),
            "primary_category": row["primary_category"],
        })

    def group(key, predicate):
        members = [c for c in comparison if predicate(c)]
        if not members:
            return {"n": 0}
        def avg(field):
            values = [m[field] for m in members if m[field] is not None]
            return round(float(np.mean(values)), 5) if values else None
        return {
            "n": len(members),
            "samples": [m["sample_id"] for m in members],
            "mean_positive_fraction": avg("positive_fraction"),
            "mean_true_px": avg("true_px"),
            "mean_dB12_in": avg("dB12_in_median"),
            "mean_dB08_in": avg("dB08_in_median"),
            "mean_dB04_in": avg("dB04_in_median"),
            "mean_residual_cloud": avg("residual_cloud_fraction"),
            "regions": {r: sum(1 for m in members if m["region"] == r)
                        for r in ("Amazon", "SE Asia")},
            "event_sizes": {s: sum(1 for m in members if m["event_size"] == s)
                            for s in ("small", "medium", "large", "very_large")},
        }

    positives_only = lambda c: c["sample_type"] == "positive"
    comparative = {
        "substantial_delta": SUBSTANTIAL,
        "E2_much_better_than_E0": group("E2>E0", lambda c: positives_only(c) and c["E2_minus_E0"] > SUBSTANTIAL),
        "E2_much_better_than_E1": group("E2>E1", lambda c: positives_only(c) and c["E2_minus_E1"] > SUBSTANTIAL),
        "E2_much_worse_than_E0": group("E2<E0", lambda c: positives_only(c) and c["E2_minus_E0"] < -SUBSTANTIAL),
        "E2_much_worse_than_E1": group("E2<E1", lambda c: positives_only(c) and c["E2_minus_E1"] < -SUBSTANTIAL),
        "unchanged": group("same", lambda c: positives_only(c) and abs(c["E2_minus_E0"]) <= SUBSTANTIAL and abs(c["E2_minus_E1"]) <= SUBSTANTIAL),
    }

    # --- top false positives -------------------------------------------------
    top_fp = sorted(table.values(), key=lambda r: -r["fp"])[:10]
    fp_rows = [{
        "sample_id": r["sample_id"], "sample_type": r["sample_type"],
        "region": r["region"], "mgrs_tile": r["mgrs_tile"],
        "fp_pixels": r["fp"], "fp_fraction": round(r["fp"] / 65536, 5),
        "true_px": r["true_px"], "pred_px": r["pred_px"],
        "water_fraction": r["water_fraction"],
        "residual_cloud_fraction": r["residual_cloud_fraction"],
        "before_cloud_pct": r["before_cloud_pct"],
        "after_cloud_pct": r["after_cloud_pct"],
        "invalid_fraction": r["invalid_fraction"],
        "dB12_out_median": r.get("dB12_out_median"),
        "dB08_out_median": r.get("dB08_out_median"),
        "primary_category": r["primary_category"],
        "note": ("TMF-negative / Sentinel-2 spectral change - TMF may itself be "
                 "incomplete here" if r["sample_type"] == "negative"
                 else "over-extent around a genuine event"),
    } for r in top_fp]

    # --- fixed qualitative grid, chosen by category membership ---------------
    def pick(predicate, key, reverse=True):
        members = [r for r in table.values() if predicate(r)]
        if not members:
            return None
        return sorted(members, key=lambda r: (key(r), r["sample_id"]),
                      reverse=reverse)[0]["sample_id"]

    wanted = [
        ("best detection", pick(lambda r: r["primary_category"] == "STRONG_DETECTION",
                                lambda r: r["iou"])),
        ("large-event detection", pick(lambda r: r["sample_type"] == "positive"
                                       and r["event_size"] in ("large", "very_large"),
                                       lambda r: r["true_px"])),
        ("small-event miss", pick(lambda r: r["primary_category"] == "SMALL_EVENT_MISS",
                                  lambda r: -r["iou"])),
        ("fragmented-event miss", pick(lambda r: r["primary_category"] == "FRAGMENTATION_MISS",
                                       lambda r: r["gt_components"])),
        ("large false positive", pick(lambda r: r["sample_type"] == "negative",
                                      lambda r: r["fp"])),
        ("complete miss", pick(lambda r: r["primary_category"] == "COMPLETE_MISS",
                               lambda r: r["true_px"])),
        ("subtle spectral change", pick(
            lambda r: r["sample_type"] == "positive"
            and r.get("dB12_in_median") is not None,
            lambda r: -abs(r["dB12_in_median"]))),
        ("haze / cloud", pick(lambda r: True,
                              lambda r: (r["residual_cloud_fraction"] or 0))),
        ("Amazon", pick(lambda r: r["region"] == "Amazon"
                        and r["sample_type"] == "positive", lambda r: r["true_px"])),
        ("SE Asia", pick(lambda r: r["region"] == "SE Asia"
                         and r["sample_type"] == "positive", lambda r: r["true_px"])),
    ]
    cases, used = [], set()
    for caption, sid in wanted:
        if sid and sid not in used:
            used.add(sid)
            cases.append((sid, caption))

    figure = os.path.join(OUT_DIR, "failure_grid.png")
    render_grid(cases, records, probs, thresholds, figure)

    payload = {
        "phase": "4B-3 comparative and qualitative analysis (read-only)",
        "thresholds": {k: thresholds[k] for k in EXPERIMENTS},
        "per_sample_comparison": sorted(comparison, key=lambda c: c["sample_id"]),
        "comparative_groups": comparative,
        "top_false_positives": fp_rows,
        "qualitative_cases": [{"sample_id": s, "caption": c} for s, c in cases],
        "case_selection_rule": ("membership in an error category from the frozen "
                                "per-sample table, then a deterministic extremum "
                                "within that category; never chosen to flatter "
                                "the model"),
        "figure": os.path.relpath(figure),
    }
    with open(os.path.join(OUT_DIR, "comparative_failure_analysis.json"), "w",
              encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print("\n=== E0/E1/E2 comparative groups (positives, |dIoU| > %.2f) ===" % SUBSTANTIAL)
    for name, block in comparative.items():
        if name == "substantial_delta" or not isinstance(block, dict):
            continue
        if block.get("n"):
            print(f"  {name:<28} n={block['n']:<3} regions {block['regions']} "
                  f"sizes {block['event_sizes']}")
            print(f"      mean dB12_in {block['mean_dB12_in']}  "
                  f"dB08_in {block['mean_dB08_in']}  dB04_in {block['mean_dB04_in']}  "
                  f"pos_frac {block['mean_positive_fraction']}")
        else:
            print(f"  {name:<28} n=0")
    print("\n=== top false positives ===")
    print("  %-30s %-9s %9s %9s %9s %s" % ("sample", "region", "FP px", "water", "resid", "category"))
    for r in fp_rows:
        print("  %-30s %-9s %9d %9.4f %9.5f %s" % (
            r["sample_id"], r["region"], r["fp_pixels"], r["water_fraction"] or 0,
            r["residual_cloud_fraction"] or 0, r["primary_category"]))
    print(f"\n[figure] {figure} ({len(cases)} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
