"""Stage 0 vs Stage 1 figures.

    fig5_stage0_vs_stage1.png           test metrics side by side + PR curves
    fig6_stage0_vs_stage1_examples.png  the same test tiles through both stages

Reads outputs/results/evaluation.json (Stage 1) and
outputs/results/baseline_evaluation.json (Stage 0). Writes figures only: no
model, checkpoint or result file is changed.

Usage:
    python -m src.viz.compare_figures
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.domains.built_environment.data.levir import LevirCDTiles, denormalize
from src import config
from src.common.model_loader import load_model
from src.eval.evaluate_baseline import (load_vit, norm_calibrated, norm_poc_minmax,
                                        patch_distance, upsample)
from src.viz.figures import ACCENT, BG, FG, _style, error_map, tile_f1

ROOT = config.ROOT
TILES = config.LEVIR_TILES
FIGS = config.FIGURES
RESULTS = config.RESULTS

S0_COLORS = {"poc_minmax": "#6E7885", "val_calibrated": "#AEB6C1"}
SHORT = {"poc_minmax": "Stage 0 - frozen ViT, POC min-max (not trained)",
         "val_calibrated": "Stage 0 - frozen ViT, val-calibrated (not trained)",
         "stage1": "Stage 1 - trained Siamese U-Net (ResNet-34)"}


def _series(b, s1):
    rows = []
    for k in ("poc_minmax", "val_calibrated"):
        v = b["variants"][k]
        rows.append((k, S0_COLORS[k], v["test"], v["test_average_precision"],
                     v["threshold"], v["pr_curve_test"]))
    rows.append(("stage1", ACCENT, s1["test"], s1["test_average_precision"],
                 s1["selected_threshold"], s1["pr_curve_test"]))
    return rows


def fig_metrics(b, s1, path):
    rows = _series(b, s1)
    names = ["Precision", "Recall", "F1", "IoU", "AP"]
    fig = plt.figure(figsize=(18, 8), facecolor=BG)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.16, top=0.83, bottom=0.14)

    ax = fig.add_subplot(gs[0, 0]); ax.set_facecolor("#111827")
    x = np.arange(len(names)); w = 0.27
    for j, (key, color, test, apv, _tau, _pr) in enumerate(rows):
        vals = [test["precision"], test["recall"], test["f1"], test["iou"], apv]
        bars = ax.bar(x + (j - 1) * w, vals, w, color=color, edgecolor="#2a2a2a", label=SHORT[key])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.3f}",
                    ha="center", va="bottom", color=FG, fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score (change class)", color="#AAAAAA", fontsize=11)
    ax.tick_params(colors=FG)
    for s in ax.spines.values():
        s.set_edgecolor("#444")
    ax.grid(axis="y", linestyle="--", alpha=0.2, color=FG)
    ax.set_title(f"LEVIR-CD test split ({b['protocol']['n_test_tiles']:,} tiles)",
                 color=FG, fontsize=12, fontweight="bold", pad=10)

    ax2 = fig.add_subplot(gs[0, 1]); ax2.set_facecolor("#111827")
    for key, color, _test, _apv, tau, pr in rows:
        ax2.plot(pr["recall"], pr["precision"], color=color, lw=2.4 if key == "stage1" else 1.8)
        i = int(np.argmin(np.abs(np.array(pr["threshold"]) - tau)))
        ax2.scatter([pr["recall"][i]], [pr["precision"][i]], s=60, color=color,
                    edgecolor=FG, linewidth=0.8, zorder=5)
    # Chance level: a random score has precision equal to the change-pixel rate.
    tt = s1["test"]
    chance = (tt["tp"] + tt["fn"]) / (tt["tp"] + tt["fp"] + tt["fn"] + tt["tn"])
    ax2.axhline(chance, color="#E2807F", lw=1.2, ls="--", zorder=1)
    ax2.text(0.02, chance - 0.01, f"chance level = change-pixel rate ({chance:.3f})",
             ha="left", va="top", color="#E2807F", fontsize=9)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1.02)
    ax2.set_xlabel("Recall", color="#AAAAAA", fontsize=11)
    ax2.set_ylabel("Precision", color="#AAAAAA", fontsize=11)
    ax2.tick_params(colors=FG)
    for s in ax2.spines.values():
        s.set_edgecolor("#444")
    ax2.grid(linestyle="--", alpha=0.2, color=FG)
    ax2.set_title("Precision-recall on test  (dot = threshold chosen on validation)",
                  color=FG, fontsize=12, fontweight="bold", pad=10)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=3,
               facecolor="#1C2333", labelcolor=FG, edgecolor="#444", fontsize=10)

    rule = b["variants"]["poc_native_rule"]["test"]
    note = ("Stage 0 is the original proof of concept: a frozen ImageNet ViT-B/16 comparing the patch "
            "embeddings of the two dates. It is not trained for change detection.\n"
            "Both stages use the same test tiles. Thresholds (and Stage 0's calibration constants) were "
            "fixed on validation and applied to test unchanged; pixels are counted globally.\n"
            f"The POC's own decision rule (per-tile mean + 1.5 std, no tuning) gives F1 {rule['f1']:.3f} "
            "on the same tiles.")
    fig.text(0.5, 0.055, note, ha="center", va="top", color="#8B9E96", fontsize=9.5)
    fig.suptitle("Earth Guardian - Stage 0 baseline vs Stage 1 trained model",
                 color=FG, fontsize=15, fontweight="bold", y=0.985)
    fig.savefig(path, dpi=140, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def pick_tiles(ds, n=3, min_frac=0.05):
    """Chosen from GROUND TRUTH only: evenly spaced among test tiles with change.

    Model output plays no part, so the examples are not cherry-picked for either
    stage.
    """
    cands = []
    for name in ds.names:
        m = np.array(Image.open(os.path.join(ds.dir, "label", name)).convert("L")) > 127
        if m.mean() >= min_frac:
            cands.append(name)
    idx = [int(round(q * (len(cands) - 1))) for q in np.linspace(1 / 6, 5 / 6, n)]
    return [cands[i] for i in idx]


@torch.no_grad()
def fig_examples(b, s1, path, device):
    ds = LevirCDTiles(TILES, "test", augment=False)
    names = pick_tiles(ds)
    items = [ds[ds.names.index(n)] for n in names]
    a = torch.stack([it["a"] for it in items]).to(device)
    bb = torch.stack([it["b"] for it in items]).to(device)
    gt = torch.stack([it["mask"] for it in items])[:, 0].numpy() > 0.5
    size = tuple(a.shape[-2:])

    # Stage 0 variant: whichever has the higher VALIDATION F1 (no test data used).
    key = max(("poc_minmax", "val_calibrated"), key=lambda k: b["variants"][k]["val"]["f1"])
    v = b["variants"][key]
    vit = load_vit(device)
    d = patch_distance(vit, a, bb)
    if key == "poc_minmax":
        s = upsample(norm_poc_minmax(d), size)
    else:
        s = upsample(norm_calibrated(d, v["calibration"]["lo"], v["calibration"]["hi"]), size)
    s0 = s[:, 0].cpu().numpy()
    p0 = s0 >= v["threshold"]
    del vit
    if device == "cuda":
        torch.cuda.empty_cache()

    model, _ = load_model(os.path.join(ROOT, s1["checkpoint"]), device)
    with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
        logits = model(a, bb)
    p1 = torch.sigmoid(logits.float())[:, 0].cpu().numpy() >= s1["selected_threshold"]

    n = len(names)
    fig, axes = plt.subplots(n, 6, figsize=(25, 4.9 * n), facecolor=BG)
    axes = np.atleast_2d(axes)
    for r in range(n):
        axes[r, 0].imshow(denormalize(items[r]["a"]));          _style(axes[r, 0], "BEFORE")
        axes[r, 1].imshow(denormalize(items[r]["b"]));          _style(axes[r, 1], "AFTER")
        axes[r, 2].imshow(gt[r], cmap="gray");                  _style(axes[r, 2], "GROUND TRUTH")
        axes[r, 3].imshow(s0[r], cmap="magma", vmin=0, vmax=1); _style(axes[r, 3], "STAGE 0 SCORE MAP")
        axes[r, 4].imshow(error_map(p0[r], gt[r]))
        _style(axes[r, 4], f"STAGE 0 ERRORS  (tile F1 {tile_f1(p0[r], gt[r]):.3f})")
        axes[r, 5].imshow(error_map(p1[r], gt[r]))
        _style(axes[r, 5], f"STAGE 1 ERRORS  (tile F1 {tile_f1(p1[r], gt[r]):.3f})")
        axes[r, 0].text(0.02, 0.02, names[r], transform=axes[r, 0].transAxes,
                        color="#AAAAAA", fontsize=7, va="bottom")

    handles = [mpatches.Patch(color="#3CC85A", label="True positive"),
               mpatches.Patch(color="#E64646", label="False positive"),
               mpatches.Patch(color="#4682EB", label="False negative"),
               mpatches.Patch(color="#191919", label="True negative")]
    fig.legend(handles=handles, loc="lower center", ncol=4, facecolor="#1C2333",
               labelcolor=FG, edgecolor="#444", fontsize=10, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        "Earth Guardian - the same LEVIR-CD test tiles through Stage 0 and Stage 1\n"
        "Tiles chosen from ground truth only (evenly spaced among test tiles with >= 5% change), "
        "not by model output\n"
        f"{SHORT[key]}, threshold {v['threshold']:.2f} (variant with the higher validation F1)   |   "
        f"{SHORT['stage1']}, threshold {s1['selected_threshold']:.2f}",
        color=FG, fontsize=13, fontweight="bold", y=0.985)
    # Explicit spacing: tight_layout let row titles overlap the image above.
    fig.subplots_adjust(left=0.01, right=0.99, top=0.89, bottom=0.05, wspace=0.06, hspace=0.22)
    fig.savefig(path, dpi=120, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def main():
    os.makedirs(FIGS, exist_ok=True)
    s1 = json.load(open(os.path.join(RESULTS, "evaluation.json")))
    bpath = os.path.join(RESULTS, "baseline_evaluation.json")
    if not os.path.exists(bpath):
        raise SystemExit(f"{bpath} missing - run: python -m src.eval.evaluate_baseline")
    b = json.load(open(bpath))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fig_metrics(b, s1, os.path.join(FIGS, "fig5_stage0_vs_stage1.png"))
    fig_examples(b, s1, os.path.join(FIGS, "fig6_stage0_vs_stage1_examples.png"), device)


if __name__ == "__main__":
    main()
