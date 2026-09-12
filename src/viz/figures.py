"""Presentation-quality figures for the midterm.

Produces:
  fig1_before_after.png          Before | After
  fig2_qualitative_*.png         Before | After | Ground Truth | Prediction | Error
  fig3_metrics.png               Precision / Recall / F1 / IoU + PR curve
  fig4_failures.png              representative failure cases

Run after training + evaluation:
    python -m src.viz.figures --checkpoint checkpoints/siamese_unet_r34_best.pt
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.domains.built_environment.data.levir import LevirCDTiles, denormalize
from src import config
from src.common.model_loader import load_model

ROOT = config.ROOT
TILES = config.LEVIR_TILES
FIGS = config.FIGURES
RESULTS = config.RESULTS

BG = "#0D1117"
FG = "white"
ACCENT = "#63C8BF"


def _style(ax, title, color=FG, size=11):
    ax.set_facecolor(BG)
    ax.set_title(title, color=color, fontsize=size, pad=7, fontweight="bold")
    ax.axis("off")


def error_map(pred, gt):
    """TP green / FP red / FN blue / TN dark. The panel reviewers look at first."""
    h, w = gt.shape
    img = np.full((h, w, 3), 25, dtype=np.uint8)
    tp = pred & gt
    fp = pred & ~gt
    fn = ~pred & gt
    img[tp] = (60, 200, 90)
    img[fp] = (230, 70, 70)
    img[fn] = (70, 130, 235)
    return img


def tile_f1(pred, gt):
    tp = float((pred & gt).sum()); fp = float((pred & ~gt).sum()); fn = float((~pred & gt).sum())
    if tp + fp + fn == 0:
        return 1.0
    return 2 * tp / (2 * tp + fp + fn + 1e-9)


@torch.no_grad()
def collect(model, device, threshold, n_scan=600, tiles_root=None):
    """Run the test split and keep per-tile results ranked by F1."""
    ds = LevirCDTiles(tiles_root or TILES, "test", augment=False)
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)
    items = []
    seen = 0
    for batch in dl:
        a = batch["a"].to(device); b = batch["b"].to(device)
        m = batch["mask"]
        with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            logits = model(a, b)
        probs = torch.sigmoid(logits.float()).cpu().numpy()[:, 0]
        for i in range(a.size(0)):
            gt = m[i, 0].numpy() > 0.5
            pred = probs[i] >= threshold
            items.append({
                "name": batch["name"][i],
                "before": denormalize(batch["a"][i]),
                "after": denormalize(batch["b"][i]),
                "gt": gt, "pred": pred, "prob": probs[i],
                "f1": tile_f1(pred, gt), "gt_px": int(gt.sum()),
            })
        seen += a.size(0)
        if seen >= n_scan:
            break
    return items


def fig1_before_after(items, path):
    picks = [it for it in items if it["gt_px"] > 1500][:3] or items[:3]
    n = len(picks)
    fig, axes = plt.subplots(n, 2, figsize=(9, 4.5 * n), facecolor=BG)
    axes = np.atleast_2d(axes)
    for r, it in enumerate(picks):
        axes[r, 0].imshow(it["before"]); _style(axes[r, 0], f"BEFORE  ({it['name']})")
        axes[r, 1].imshow(it["after"]);  _style(axes[r, 1], "AFTER")
    fig.suptitle("Earth Guardian - LEVIR-CD bi-temporal pairs",
                 color=FG, fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def fig2_qualitative(items, path, picks, title, subtitle=None):
    n = len(picks)
    fig, axes = plt.subplots(n, 5, figsize=(21, 4.3 * n), facecolor=BG)
    axes = np.atleast_2d(axes)
    for r, it in enumerate(picks):
        axes[r, 0].imshow(it["before"]);            _style(axes[r, 0], "BEFORE")
        axes[r, 1].imshow(it["after"]);             _style(axes[r, 1], "AFTER")
        axes[r, 2].imshow(it["gt"], cmap="gray");   _style(axes[r, 2], "GROUND TRUTH")
        axes[r, 3].imshow(it["pred"], cmap="gray"); _style(axes[r, 3], "PREDICTION")
        axes[r, 4].imshow(error_map(it["pred"], it["gt"]))
        _style(axes[r, 4], f"ERROR MAP   (tile F1 {it['f1']:.3f})")
        axes[r, 0].text(0.02, 0.02, it["name"], transform=axes[r, 0].transAxes,
                        color="#AAAAAA", fontsize=7, va="bottom")
    handles = [mpatches.Patch(color="#3CC85A", label="True positive"),
               mpatches.Patch(color="#E64646", label="False positive"),
               mpatches.Patch(color="#4682EB", label="False negative"),
               mpatches.Patch(color="#191919", label="True negative")]
    fig.legend(handles=handles, loc="lower center", ncol=4, facecolor="#1C2333",
               labelcolor=FG, edgecolor="#444", fontsize=10,
               bbox_to_anchor=(0.5, -0.012))
    full = title + (f"\n{subtitle}" if subtitle else "")
    fig.suptitle(full, color=FG, fontsize=14, fontweight="bold", y=0.998)
    fig.tight_layout(rect=[0, 0.022, 1, 0.99])
    fig.savefig(path, dpi=130, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def fig3_metrics(ev, path):
    test = ev["test"]
    fig = plt.figure(figsize=(16, 7), facecolor=BG)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=0.22)

    # --- bars ---
    ax = fig.add_subplot(gs[0, 0]); ax.set_facecolor("#111827")
    names = ["Precision", "Recall", "F1", "IoU"]
    vals = [test["precision"], test["recall"], test["f1"], test["iou"]]
    cols = ["#8FA4DA", "#D9A85C", ACCENT, "#B98FDA"]
    bars = ax.bar(names, vals, color=cols, width=0.6, edgecolor="#2a2a2a")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.018, f"{v:.4f}",
                ha="center", color=FG, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score (change class)", color="#AAAAAA", fontsize=11)
    ax.tick_params(colors=FG, labelsize=11)
    for s in ax.spines.values():
        s.set_edgecolor("#444")
    ax.grid(axis="y", linestyle="--", alpha=0.2, color=FG)
    ax.set_title(f"LEVIR-CD TEST  |  {ev['model_name']}  |  "
                 f"threshold {ev['selected_threshold']:.2f} (chosen on val)",
                 color=FG, fontsize=12, fontweight="bold", pad=10)

    # --- PR curve ---
    ax2 = fig.add_subplot(gs[0, 1]); ax2.set_facecolor("#111827")
    pr = ev["pr_curve_test"]
    ax2.plot(pr["recall"], pr["precision"], color=ACCENT, lw=2.2)
    i = int(np.argmin(np.abs(np.array(pr["threshold"]) - ev["selected_threshold"])))
    ax2.scatter([pr["recall"][i]], [pr["precision"][i]], s=90, color="#E2807F",
                zorder=5, label=f"operating point (tau={ev['selected_threshold']:.2f})")
    ax2.set_xlabel("Recall", color="#AAAAAA", fontsize=11)
    ax2.set_ylabel("Precision", color="#AAAAAA", fontsize=11)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1.02)
    ax2.tick_params(colors=FG, labelsize=10)
    for s in ax2.spines.values():
        s.set_edgecolor("#444")
    ax2.grid(linestyle="--", alpha=0.2, color=FG)
    ax2.legend(facecolor="#1C2333", labelcolor=FG, edgecolor="#444", fontsize=9, loc="lower left")
    ax2.set_title(f"Precision-Recall curve  |  AP = {ev['test_average_precision']:.4f}",
                  color=FG, fontsize=12, fontweight="bold", pad=10)

    note = (f"Pixel accuracy {test['pixel_accuracy']:.4f} is shown for contrast only: "
            f"~{100*(test['tn']+test['fn'])/(test['tp']+test['tn']+test['fp']+test['fn']):.1f}% "
            f"of pixels are unchanged, so predicting \"no change\" everywhere would score "
            f"about that much at F1 = 0.  |  "
            f"TP {test['tp']:,}   FP {test['fp']:,}   FN {test['fn']:,}")
    fig.text(0.5, -0.015, note, ha="center", color="#8B9E96", fontsize=9.5)
    fig.suptitle("Earth Guardian - Quantitative Evaluation vs LEVIR-CD Ground Truth",
                 color=FG, fontsize=15, fontweight="bold", y=1.02)
    fig.savefig(path, dpi=140, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=config.DEFAULT_CHECKPOINT_REL)
    ap.add_argument("--n-scan", type=int, default=600)
    ap.add_argument("--tiles", default=None, help="override tile root")
    args = ap.parse_args()

    os.makedirs(FIGS, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck_path = os.path.join(ROOT, args.checkpoint)
    model, ck = load_model(ck_path, device)

    ev_path = os.path.join(RESULTS, "evaluation.json")
    if not os.path.exists(ev_path):
        raise SystemExit(f"{ev_path} missing - run: python -m src.eval.evaluate")
    ev = json.load(open(ev_path))
    tau = ev["selected_threshold"]

    items = collect(model, device, tau, args.n_scan, args.tiles)
    with_change = [it for it in items if it["gt_px"] > 200]
    with_change.sort(key=lambda d: -d["f1"])

    fig1_before_after(with_change, os.path.join(FIGS, "fig1_before_after.png"))

    best = with_change[:3]
    fig2_qualitative(items, os.path.join(FIGS, "fig2_qualitative_success.png"), best,
                     "Earth Guardian - Prediction vs Ground Truth (strong cases)",
                     f"LEVIR-CD test split  |  {ev['model_name']}  |  threshold {tau:.2f}")

    worst = with_change[-3:]
    fig2_qualitative(items, os.path.join(FIGS, "fig4_failures.png"), worst,
                     "Earth Guardian - Representative FAILURE cases",
                     "Shown deliberately: these are the weakest test tiles, not the best")

    fig3_metrics(ev, os.path.join(FIGS, "fig3_metrics.png"))
    print(f"\nAll figures -> {FIGS}")


if __name__ == "__main__":
    main()
