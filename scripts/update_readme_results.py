"""Write the measured metrics into README.md between the RESULTS markers.

Numbers are read from outputs/results/evaluation.json and never typed by hand,
so the README cannot drift from what was actually measured.

Usage:  python scripts/update_readme_results.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config  # noqa: E402

ROOT = config.ROOT
EV = config.EVALUATION_JSON
README = os.path.join(ROOT, "README.md")
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def main():
    if not os.path.exists(EV):
        raise SystemExit(f"{EV} not found - run: python -m src.eval.evaluate")
    ev = json.load(open(EV))
    t, v = ev["test"], ev["val"]
    hist_path = os.path.join(ROOT, "checkpoints",
                             f"{os.path.basename(ev['checkpoint']).replace('_best.pt','')}_history.json")
    train_time = epochs = device = "n/a"
    if os.path.exists(hist_path):
        h = json.load(open(hist_path))
        train_time = h.get("total_time", "n/a")
        epochs = f"{h.get('best_epoch')} (best) of {len(h.get('history', []))} run"
        device = h.get("device", "n/a")

    comparison = ""
    bpath = config.BASELINE_EVALUATION_JSON
    if os.path.exists(bpath):
        b = json.load(open(bpath))
        pm, vc, nr = (b["variants"][k] for k in ("poc_minmax", "val_calibrated", "poc_native_rule"))

        def row(stage, method, trained, m, apv):
            ap_txt = f"{apv:.4f}" if apv is not None else "n/a"
            return (f"| {stage} | {method} | {trained} | {m['precision']:.4f} | "
                    f"{m['recall']:.4f} | {m['f1']:.4f} | {m['iou']:.4f} | {ap_txt} |")

        comparison = "\n".join([
            "",
            "### Stage 0 vs Stage 1 — same test tiles, same protocol",
            "",
            "| Stage | Method | Trained? | Precision | Recall | F1 | IoU | AP |",
            "|---|---|---|---|---|---|---|---|",
            row("0", "Frozen ImageNet ViT-B/16 feature distance, per-tile min-max (as in the POC)",
                "No", pm["test"], pm["test_average_precision"]),
            row("0", "Frozen ImageNet ViT-B/16 feature distance, calibrated on validation",
                "No", vc["test"], vc["test_average_precision"]),
            row("**1**", "**Siamese U-Net (ResNet-34), trained on LEVIR-CD**", "**Yes**",
                t, ev["test_average_precision"]),
            "",
            "> **Stage 0 is not a trained change detector.** It is the original proof of",
            "> concept: a frozen ImageNet ViT-B/16 whose patch embeddings are compared",
            "> between the two dates, with the 14x14 distance map upsampled to 256x256.",
            "> It is kept as a control. Every row uses the same "
            f"{b['protocol']['n_test_tiles']:,} test tiles.",
            "> Thresholds, and Stage 0's calibration constants, were fixed on validation",
            "> and applied to test unchanged. The POC's own decision rule (per-tile",
            f"> mean + 1.5 std, no tuning) scores F1 {nr['test']['f1']:.4f} on the same tiles.",
            "",
            "![Stage 0 vs Stage 1](outputs/figures/fig5_stage0_vs_stage1.png)",
            "![Stage 0 vs Stage 1 examples](outputs/figures/fig6_stage0_vs_stage1_examples.png)",
        ])

    block = f"""{START}
**LEVIR-CD test split** — {ev['n_test_tiles']:,} tiles of 256x256, ground truth
from the official annotations. Threshold **tau = {ev['selected_threshold']:.3f}**,
selected on the validation split and applied to test unchanged.

| Metric (change class) | Test | Validation |
|---|---|---|
| **Precision** | **{t['precision']:.4f}** | {v['precision']:.4f} |
| **Recall**    | **{t['recall']:.4f}** | {v['recall']:.4f} |
| **F1**        | **{t['f1']:.4f}** | {v['f1']:.4f} |
| **IoU**       | **{t['iou']:.4f}** | {v['iou']:.4f} |
| Average precision | {ev['test_average_precision']:.4f} | {ev['val_average_precision']:.4f} |
| Pixel accuracy | {t['pixel_accuracy']:.4f} | {v['pixel_accuracy']:.4f} |

Confusion (test, pixels): TP {t['tp']:,} · FP {t['fp']:,} · FN {t['fn']:,} · TN {t['tn']:,}

> **Pixel accuracy is reported only to show that it is uninformative.** About
> {100*(t['tn']+t['fp'])/(t['tp']+t['tn']+t['fp']+t['fn']):.1f}% of test pixels are unchanged, so a model that
> predicts "no change" everywhere scores roughly that accuracy at F1 = 0.
> Precision, Recall, F1 and IoU are computed for the **change class** only.

> **Threshold transfer.** The test-optimal threshold would have been
> {ev['test_oracle_threshold']['threshold']:.3f} (F1 {ev['test_oracle_threshold']['f1']:.4f}). We do not use it;
> the gap of {ev['test_oracle_threshold']['f1'] - t['f1']:+.4f} F1 is the honest cost of selecting the
> threshold on validation, and is reported rather than hidden.

Model `{ev['model_name']}` v{ev['version']} · encoder `{ev['encoder']}` ·
best epoch {epochs} · trained on {device} · training time {train_time} ·
test inference {ev['test_seconds']:.1f}s for {ev['n_test_tiles']:,} tiles.
{comparison}

![Metrics](outputs/figures/fig3_metrics.png)
![Qualitative](outputs/figures/fig2_qualitative_success.png)
![Failures](outputs/figures/fig4_failures.png)
{END}"""

    s = open(README, encoding="utf-8").read()
    i, j = s.index(START), s.index(END) + len(END)
    open(README, "w", encoding="utf-8").write(s[:i] + block + s[j:])
    print(f"README results updated: test F1 {t['f1']:.4f}  IoU {t['iou']:.4f}")


if __name__ == "__main__":
    main()
