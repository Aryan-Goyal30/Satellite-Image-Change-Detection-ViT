"""Write the measured metrics into README.md between the RESULTS markers.

Numbers are read from outputs/results/evaluation.json and never typed by hand,
so the README cannot drift from what was actually measured.

Usage:  python scripts/update_readme_results.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EV = os.path.join(ROOT, "outputs", "results", "evaluation.json")
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
