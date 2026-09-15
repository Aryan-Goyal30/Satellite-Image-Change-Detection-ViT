"""Evaluate the Stage 3B-2 region direction classifier on the official test split.

Primary result = region classifier performance on S2Looking GROUND-TRUTH regions.
This script deliberately does NOT run the frozen LEVIR detector: an end-to-end
number would confound detector domain shift with classifier quality, and is not
the directionality metric.

Reported
--------
per-class precision / recall / F1, macro-F1 (headline), balanced accuracy,
accuracy, confusion matrix, ECE, and an abstention analysis.

Abstention
----------
UNCERTAIN is an *abstention*, never a ground-truth class: no region is labelled
"uncertain" anywhere in the data. Confidence is max softmax probability. The
operating threshold is chosen on VALIDATION ONLY - the test split is never used
to pick it - as the smallest threshold reaching TARGET_COVERAGE on validation.
The full accuracy/F1-versus-coverage curve is reported on test for transparency.

Usage:
    python scripts/evaluate_direction_classifier.py --mode both
    python scripts/evaluate_direction_classifier.py --summary
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config  # noqa: E402
from src.domains.built_environment.data.s2looking import TEMPORAL_ORDERING  # noqa: E402
from src.domains.built_environment.direction.dataset import RegionCrops  # noqa: E402
from src.domains.built_environment.direction.model import (  # noqa: E402
    CLASS_NAMES, build_model)

CROPS_DIR = os.path.join(config.S2LOOKING_RAW, "crops")
TARGET_COVERAGE = 0.90          # validation operating point for abstention
ECE_BINS = 15


def confusion(y_true, y_pred, k=2):
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def class_metrics(cm):
    out, f1s, recalls = {}, [], []
    for c, name in enumerate(CLASS_NAMES):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        out[name] = {"precision": round(p, 6), "recall": round(r, 6),
                     "f1": round(f1, 6), "support": int(cm[c, :].sum()),
                     "tp": tp, "fp": fp, "fn": fn}
        f1s.append(f1)
        recalls.append(r)
    return out, float(np.mean(f1s)), float(np.mean(recalls))


def expected_calibration_error(conf, correct, bins=ECE_BINS):
    """Equal-width binning of max-probability confidence against accuracy."""
    edges = np.linspace(0.5, 1.0, bins + 1)      # 2-class max-prob lives in [0.5, 1]
    ece, detail = 0.0, []
    n = len(conf)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        if not m.any():
            detail.append({"bin": [round(lo, 4), round(hi, 4)], "n": 0})
            continue
        acc = float(correct[m].mean())
        avg_conf = float(conf[m].mean())
        ece += (m.sum() / n) * abs(acc - avg_conf)
        detail.append({"bin": [round(lo, 4), round(hi, 4)], "n": int(m.sum()),
                       "accuracy": round(acc, 6), "confidence": round(avg_conf, 6)})
    return float(ece), detail


def abstention_curve(conf, y_true, y_pred, thresholds):
    rows = []
    for t in thresholds:
        keep = conf >= t
        n = int(keep.sum())
        if n == 0:
            rows.append({"threshold": round(float(t), 4), "coverage": 0.0})
            continue
        cm = confusion(y_true[keep], y_pred[keep])
        _, mf1, bal = class_metrics(cm)
        rows.append({
            "threshold": round(float(t), 4),
            "coverage": round(n / len(conf), 6),
            "retained": n,
            "accuracy": round(float((y_true[keep] == y_pred[keep]).mean()), 6),
            "macro_f1": round(mf1, 6),
            "balanced_accuracy": round(bal, 6),
        })
    return rows


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x)
        probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def evaluate_mode(mode, device, workers=4):
    ckpt_path = os.path.join(config.CHECKPOINTS, f"direction_resnet18_{mode}_best.pt")
    if not os.path.exists(ckpt_path):
        print(f"NOT FOUND: {ckpt_path}")
        return None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(mode, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])

    loaders = {}
    for split in ("val", "test"):
        ds = RegionCrops(CROPS_DIR, split, mode=mode, augment=False)
        loaders[split] = (ds, DataLoader(ds, batch_size=256, shuffle=False,
                                         num_workers=workers, pin_memory=True))

    out = {"mode": mode,
           "checkpoint": os.path.relpath(ckpt_path, config.ROOT),
           "checkpoint_sha256": hashlib.sha256(open(ckpt_path, "rb").read()).hexdigest(),
           "trained_epoch": ckpt.get("epoch"),
           "val_macro_f1_at_selection": ckpt.get("val_macro_f1"),
           "temporal_ordering": TEMPORAL_ORDERING,
           "abstention": {"definition":
                          "UNCERTAIN = abstention when max softmax probability < "
                          "threshold. It is not a ground-truth class; no region is "
                          "labelled uncertain in the data.",
                          "threshold_selected_on": "validation",
                          "target_coverage_on_validation": TARGET_COVERAGE}}

    # ---- validation: threshold selection only -----------------------------
    vp, vy = predict(model, loaders["val"][1], device)
    v_pred, v_conf = vp.argmax(1), vp.max(1)
    grid = np.round(np.arange(0.50, 1.0001, 0.005), 4)
    v_curve = abstention_curve(v_conf, vy, v_pred, grid)
    feasible = [r for r in v_curve if r.get("coverage", 0) >= TARGET_COVERAGE]
    tau = max(r["threshold"] for r in feasible) if feasible else 0.5
    out["validation"] = {
        "counts": loaders["val"][0].class_counts(),
        "accuracy": round(float((vy == v_pred).mean()), 6),
        "selected_threshold": float(tau),
        "coverage_at_selected": next(
            (r["coverage"] for r in v_curve if r["threshold"] == tau), None),
    }

    # ---- test: untouched -------------------------------------------------
    tp_, ty = predict(model, loaders["test"][1], device)
    t_pred, t_conf = tp_.argmax(1), tp_.max(1)
    cm = confusion(ty, t_pred)
    per_class, mf1, bal = class_metrics(cm)
    correct = (ty == t_pred).astype(np.float64)
    ece, ece_detail = expected_calibration_error(t_conf, correct)

    keep = t_conf >= tau
    cm_kept = confusion(ty[keep], t_pred[keep]) if keep.any() else np.zeros((2, 2), np.int64)
    kept_per_class, kept_mf1, kept_bal = class_metrics(cm_kept)

    out["test"] = {
        "counts": loaders["test"][0].class_counts(),
        "regions": int(len(ty)),
        "accuracy": round(float(correct.mean()), 6),
        "macro_f1": round(mf1, 6),
        "balanced_accuracy": round(bal, 6),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": {"rows": "true", "cols": "predicted",
                                    "order": list(CLASS_NAMES)},
        "ece": round(ece, 6),
        "ece_bins": ece_detail,
        "mean_confidence": round(float(t_conf.mean()), 6),
        "abstention_at_selected_threshold": {
            "threshold": float(tau),
            "coverage": round(float(keep.mean()), 6),
            "retained": int(keep.sum()),
            "abstained": int((~keep).sum()),
            "accuracy": round(float((ty[keep] == t_pred[keep]).mean()), 6) if keep.any() else None,
            "macro_f1": round(kept_mf1, 6),
            "balanced_accuracy": round(kept_bal, 6),
            "per_class": kept_per_class,
        },
        "abstention_curve": abstention_curve(
            t_conf, ty, t_pred, np.round(np.arange(0.50, 1.0001, 0.025), 4)),
    }

    path = os.path.join(config.S2LOOKING_META, f"direction_eval_{mode}.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=2)
    print(f"\n=== {mode} ===")
    print(f"  test regions {out['test']['regions']}  accuracy {out['test']['accuracy']:.4f}")
    print(f"  MACRO-F1 {out['test']['macro_f1']:.4f}   balanced acc {out['test']['balanced_accuracy']:.4f}")
    for name, m in per_class.items():
        print(f"    {name:13s} P {m['precision']:.4f}  R {m['recall']:.4f}  "
              f"F1 {m['f1']:.4f}  n={m['support']}")
    print(f"  confusion (rows=true {CLASS_NAMES}): {cm.tolist()}")
    print(f"  ECE {ece:.4f}   mean confidence {out['test']['mean_confidence']:.4f}")
    a = out["test"]["abstention_at_selected_threshold"]
    print(f"  abstention tau={a['threshold']:.3f} (from val): coverage {a['coverage']:.4f}  "
          f"accuracy {a['accuracy']}  macro-F1 {a['macro_f1']:.4f}")
    print(f"  [write] {path}")
    return out


def summary():
    rows = []
    for mode in ("both", "before", "after"):
        p = os.path.join(config.S2LOOKING_META, f"direction_eval_{mode}.json")
        if os.path.exists(p):
            rows.append(json.load(open(p, encoding="utf-8")))
    if not rows:
        print("no evaluations found")
        return 1
    print(f"{'mode':8s} {'macro-F1':>9s} {'bal-acc':>8s} {'accuracy':>9s} {'ECE':>7s}")
    for r in rows:
        t = r["test"]
        print(f"{r['mode']:8s} {t['macro_f1']:9.4f} {t['balanced_accuracy']:8.4f} "
              f"{t['accuracy']:9.4f} {t['ece']:7.4f}")
    out = {"comparison": [{"mode": r["mode"], **{k: r["test"][k] for k in
                           ("macro_f1", "balanced_accuracy", "accuracy", "ece")}}
                          for r in rows],
           "primary": "both",
           "note": "BEFORE-only and AFTER-only are ablations. The primary model is "
                   "the 6-channel BEFORE+AFTER model."}
    path = os.path.join(config.S2LOOKING_META, "direction_ablations.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=2)
    print(f"\n[write] {path}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("both", "before", "after"))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    if args.summary:
        return summary()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modes = ("both", "before", "after") if args.all else (args.mode or "both",)
    for mode in modes:
        evaluate_mode(mode, device, args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
