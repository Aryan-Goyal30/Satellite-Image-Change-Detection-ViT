"""Change-detection metrics.

IMPORTANT — aggregation convention:
    All metrics are computed from GLOBALLY accumulated TP/FP/FN/TN over the whole
    split, not by averaging per-image scores. This is the convention used by the
    LEVIR-CD literature. Per-image averaging inflates results, because tiles with
    no change pixels produce degenerate per-image F1 values that dominate the
    mean. Using the global convention is what makes our numbers comparable to
    published ones.

Reported:
    precision, recall, f1, iou  -- all for the CHANGE class
    pixel accuracy              -- reported only to show it is uninformative
                                   (~95% by predicting "no change" everywhere)
"""
import numpy as np
import torch


class ConfusionAccumulator:
    """Streaming TP/FP/FN/TN over a dataset at a fixed threshold."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.tp = self.fp = self.fn = self.tn = 0

    @torch.no_grad()
    def update(self, probs, target):
        pred = (probs >= self.threshold)
        gt = (target > 0.5)
        self.tp += int((pred & gt).sum())
        self.fp += int((pred & ~gt).sum())
        self.fn += int((~pred & gt).sum())
        self.tn += int((~pred & ~gt).sum())

    def compute(self):
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        eps = 1e-9
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = tp / (tp + fp + fn + eps)
        acc = (tp + tn) / (tp + tn + fp + fn + eps)
        return {
            "precision": precision, "recall": recall, "f1": f1, "iou": iou,
            "pixel_accuracy": acc,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "threshold": self.threshold,
        }


class ThresholdSweep:
    """Accumulate histograms so P/R/F1 can be evaluated at many thresholds.

    Used to pick the operating threshold on VALIDATION (never on test) and to
    produce a precision-recall curve, which is the threshold-free view.
    """

    def __init__(self, n_bins=201):
        self.n_bins = n_bins
        self.edges = np.linspace(0.0, 1.0, n_bins)
        self.pos = np.zeros(n_bins, dtype=np.int64)   # positives per prob bin
        self.neg = np.zeros(n_bins, dtype=np.int64)

    @torch.no_grad()
    def update(self, probs, target):
        p = probs.detach().float().reshape(-1).cpu().numpy()
        t = (target.detach().reshape(-1).cpu().numpy() > 0.5)
        idx = np.clip((p * (self.n_bins - 1)).astype(np.int64), 0, self.n_bins - 1)
        self.pos += np.bincount(idx[t], minlength=self.n_bins)
        self.neg += np.bincount(idx[~t], minlength=self.n_bins)

    def curve(self):
        """Return (thresholds, precision, recall, f1) arrays."""
        # counts at prob >= threshold => reverse cumulative sum
        tp = np.cumsum(self.pos[::-1])[::-1].astype(np.float64)
        fp = np.cumsum(self.neg[::-1])[::-1].astype(np.float64)
        total_pos = self.pos.sum()
        fn = total_pos - tp
        eps = 1e-9
        # Where nothing is predicted positive (tp + fp == 0) precision is
        # undefined. Use the standard convention P = 1 (as sklearn does at
        # recall 0). Defining it as 0 drew a false diagonal from the origin on
        # the PR plot. F1 and AP are unaffected: recall is 0 at those points.
        predicted = tp + fp
        precision = np.where(predicted > 0, tp / np.maximum(predicted, 1), 1.0)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        return self.edges, precision, recall, f1

    def best_f1(self):
        th, p, r, f = self.curve()
        i = int(np.argmax(f))
        return {"threshold": float(th[i]), "f1": float(f[i]),
                "precision": float(p[i]), "recall": float(r[i])}

    def average_precision(self):
        """AP = sum_i (R_i - R_{i-1}) * P_i, the standard step interpolation.

        Deliberately NOT a trapezoid over (recall, precision): many thresholds
        can share an identical recall value, which makes trapezoidal integration
        collapse to near-zero. The step form is what sklearn's average_precision
        computes and is the convention in the detection literature.
        """
        _, p, r, _ = self.curve()
        # curve() is indexed by ascending threshold, so recall descends.
        # Reverse to get recall ascending.
        r = r[::-1]
        p = p[::-1]
        prev_r = 0.0
        ap = 0.0
        for pi, ri in zip(p, r):
            if ri > prev_r:
                ap += (ri - prev_r) * pi
                prev_r = ri
        return float(ap)


def format_metrics(m, title=""):
    lines = []
    if title:
        lines.append(title)
        lines.append("-" * len(title))
    lines.append(f"  Precision      : {m['precision']:.4f}")
    lines.append(f"  Recall         : {m['recall']:.4f}")
    lines.append(f"  F1             : {m['f1']:.4f}")
    lines.append(f"  IoU            : {m['iou']:.4f}")
    lines.append(f"  Pixel accuracy : {m['pixel_accuracy']:.4f}   (uninformative - see note)")
    lines.append(f"  threshold      : {m['threshold']:.3f}")
    lines.append(f"  TP={m['tp']:,}  FP={m['fp']:,}  FN={m['fn']:,}  TN={m['tn']:,}")
    return "\n".join(lines)
