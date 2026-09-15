"""Region-level detection metric for environmental change (Phase 4B-1B).

Pixel metrics answer "how much of the changed area was found". They do not
answer "how many distinct clearings were found", which is the question an
operator actually asks. This module supplies the second answer, built on the
project's existing region protocol rather than a new one.

Regions come from `src.common.regions.extract_regions`: 8-connected components
of the binary mask, components below `min_area_px` discarded. The same function
defines regions for the built-environment engine, so "region" means the same
thing across the project.

The matching criterion, stated explicitly
-----------------------------------------
A predicted region is a TRUE POSITIVE when it overlaps a ground-truth region
with

    IoU(prediction, ground truth) >= MATCH_IOU

Matching is greedy in descending predicted area, and each ground-truth region
may be claimed at most once, so two predictions splitting one clearing yield one
TP and one FP rather than two TPs.

    region precision = TP / number of predicted regions
    region recall    = TP / number of ground-truth regions
    region F1        = harmonic mean of the two

MATCH_IOU is 0.25, not 0.5. The label is 30 m-quantised supervision resampled
onto a 10 m grid, so its boundaries are accurate to roughly one TMF pixel - about
three sample pixels. Demanding 0.5 IoU against a boundary that is itself only
30 m-accurate would measure label quantisation more than detection. 0.25 is
lenient enough to tolerate that quantisation while still requiring substantial
overlap; the threshold is reported alongside every result because the number
means nothing without it.
"""
from __future__ import annotations

import numpy as np

from src import config
from src.common.regions import extract_regions

#: Overlap required to call a predicted region a detection. See the module note.
MATCH_IOU = 0.25

#: Components smaller than this are noise, per the project-wide default.
MIN_AREA_PX = config.DEFAULT_MIN_AREA_PX

CRITERION = (f"greedy matching by descending predicted area; a prediction is a "
             f"true positive when IoU with an unclaimed ground-truth region is "
             f">= {MATCH_IOU}; each ground-truth region matches at most once")


def _component_masks(mask: np.ndarray, min_area_px: int):
    """Boolean mask per surviving connected component."""
    import cv2
    filtered, regions = extract_regions(mask.astype(np.uint8),
                                        min_area_px=min_area_px)
    if not regions:
        return []
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        filtered.astype(np.uint8), connectivity=8)
    return [labels == i for i in range(1, n)]


def region_confusion(pred_mask: np.ndarray, true_mask: np.ndarray,
                     min_area_px: int = MIN_AREA_PX,
                     match_iou: float = MATCH_IOU) -> dict:
    """Region-level TP/FP/FN for one sample."""
    predicted = _component_masks(pred_mask, min_area_px)
    truth = _component_masks(true_mask, min_area_px)
    predicted.sort(key=lambda m: -int(m.sum()))

    claimed = set()
    tp = 0
    for component in predicted:
        best, best_iou = None, 0.0
        for index, gt in enumerate(truth):
            if index in claimed:
                continue
            union = np.logical_or(component, gt).sum()
            if not union:
                continue
            iou = float(np.logical_and(component, gt).sum()) / float(union)
            if iou > best_iou:
                best, best_iou = index, iou
        if best is not None and best_iou >= match_iou:
            claimed.add(best)
            tp += 1
    return {"tp": tp, "fp": len(predicted) - tp, "fn": len(truth) - len(claimed),
            "n_pred": len(predicted), "n_true": len(truth)}


class RegionAccumulator:
    """Region-level TP/FP/FN accumulated over a split."""

    def __init__(self, min_area_px: int = MIN_AREA_PX, match_iou: float = MATCH_IOU):
        self.min_area_px = min_area_px
        self.match_iou = match_iou
        self.tp = self.fp = self.fn = 0
        self.n_pred = self.n_true = 0

    def update(self, pred_mask: np.ndarray, true_mask: np.ndarray) -> None:
        result = region_confusion(pred_mask, true_mask, self.min_area_px,
                                  self.match_iou)
        self.tp += result["tp"]
        self.fp += result["fp"]
        self.fn += result["fn"]
        self.n_pred += result["n_pred"]
        self.n_true += result["n_true"]

    def compute(self) -> dict:
        eps = 1e-9
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        return {
            "region_precision": precision,
            "region_recall": recall,
            "region_f1": 2 * precision * recall / (precision + recall + eps),
            "region_tp": self.tp, "region_fp": self.fp, "region_fn": self.fn,
            "predicted_regions": self.n_pred, "ground_truth_regions": self.n_true,
            "match_iou": self.match_iou, "min_area_px": self.min_area_px,
            "criterion": CRITERION,
        }
