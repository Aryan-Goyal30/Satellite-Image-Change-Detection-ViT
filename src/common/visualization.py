"""Shared presentation helpers.

Rendering and ground-truth comparison used by applications. Applications call
into this module; this module knows nothing about them:

    application  ->  src.common.visualization        (never the reverse)

It must NOT import Streamlit or any other UI framework, so the CLI, the current
demo and a future frontend all render and score identically.

Inputs are the Earth Guardian result contract (ChangeResult) plus, optionally, a
ground-truth mask. No model, dataset or engine is imported here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Error-map colours (RGB), unchanged from the original inline implementation.
TP_COLOR = (60, 200, 90)
FP_COLOR = (230, 70, 70)
FN_COLOR = (70, 130, 235)
BACKGROUND_LEVEL = 25


def error_map(pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    """Colour-coded TP / FP / FN map over a dark background.

    Identical arithmetic and colours to the previous inline implementation.
    """
    h, w = pred_mask.shape
    tp = pred_mask & gt_mask
    fp = pred_mask & ~gt_mask
    fn = ~pred_mask & gt_mask
    img = np.full((h, w, 3), BACKGROUND_LEVEL, dtype=np.uint8)
    img[tp] = TP_COLOR
    img[fp] = FP_COLOR
    img[fn] = FN_COLOR
    return img


def f1_from_masks(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """F1 of the change class for one pair of masks.

    Kept bit-identical to the previous inline expression, including the 1e-9
    guard and the float cast of the intersection.
    """
    tp = pred_mask & gt_mask
    fp = pred_mask & ~gt_mask
    fn = ~pred_mask & gt_mask
    inter = float(tp.sum())
    return 2 * inter / (2 * inter + fp.sum() + fn.sum() + 1e-9)


def overlay(base_rgb: np.ndarray, mask: np.ndarray,
            color=(255, 40, 40), alpha: float = 0.45) -> np.ndarray:
    """Blend a binary mask over an image as a translucent colour layer.

    Moved here from the Built Environment engine so applications import
    rendering helpers from the shared layer, not from a domain. The engine
    re-exports it for existing callers. Arithmetic unchanged.
    """
    out = base_rgb.astype(np.float32).copy()
    col = np.array(color, dtype=np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * col
    return out.clip(0, 255).astype(np.uint8)


def draw_bboxes(image: np.ndarray, boxes, color=(255, 214, 10), thickness: int = 2) -> np.ndarray:
    """Outline bounding boxes on an RGB image. Pure presentation, no analysis.

    `boxes` is a sequence of (x, y, w, h) in pixel coordinates, as carried by
    ChangeResult regions.
    """
    out = image.copy()
    if out.ndim == 2:
        # Single-channel views (mask, probability map) are promoted to RGB so a
        # coloured outline can be drawn on them.
        out = np.stack([out] * 3, axis=-1)
    h_img, w_img = out.shape[:2]
    col = np.array(color, dtype=out.dtype)
    for x, y, w, h in boxes:
        x0, y0 = max(int(x), 0), max(int(y), 0)
        x1, y1 = min(int(x + w), w_img), min(int(y + h), h_img)
        if x1 <= x0 or y1 <= y0:
            continue
        t = max(1, int(thickness))
        out[y0:min(y0 + t, y1), x0:x1] = col          # top
        out[max(y1 - t, y0):y1, x0:x1] = col          # bottom
        out[y0:y1, x0:min(x0 + t, x1)] = col          # left
        out[y0:y1, max(x1 - t, x0):x1] = col          # right
    return out


@dataclass(frozen=True)
class GroundTruthComparison:
    """Everything an application needs to render a ground-truth comparison."""
    error_map: np.ndarray
    f1: float
    tp: int
    fp: int
    fn: int

    @property
    def caption(self) -> str:
        return f"Error map - green TP / red FP / blue FN (F1 {self.f1:.3f})"

    def to_dict(self) -> dict:
        """JSON-safe summary. The image array stays out by design."""
        return {"f1": self.f1, "tp": self.tp, "fp": self.fp, "fn": self.fn}


def compare_to_ground_truth(result, gt_mask: np.ndarray,
                            layer: Optional[str] = None) -> GroundTruthComparison:
    """Compare a ChangeResult layer against a ground-truth mask.

    Args:
        result:   a ChangeResult
        gt_mask:  boolean ground-truth mask, same shape as the prediction
        layer:    layer name; defaults to the result's primary layer
    """
    pred = result.layer(layer).mask if layer else result.primary_layer.mask
    if pred.shape != gt_mask.shape:
        raise ValueError(
            f"prediction {pred.shape} and ground truth {gt_mask.shape} differ in shape")
    tp = pred & gt_mask
    fp = pred & ~gt_mask
    fn = ~pred & gt_mask
    return GroundTruthComparison(
        error_map=error_map(pred, gt_mask),
        f1=f1_from_masks(pred, gt_mask),
        tp=int(tp.sum()), fp=int(fp.sum()), fn=int(fn.sum()),
    )
