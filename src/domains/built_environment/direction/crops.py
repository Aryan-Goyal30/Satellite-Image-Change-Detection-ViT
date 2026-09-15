"""Canonical crop and preprocessing protocol for the direction classifier.

**Single source of truth.** Imported by both the training crop builder
(`scripts/build_s2looking_crops.py`) and any future production inference path,
so the two cannot drift apart. Production code must never import from
`scripts/`; the dependency runs the other way.

This module is deliberately small and has no Streamlit, dataset, memmap or
engine dependency. It does not load, cache or own a model.

Protocol (fixed by what the trained checkpoint expects)
------------------------------------------------------
    crop size        128 x 128
    context          0.25 of the bbox width/height added on EACH side
    minimum side     16 px
    channel order    0:3 = BEFORE RGB, 3:6 = AFTER RGB
    resampling       PIL bilinear
    normalisation    ImageNet mean/std, applied independently to each date

Under the project's ratified temporal convention (a project decision, **not**
an author-documented property of S2Looking - see
docs/S2LOOKING_TEMPORAL_SEMANTICS.md) BEFORE is `Image2` and AFTER is `Image1`.
This module takes BEFORE and AFTER as given; resolving which folder is which is
the caller's job.

Geometry
--------
1. take the region bbox (x, y, w, h)
2. expand by CONTEXT_FRACTION of w and h on each side
3. square it off about the bbox centre, so the resize does not distort aspect
4. clamp the side to the scene, then SHIFT the box inside the scene rather than
   shrinking it
5. crop and resize to CROP_SIZE x CROP_SIZE (bilinear)

Source images are never resized before cropping: the box is computed in the
source pixel grid and the crop is taken from it directly.

Known edge case (pre-existing, deliberately preserved)
------------------------------------------------------
`x0`/`y0` and `side` are rounded to int independently, so when the box is
clamped hard against an image edge the pair can overshoot by at most one pixel;
PIL's ``.crop()`` zero-pads that sliver. This needs `side` to approach the image
dimension, which region-sized bboxes in a 1024x1024 scene never do - none of the
40,221 generated S2Looking regions trigger it. The behaviour is reproduced
exactly rather than corrected, because changing it would invalidate the crops
the classifier was trained on.
"""
from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image

from src.common.preprocessing import IMAGENET_MEAN, IMAGENET_STD

# --- protocol constants (must match the trained checkpoint) ----------------
CROP_SIZE = 128
CONTEXT_FRACTION = 0.25
MIN_SIDE = 16                  # guards against degenerate crops from tiny bboxes
N_CHANNELS = 6
BEFORE_SLICE = slice(0, 3)
AFTER_SLICE = slice(3, 6)
RESAMPLE = Image.BILINEAR

#: Machine-readable record of the protocol, for provenance/model cards.
PROTOCOL = {
    "crop_size": CROP_SIZE,
    "context_fraction": CONTEXT_FRACTION,
    "min_side_px": MIN_SIDE,
    "channels": "0:3 = BEFORE RGB, 3:6 = AFTER RGB",
    "resample": "PIL bilinear",
    "normalisation": "ImageNet mean/std per date",
}

# 6-channel normaliser: the same ImageNet statistics for each date. Built
# exactly as RegionCrops builds it, so the two agree bit for bit.
_MEAN6 = np.concatenate([np.asarray(IMAGENET_MEAN, dtype=np.float32),
                         np.asarray(IMAGENET_MEAN, dtype=np.float32)]).reshape(6, 1, 1)
_STD6 = np.concatenate([np.asarray(IMAGENET_STD, dtype=np.float32),
                        np.asarray(IMAGENET_STD, dtype=np.float32)]).reshape(6, 1, 1)

ImageLike = Union[Image.Image, np.ndarray]


# ----------------------------------------------------------------- geometry
def crop_box(x: float, y: float, w: float, h: float,
             img_w: int, img_h: int) -> Tuple[int, int, int]:
    """Square, context-expanded box clamped into the scene.

    Returns ``(x0, y0, side)`` in source pixels. This is the canonical geometry;
    it is a verbatim port of the implementation the training crops were built
    with, and must not be "improved" without regenerating the crops.
    """
    mx, my = CONTEXT_FRACTION * w, CONTEXT_FRACTION * h
    x0, x1 = x - mx, x + w + mx
    y0, y1 = y - my, y + h + my
    side = max(x1 - x0, y1 - y0, MIN_SIDE)
    side = min(side, img_w, img_h)            # cannot exceed the scene
    cx, cy = x + w / 2.0, y + h / 2.0
    x0 = cx - side / 2.0
    y0 = cy - side / 2.0
    x0 = min(max(0.0, x0), img_w - side)      # shift, do not shrink
    y0 = min(max(0.0, y0), img_h - side)
    return int(round(x0)), int(round(y0)), int(round(side))


# -------------------------------------------------------------- image input
def as_pil_rgb(image: ImageLike) -> Image.Image:
    """PIL RGB view of a PIL image or a uint8 array. Never resizes."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        raise TypeError(
            f"expected a uint8 image, got dtype {arr.dtype}. Convert explicitly "
            "rather than relying on a silent cast.")
    return Image.fromarray(arr).convert("RGB")


def crop_image(image: ImageLike, box: Sequence[int]) -> np.ndarray:
    """One image cropped to `box` and resized to CROP_SIZE. uint8 HWC, 3 channels."""
    x0, y0, side = box
    img = as_pil_rgb(image)
    return np.asarray(
        img.crop((x0, y0, x0 + side, y0 + side))
           .resize((CROP_SIZE, CROP_SIZE), RESAMPLE),
        dtype=np.uint8)


def crop_pair(before: ImageLike, after: ImageLike,
              bbox_xywh: Sequence[int]) -> np.ndarray:
    """BEFORE and AFTER crops for one region, stacked.

    Returns uint8 ``(CROP_SIZE, CROP_SIZE, 6)`` with BEFORE in channels 0:3 and
    AFTER in 3:6 - the ordering the trained classifier expects.

    The box is computed from the BEFORE image's size; the two dates must be the
    same size, which is already a precondition of change detection.
    """
    b = as_pil_rgb(before)
    a = as_pil_rgb(after)
    if b.size != a.size:
        raise ValueError(
            f"BEFORE and AFTER differ in size: {b.size} vs {a.size}. The two "
            "dates must cover the same extent at the same size.")
    x, y, w, h = bbox_xywh
    box = crop_box(x, y, w, h, b.width, b.height)
    out = np.empty((CROP_SIZE, CROP_SIZE, N_CHANNELS), dtype=np.uint8)
    out[:, :, BEFORE_SLICE] = crop_image(b, box)
    out[:, :, AFTER_SLICE] = crop_image(a, box)
    return out


# ------------------------------------------------------------ model input
def to_model_input(crop_hwc6: np.ndarray, batched: bool = True) -> torch.Tensor:
    """Normalised model input from a stacked uint8 crop.

    Arithmetic is identical to RegionCrops.__getitem__ with augmentation off:
    transpose to CHW, cast to float32, divide by 255, then subtract the
    6-channel ImageNet mean and divide by the 6-channel std.

    Returns float32 ``(1, 6, 128, 128)`` when `batched`, else ``(6, 128, 128)``.
    """
    crop = np.asarray(crop_hwc6)
    if crop.ndim != 3 or crop.shape[2] != N_CHANNELS:
        raise ValueError(
            f"expected a (H, W, {N_CHANNELS}) stacked crop, got shape {crop.shape}")
    if crop.dtype != np.uint8:
        raise TypeError(f"expected uint8 crop, got dtype {crop.dtype}")
    arr = np.ascontiguousarray(crop.transpose(2, 0, 1), dtype=np.float32) / 255.0
    arr = (arr - _MEAN6) / _STD6
    t = torch.from_numpy(arr)
    return t.unsqueeze(0) if batched else t


def region_model_input(before: ImageLike, after: ImageLike,
                       bbox_xywh: Sequence[int],
                       batched: bool = True) -> torch.Tensor:
    """Image pair + one region bbox -> the exact input the classifier expects.

    The whole single-region inference preprocessing path in one call. It does
    not load or run a model.
    """
    return to_model_input(crop_pair(before, after, bbox_xywh), batched=batched)


__all__ = [
    "CROP_SIZE", "CONTEXT_FRACTION", "MIN_SIDE", "N_CHANNELS",
    "BEFORE_SLICE", "AFTER_SLICE", "RESAMPLE", "PROTOCOL",
    "crop_box", "as_pil_rgb", "crop_image", "crop_pair",
    "to_model_input", "region_model_input",
]
