"""Sliding-window inference over images larger than a model's training tile.

Extracted verbatim from the Built Environment engine so that a second domain
does not grow a second copy of the same blending arithmetic. The operations and
their order are unchanged, so results are bit-identical to the previous
in-engine implementation.

Why Hann weighting
------------------
A model trained on 256 x 256 crops sees less context near a tile's border than
at its centre, so abutting tiles disagree along their shared edge and leave a
visible seam. Overlapping the tiles and weighting each one by a 2D Hann window
makes every pixel a smooth blend of the tiles that saw it, weighted towards the
tile that saw it most centrally.

What this module does NOT do
----------------------------
It does not normalise, batch, or know anything about a model: the caller passes
a `predict` callable that turns a list of co-located tiles into one probability
map. Normalisation is part of a model's input contract and stays in its domain.
"""
from __future__ import annotations

import numpy as np


def hann2d(size: int) -> np.ndarray:
    """2D Hann window for seamless blending of overlapping tiles."""
    w = np.hanning(size + 2)[1:-1]          # drop the zero endpoints
    win = np.outer(w, w).astype(np.float32)
    return np.maximum(win, 1e-3)            # never exactly zero


def padded_size(extent: int, tile: int, stride: int) -> int:
    """Smallest size >= extent that a whole number of tiles covers exactly."""
    return max(tile, int(np.ceil(max(extent - tile, 0) / stride)) * stride + tile)


def sliding_window_probability(arrays, predict, tile: int = 256,
                               overlap: int = 64) -> np.ndarray:
    """Blend per-tile probability maps into one full-resolution map.

    Args:
        arrays:  co-located H x W x C arrays (e.g. the two dates). All must
                 share H and W; channel counts may differ.
        predict: callable(list_of_tiles) -> tile x tile float array in [0, 1].
                 One call per window position.
        tile:    the model's native input size.
        overlap: pixels shared between neighbouring tiles.

    Returns an H x W float32 array cropped back to the input extent.
    """
    if not arrays:
        raise ValueError("sliding_window_probability needs at least one array")
    H, W = arrays[0].shape[:2]
    for other in arrays[1:]:
        if other.shape[:2] != (H, W):
            raise ValueError(
                f"Array extents differ: {(H, W)} vs {other.shape[:2]}. "
                "The two dates must cover the same extent at the same size.")

    stride = tile - overlap
    # pad so every position is covered by a full tile
    ph = padded_size(H, tile, stride)
    pw = padded_size(W, tile, stride)
    # 'symmetric' rather than 'reflect': when the image is smaller than one
    # tile the pad width exceeds the dimension, which older numpy rejects in
    # reflect mode. The padded margin is cropped off again below, so the
    # choice only has to be safe, not principled.
    pad = ((0, ph - H), (0, pw - W), (0, 0))
    padded = [np.pad(a, pad, mode="symmetric") for a in arrays]

    acc = np.zeros((ph, pw), dtype=np.float32)
    wsum = np.zeros((ph, pw), dtype=np.float32)
    win = hann2d(tile)

    rows = list(range(0, ph - tile + 1, stride))
    cols = list(range(0, pw - tile + 1, stride))
    for r in rows:
        for c in cols:
            prob = predict([a[r:r + tile, c:c + tile] for a in padded])
            acc[r:r + tile, c:c + tile] += prob * win
            wsum[r:r + tile, c:c + tile] += win

    return (acc / np.maximum(wsum, 1e-6))[:H, :W]
