"""Native ~30 m spatial support for the environmental experiments (Phase 4B-4).

Why this module exists
----------------------
TMF supervision has ~30 m native support; the model predicts on a 10 m grid.
Phase 4B-3 measured the consequence: 272 ground-truth components (median 66 px)
against 80 predicted (median 446 px) while predicted AREA was 1.18x the ground
truth. The topology disagrees while the extent roughly agrees, which is what a
30 m label resampled to 10 m would produce. E3 tests whether moving the loss and
the primary evaluation to ~30 m support resolves that.

How the v24 label was actually built
------------------------------------
Each 10 m pixel centre was converted to lon/lat and NEAREST-NEIGHBOUR sampled
from the 30 m TMF raster, which lives in EPSG:4326 while the imagery lives on a
UTM metre grid. The label is therefore piecewise-constant over irregular
footprints, not over an axis-aligned 3 x 3 lattice.

Measured on v24 test samples:

    mean 10 m pixels per distinct TMF pixel   8.56   (9 would be exact)
    mean distinct TMF pixels per 3 x 3 block  2.97   (1.00 would be aligned)

So a fixed 3 x 3 block grid reproduces the SCALE of TMF support but NOT the
identity of individual TMF pixels - the two grids share neither origin nor
orientation, and no fixed lattice can align them. This is stated plainly rather
than implied away: the pooled target is a scale-matched approximation of native
support, which is what the hypothesis concerns, and it is not a pixel-exact
reconstruction of the TMF grid.

Edge handling
-------------
256 = 3 x 85 + 1, so the crop does not divide evenly. The input is padded
symmetrically by one pixel with edge replication to 258 = 3 x 86 before pooling.
No pixel is discarded and no spatial information is silently cropped; the only
cost is that the first and last block along each axis contain one replicated
pixel.
"""
from __future__ import annotations

import numpy as np

#: Pooling geometry. 10 m x 3 = 30 m, matching TMF's native support.
BLOCK = 3
PAD = 1
TILE = 256
GRID = (TILE + 2 * PAD) // BLOCK        # 86

#: A 30 m cell is positive when most of it is covered by 10 m positives.
#: Majority is the rule that recovers the underlying TMF value wherever a block
#: does sit inside one TMF footprint.
LABEL_MAJORITY = 0.5

DESCRIPTION = {
    "block": BLOCK, "pad": PAD, "grid": [GRID, GRID],
    "pooling": "mean over 3x3 blocks after symmetric edge padding of 1 (256 -> 258 = 3*86)",
    "label_rule": f"30 m cell positive when >= {LABEL_MAJORITY} of its 10 m pixels are positive",
    "edge_handling": "edge replication, no pixel discarded",
    "alignment_caveat": ("3x3 blocks reproduce the SCALE of TMF support, not the "
                         "identity of TMF pixels: measured 2.97 distinct TMF "
                         "pixels per block (1.00 would be aligned), because the "
                         "TMF EPSG:4326 grid and the UTM imagery grid share "
                         "neither origin nor orientation"),
    "physical_cell_m": 30,
}


def pool_mean_torch(x):
    """Mean-pool a B,C,256,256 tensor to B,C,86,86 at ~30 m support."""
    import torch.nn.functional as F
    padded = F.pad(x, (PAD, PAD, PAD, PAD), mode="replicate")
    return F.avg_pool2d(padded, BLOCK, stride=BLOCK)


def pool_mean_numpy(array: np.ndarray) -> np.ndarray:
    """Mean-pool a 256x256 array to 86x86 fractional coverage."""
    padded = np.pad(np.asarray(array, dtype=np.float32),
                    ((PAD, PAD), (PAD, PAD)), mode="edge")
    return padded.reshape(GRID, BLOCK, GRID, BLOCK).mean(axis=(1, 3))


def label_30m(label: np.ndarray, majority: float = LABEL_MAJORITY) -> np.ndarray:
    """Binary 30 m-support target from the frozen 10 m label.

    The 10 m label itself is never modified - this is a view of it at the
    support the supervision actually has.
    """
    return (pool_mean_numpy(label) >= majority).astype(np.float32)


#: Component-area floor at 30 m, chosen to match the PHYSICAL area of the 10 m
#: floor rather than the pixel count. 32 px x 100 m2 = 3,200 m2; at 900 m2 per
#: 30 m pixel that is 3.56, rounded up to 4. Reusing 32 unchanged would impose a
#: 9x larger physical floor and delete components purely by changing units.
MIN_AREA_PX_30M = 4
