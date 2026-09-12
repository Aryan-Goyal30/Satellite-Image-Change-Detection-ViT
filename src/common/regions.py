"""Region extraction from a binary change mask.

Pure post-processing: no model, no dataset, no UI. Applications and engines both
use this so region semantics are defined in exactly one place.

What a region is
----------------
A region is a **connected component of the binary change mask** - a contiguous
patch of pixels the model scored above the decision threshold. It is an area of
detected change. It is NOT an identified building, and the engine cannot say
whether the change was construction or demolition.

Region confidence
-----------------
`confidence` is the **mean predicted change probability over the pixels of that
region**, taken from the model's probability map. It is a summary of how
strongly the model scored that patch, on the same 0-1 scale as the threshold.
It is not a calibrated probability that the region is "really" a change, and it
is not validated per region - only the pixel-level metrics are. When no
probability map is supplied, confidence stays None rather than being invented.

Minimum area
------------
Components smaller than `min_area_px` are discarded as noise and removed from
the mask, so the reported changed-pixel count always matches the regions shown.
The default lives in src/config.py so it is configurable in one place.

Geography
---------
When a GeoRef with a metric scale is supplied, each region also carries
`area_m2` and its centroid in CRS coordinates. Without one, both stay None.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from src import config
from src.core.types import GeoRef, Region

DEFAULT_MIN_AREA_PX = config.DEFAULT_MIN_AREA_PX


def _connected_components(mask: np.ndarray):
    """8-connected labelling. Uses OpenCV, already a project dependency."""
    import cv2
    return cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)


def pixel_to_crs(transform, x: float, y: float):
    """Map a pixel coordinate to CRS coordinates using a GDAL-style transform.

    transform = (origin_x, pixel_width, row_rotation,
                 origin_y, column_rotation, pixel_height)
    """
    ox, sx, ry, oy, rx, sy = transform
    return (ox + x * sx + y * ry, oy + x * rx + y * sy)


def extract_regions(mask: np.ndarray,
                    score_map: Optional[np.ndarray] = None,
                    min_area_px: int = DEFAULT_MIN_AREA_PX,
                    georef: Optional[GeoRef] = None,
                    layer: Optional[str] = None):
    """Connected components of `mask`, filtered by area.

    Returns ``(filtered_mask, regions)`` where `filtered_mask` has the discarded
    components removed, so quantities computed from it agree with the regions.

    Region IDs are deterministic: components are ordered by descending pixel
    area, ties broken by the labelling order, then numbered from 1.
    """
    n, labels, stats, centroids = _connected_components(mask)

    import cv2
    kept = []
    for i in range(1, n):                       # label 0 is background
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area_px:
            labels[labels == i] = 0
            continue
        kept.append((
            area,
            (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
             int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])),
            (round(float(centroids[i][0]), 1), round(float(centroids[i][1]), 1)),
            i,
        ))

    # Stable sort on descending area: deterministic, and ties keep label order.
    kept.sort(key=lambda r: -r[0])

    filtered_mask = labels > 0
    has_scale = georef is not None and georef.has_scale

    regions = []
    for new_id, (area, bbox, centroid, label_value) in enumerate(kept, 1):
        confidence = None
        if score_map is not None:
            component = labels == label_value
            if component.any():
                confidence = round(float(score_map[component].mean()), 4)

        area_m2 = None
        centroid_crs = None
        if has_scale:
            area_m2 = round(area * georef.gsd_m * georef.gsd_m, 2)
        if georef is not None and georef.transform is not None:
            cx, cy = pixel_to_crs(georef.transform, centroid[0], centroid[1])
            centroid_crs = (round(cx, 3), round(cy, 3))

        regions.append(Region(
            id=new_id, area_px=area, bbox_xywh=list(bbox), centroid_xy=list(centroid),
            confidence=confidence, layer=layer, area_m2=area_m2,
            centroid_crs_xy=centroid_crs))

    return filtered_mask, regions
