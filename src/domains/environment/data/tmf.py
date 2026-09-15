"""JRC Tropical Moist Forest labels for the Environment domain (Phase 4A).

Source: JRC TMF, Vancutsem et al., Science Advances 7:eabe1603 (2021).
Distributed free of charge without restriction of use, citation required.

Product facts verified on the downloaded tile rather than assumed:

    format      BigTIFF, uint16, ~37107 x 37246 px per 10 x 10 degree tile
    CRS         EPSG:4326 (geographic - degrees, NOT metres)
    pixel       0.00026949458523585647 deg  (~30 m near the equator)

`DeforestationYear` is, in JRC's own words, "the year when the forest cover has
been deforested for the first time (followed or not by a regrowth)". Values are
discrete years 1982-2025, with 0 meaning no deforestation recorded.

What the label is and is not
----------------------------
The positive rule is equality on the deforestation year. Degradation and
regrowth live in *different* TMF layers and are never positives here, so they
cannot leak in. JRC's definition does however include conversion to tree
plantations and to water, so "deforestation" here means forest cover loss, not
necessarily illegal or permanent clearing - that caveat belongs in any claim
made about a model trained on it.

The label is produced at 30 m and nearest-neighbour sampled onto the 10 m
Sentinel-2 grid. The result is a 10 m array carrying **30 m-quantised
supervision**: its boundaries are accurate to about one TMF pixel, not to 10 m.
"""
from __future__ import annotations

import numpy as np

LAYER = "DeforestationYear"
DATASET = "JRC Tropical Moist Forest (TMF)"
CITATION = ("Vancutsem, C. et al. Long-term (1990-2019) monitoring of forest "
            "cover changes in the humid tropics. Science Advances 7, eabe1603 (2021)")
LICENCE = "Free of charge, without restriction of use; citation required (JRC)"

#: Discrete years the layer can carry; 0 means "no deforestation recorded".
MIN_YEAR = 1982
MAX_YEAR = 2025
NO_DEFORESTATION = 0

#: Native resolution of the distributed product.
NATIVE_RESOLUTION_M = 30
LABEL_RESAMPLE = "nearest-neighbour, 30 m -> 10 m (30 m-quantised supervision)"

#: Transition-map code for disturbances that began 2023-2025 and cannot yet be
#: attributed to degradation or deforestation. It lives in the transition map,
#: not in DeforestationYear, so it is never a positive here - recorded so the
#: exclusion is explicit rather than accidental.
ONGOING_DISTURBANCE_CLASS = 50
EXCLUDED_CLASSES = (ONGOING_DISTURBANCE_CLASS,)


def geotransform(origin_lon: float, origin_lat: float, pixel_deg: float) -> dict:
    """Bundle a TMF tile's geographic grid definition."""
    return {"origin_lon": origin_lon, "origin_lat": origin_lat,
            "pixel_deg": pixel_deg}


def lonlat_to_pixel(lon, lat, grid: dict):
    """Geographic coordinates -> TMF (col, row), nearest pixel."""
    col = np.round((np.asarray(lon) - grid["origin_lon"]) / grid["pixel_deg"])
    row = np.round((np.asarray(lat) - grid["origin_lat"]) / -grid["pixel_deg"])
    return col.astype(np.int64), row.astype(np.int64)


def label_from_years(years: np.ndarray, year: int) -> np.ndarray:
    """Binary target: 1 where TMF records first deforestation in `year`.

    Equality, never a range: a pixel deforested in a different year is a
    negative, because the Sentinel-2 pair only brackets this one year.
    """
    if not (MIN_YEAR <= year <= MAX_YEAR):
        raise ValueError(
            f"year {year} outside the TMF range {MIN_YEAR}-{MAX_YEAR}")
    return (np.asarray(years) == year).astype(np.uint8)


def positive_fraction(label: np.ndarray) -> float:
    return float(np.asarray(label).mean())


def describe_years(years: np.ndarray) -> dict:
    """Year histogram of a window, for manifest provenance."""
    values, counts = np.unique(np.asarray(years), return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}
