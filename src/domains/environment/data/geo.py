"""WGS84 <-> UTM point conversion.

Phase 4A needs this because the two data sources live in different coordinate
systems: JRC TMF is distributed in EPSG:4326 (degrees) while Sentinel-2 L2A is
in a UTM zone (metres, here EPSG:32721). Aligning a 30 m geographic label grid
to a 10 m UTM imagery grid therefore requires converting COORDINATES.

This is point conversion only. Nothing here warps, resamples or reprojects an
image - the imagery is always read on its own native UTM grid, and only the
sample positions are converted. That keeps the project free of a GDAL/pyproj
dependency, which src/common/georef.py also deliberately avoids.

Accuracy: the standard Snyder series for the transverse Mercator projection.
Round-trip error is 0.0000 m at test points spanning the tile, and the derived
corners agree with the Sentinel-2 product's own STAC bbox to within a pixel
(see tests/test_phase4a.py).
"""
from __future__ import annotations

import math

# WGS84 ellipsoid
A = 6378137.0
F = 1 / 298.257223563
E2 = F * (2 - F)
EP2 = E2 / (1 - E2)
K0 = 0.9996
FALSE_EASTING = 500000.0
FALSE_NORTHING_SOUTH = 10000000.0


def zone_central_meridian(zone: int) -> float:
    """Central meridian of a UTM zone, in degrees."""
    return (zone - 1) * 6 - 180 + 3


def utm_to_lonlat(easting: float, northing: float, zone: int,
                  south: bool = True) -> tuple:
    """UTM (metres) -> (lon, lat) in degrees."""
    x = easting - FALSE_EASTING
    y = northing - (FALSE_NORTHING_SOUTH if south else 0.0)

    e1 = (1 - math.sqrt(1 - E2)) / (1 + math.sqrt(1 - E2))
    m = y / K0
    mu = m / (A * (1 - E2 / 4 - 3 * E2 ** 2 / 64 - 5 * E2 ** 3 / 256))
    phi1 = (mu
            + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * math.sin(2 * mu)
            + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * math.sin(4 * mu)
            + (151 * e1 ** 3 / 96) * math.sin(6 * mu)
            + (1097 * e1 ** 4 / 512) * math.sin(8 * mu))

    c1 = EP2 * math.cos(phi1) ** 2
    t1 = math.tan(phi1) ** 2
    n1 = A / math.sqrt(1 - E2 * math.sin(phi1) ** 2)
    r1 = A * (1 - E2) / (1 - E2 * math.sin(phi1) ** 2) ** 1.5
    d = x / (n1 * K0)

    lat = phi1 - (n1 * math.tan(phi1) / r1) * (
        d ** 2 / 2
        - (5 + 3 * t1 + 10 * c1 - 4 * c1 ** 2 - 9 * EP2) * d ** 4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1 ** 2 - 252 * EP2 - 3 * c1 ** 2)
        * d ** 6 / 720)
    lon = (d
           - (1 + 2 * t1 + c1) * d ** 3 / 6
           + (5 - 2 * c1 + 28 * t1 - 3 * c1 ** 2 + 8 * EP2 + 24 * t1 ** 2)
           * d ** 5 / 120) / math.cos(phi1)

    lon0 = math.radians(zone_central_meridian(zone))
    return math.degrees(lon0 + lon), math.degrees(lat)


def grid_lonlat(origin_e: float, origin_n: float, col0: int, row0: int,
                width: int, height: int, gsd: float, zone: int,
                south: bool = True):
    """Longitude/latitude of every pixel centre in a UTM raster window.

    Returns two `height` x `width` float arrays. Used to look up a geographic
    label raster (TMF, EPSG:4326) at the position of each imagery pixel, which
    is how a 30 m geographic grid is aligned to a 10 m UTM grid without
    reprojecting either raster.

    The conversion is per pixel rather than interpolated from the corners: over
    a 2.56 km window the error from interpolating would be small but not zero,
    and the label is already only 30 m-accurate, so there is no reason to add a
    second approximation on top of the first.
    """
    import numpy as np

    lons = np.empty((height, width), dtype=np.float64)
    lats = np.empty((height, width), dtype=np.float64)
    for row in range(height):
        northing = origin_n - (row0 + row + 0.5) * gsd
        for col in range(width):
            easting = origin_e + (col0 + col + 0.5) * gsd
            lon, lat = utm_to_lonlat(easting, northing, zone, south=south)
            lons[row, col] = lon
            lats[row, col] = lat
    return lons, lats


def lonlat_to_utm(lon: float, lat: float, zone: int,
                  south: bool = True) -> tuple:
    """(lon, lat) in degrees -> UTM (easting, northing) in metres."""
    lo = math.radians(lon)
    la = math.radians(lat)
    lon0 = math.radians(zone_central_meridian(zone))

    n = A / math.sqrt(1 - E2 * math.sin(la) ** 2)
    t = math.tan(la) ** 2
    c = EP2 * math.cos(la) ** 2
    a_ = math.cos(la) * (lo - lon0)
    m = A * ((1 - E2 / 4 - 3 * E2 ** 2 / 64 - 5 * E2 ** 3 / 256) * la
             - (3 * E2 / 8 + 3 * E2 ** 2 / 32 + 45 * E2 ** 3 / 1024) * math.sin(2 * la)
             + (15 * E2 ** 2 / 256 + 45 * E2 ** 3 / 1024) * math.sin(4 * la)
             - (35 * E2 ** 3 / 3072) * math.sin(6 * la))

    easting = K0 * n * (a_
                        + (1 - t + c) * a_ ** 3 / 6
                        + (5 - 18 * t + t ** 2 + 72 * c - 58 * EP2) * a_ ** 5 / 120
                        ) + FALSE_EASTING
    northing = K0 * (m + n * math.tan(la) * (
        a_ ** 2 / 2
        + (5 - t + 9 * c + 4 * c ** 2) * a_ ** 4 / 24
        + (61 - 58 * t + t ** 2 + 600 * c - 330 * EP2) * a_ ** 6 / 720))
    if south:
        northing += FALSE_NORTHING_SOUTH
    return easting, northing
