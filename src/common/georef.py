"""GeoTIFF georeferencing, read from TIFF tags with Pillow.

No new dependency: Pillow already exposes the GeoTIFF tags this needs
(ModelPixelScale, ModelTiepoint, ModelTransformation, GeoKeyDirectory). rasterio
and GDAL would pull in a large binary stack for metadata this module can read
directly, so they are deliberately not used.

What is read
------------
    CRS              from the GeoKey directory, as "EPSG:<code>" when declared
    transform        a GDAL-style 6-tuple derived from the tags
    pixel size       the raster's pixel dimensions in CRS units
    gsd_m            ONLY when the CRS is projected and its linear unit is metre
    band count       from the image itself

When metres are available
-------------------------
`gsd_m` is set only if all of these hold:

    * a projected coordinate system is declared (GTModelType = projected)
    * the linear-unit key is absent or states metre (EPSG 9001)
    * pixel width and height are positive and equal to within 0.1 %

A geographic CRS (degrees, e.g. EPSG:4326) never yields `gsd_m`: converting
degrees to metres depends on latitude and would be an assumption, not a
measurement. Without `gsd_m` no square-metre figure is ever produced.

This module does not reproject, resample or register anything. It reads metadata
and reports honestly when a safe geographic interpretation is unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.core.types import GeoRef

# GeoTIFF TIFF tag numbers
MODEL_PIXEL_SCALE = 33550
MODEL_TIEPOINT = 33922
MODEL_TRANSFORMATION = 34264
GEO_KEY_DIRECTORY = 34735

# GeoKey IDs
GT_MODEL_TYPE = 1024        # 1 = projected, 2 = geographic, 3 = geocentric
GEOGRAPHIC_TYPE = 2048      # EPSG code of a geographic CRS
PROJECTED_CS_TYPE = 3072    # EPSG code of a projected CRS
PROJ_LINEAR_UNITS = 3076    # 9001 = metre

MODEL_PROJECTED = 1
MODEL_GEOGRAPHIC = 2
LINEAR_UNIT_METRE = 9001
UNDEFINED_EPSG = {0, 32767}


@dataclass(frozen=True)
class RasterInfo:
    """Everything read from one image, georeferenced or not."""
    width: int
    height: int
    band_count: int
    mode: str
    georef: Optional[GeoRef] = None
    is_geotiff: bool = False
    scale_note: str = ""      # why a metric scale is or is not available

    @property
    def has_crs(self) -> bool:
        return self.georef is not None and self.georef.crs is not None

    @property
    def has_metric_scale(self) -> bool:
        return self.georef is not None and self.georef.has_scale

    def to_dict(self) -> dict:
        return {"width": self.width, "height": self.height,
                "band_count": self.band_count, "mode": self.mode,
                "is_geotiff": self.is_geotiff,
                "georef": self.georef.to_dict() if self.georef else None,
                "scale_note": self.scale_note}


def _parse_geokeys(directory) -> dict:
    """GeoKeyDirectory -> {key_id: value} for the immediate (in-directory) values."""
    if not directory or len(directory) < 4:
        return {}
    values = list(directory)
    n_keys = int(values[3])
    keys = {}
    for k in range(n_keys):
        base = 4 + k * 4
        if base + 3 >= len(values):
            break
        key_id, tag_location, count, value_offset = values[base:base + 4]
        # tag_location 0 means the value is stored inline in value_offset.
        if int(tag_location) == 0 and int(count) == 1:
            keys[int(key_id)] = int(value_offset)
    return keys


def _transform_from_tags(tags):
    """GDAL-style 6-tuple from pixel-scale + tiepoint, or the 4x4 transform tag."""
    scale = tags.get(MODEL_PIXEL_SCALE)
    tie = tags.get(MODEL_TIEPOINT)
    if scale and tie and len(scale) >= 2 and len(tie) >= 6:
        sx, sy = float(scale[0]), float(scale[1])
        i, j, _k, x, y, _z = [float(v) for v in tie[:6]]
        # origin of pixel (0, 0); y decreases down the raster
        origin_x = x - i * sx
        origin_y = y + j * sy
        return (origin_x, sx, 0.0, origin_y, 0.0, -sy), (sx, sy)

    matrix = tags.get(MODEL_TRANSFORMATION)
    if matrix and len(matrix) >= 8:
        m = [float(v) for v in matrix]
        # row-major 4x4: [a b _ c ; d e _ f ; ...]
        transform = (m[3], m[0], m[1], m[7], m[4], m[5])
        return transform, (abs(m[0]), abs(m[5]))

    return None, None


def read_raster_info(img) -> RasterInfo:
    """Inspect a PIL image, extracting GeoTIFF metadata when present."""
    width, height = img.size
    bands = len(img.getbands())
    tags = getattr(img, "tag_v2", None)

    if not tags or MODEL_PIXEL_SCALE not in tags and MODEL_TRANSFORMATION not in tags \
            and GEO_KEY_DIRECTORY not in tags:
        return RasterInfo(width, height, bands, img.mode, None, False,
                          "Not a georeferenced image: pixel counts only.")

    transform, pixel_size = _transform_from_tags(tags)
    keys = _parse_geokeys(tags.get(GEO_KEY_DIRECTORY))

    model_type = keys.get(GT_MODEL_TYPE)
    projected_epsg = keys.get(PROJECTED_CS_TYPE)
    geographic_epsg = keys.get(GEOGRAPHIC_TYPE)
    linear_units = keys.get(PROJ_LINEAR_UNITS)

    crs = None
    units = None
    if projected_epsg and projected_epsg not in UNDEFINED_EPSG:
        crs = f"EPSG:{projected_epsg}"
    elif geographic_epsg and geographic_epsg not in UNDEFINED_EPSG:
        crs = f"EPSG:{geographic_epsg}"

    gsd_m = None
    note = ""
    is_projected = model_type == MODEL_PROJECTED or (
        model_type is None and projected_epsg not in (None, *UNDEFINED_EPSG))

    if crs is None:
        note = ("This GeoTIFF declares no CRS, so geographic conversion is "
                "unavailable. Pixel counts only.")
    elif model_type == MODEL_GEOGRAPHIC or (geographic_epsg and not projected_epsg):
        units = "degrees"
        note = (f"{crs} is a geographic CRS: pixel size is in degrees. Converting "
                f"to metres depends on latitude, so no square-metre figure is "
                f"reported.")
    elif not is_projected:
        note = "Coordinate system type is not projected; metric area unavailable."
    elif linear_units is not None and linear_units != LINEAR_UNIT_METRE:
        units = f"EPSG unit {linear_units}"
        note = (f"Projected CRS uses linear unit {linear_units}, not metre; no "
                f"square-metre figure is reported.")
    elif pixel_size is None:
        note = "No pixel scale in the file; metric area unavailable."
    else:
        sx, sy = pixel_size
        if sx <= 0 or sy <= 0:
            note = "Pixel scale is not positive; metric area unavailable."
        elif abs(sx - sy) / max(sx, sy) > 1e-3:
            units = "metre"
            note = (f"Pixel is not square ({sx:g} x {sy:g} m); a single ground "
                    f"sample distance would be misleading, so no square-metre "
                    f"figure is reported.")
        else:
            units = "metre"
            gsd_m = float(sx)
            note = f"Projected CRS {crs} with {sx:g} m pixels."

    georef = GeoRef(crs=crs, transform=transform, gsd_m=gsd_m, units=units,
                    pixel_size=pixel_size)
    return RasterInfo(width, height, bands, img.mode, georef, True, note)


def transforms_match(a, b, rel_tol: float = 1e-9) -> bool:
    """True when two geotransforms describe the same pixel grid."""
    if a is None or b is None:
        return a is b
    if len(a) != len(b):
        return False
    for u, v in zip(a, b):
        scale = max(abs(u), abs(v), 1.0)
        if abs(u - v) > rel_tol * scale:
            return False
    return True


def pair_geo_issues(before: RasterInfo, after: RasterInfo) -> list:
    """Geospatial compatibility problems between two rasters.

    Returns ``(level, message)`` pairs. "error" means the pair must not be
    analysed as if it were aligned; "warning" means analysis is safe but some
    geographic output is unavailable.

    No reprojection or registration is attempted: incompatible inputs are
    reported, not silently corrected.
    """
    issues = []
    if not (before.is_geotiff or after.is_geotiff):
        return issues

    if before.is_geotiff != after.is_geotiff:
        issues.append(("warning",
            "Only one image is a GeoTIFF. Geographic output needs georeferencing "
            "on both; reporting pixel counts only."))
        return issues

    b_crs = before.georef.crs if before.georef else None
    a_crs = after.georef.crs if after.georef else None
    if b_crs and a_crs and b_crs != a_crs:
        issues.append(("error",
            f"The two GeoTIFFs use different coordinate reference systems "
            f"({b_crs} and {a_crs}). Earth Guardian does not reproject; supply "
            f"both images in the same CRS."))
    elif not (b_crs and a_crs):
        issues.append(("warning",
            "At least one GeoTIFF declares no CRS, so geographic conversion is "
            "unavailable. Analysis will report pixel counts only."))

    b_t = before.georef.transform if before.georef else None
    a_t = after.georef.transform if after.georef else None
    if b_t and a_t and not transforms_match(b_t, a_t):
        issues.append(("error",
            "The two GeoTIFFs have different geotransforms, so their pixels do "
            "not cover the same ground. Earth Guardian does not resample or "
            "register; supply images already on the same grid."))

    return issues


def pair_georef(before: RasterInfo, after: RasterInfo) -> Optional[GeoRef]:
    """The GeoRef to attach to a result, or None when it would be unsafe."""
    if not (before.is_geotiff and after.is_geotiff):
        return None
    if before.georef is None or after.georef is None:
        return None
    if before.georef.crs != after.georef.crs:
        return None
    if not transforms_match(before.georef.transform, after.georef.transform):
        return None
    return before.georef
