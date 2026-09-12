"""Focused tests for Stage 3A: region intelligence and the GeoTIFF foundation.

pytest is not a project dependency, so these are plain functions that assert.
Run either way:

    python tests/test_stage3a.py          # standalone runner
    pytest tests/test_stage3a.py          # if pytest is available

No model weights are needed: region and georeference logic is pure and is
exercised on tiny synthetic fixtures.
"""
import io
import os
import sys

import numpy as np
from PIL import Image, TiffImagePlugin

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.common.georef import (GEO_KEY_DIRECTORY, MODEL_PIXEL_SCALE, MODEL_TIEPOINT,
                               pair_geo_issues, pair_georef, read_raster_info,
                               transforms_match)
from src.common.image_input import open_image, validate_pair
from src.common.regions import DEFAULT_MIN_AREA_PX, extract_regions, pixel_to_crs
from src.core.types import GeoRef, Quantities, Region


# --------------------------------------------------------------- fixtures
def make_geotiff(size=(16, 16), pixel_scale=(0.5, 0.5), origin=(500000.0, 4649776.0),
                 epsg=32610, model_type=1, linear_units=9001, with_geokeys=True,
                 bands=3):
    """A tiny deterministic GeoTIFF in memory."""
    w, h = size
    arr = np.zeros((h, w, bands), np.uint8) if bands > 1 else np.zeros((h, w), np.uint8)
    arr = arr + np.arange(h, dtype=np.uint8).reshape(h, 1, 1) if bands > 1 else arr
    info = TiffImagePlugin.ImageFileDirectory_v2()
    if pixel_scale is not None:
        info[MODEL_PIXEL_SCALE] = (pixel_scale[0], pixel_scale[1], 0.0)
        info[MODEL_TIEPOINT] = (0.0, 0.0, 0.0, origin[0], origin[1], 0.0)
    if with_geokeys:
        keys = [1024, 0, 1, model_type]
        if model_type == 1:
            keys += [3072, 0, 1, epsg]
            if linear_units is not None:
                keys += [3076, 0, 1, linear_units]
        else:
            keys += [2048, 0, 1, epsg]
        n = len(keys) // 4
        info[GEO_KEY_DIRECTORY] = tuple([1, 1, 0, n] + keys)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="TIFF", tiffinfo=info)
    buf.seek(0)
    return buf


def make_png(size=(16, 16)):
    buf = io.BytesIO()
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (size[1], size[0], 3),
                                                      dtype=np.uint8)).save(buf, "PNG")
    buf.seek(0)
    return buf


# ----------------------------------------------------------------- regions
def test_region_extraction_basic():
    mask = np.zeros((40, 40), bool)
    mask[5:15, 5:15] = True          # 100 px
    mask[25:31, 25:35] = True        # 60 px
    filtered, regions = extract_regions(mask, min_area_px=1)
    assert len(regions) == 2
    assert [r.area_px for r in regions] == [100, 60]      # sorted by area desc
    assert [r.id for r in regions] == [1, 2]
    assert filtered.sum() == 160


def test_bbox_and_centroid_correct():
    mask = np.zeros((40, 40), bool)
    mask[10:20, 4:14] = True          # x 4..13, y 10..19
    _, regions = extract_regions(mask, min_area_px=1)
    r = regions[0]
    assert list(r.bbox_xywh) == [4, 10, 10, 10]
    assert list(r.centroid_xy) == [8.5, 14.5]
    assert r.area_px == 100


def test_min_area_filter_removes_noise_and_mask_agrees():
    mask = np.zeros((30, 30), bool)
    mask[2:12, 2:12] = True           # 100 px, kept
    mask[20, 20] = True               # 1 px, noise
    mask[25:27, 25:27] = True         # 4 px, noise
    filtered, regions = extract_regions(mask, min_area_px=10)
    assert len(regions) == 1
    # the filtered mask must match the regions that survived
    assert filtered.sum() == 100
    assert not filtered[20, 20] and not filtered[25, 25]


def test_default_min_area_is_single_sourced():
    assert DEFAULT_MIN_AREA_PX == config.DEFAULT_MIN_AREA_PX == 32


def test_region_ids_are_deterministic():
    rng = np.random.default_rng(3)
    mask = np.zeros((64, 64), bool)
    for y, x, s in [(2, 2, 6), (30, 30, 9), (10, 40, 7), (45, 5, 8)]:
        mask[y:y + s, x:x + s] = True
    runs = [[(r.id, r.area_px, tuple(r.bbox_xywh)) for r in extract_regions(mask, min_area_px=1)[1]]
            for _ in range(5)]
    assert all(run == runs[0] for run in runs)
    areas = [a for _, a, _ in runs[0]]
    assert areas == sorted(areas, reverse=True)


def test_region_confidence_is_mean_probability_over_region():
    mask = np.zeros((20, 20), bool)
    mask[2:6, 2:6] = True                      # 16 px
    prob = np.zeros((20, 20), np.float32)
    prob[2:6, 2:6] = 0.8
    prob[3, 3] = 0.4                           # one lower pixel
    _, regions = extract_regions(mask, score_map=prob, min_area_px=1)
    expected = round(float(prob[mask].mean()), 4)
    assert regions[0].confidence == expected
    assert 0.0 <= regions[0].confidence <= 1.0


def test_confidence_absent_without_score_map():
    mask = np.zeros((10, 10), bool)
    mask[1:5, 1:5] = True
    _, regions = extract_regions(mask, min_area_px=1)
    assert regions[0].confidence is None
    assert "confidence" not in regions[0].to_dict()


def test_regions_have_no_geography_without_georef():
    mask = np.zeros((10, 10), bool)
    mask[1:5, 1:5] = True
    _, regions = extract_regions(mask, min_area_px=1)
    assert regions[0].area_m2 is None and regions[0].centroid_crs_xy is None


def test_region_geography_with_metric_georef():
    mask = np.zeros((10, 10), bool)
    mask[0:2, 0:2] = True                       # 4 px
    georef = GeoRef(crs="EPSG:32610", transform=(1000.0, 0.5, 0.0, 2000.0, 0.0, -0.5),
                    gsd_m=0.5, units="metre", pixel_size=(0.5, 0.5))
    _, regions = extract_regions(mask, min_area_px=1, georef=georef)
    r = regions[0]
    assert r.area_m2 == 4 * 0.5 * 0.5           # 1.0 m2
    assert r.centroid_crs_xy == (1000.0 + 0.5 * 0.5, 2000.0 - 0.5 * 0.5)


def test_pixel_to_crs_matches_transform():
    t = (100.0, 2.0, 0.0, 900.0, 0.0, -2.0)
    assert pixel_to_crs(t, 0, 0) == (100.0, 900.0)
    assert pixel_to_crs(t, 3, 4) == (106.0, 892.0)


def test_empty_mask_gives_no_regions():
    filtered, regions = extract_regions(np.zeros((20, 20), bool), min_area_px=1)
    assert regions == [] and filtered.sum() == 0


# ---------------------------------------------------------------- georef
def test_geotiff_projected_metre_gives_gsd():
    info = read_raster_info(open_image(make_geotiff()))
    assert info.is_geotiff and info.band_count == 3
    assert info.georef.crs == "EPSG:32610"
    assert info.georef.gsd_m == 0.5
    assert info.georef.units == "metre"
    assert info.has_metric_scale


def test_geotiff_transform_is_correct():
    info = read_raster_info(open_image(make_geotiff(pixel_scale=(2.0, 2.0),
                                                    origin=(1000.0, 5000.0))))
    assert info.georef.transform == (1000.0, 2.0, 0.0, 5000.0, 0.0, -2.0)


def test_geographic_crs_never_yields_metres():
    info = read_raster_info(open_image(
        make_geotiff(pixel_scale=(0.0001, 0.0001), epsg=4326, model_type=2)))
    assert info.georef.crs == "EPSG:4326"
    assert info.georef.gsd_m is None
    assert info.georef.units == "degrees"
    assert not info.has_metric_scale
    assert "latitude" in info.scale_note


def test_missing_crs_reports_unavailable():
    info = read_raster_info(open_image(make_geotiff(with_geokeys=False)))
    assert info.is_geotiff
    assert info.georef.crs is None
    assert info.georef.gsd_m is None
    assert "no CRS" in info.scale_note


def test_non_metre_linear_unit_rejected():
    info = read_raster_info(open_image(make_geotiff(linear_units=9002)))  # feet
    assert info.georef.gsd_m is None
    assert "not metre" in info.scale_note


def test_non_square_pixels_rejected():
    info = read_raster_info(open_image(make_geotiff(pixel_scale=(0.5, 0.9))))
    assert info.georef.gsd_m is None
    assert "not square" in info.scale_note


def test_plain_png_is_not_georeferenced():
    info = read_raster_info(open_image(make_png()))
    assert not info.is_geotiff and info.georef is None
    assert not info.has_metric_scale


def test_pair_with_different_crs_is_an_error():
    a = read_raster_info(open_image(make_geotiff(epsg=32610)))
    b = read_raster_info(open_image(make_geotiff(epsg=32611)))
    issues = pair_geo_issues(a, b)
    assert any(level == "error" and "coordinate reference" in msg for level, msg in issues)
    assert pair_georef(a, b) is None


def test_pair_with_different_transform_is_an_error():
    a = read_raster_info(open_image(make_geotiff(origin=(500000.0, 4649776.0))))
    b = read_raster_info(open_image(make_geotiff(origin=(600000.0, 4649776.0))))
    issues = pair_geo_issues(a, b)
    assert any(level == "error" and "geotransform" in msg for level, msg in issues)
    assert pair_georef(a, b) is None


def test_matching_pair_yields_a_georef():
    a = read_raster_info(open_image(make_geotiff()))
    b = read_raster_info(open_image(make_geotiff()))
    assert pair_geo_issues(a, b) == []
    georef = pair_georef(a, b)
    assert georef is not None and georef.gsd_m == 0.5


def test_mixed_geotiff_and_png_warns_and_gives_no_georef():
    a = read_raster_info(open_image(make_geotiff()))
    b = read_raster_info(open_image(make_png()))
    issues = pair_geo_issues(a, b)
    assert any(level == "warning" for level, _ in issues)
    assert pair_georef(a, b) is None


def test_transforms_match_helper():
    t = (1.0, 0.5, 0.0, 2.0, 0.0, -0.5)
    assert transforms_match(t, tuple(t))
    assert not transforms_match(t, (1.0, 0.5, 0.0, 2.5, 0.0, -0.5))


# ------------------------------------------------------------- validation
def test_validate_geotiff_pair_passes_and_exposes_georef():
    a, b = open_image(make_geotiff((300, 300))), open_image(make_geotiff((300, 300)))
    v = validate_pair(a, b)
    assert v.ok
    assert v.is_georeferenced and v.georef.gsd_m == 0.5


def test_validate_blocks_incompatible_geotiff_pair():
    a = open_image(make_geotiff((300, 300), epsg=32610))
    b = open_image(make_geotiff((300, 300), epsg=32611))
    v = validate_pair(a, b)
    assert not v.ok
    assert v.georef is None


def test_validate_blocks_mismatched_dimensions():
    v = validate_pair(open_image(make_geotiff((300, 300))),
                      open_image(make_geotiff((256, 300))))
    assert not v.ok
    assert any("same dimensions" in i.message for i in v.errors)


def test_validate_blocks_more_than_four_bands():
    class FakeMultispectral:
        mode = "RGB"
        size = (300, 300)

        def getbands(self):
            return ("B1", "B2", "B3", "B4", "B5", "B6")

        def convert(self, _mode):
            return Image.fromarray(np.zeros((300, 300, 3), np.uint8))

    v = validate_pair(FakeMultispectral(), FakeMultispectral())
    assert not v.ok
    assert any("multispectral" in i.message for i in v.errors)


def test_png_pair_reports_no_metric_scale():
    v = validate_pair(open_image(make_png((300, 300))), open_image(make_png((300, 300))))
    assert v.ok and v.georef is None


# ------------------------------------------------------------- quantities
def test_quantities_area_only_with_scale():
    mask = np.zeros((10, 10), bool)
    mask[0:2, 0:5] = True                       # 10 px
    assert Quantities.from_mask(mask).area_m2 is None
    assert Quantities.from_mask(mask, GeoRef()).area_m2 is None
    assert Quantities.from_mask(mask, GeoRef(gsd_m=2.0)).area_m2 == 40.0


def test_region_to_dict_omits_absent_optionals():
    d = Region(id=1, area_px=5, bbox_xywh=[0, 0, 2, 3], centroid_xy=[1.0, 1.5]).to_dict()
    assert set(d) == {"id", "area_px", "bbox_xywh", "centroid_xy"}


# ------------------------------------------------------------------ runner
def _run():
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS] {name}")
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run())
