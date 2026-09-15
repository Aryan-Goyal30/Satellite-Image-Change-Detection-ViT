"""Focused tests for Phase 4A Milestone 1: the environmental sample pipeline.

pytest is not a project dependency, so these are plain functions that assert.
Run either way:

    python tests/test_phase4a.py
    pytest tests/test_phase4a.py

NO NETWORK ACCESS. Everything here is deterministic and runs on tiny synthetic
arrays, except two checks that use published Sentinel-2 product metadata as
literal constants to validate the coordinate conversion.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.domains.environment.data import geo, sentinel2, tmf


# ------------------------------------------------------------ band contract
def test_band_order_is_fixed_and_documented():
    assert sentinel2.BANDS == ("B02", "B03", "B04", "B08", "B11", "B12")
    assert sentinel2.BAND_INDEX["B02"] == 0
    assert sentinel2.BAND_INDEX["B12"] == 5
    assert len(sentinel2.BANDS) == 6


def test_native_resolutions_match_the_mission_spec():
    for band in ("B02", "B03", "B04", "B08"):
        assert sentinel2.BAND_RESOLUTION_M[band] == 10
    for band in ("B11", "B12"):
        assert sentinel2.BAND_RESOLUTION_M[band] == 20


def test_rgb_composite_is_red_green_blue_in_that_order():
    assert sentinel2.RGB_BANDS == ("B04", "B03", "B02")


def test_red_edge_bands_are_not_included_yet():
    for band in ("B05", "B06", "B07"):
        assert band not in sentinel2.BANDS


# --------------------------------------------------------- 20 m -> 10 m
def test_upsample_doubles_each_axis():
    a = np.arange(4, dtype=np.uint16).reshape(2, 2)
    up = sentinel2.upsample_nearest(a, 2)
    assert up.shape == (4, 4)
    assert up.dtype == a.dtype


def test_upsample_is_nearest_neighbour_block_replication():
    """Each 20 m pixel must become a 2x2 block of identical 10 m pixels."""
    a = np.array([[10, 20], [30, 40]], dtype=np.uint16)
    up = sentinel2.upsample_nearest(a, 2)
    assert np.array_equal(up, np.array([[10, 10, 20, 20],
                                        [10, 10, 20, 20],
                                        [30, 30, 40, 40],
                                        [30, 30, 40, 40]], dtype=np.uint16))


def test_upsample_invents_no_new_values():
    """Categorical layers (SCL) must never be interpolated into new classes."""
    scl = np.array([[4, 9], [3, 5]], dtype=np.uint8)
    up = sentinel2.upsample_nearest(scl, 2)
    assert set(np.unique(up).tolist()) <= set(np.unique(scl).tolist())


def test_upsampled_20m_window_aligns_with_the_10m_grid():
    """A 128x128 window at 20 m must cover exactly the 256x256 window at 10 m."""
    twenty = np.zeros((128, 128), np.uint16)
    assert sentinel2.upsample_nearest(twenty, 2).shape == (256, 256)


# ------------------------------------------------------------------- SCL
def test_scl_invalid_classes_are_exactly_the_specified_set():
    assert sentinel2.SCL_INVALID == frozenset({0, 1, 3, 8, 9, 10, 11})


def test_scl_valid_classes_are_not_masked():
    """Vegetation, bare soil, water, dark area and low-prob cloud stay usable."""
    for keep in (2, 4, 5, 6, 7):
        assert keep not in sentinel2.SCL_INVALID


def test_invalid_mask_flags_only_invalid_classes():
    scl = np.array([[4, 5, 6], [3, 9, 11], [0, 1, 10]], dtype=np.uint8)
    mask = sentinel2.invalid_from_scl(scl)
    assert mask.tolist() == [[False, False, False],
                             [True, True, True],
                             [True, True, True]]
    assert mask.sum() == 6


def test_cloud_shadow_is_treated_as_invalid():
    """Shadow (3) is easy to forget and is not a cloud class."""
    assert 3 in sentinel2.SCL_INVALID
    assert bool(sentinel2.invalid_from_scl(np.array([[3]], np.uint8))[0, 0])


# ---------------------------------------------------------- reflectance
def test_reflectance_scaling_uses_the_quantification_value():
    dn = np.array([[1000]], dtype=np.uint16)
    out = sentinel2.to_reflectance(dn, offset_applied=True)
    assert np.isclose(out[0, 0], 0.1)


def test_offset_is_not_applied_twice_when_distributor_already_did_it():
    """The baseline-04.00 trap: double-correcting shifts everything by 0.1."""
    dn = np.array([[2000]], dtype=np.uint16)
    already = sentinel2.to_reflectance(dn, offset_applied=True, baseline="05.00")
    not_yet = sentinel2.to_reflectance(dn, offset_applied=False, baseline="05.00")
    assert np.isclose(already[0, 0], 0.2)
    assert np.isclose(not_yet[0, 0], 0.1)
    assert not np.isclose(already[0, 0], not_yet[0, 0])


def test_offset_is_not_applied_to_pre_baseline_04_products():
    """The opposite trap, and the one that actually bit.

    A baseline 02.13 product never carried the offset, so there is nothing to
    undo. Deciding from earthsearch:boa_offset_applied alone subtracts 1000 from
    24 of the Milestone 2 BEFORE scenes and drives tropical-forest visible bands
    negative - an impossible surface reflectance.
    """
    dn = np.array([[420]], dtype=np.uint16)          # typical forest blue band
    out = sentinel2.to_reflectance(dn, offset_applied=False, baseline="02.13")
    assert np.isclose(out[0, 0], 0.042)
    assert out[0, 0] > 0


def test_offset_rule_uses_baseline_and_flag_together():
    assert not sentinel2.needs_offset(True, "05.00")
    assert not sentinel2.needs_offset(True, "02.13")
    assert not sentinel2.needs_offset(False, "00.01")
    assert not sentinel2.needs_offset(False, "03.01")
    assert sentinel2.needs_offset(False, "04.00")
    assert sentinel2.needs_offset(False, "05.00")


def test_baseline_is_required_when_the_offset_flag_is_false():
    """The one case the flag cannot resolve alone must refuse to guess."""
    try:
        sentinel2.needs_offset(False)
    except ValueError:
        return
    raise AssertionError("expected ValueError when the baseline is unknown")


def test_offset_constant_matches_the_published_baseline_shift():
    assert sentinel2.BOA_ADD_OFFSET == -1000.0
    assert sentinel2.QUANTIFICATION_VALUE == 10000.0


# ----------------------------------------------------------- TMF target
def test_label_is_equality_on_the_deforestation_year():
    years = np.array([[2019, 2020], [2021, 0]], dtype=np.uint16)
    label = tmf.label_from_years(years, 2020)
    assert label.tolist() == [[0, 1], [0, 0]]
    assert label.dtype == np.uint8


def test_other_years_are_negatives_not_positives():
    """The S2 pair brackets one year only, so neighbouring years are negative."""
    years = np.array([[2018, 2019, 2020, 2021, 2022]], dtype=np.uint16)
    assert tmf.label_from_years(years, 2020).sum() == 1


def test_zero_means_no_deforestation_and_is_never_positive():
    years = np.zeros((8, 8), dtype=np.uint16)
    assert tmf.label_from_years(years, 2020).sum() == 0
    assert tmf.NO_DEFORESTATION == 0


def test_year_outside_the_tmf_range_is_rejected():
    years = np.zeros((2, 2), dtype=np.uint16)
    for bad in (1981, 2026):
        try:
            tmf.label_from_years(years, bad)
        except ValueError as e:
            assert "outside the TMF range" in str(e)
        else:
            raise AssertionError(f"year {bad} should be rejected")


def test_ongoing_disturbance_class_50_is_explicitly_excluded():
    """Class 50 (2023-25, unattributed) must never count as deforestation."""
    assert tmf.ONGOING_DISTURBANCE_CLASS == 50
    assert 50 in tmf.EXCLUDED_CLASSES
    # It lives in the transition map, so a DeforestationYear of 50 is not a
    # valid year and cannot be requested as a target at all.
    try:
        tmf.label_from_years(np.array([[50]], np.uint16), 50)
    except ValueError:
        return
    raise AssertionError("class 50 must not be usable as a deforestation year")


def test_degradation_and_regrowth_cannot_leak_into_the_target():
    """They are separate TMF layers; only DeforestationYear is read."""
    assert tmf.LAYER == "DeforestationYear"
    years = np.array([[2020, 2020]], dtype=np.uint16)
    assert tmf.label_from_years(years, 2020).sum() == 2


def test_positive_fraction_matches_the_mask():
    label = np.zeros((10, 10), np.uint8)
    label[:2, :5] = 1
    assert abs(tmf.positive_fraction(label) - 0.10) < 1e-9


# --------------------------------------------- 30 m -> 10 m label sampling
def test_lonlat_to_pixel_is_nearest_neighbour():
    grid = tmf.geotransform(-60.0, 0.0, 0.00026949458523585647)
    col, row = tmf.lonlat_to_pixel(-60.0, 0.0, grid)
    assert int(col) == 0 and int(row) == 0
    # half a pixel east/south rounds to the neighbouring pixel
    col, row = tmf.lonlat_to_pixel(-60.0 + 0.00026949458523585647, 0.0, grid)
    assert int(col) == 1


def test_label_sampling_quantises_to_30m_blocks():
    """10 m samples within half a cell of a 30 m centre resolve to that cell.

    Nearest-neighbour rounds at the half-pixel boundary, so the property to
    assert is about distance from the cell CENTRE - not that any three
    consecutive 10 m steps share a cell, which is false by construction when a
    step crosses the boundary.
    """
    pixel = 0.00026949458523585647
    grid = tmf.geotransform(-60.0, 0.0, pixel)
    centre_lon = -60.0 + 5 * pixel               # centre of TMF cell 5
    step = pixel / 3.0                           # ~10 m in degrees
    cols = [int(tmf.lonlat_to_pixel(centre_lon + k * step, 0.0, grid)[0])
            for k in (-1, 0, 1)]
    assert cols == [5, 5, 5], f"expected cell 5 three times, got {cols}"


def test_label_sampling_crosses_to_the_next_cell_beyond_half_a_pixel():
    """The complement: past the half-pixel boundary it must switch cells."""
    pixel = 0.00026949458523585647
    grid = tmf.geotransform(-60.0, 0.0, pixel)
    just_over = -60.0 + (5 + 0.51) * pixel
    assert int(tmf.lonlat_to_pixel(just_over, 0.0, grid)[0]) == 6


def test_label_resample_is_documented_as_30m_quantised():
    assert "30 m" in tmf.LABEL_RESAMPLE
    assert "nearest" in tmf.LABEL_RESAMPLE.lower()
    assert tmf.NATIVE_RESOLUTION_M == 30


# ----------------------------------------------------- coordinate conversion
def test_utm_lonlat_round_trip_is_sub_millimetre():
    for lon, lat in [(-54.913025, -2.996780), (-55.2002, -3.7057),
                     (-54.2116, -2.7110)]:
        e, n = geo.lonlat_to_utm(lon, lat, 21, south=True)
        lon2, lat2 = geo.utm_to_lonlat(e, n, 21, south=True)
        assert abs(lon2 - lon) < 1e-9, (lon, lon2)
        assert abs(lat2 - lat) < 1e-9, (lat, lat2)


def test_conversion_matches_the_product_stac_bbox():
    """Independent check against S2B_21MYS_20190810 published metadata.

    The COG's own tags give origin (699960, 9700000) at 10 m over 10980 px;
    STAC reports bbox lon -55.2002..-54.2116, lat -3.7057..-2.7110. The two
    must agree to about a pixel.
    """
    origin_e, origin_n, res, size = 699960.0, 9700000.0, 10.0, 10980
    sw_lon, sw_lat = geo.utm_to_lonlat(origin_e, origin_n - size * res, 21)
    ne_lon, ne_lat = geo.utm_to_lonlat(origin_e + size * res, origin_n, 21)
    assert abs(sw_lon - (-55.2002)) < 0.01, sw_lon
    assert abs(sw_lat - (-3.7057)) < 0.01, sw_lat
    assert abs(ne_lon - (-54.2116)) < 0.01, ne_lon
    assert abs(ne_lat - (-2.7110)) < 0.01, ne_lat


def test_zone_21_central_meridian():
    assert geo.zone_central_meridian(21) == -57


def test_southern_hemisphere_false_northing_is_applied():
    e_s, n_s = geo.lonlat_to_utm(-57.0, -3.0, 21, south=True)
    e_n, n_n = geo.lonlat_to_utm(-57.0, -3.0, 21, south=False)
    assert abs(n_s - n_n - 10000000.0) < 1e-6
    assert abs(e_s - e_n) < 1e-9


# --------------------------------------------------- module hygiene
def test_data_modules_pull_in_no_heavy_dependencies():
    """Phase 4A data access must not drag in torch, streamlit or rasterio."""
    import subprocess
    code = ("import sys;"
            "import src.domains.environment.data.sentinel2 as s;"
            "import src.domains.environment.data.tmf as t;"
            "import src.domains.environment.data.geo as g;"
            "bad=[m for m in ('torch','streamlit','rasterio','osgeo','pyproj')"
            " if m in sys.modules];"
            "print(','.join(bad))")
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True,
                         cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert out.stdout.strip() == "", f"unexpected imports: {out.stdout.strip()}"


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
