"""Offline tests for Phase 4A Milestone 2: environmental dataset construction.

No test here touches the network. The on-disk checks read the built dataset
from data/environment/dataset/manifest.json; run
scripts/build_environment_dataset.py before them, since a dataset audit suite
that silently passes with no dataset present would be worthless.
"""
import datetime
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config                                       # noqa: E402
from src.domains.environment.data import dataset as ds       # noqa: E402
from src.domains.environment.data import events as ev        # noqa: E402
from src.domains.environment.data import geo                 # noqa: E402
from src.domains.environment.data import sentinel2 as s2     # noqa: E402
from src.domains.environment.data import stac                # noqa: E402
from src.domains.environment.data import tmf                 # noqa: E402

#: The Milestone 2.1 validation dataset. The pre-2.1 dataset is kept alongside
#: it for the temporal comparison, but it predates the fields this suite now
#: requires, so it is deliberately NOT what these tests validate.
DATASET_DIR = os.environ.get(
    "EG_DATASET_DIR",
    os.path.join(config.DATA_DIR, "environment", "dataset_v24"))
MANIFEST_PATH = os.path.join(DATASET_DIR, "manifest.json")
_CACHE = {}


def manifest():
    if "m" not in _CACHE:
        if not os.path.exists(MANIFEST_PATH):
            raise FileNotFoundError(
                f"dataset not built: {MANIFEST_PATH} - run "
                "scripts/build_environment_dataset.py first")
        with open(MANIFEST_PATH, encoding="utf-8") as fh:
            _CACHE["m"] = json.load(fh)
    return _CACHE["m"]


def _scene(month, day, year, cloud=5.0, epsg=32721, zone=21, square="YS",
           band="M"):
    return stac.Scene(
        stac_id=f"S2B_{zone}{band}{square}_{year}{month:02d}{day:02d}_0_L2A",
        product_id=f"S2B_MSIL2A_{year}{month:02d}{day:02d}T140000",
        dt=datetime.datetime(year, month, day, 14, 0, 0),
        cloud_pct=cloud, nodata_pct=0.0, baseline="05.00", offset_applied=True,
        epsg=epsg, utm_zone=zone, lat_band=band, grid_square=square,
        base_url="https://example.invalid/")


def _event(index, row, col, positive_px, tile="N0_W60", year=2020):
    return {"event_id": f"{tile}_{year}_{index:04d}", "tmf_tile": tile,
            "year": year, "n_blocks": 1, "positive_px": positive_px,
            "centroid_block": (float(row), float(col)),
            "bbox_block": (row, col, row, col)}


# ----------------------------------------------------------------- band order
def test_band_order_is_the_protocol_order():
    assert s2.BANDS == ("B02", "B03", "B04", "B08", "B11", "B12")


def test_band_index_matches_band_order():
    assert [s2.BAND_INDEX[b] for b in s2.BANDS] == list(range(6))


def test_every_band_has_a_declared_resolution():
    assert all(b in s2.BAND_RESOLUTION_M for b in s2.BANDS)
    assert {s2.BAND_RESOLUTION_M[b] for b in s2.BANDS} == {10, 20}


def test_scl_invalid_set_is_exactly_the_protocol():
    assert s2.SCL_INVALID == frozenset({0, 1, 3, 8, 9, 10, 11})


# -------------------------------------------------------------- label + gates
def test_label_is_equality_on_the_year_not_a_range():
    years = np.array([[2019, 2020], [2021, 0]], np.uint16)
    assert ev is not None
    assert tmf.label_from_years(years, 2020).tolist() == [[0, 1], [0, 0]]


def test_label_rejects_year_outside_tmf_range():
    try:
        tmf.label_from_years(np.zeros((2, 2), np.uint16), 1900)
    except ValueError:
        return
    raise AssertionError("expected ValueError for a year outside 1982-2025")


def test_positive_fraction_is_the_mean_of_the_label():
    label = np.zeros((256, 256), np.uint8)
    label[:64] = 1
    assert abs(tmf.positive_fraction(label) - 0.25) < 1e-12


def test_positive_bucket_boundaries():
    assert ds.positive_bucket(0.004) == "<1%"
    assert ds.positive_bucket(0.01) == "1-5%"
    assert ds.positive_bucket(0.05) == "5-25%"
    assert ds.positive_bucket(0.25) == "25-50%"
    assert ds.positive_bucket(0.5) == ">50%"
    assert ds.positive_bucket(0.61891) == ">50%"


def test_invalid_gate_is_five_percent_inclusive():
    assert ds.invalid_gate(0.05)
    assert not ds.invalid_gate(0.0501)


# ----------------------------------------------------------------- class 50
def test_class50_flagged_when_the_value_appears():
    assert ds.class50_ambiguous({0: 10, 50: 1}, 2020)


def test_class50_flagged_for_unattributed_years():
    assert ds.class50_ambiguous({0: 10}, 2023)
    assert ds.class50_ambiguous({0: 10}, 2025)


def test_class50_clear_for_the_target_years_in_use():
    for year in (2019, 2020, 2021, 2022):
        assert not ds.class50_ambiguous({0: 10, year: 5}, year)


def test_ongoing_disturbance_class_is_recorded_as_excluded():
    assert tmf.ONGOING_DISTURBANCE_CLASS == 50
    assert 50 in tmf.EXCLUDED_CLASSES


# ------------------------------------------------------------ negative subtype
def test_negative_subtype_stable_forest():
    assert ev.negative_subtype({0: 65536}, 2020) == "stable_forest"


def test_negative_subtype_prior_deforestation():
    assert ev.negative_subtype({0: 100, 2005: 20}, 2020) == "prior_deforestation"


def test_negative_subtype_later_deforestation():
    assert ev.negative_subtype({0: 100, 2022: 20}, 2020) == "later_deforestation"


def test_negative_subtype_mixed_history():
    assert ev.negative_subtype({0: 10, 2005: 5, 2022: 5}, 2020) == "mixed_history"


def test_negative_subtypes_are_a_closed_set():
    assert set(ev.NEGATIVE_SUBTYPES) == {
        "stable_forest", "prior_deforestation", "later_deforestation",
        "mixed_history"}


# --------------------------------------------------------- rejection ledger
def test_ledger_counts_by_reason():
    ledger = ds.RejectionLedger()
    ledger.reject("a", "cloud")
    ledger.reject("b", "cloud")
    ledger.reject("c", "scl_invalid")
    assert ledger.counts["cloud"] == 2
    assert ledger.counts["scl_invalid"] == 1
    assert ledger.total == 3


def test_ledger_buckets_an_unknown_reason_rather_than_losing_it():
    ledger = ds.RejectionLedger()
    ledger.reject("a", "something_new")
    assert ledger.counts["other"] == 1
    assert ledger.total == 1


def test_ledger_records_every_rejection_individually():
    ledger = ds.RejectionLedger()
    for i in range(5):
        ledger.reject(f"cand{i}", "no_s2_pair", "detail")
    assert len(ledger.records) == 5
    assert {r["candidate_id"] for r in ledger.records} == {f"cand{i}" for i in range(5)}


# --------------------------------------------------------- temporal pairing
def test_in_season_boundaries():
    assert stac.in_season(datetime.date(2020, 8, 1))
    assert stac.in_season(datetime.date(2020, 9, 15))
    assert not stac.in_season(datetime.date(2020, 7, 31))
    assert not stac.in_season(datetime.date(2020, 9, 16))


def test_acceptable_rejects_cloud_at_or_above_the_limit():
    assert stac.acceptable(_scene(8, 10, 2019, cloud=19.9))
    assert not stac.acceptable(_scene(8, 10, 2019, cloud=20.0))
    assert not stac.acceptable(_scene(10, 10, 2019, cloud=1.0))   # out of season


def test_select_pair_minimises_day_of_year_separation():
    """The corrected rule. Gaps here are 2, 27, 38 and 9 days; 2 must win."""
    scenes = [_scene(8, 5, 2019), _scene(9, 10, 2019),
              _scene(8, 3, 2021), _scene(9, 1, 2021)]
    t1, t2, reason, info = stac.select_pair(scenes, 2020)
    assert reason is None
    assert (t1.dt.month, t1.dt.day) == (8, 5)
    assert (t2.dt.month, t2.dt.day) == (8, 3)
    assert info["doy_difference"] == 2


def test_select_pair_no_longer_takes_last_before_and_first_after():
    """Guards against regressing to the rule that caused the seasonal bias.

    The old rule would pick 10 Sep 2019 with 3 Aug 2021 - a 38-day gap in the
    September-to-August direction. The new rule must not.
    """
    scenes = [_scene(8, 5, 2019), _scene(9, 10, 2019),
              _scene(8, 3, 2021), _scene(9, 1, 2021)]
    t1, t2, _, info = stac.select_pair(scenes, 2020)
    assert not (t1.dt.day == 10 and t1.dt.month == 9 and t2.dt.day == 3)
    assert info["doy_difference"] < 38


def test_select_pair_tie_break_prefers_lower_cloud():
    """Both BEFORE scenes sit 2 days from the AFTER scene; cloud decides."""
    scenes = [_scene(8, 5, 2019, cloud=15.0), _scene(8, 9, 2019, cloud=1.0),
              _scene(8, 7, 2021, cloud=1.0)]
    t1, _, reason, _ = stac.select_pair(scenes, 2020)
    assert reason is None
    assert t1.dt.day == 9


def test_select_pair_tie_break_is_deterministic_when_cloud_also_ties():
    scenes = [_scene(8, 5, 2019, cloud=5.0), _scene(8, 9, 2019, cloud=5.0),
              _scene(8, 7, 2021, cloud=5.0)]
    first = stac.select_pair(scenes, 2020)
    second = stac.select_pair(list(reversed(scenes)), 2020)
    assert first[0].stac_id == second[0].stac_id
    assert first[0].dt.day == 5          # lexicographically smaller STAC id


def test_select_pair_never_uses_the_target_year():
    scenes = [_scene(8, 5, 2019), _scene(8, 20, 2020), _scene(8, 3, 2021)]
    t1, t2, reason, _ = stac.select_pair(scenes, 2020)
    assert reason is None
    assert t1.dt.year == 2019 and t2.dt.year == 2021


def test_select_pair_keeps_the_y_minus_1_to_y_plus_1_bracket():
    scenes = [_scene(8, 5, 2019), _scene(8, 6, 2021)]
    t1, t2, reason, _ = stac.select_pair(scenes, 2020)
    assert reason is None
    assert t1.dt.year == 2020 - 1
    assert t2.dt.year == 2020 + 1


def test_select_pair_reports_no_pair_when_a_side_is_empty():
    _, _, reason, info = stac.select_pair([_scene(8, 5, 2019)], 2020)
    assert reason == "no_s2_pair"
    assert info["n_after_candidates"] == 0


def test_select_pair_rejects_a_crs_mismatch():
    scenes = [_scene(8, 5, 2019), _scene(8, 3, 2021, epsg=32722, zone=22)]
    _, _, reason, _ = stac.select_pair(scenes, 2020)
    assert reason == "crs_grid"


def test_cloudy_scenes_cannot_be_selected():
    scenes = [_scene(8, 5, 2019), _scene(8, 4, 2019, cloud=55.0),
              _scene(8, 3, 2021)]
    t1, _, reason, _ = stac.select_pair(scenes, 2020)
    assert reason is None
    assert t1.dt.day == 5          # the nearer scene was too cloudy to use


def test_select_pair_records_its_provenance():
    scenes = [_scene(8, 5, 2019), _scene(9, 10, 2019), _scene(8, 3, 2021)]
    _, _, _, info = stac.select_pair(scenes, 2020)
    for key in ("selection_rule", "seasonal_window", "n_before_candidates",
                "n_after_candidates", "before_candidate_dates",
                "after_candidate_dates", "n_pairs_considered", "before_doy",
                "after_doy", "doy_difference", "selected_reason"):
        assert key in info, key
    assert info["n_before_candidates"] == 2
    assert info["n_pairs_considered"] == 2


def test_selection_is_not_biased_toward_later_before_scenes():
    """Across many synthetic layouts the chosen BEFORE must not sit
    systematically late in the window, which is what produced the old bias."""
    late = 0
    trials = 0
    for before_day in range(1, 15):
        for after_day in range(1, 15):
            scenes = [_scene(8, before_day, 2019), _scene(9, 10, 2019),
                      _scene(8, after_day, 2021), _scene(9, 12, 2021)]
            t1, t2, reason, _ = stac.select_pair(scenes, 2020)
            if reason:
                continue
            trials += 1
            late += int(t1.dt.month == 9 and t2.dt.month == 8)
    assert trials > 0
    assert late / trials < 0.5, f"{late}/{trials} pairs still September->August"


def test_seasonal_window_bounds_the_doy_gap_by_construction():
    worst = 0
    for year_a in (2019, 2020, 2021, 2022):
        for year_b in (2019, 2020, 2021, 2022):
            for month, day in ((8, 1), (8, 31), (9, 1), (9, 15)):
                for m2, d2 in ((8, 1), (8, 31), (9, 1), (9, 15)):
                    a = _scene(month, day, year_a)
                    b = _scene(m2, d2, year_b)
                    worst = max(worst, stac.doy_gap(a, b))
    assert worst <= stac.SEASONAL_MAX_DOY_GAP, worst


# ------------------------------------------------------- events and sampling
def test_label_components_uses_four_connectivity():
    mask = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], bool)
    assert int(ev.label_components(mask).max()) == 3


def test_label_components_merges_an_l_shape_into_one():
    mask = np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], bool)
    assert int(ev.label_components(mask).max()) == 1


def test_label_components_labels_two_separate_blobs():
    mask = np.zeros((10, 10), bool)
    mask[0:3, 0:3] = True
    mask[7:10, 7:10] = True
    assert int(ev.label_components(mask).max()) == 2


def test_extract_events_discards_speckle_below_the_threshold():
    counts = np.zeros((20, 20), np.int32)
    counts[5, 5] = 10                       # below MIN_BLOCK_POSITIVES
    counts[10:12, 10:12] = 500              # a real event
    found = ev.extract_events(counts, 2020, "N0_W60")
    assert len(found) == 1
    assert found[0]["positive_px"] == 2000


def test_extract_events_centroid_is_positive_weighted():
    counts = np.zeros((20, 20), np.int32)
    counts[5, 5] = 100
    counts[5, 6] = 900
    found = ev.extract_events(counts, 2020, "N0_W60")
    row, col = found[0]["centroid_block"]
    assert abs(row - 5.0) < 1e-9
    assert col > 5.8                        # pulled toward the heavier block


def test_size_buckets_partition_the_range():
    assert ev.size_bucket(999) == "small"
    assert ev.size_bucket(1000) == "medium"
    assert ev.size_bucket(10000) == "large"
    assert ev.size_bucket(100000) == "very_large"


def test_sample_events_is_deterministic_for_a_seed():
    events = [_event(i, i * 20, i * 20, 500 * (i + 1)) for i in range(20)]
    a = ev.sample_events(events, 6, seed=7)
    b = ev.sample_events(events, 6, seed=7)
    assert [e["event_id"] for e in a] == [e["event_id"] for e in b]


def test_sample_events_mixes_size_buckets():
    events = ([_event(i, i * 20, 0, 500) for i in range(5)] +
              [_event(10 + i, i * 20, 40, 5000) for i in range(5)] +
              [_event(20 + i, i * 20, 80, 50000) for i in range(5)])
    picked = ev.sample_events(events, 6, seed=3)
    buckets = {ev.size_bucket(e["positive_px"]) for e in picked}
    assert len(buckets) >= 2, buckets


def test_sample_events_enforces_spatial_separation():
    events = [_event(i, 10, 10 + i, 5000) for i in range(6)]   # all adjacent
    picked = ev.sample_events(events, 6, seed=1, min_separation_blocks=8)
    assert len(picked) == 1


def test_sample_events_never_exceeds_the_request():
    events = [_event(i, i * 20, i * 20, 5000) for i in range(30)]
    assert len(ev.sample_events(events, 4, seed=1)) <= 4


def test_negative_erosion_excludes_blocks_touching_positives():
    counts = {2020: np.zeros((9, 9), np.int32)}
    counts[2020][4, 4] = 5000
    any_def = np.zeros((9, 9), np.int32)
    picked = ev.select_negative_blocks(counts, 2020, any_def, 8, seed=1,
                                       min_separation_blocks=1)
    for candidate in picked:
        row, col = candidate["centroid_block"]
        assert abs(row - 4) > 1 or abs(col - 4) > 1


def test_block_to_lonlat_centres_on_the_block():
    grid = tmf.geotransform(-60.0, 0.0, 0.001)
    lon, lat = ev.block_to_lonlat(0, 0, grid, block=10)
    assert abs(lon - (-60.0 + 0.005)) < 1e-12
    assert abs(lat - (-0.005)) < 1e-12


# -------------------------------------------------------------------- splits
def test_haversine_matches_a_known_distance():
    km = ds.haversine_km(0.0, 0.0, 1.0, 0.0)
    assert abs(km - 111.19) < 0.1, km


def test_tiles_closer_than_the_threshold_merge_into_one_block():
    centres = {"A": (-60.0, -3.0), "B": (-59.9, -3.0), "C": (-50.0, -3.0)}
    blocks = ds.group_tiles_into_blocks(centres, min_km=50.0)
    assert len(blocks) == 2
    assert ["A", "B"] in blocks


def test_assign_splits_keeps_a_block_whole():
    blocks = [["A", "B"], ["C"], ["D"]]
    weights = {"A": 5, "B": 5, "C": 4, "D": 3}
    assigned = ds.assign_splits(blocks, weights)
    placed = {t: s for s, tiles in assigned.items() for t in tiles}
    assert placed["A"] == placed["B"]
    assert sum(len(v) for v in assigned.values()) == 4


def test_split_integrity_detects_an_overlap():
    result = ds.split_integrity({"train": ["A"], "val": ["A"], "test": ["B"]})
    assert not result["disjoint"]
    assert result["intersections"]["train&val"] == ["A"]


def test_split_integrity_accepts_disjoint_sets():
    result = ds.split_integrity({"train": ["A"], "val": ["B"], "test": ["C"]})
    assert result["disjoint"]


def test_min_split_separation_reports_the_closest_pair():
    centres = {"A": (-60.0, -3.0), "B": (-59.0, -3.0)}
    out = ds.min_split_separation_km({"train": ["A"], "val": ["B"], "test": []},
                                     centres)
    assert out["train-val"] is not None
    assert out["test-val" if "test-val" in out else "val-test"] is None
    assert abs(out["minimum"] - out["train-val"]) < 1e-9


# ----------------------------------------------------------------- geometry
def test_grid_lonlat_agrees_with_pointwise_conversion():
    lons, lats = geo.grid_lonlat(800000.0, 9650000.0, 0, 0, 3, 3, 10.0, 21,
                                 south=True)
    lon, lat = geo.utm_to_lonlat(800000.0 + 1.5 * 10.0,
                                 9650000.0 - 1.5 * 10.0, 21, south=True)
    assert abs(lons[1, 1] - lon) < 1e-12
    assert abs(lats[1, 1] - lat) < 1e-12


def test_grid_lonlat_shape_matches_the_window():
    lons, lats = geo.grid_lonlat(800000.0, 9650000.0, 10, 20, 8, 4, 10.0, 21)
    assert lons.shape == (4, 8) and lats.shape == (4, 8)


# ------------------------------------------------------- built dataset checks
def test_manifest_exists_and_declares_the_protocol():
    protocol = manifest()["protocol"]
    assert protocol["bands"] == list(s2.BANDS)
    assert protocol["tile_size"] == [256, 256]
    assert protocol["gsd_m"] == 10.0
    assert protocol["cloud_max_pct"] == 20.0
    assert protocol["invalid_max_fraction"] == 0.05
    assert protocol["scl_invalid_classes"] == sorted(s2.SCL_INVALID)


def test_every_sample_record_has_the_required_fields():
    missing = {r["sample_id"]: ds.validate_record(r) for r in manifest()["samples"]}
    offenders = {k: v for k, v in missing.items() if v}
    assert not offenders, offenders


def test_sample_ids_are_unique():
    ids = [r["sample_id"] for r in manifest()["samples"]]
    assert len(ids) == len(set(ids))


def test_split_mgrs_sets_are_disjoint():
    result = ds.split_integrity(manifest()["split_tiles"])
    assert result["disjoint"], result["intersections"]


def test_no_mgrs_tile_spans_two_splits():
    placement = {}
    for record in manifest()["samples"]:
        placement.setdefault(record["mgrs_tile"], set()).add(record["split"])
    offenders = {t: s for t, s in placement.items() if len(s) > 1}
    assert not offenders, offenders


def test_every_sample_is_assigned_to_a_split():
    assert all(r["split"] in ds.SPLITS for r in manifest()["samples"])


def test_sample_arrays_have_the_declared_dimensions():
    for record in manifest()["samples"]:
        for name, shape in (("before", (256, 256, 6)), ("after", (256, 256, 6))):
            path = os.path.join(DATASET_DIR, f"{record['sample_id']}_{name}.npy")
            array = np.load(path, mmap_mode="r")
            assert array.shape == shape, (path, array.shape)
            assert array.dtype == np.uint16, (path, array.dtype)


def test_label_arrays_have_the_declared_dimensions():
    for record in manifest()["samples"]:
        path = os.path.join(DATASET_DIR, f"{record['sample_id']}_label.npy")
        array = np.load(path, mmap_mode="r")
        assert array.shape == (256, 256), (path, array.shape)
        assert array.dtype == np.uint8


def test_invalid_masks_have_the_declared_dimensions():
    for record in manifest()["samples"]:
        path = os.path.join(DATASET_DIR, f"{record['sample_id']}_invalid.npy")
        array = np.load(path, mmap_mode="r")
        assert array.shape == (256, 256), (path, array.shape)
        assert set(np.unique(np.asarray(array)).tolist()) <= {0, 1}


def test_manifest_positive_fraction_matches_the_label_array():
    for record in manifest()["samples"]:
        path = os.path.join(DATASET_DIR, f"{record['sample_id']}_label.npy")
        actual = float(np.load(path, mmap_mode="r").mean())
        assert abs(actual - record["positive_fraction"]) < 1e-5, record["sample_id"]


def test_manifest_invalid_fraction_matches_the_mask_array():
    for record in manifest()["samples"]:
        path = os.path.join(DATASET_DIR, f"{record['sample_id']}_invalid.npy")
        actual = float(np.load(path, mmap_mode="r").mean())
        assert abs(actual - record["invalid_fraction"]) < 1e-5, record["sample_id"]


def test_every_accepted_sample_passes_the_invalid_gate():
    for record in manifest()["samples"]:
        assert ds.invalid_gate(record["invalid_fraction"]), record["sample_id"]


def test_every_accepted_sample_passes_the_cloud_gate():
    for record in manifest()["samples"]:
        assert record["before_cloud_pct"] < 20.0
        assert record["after_cloud_pct"] < 20.0


def test_negatives_carry_no_positive_pixels():
    for record in manifest()["samples"]:
        if record["sample_type"] == "negative":
            assert record["positive_px"] == 0, record["sample_id"]
            assert record["negative_source"] in ev.NEGATIVE_SUBTYPES


def test_positives_carry_positive_pixels():
    for record in manifest()["samples"]:
        if record["sample_type"] == "positive":
            assert record["positive_px"] > 0, record["sample_id"]


def test_pairs_bracket_the_target_year():
    for record in manifest()["samples"]:
        year = record["tmf_year"]
        assert record["before_datetime"].startswith(str(year - 1)), record["sample_id"]
        assert record["after_datetime"].startswith(str(year + 1)), record["sample_id"]


def test_pairs_respect_the_seasonal_window():
    for record in manifest()["samples"]:
        assert record["doy_gap_days"] <= stac.SEASONAL_MAX_DOY_GAP, record["sample_id"]


def test_rejection_counts_match_the_records():
    rejections = manifest()["rejections"]
    assert rejections["total"] == sum(rejections["counts"].values())
    assert rejections["total"] == len(rejections["records"])


def test_every_candidate_is_accounted_for():
    data = manifest()
    assert data["candidates_attempted"] >= (len(data["samples"]) +
                                            data["rejections"]["total"])


def test_rejection_reasons_are_from_the_declared_set():
    for record in manifest()["rejections"]["records"]:
        assert record["reason"] in ds.REJECT_REASONS, record


# ------------------------------------------------ Milestone 2.1: eligibility
def test_minimum_positive_pixels_is_the_documented_threshold():
    assert ds.MIN_POSITIVE_PIXELS == 200


def test_positive_below_the_threshold_is_a_small_event_not_a_deletion():
    assert ds.training_eligibility("positive", 9) == ds.SMALL_EVENT
    assert ds.training_eligibility("positive", 199) == ds.SMALL_EVENT


def test_positive_at_or_above_the_threshold_is_training_eligible():
    assert ds.training_eligibility("positive", 200) == ds.TRAINING_ELIGIBLE
    assert ds.training_eligibility("positive", 47335) == ds.TRAINING_ELIGIBLE


def test_negatives_are_always_training_eligible():
    """Zero positive pixels is the definition of a negative, not a defect."""
    assert ds.training_eligibility("negative", 0) == ds.TRAINING_ELIGIBLE


def test_eligibility_threshold_is_configurable():
    assert ds.training_eligibility("positive", 50, minimum=10) == ds.TRAINING_ELIGIBLE


# -------------------------------------------- Milestone 2.1: residual cloud
def _stack(**bands):
    out = np.zeros((16, 16, 6), np.float32)
    for name, value in bands.items():
        out[:, :, s2.BAND_INDEX[name]] = value
    return out


def test_residual_cloud_fraction_counts_bright_blue_pixels():
    ref = _stack(B02=0.02)
    ref[:8, :, s2.BAND_INDEX["B02"]] = 0.40          # half the window is haze
    assert abs(s2.residual_cloud_fraction(ref) - 0.5) < 1e-6


def test_residual_cloud_spares_bright_bare_soil():
    """Soil is red- and SWIR-sloped and stays low in blue; it must not trip."""
    ref = _stack(B02=0.09, B03=0.14, B04=0.22, B08=0.30, B11=0.38, B12=0.33)
    assert s2.residual_cloud_fraction(ref) == 0.0
    assert not s2.residual_cloud_suspect(s2.residual_cloud_fraction(ref))


def test_residual_cloud_flags_haze():
    ref = _stack(B02=0.19, B03=0.21, B04=0.22, B08=0.26, B11=0.20, B12=0.15)
    assert s2.residual_cloud_fraction(ref) == 1.0
    assert s2.residual_cloud_suspect(s2.residual_cloud_fraction(ref))


def test_residual_cloud_respects_the_valid_mask():
    ref = _stack(B02=0.02)
    ref[:8, :, s2.BAND_INDEX["B02"]] = 0.40
    valid = np.zeros((16, 16), bool)
    valid[8:, :] = True                              # look only at clean half
    assert s2.residual_cloud_fraction(ref, valid) == 0.0


def test_residual_cloud_treats_a_fully_masked_window_as_suspect():
    assert s2.residual_cloud_fraction(_stack(B02=0.02),
                                      np.zeros((16, 16), bool)) == 1.0


def test_residual_cloud_threshold_sits_between_soil_and_haze():
    """Calibration claim, asserted so it cannot drift unnoticed."""
    assert 0.0024 < s2.RESIDUAL_CLOUD_MAX_FRACTION < 0.0959


def test_residual_cloud_is_a_declared_reject_reason():
    assert "residual_cloud" in ds.REJECT_REASONS


# ------------------------------------------ Milestone 2.1: negative targeting
def _counts(shape=(9, 9)):
    return {y: np.zeros(shape, np.int32) for y in (2019, 2020, 2021, 2022)}


def test_negative_pools_exclude_the_target_year_neighbourhood():
    counts = _counts()
    counts[2020][4, 4] = 5000
    pools = ev.negative_pools(counts, 2020, np.zeros((9, 9), np.int32))
    for mask in pools.values():
        assert not mask[3:6, 3:6].any()


def test_negative_pools_identify_later_deforestation():
    counts = _counts()
    counts[2022][4, 4] = 5000
    any_def = np.zeros((9, 9), np.int32)
    any_def[4, 4] = 5000
    pools = ev.negative_pools(counts, 2020, any_def)
    assert pools["later_deforestation"][4, 4]
    assert not pools["prior_deforestation"][4, 4]
    assert not pools["stable_forest"][4, 4]


def test_negative_pools_identify_stable_forest():
    pools = ev.negative_pools(_counts(), 2020, np.zeros((9, 9), np.int32))
    assert pools["stable_forest"][4, 4]
    assert not pools["later_deforestation"].any()


def test_pool_shares_over_weight_later_deforestation():
    shares = dict(ev.NEGATIVE_POOL_SHARES)
    assert shares["later_deforestation"] > shares["prior_deforestation"]
    assert abs(sum(shares.values()) - 1.0) < 1e-9


def test_select_negative_blocks_draws_later_deforestation_when_available():
    counts = _counts((40, 40))
    counts[2022][5:35:4, 5:35:4] = 5000              # plenty of later-year blocks
    any_def = np.zeros((40, 40), np.int32)
    any_def[5:35:4, 5:35:4] = 5000
    picked = ev.select_negative_blocks(counts, 2020, any_def, 6, seed=1,
                                       min_separation_blocks=2)
    assert picked
    assert any(c["pool"] == "later_deforestation" for c in picked)


def test_select_negative_blocks_is_deterministic():
    counts = _counts((30, 30))
    any_def = np.zeros((30, 30), np.int32)
    a = ev.select_negative_blocks(counts, 2020, any_def, 4, seed=5)
    b = ev.select_negative_blocks(counts, 2020, any_def, 4, seed=5)
    assert [c["centroid_block"] for c in a] == [c["centroid_block"] for c in b]


def test_manifest_schema_declares_the_milestone_21_fields():
    for field in ("before_doy", "after_doy", "doy_difference", "selection_rule",
                  "selection_seed", "training_eligibility", "min_positive_pixels",
                  "residual_cloud_fraction", "residual_cloud_threshold",
                  "residual_cloud_method", "negative_subtype"):
        assert field in ds.MANIFEST_FIELDS, field


def test_manifest_records_a_baseline_for_both_dates():
    """Without the baseline the BOA offset cannot be decided correctly."""
    for record in manifest()["samples"]:
        assert record["before_baseline"], record["sample_id"]
        assert record["after_baseline"], record["sample_id"]


def test_converted_reflectance_is_physically_plausible():
    """The check that catches a catastrophic BOA-offset error.

    Asserted on PERCENTILES, not on the extremes. An earlier version used an
    invented 1.6 ceiling on the absolute maximum and failed on a window whose
    four brightest pixels out of 393,216 were specular - L2A reflectance
    legitimately exceeds 1.0 over bright and specular surfaces. A whole-image
    offset error, the thing worth catching, moves the whole distribution, so
    percentiles detect it and a handful of bright pixels cannot trip it.
    """
    for record in manifest()["samples"]:
        for side in ("before", "after"):
            dn = np.load(os.path.join(DATASET_DIR,
                                      f"{record['sample_id']}_{side}.npy"))
            ref = s2.to_reflectance(dn, record[f"{side}_offset_applied"],
                                    record[f"{side}_baseline"])
            p01, p999 = np.percentile(ref, [0.1, 99.9])
            assert p01 >= -0.01, (record["sample_id"], side, float(p01))
            assert p999 <= 1.0, (record["sample_id"], side, float(p999))


def test_no_sample_converts_to_mostly_negative_reflectance():
    """The guard that replaces the expired 'nothing needs a correction' test.

    v1 happened to contain no baseline >=04.00 product with the offset still
    outstanding, so asserting that was really asserting a property of a
    43-sample dataset. What must hold universally is that the conversion never
    produces a physically impossible image.
    """
    for record in manifest()["samples"]:
        for side in ("before", "after"):
            dn = np.load(os.path.join(DATASET_DIR,
                                      f"{record['sample_id']}_{side}.npy"),
                         mmap_mode="r")
            ref = s2.to_reflectance(np.asarray(dn),
                                    record[f"{side}_offset_applied"],
                                    record[f"{side}_baseline"])
            share = float((ref < 0).mean())
            assert share <= s2.MAX_NEGATIVE_REFLECTANCE_FRACTION, (
                record["sample_id"], side, share)


def test_offset_decision_obeys_metadata_when_the_data_agrees():
    dn = np.full((8, 8, 6), 2000, np.uint16)      # floor well above the offset
    d = s2.offset_decision(dn, offset_applied=False, baseline="05.00")
    assert d["metadata_says"] and d["applied"] and not d["overridden"]


def test_offset_decision_overrides_metadata_when_the_data_contradicts_it():
    """The env_N10_E100_2021_0138 case: flag says correct me, DN say otherwise."""
    dn = np.full((8, 8, 6), 250, np.uint16)       # already-corrected forest DN
    d = s2.offset_decision(dn, offset_applied=False, baseline="04.00")
    assert d["metadata_says"], "metadata rule should ask for the offset"
    assert d["overridden"] and not d["applied"]
    assert d["negative_fraction"] > 0.5


def test_offset_decision_is_inert_when_no_offset_is_due():
    d = s2.offset_decision(np.full((4, 4, 6), 300, np.uint16), True, "05.00")
    assert not d["metadata_says"] and not d["applied"] and not d["overridden"]


def test_guarded_conversion_never_returns_a_mostly_negative_image():
    dn = np.full((8, 8, 6), 250, np.uint16)
    ref = s2.to_reflectance(dn, offset_applied=False, baseline="04.00")
    assert ref.min() >= 0.0
    unguarded = s2.to_reflectance(dn, False, "04.00", trust_metadata=True)
    assert unguarded.min() < 0.0, "the unguarded path must still show the fault"


def test_negative_reflectance_bound_separates_the_observed_cases():
    """519 of 520 v2.2 windows gave exactly 0 negatives; the fault gave 0.666."""
    assert 0.0 < s2.MAX_NEGATIVE_REFLECTANCE_FRACTION < 0.6659


# ------------------------------------------------ Milestone 2.3: water gate
def _water_stack(ndwi_positive_rows):
    """Build a stack whose first N rows read as water (green > NIR)."""
    ref = np.zeros((16, 16, 6), np.float32)
    ref[:, :, s2.BAND_INDEX["B03"]] = 0.05
    ref[:, :, s2.BAND_INDEX["B08"]] = 0.30          # vegetation: NIR >> green
    ref[:ndwi_positive_rows, :, s2.BAND_INDEX["B03"]] = 0.09
    ref[:ndwi_positive_rows, :, s2.BAND_INDEX["B08"]] = 0.02   # water
    return ref


def test_water_fraction_detects_open_water():
    assert abs(s2.water_fraction(_water_stack(16)) - 1.0) < 1e-6


def test_water_fraction_is_zero_over_vegetation():
    assert s2.water_fraction(_water_stack(0)) == 0.0


def test_water_fraction_is_proportional():
    assert abs(s2.water_fraction(_water_stack(8)) - 0.5) < 1e-6


def test_water_gate_rejects_only_above_the_threshold():
    assert not s2.water_dominated(0.50)
    assert s2.water_dominated(0.51)
    assert s2.water_dominated(1.0)


def test_water_threshold_sits_in_the_observed_gap():
    """Land samples ran to 0.009; water samples started at 0.558."""
    assert 0.009 < s2.WATER_MAX_FRACTION < 0.558


def test_water_fraction_respects_the_valid_mask():
    ref = _water_stack(8)
    valid = np.zeros((16, 16), bool)
    valid[8:, :] = True
    assert s2.water_fraction(ref, valid) == 0.0


def test_water_dominated_is_a_declared_reject_reason():
    assert "water_dominated" in ds.REJECT_REASONS


# -------------------------------------------- Milestone 2.2: hemisphere
def test_southern_bands_are_detected():
    for band in ("C", "H", "L", "M"):
        assert _scene(8, 5, 2019, band=band).southern, band


def test_northern_bands_are_detected():
    for band in ("N", "P", "Q", "T", "X"):
        assert not _scene(8, 5, 2019, band=band).southern, band


def test_hemisphere_boundary_is_between_m_and_n():
    """MGRS skips I and O; M is the last southern band, N the first northern."""
    assert _scene(8, 5, 2019, band="M").southern
    assert not _scene(8, 5, 2019, band="N").southern


def test_northern_and_southern_utm_differ_by_the_false_northing():
    """The error this guards against: ~90 degrees of latitude, not a rounding
    difference."""
    north = geo.utm_to_lonlat(500000.0, 1100000.0, 47, south=False)
    south = geo.utm_to_lonlat(500000.0, 1100000.0, 47, south=True)
    assert north[1] > 0 and south[1] < 0
    assert abs(north[1] - south[1]) > 70


def test_grid_lonlat_honours_the_hemisphere_flag():
    n_lons, n_lats = geo.grid_lonlat(500000.0, 1100000.0, 0, 0, 2, 2, 10.0, 47,
                                     south=False)
    s_lons, s_lats = geo.grid_lonlat(500000.0, 1100000.0, 0, 0, 2, 2, 10.0, 47,
                                     south=True)
    assert n_lats[0, 0] > 0 > s_lats[0, 0]


# ------------------------------------- Milestone 2.2: expansion and accounting
def test_milestone_22_reject_reasons_are_declared():
    for reason in ("insufficient_positive_px", "duplicate_event",
                   "residual_cloud"):
        assert reason in ds.REJECT_REASONS, reason


def test_reject_reason_aliases_resolve_to_real_reasons():
    """The spec names some reasons differently; the aliases must not dangle."""
    for alias, actual in ds.REJECT_REASON_ALIASES.items():
        assert actual in ds.REJECT_REASONS, (alias, actual)


def test_assign_splits_respects_the_requested_ratios():
    blocks = [[f"T{i:02d}"] for i in range(20)]
    weights = {f"T{i:02d}": 10 for i in range(20)}
    out = ds.assign_splits(blocks, weights, ratios=(0.70, 0.15, 0.15))
    got = {s: sum(weights[t] for t in out[s]) for s in ds.SPLITS}
    total = sum(got.values())
    assert total == 200
    assert abs(got["train"] / total - 0.70) < 0.12, got
    assert got["train"] > got["val"] and got["train"] > got["test"]


def test_assign_splits_still_places_every_block():
    blocks = [["A", "B"], ["C"], ["D"], ["E"]]
    weights = {t: 1 for t in "ABCDE"}
    out = ds.assign_splits(blocks, weights, ratios=(0.70, 0.15, 0.15))
    assert sorted(t for v in out.values() for t in v) == list("ABCDE")


def test_no_event_is_sampled_more_than_once():
    ids = [r["event_id"] for r in manifest()["samples"] if r["event_id"]]
    assert len(ids) == len(set(ids)), "an event produced two samples"


def test_no_window_is_sampled_more_than_once():
    aois = [r["aoi_id"] for r in manifest()["samples"]]
    assert len(aois) == len(set(aois)), "a window was sampled twice"


def test_every_sample_records_a_forest_context():
    for record in manifest()["samples"]:
        assert record.get("forest_context"), record["sample_id"]


def test_dataset_spans_more_than_one_tmf_tile():
    assert len({r["tmf_tile"] for r in manifest()["samples"]}) >= 2


def test_small_events_are_retained_not_deleted():
    """Sub-threshold positives stay on disk; they are excluded from training,
    not discarded from the archive."""
    for record in manifest()["samples"]:
        if record.get("training_eligibility") == ds.SMALL_EVENT:
            path = os.path.join(DATASET_DIR, f"{record['sample_id']}_label.npy")
            assert os.path.exists(path), record["sample_id"]
            assert 0 < record["positive_px"] < ds.MIN_POSITIVE_PIXELS


def test_training_eligible_positives_meet_the_pixel_threshold():
    for record in manifest()["samples"]:
        if (record["sample_type"] == "positive"
                and record.get("training_eligibility") == ds.TRAINING_ELIGIBLE):
            assert record["positive_px"] >= ds.MIN_POSITIVE_PIXELS, record["sample_id"]


def test_no_positive_sample_has_an_empty_label():
    for record in manifest()["samples"]:
        if record["sample_type"] == "positive":
            assert record["positive_px"] > 0, record["sample_id"]


# ------------------------------- Milestone 2.4: class-aware split optimiser
def _split_fixture():
    """Nine single-tile blocks with deliberately lopsided class composition."""
    blocks = [[f"T{i:02d}"] for i in range(9)]
    stats = {}
    for i in range(9):
        # tiles 0-3 all positive, tiles 4-8 all negative
        stats[f"T{i:02d}"] = {"n": 10, "pos": 10 if i < 4 else 0}
    return blocks, stats


def test_split_cost_prefers_mirroring_the_dataset_rate():
    _, stats = _split_fixture()
    total = sum(s["n"] for s in stats.values())
    mirrored = {"train": ["T00", "T01", "T04", "T05", "T06"],
                "val": ["T02", "T07"], "test": ["T03", "T08"]}
    lopsided = {"train": ["T00", "T01", "T02", "T03", "T04"],
                "val": ["T05", "T06"], "test": ["T07", "T08"]}
    assert (ds.split_cost(mirrored, stats, (0.70, 0.15, 0.15), total)
            < ds.split_cost(lopsided, stats, (0.70, 0.15, 0.15), total))


def test_balanced_split_improves_on_the_class_blind_packer():
    blocks, stats = _split_fixture()
    total = sum(s["n"] for s in stats.values())
    weights = {t: stats[t]["n"] for t in stats}
    blind = ds.assign_splits(blocks, weights, (0.70, 0.15, 0.15))
    aware = ds.assign_splits_balanced(blocks, stats, seed=0)
    assert (ds.split_cost(aware, stats, (0.70, 0.15, 0.15), total)
            <= ds.split_cost(blind, stats, (0.70, 0.15, 0.15), total))


def test_balanced_split_keeps_every_tile_exactly_once():
    blocks, stats = _split_fixture()
    out = ds.assign_splits_balanced(blocks, stats, seed=1)
    placed = sorted(t for v in out.values() for t in v)
    assert placed == sorted(stats)


def test_balanced_split_keeps_blocks_whole():
    """A multi-tile block must never be torn across splits."""
    blocks = [["A1", "A2", "A3"], ["B1"], ["C1"], ["D1"]]
    stats = {t: {"n": 5, "pos": 2} for b in blocks for t in b}
    out = ds.assign_splits_balanced(blocks, stats, seed=3)
    placement = {t: s for s, tiles in out.items() for t in tiles}
    assert placement["A1"] == placement["A2"] == placement["A3"]


def test_balanced_split_never_empties_a_split():
    blocks, stats = _split_fixture()
    out = ds.assign_splits_balanced(blocks, stats, seed=5)
    assert all(out[s] for s in ds.SPLITS)


def test_balanced_split_is_deterministic():
    blocks, stats = _split_fixture()
    a = ds.assign_splits_balanced(blocks, stats, seed=7)
    b = ds.assign_splits_balanced(blocks, stats, seed=7)
    assert a == b


def test_balanced_split_produces_disjoint_sets():
    blocks, stats = _split_fixture()
    out = ds.assign_splits_balanced(blocks, stats, seed=2)
    assert ds.split_integrity(out)["disjoint"]


def test_class_weight_is_above_one():
    """Size is exactly satisfiable by the greedy start, so class must outweigh
    it or the search sits in the starting local minimum."""
    assert ds.CLASS_BALANCE_WEIGHT > 1.0


def test_land_pool_shares_demote_the_ocean_prone_pool():
    standard = dict(ev.NEGATIVE_POOL_SHARES)
    land = dict(ev.LAND_POOL_SHARES)
    assert land["stable_forest"] < standard["stable_forest"]
    assert abs(sum(land.values()) - 1.0) < 1e-9
    assert land["later_deforestation"] + land["prior_deforestation"] == 0.80


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
