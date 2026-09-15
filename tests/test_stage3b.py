"""Focused tests for Stage 3B-1: region-level construction/demolition targets.

pytest is not a project dependency, so these are plain functions that assert.
Run either way:

    python tests/test_stage3b.py          # standalone runner
    pytest tests/test_stage3b.py          # if pytest is available

No dataset and no model weights are needed: the label logic is pure and is
exercised on tiny synthetic masks. The existing Stage 1 and Stage 3A tests are
untouched.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.common.regions import extract_regions
from src.domains.built_environment.data.s2looking import (
    AFTER_IMAGE_DIR, BEFORE_IMAGE_DIR, CONSTRUCTION, CONSTRUCTION_LABEL_DIR,
    DEMOLITION, DEMOLITION_LABEL_DIR, MIXED, PURITY_THRESHOLD, TEMPORAL_ORDERING,
    binarize, region_targets, split_integrity, summarize_regions)


# --------------------------------------------------------------- fixtures
def blank(h=40, w=40):
    return np.zeros((h, w), bool)


def split_block(n_construction, n_demolition, top=2, left=2, width=10):
    """One solid block whose pixels are split between the two annotation maps.

    Filling row-major keeps the block a single 8-connected component while the
    construction/demolition ratio is controlled exactly.
    """
    total = n_construction + n_demolition
    l1, l2 = blank(), blank()
    for k in range(total):
        r, c = top + k // width, left + k % width
        (l1 if k < n_construction else l2)[r, c] = True
    return l1, l2


# --------------------------------------------------------------- targets
def test_construction_only_component():
    l1, l2 = blank(), blank()
    l1[2:10, 2:10] = True                       # 64 px, all newly built
    regions = region_targets(l1, l2)
    assert len(regions) == 1
    r = regions[0]
    assert r["target"] == CONSTRUCTION
    assert r["area_px"] == 64
    assert r["construction_pixels"] == 64
    assert r["demolition_pixels"] == 0
    assert r["purity"] == 1.0


def test_demolition_only_component():
    l1, l2 = blank(), blank()
    l2[2:10, 2:10] = True                       # 64 px, all demolished
    regions = region_targets(l1, l2)
    assert len(regions) == 1
    assert regions[0]["target"] == DEMOLITION
    assert regions[0]["demolition_pixels"] == 64
    assert regions[0]["construction_pixels"] == 0


def test_mixed_component_is_mixed_not_uncertain():
    l1, l2 = split_block(50, 50)
    regions = region_targets(l1, l2)
    assert len(regions) == 1
    assert regions[0]["target"] == MIXED
    assert regions[0]["target"] != "uncertain"
    assert regions[0]["purity"] == 0.5


def test_purity_exactly_at_threshold_is_classified():
    l1, l2 = split_block(95, 5)                 # purity == 0.95
    r = region_targets(l1, l2)[0]
    assert r["purity"] == PURITY_THRESHOLD
    assert r["target"] == CONSTRUCTION


def test_purity_just_below_threshold_is_mixed():
    l1, l2 = split_block(94, 6)                 # purity == 0.94
    r = region_targets(l1, l2)[0]
    assert r["purity"] < PURITY_THRESHOLD
    assert r["target"] == MIXED


def test_purity_threshold_applies_to_demolition_too():
    l1, l2 = split_block(5, 95)
    r = region_targets(l1, l2)[0]
    assert r["target"] == DEMOLITION
    assert r["purity"] == PURITY_THRESHOLD


def test_equal_pixel_counts_are_mixed():
    l1, l2 = split_block(16, 16)
    assert region_targets(l1, l2)[0]["target"] == MIXED


# ----------------------------------------------------------- area filter
def test_min_area_filtering_uses_project_default():
    assert config.DEFAULT_MIN_AREA_PX == 32
    l1, l2 = blank(), blank()
    l1[2:6, 2:10] = True                        # 4x8 = 32 px  -> kept
    l1[20:24, 20:28] = True                     # 4x8 = 32 px
    l1[23, 27] = False                          # -> 31 px     -> discarded
    regions = region_targets(l1, l2)
    assert len(regions) == 1
    assert regions[0]["area_px"] == 32


def test_min_area_is_configurable():
    l1, l2 = blank(), blank()
    l1[2:6, 2:10] = True                        # 32 px
    assert region_targets(l1, l2, min_area_px=33) == []
    assert len(region_targets(l1, l2, min_area_px=32)) == 1


def test_empty_scene_yields_no_regions():
    assert region_targets(blank(), blank()) == []


# ------------------------------------------------------- ids / ordering
def test_region_ids_deterministic_and_area_ordered():
    l1, l2 = blank(60, 60), blank(60, 60)
    l1[2:8, 2:8] = True                         # 36 px
    l1[20:32, 20:32] = True                     # 144 px  (largest)
    l2[40:48, 40:48] = True                     # 64 px
    runs = [region_targets(l1, l2) for _ in range(5)]
    assert all(r == runs[0] for r in runs), "region extraction is not deterministic"
    assert [r["region_id"] for r in runs[0]] == [1, 2, 3]
    assert [r["area_px"] for r in runs[0]] == [144, 64, 36]
    assert [r["target"] for r in runs[0]] == [CONSTRUCTION, DEMOLITION, CONSTRUCTION]


def test_geometry_matches_production_region_extraction():
    """Stage 3B regions must be the same objects production already reports."""
    l1, l2 = blank(60, 60), blank(60, 60)
    l1[2:8, 2:8] = True
    l1[20:32, 20:32] = True
    l2[40:48, 40:48] = True
    prep = region_targets(l1, l2)
    _mask, production = extract_regions(l1 | l2)
    assert len(prep) == len(production)
    for p, q in zip(prep, production):
        assert p["region_id"] == q.id
        assert p["area_px"] == q.area_px
        assert [p["bbox_x"], p["bbox_y"], p["bbox_w"], p["bbox_h"]] == list(q.bbox_xywh)
        assert [p["centroid_x"], p["centroid_y"]] == list(q.centroid_xy)


def test_eight_connectivity_joins_diagonal_pixels():
    l1, l2 = blank(), blank()
    l1[2:10, 2:10] = True                       # 64 px
    l1[10:18, 10:18] = True                     # touches only at a diagonal
    regions = region_targets(l1, l2)
    assert len(regions) == 1, "8-connectivity should merge diagonally touching blocks"
    assert regions[0]["area_px"] == 128


# ------------------------------------------------------------- overlap
def test_overlap_pixels_are_counted_not_hidden():
    l1, l2 = blank(), blank()
    l1[2:10, 2:10] = True
    l2[2:10, 2:6] = True                        # 32 px annotated in BOTH maps
    r = region_targets(l1, l2)[0]
    assert r["overlap_pixels"] == 32
    assert r["area_px"] == 64                   # union
    assert r["construction_pixels"] + r["demolition_pixels"] == 96  # counted twice


# --------------------------------------------------------- malformed input
def test_mismatched_label_dimensions_raise():
    try:
        region_targets(blank(40, 40), blank(40, 41))
    except ValueError as e:
        assert "shape" in str(e)
    else:
        raise AssertionError("mismatched label dimensions must raise")


def test_unexpected_label_values_raise():
    bad = np.zeros((40, 40), np.uint8)
    bad[2:10, 2:10] = 255                       # not yet binarized
    try:
        region_targets(bad, np.zeros((40, 40), np.uint8))
    except ValueError as e:
        assert "0/1" in str(e) or "binarize" in str(e).lower()
    else:
        raise AssertionError("unexpected label values must raise")


def test_three_dimensional_masks_raise():
    try:
        region_targets(np.zeros((40, 40, 3), bool), np.zeros((40, 40, 3), bool))
    except ValueError as e:
        assert "2-D" in str(e)
    else:
        raise AssertionError("3-D masks must raise")


def test_invalid_purity_threshold_raises():
    try:
        region_targets(blank(), blank(), purity_threshold=0.0)
    except ValueError as e:
        assert "purity" in str(e)
    else:
        raise AssertionError("invalid purity threshold must raise")


# ------------------------------------------------------------- binarize
def test_binarize_collapses_colour_axis_and_thresholds():
    a = np.zeros((10, 10, 3), np.uint8)
    a[2:5, 2:5] = 255
    m = binarize(a)
    assert m.dtype == bool and m.shape == (10, 10)
    assert int(m.sum()) == 9


def test_binarize_decodes_blue_channel_label1():
    """S2Looking label1 (construction) lives in the BLUE channel only.

    Collapsing RGB to channel 0 would return an empty mask and silently erase
    every construction annotation in the dataset.
    """
    a = np.zeros((10, 10, 3), np.uint8)
    a[2:5, 2:5, 2] = 255
    m = binarize(a)
    assert m.shape == (10, 10)
    assert int(m.sum()) == 9, "blue-channel annotation was lost"


def test_binarize_decodes_red_channel_label2():
    """S2Looking label2 (demolition) lives in the RED channel only."""
    a = np.zeros((10, 10, 3), np.uint8)
    a[2:6, 2:6, 0] = 255
    assert int(binarize(a).sum()) == 16


def test_binarize_passes_through_bool():
    m = np.zeros((4, 4), bool)
    m[1, 1] = True
    assert binarize(m) is m


def test_binarize_ignores_edge_antialiasing():
    a = np.zeros((10, 10), np.uint8)
    a[2:5, 2:5] = 255
    a[5, 2] = 100                               # antialiased edge pixel
    assert int(binarize(a).sum()) == 9


# ----------------------------------------------------------- statistics
def _rec(scene, split, target, area, n_c, n_d, region_id=1):
    return {"scene_id": scene, "split": split, "region_id": region_id,
            "target": target, "area_px": area,
            "construction_pixels": n_c, "demolition_pixels": n_d,
            "overlap_pixels": 0, "purity": 1.0}


def test_summary_counts_and_percentages():
    records = [
        _rec("a", "train", CONSTRUCTION, 100, 100, 0),
        _rec("a", "train", CONSTRUCTION, 200, 200, 0),
        _rec("b", "train", DEMOLITION, 50, 0, 50),
        _rec("c", "test", MIXED, 80, 40, 40),
    ]
    s = summarize_regions(records)
    assert s["overall"]["regions"] == 4
    assert s["overall"][CONSTRUCTION] == 2
    assert s["overall"][DEMOLITION] == 1
    assert s["overall"][MIXED] == 1
    assert s["overall"][CONSTRUCTION + "_pct"] == 50.0
    assert s["overall"][DEMOLITION + "_pct"] == 25.0
    assert s["parameters"]["purity_threshold"] == PURITY_THRESHOLD
    assert s["parameters"]["min_area_px"] == config.DEFAULT_MIN_AREA_PX
    assert s["parameters"]["connectivity"] == 8


def test_summary_per_split_and_empty_scenes_counted():
    records = [
        _rec("a", "train", CONSTRUCTION, 100, 100, 0),
        _rec("b", "val", DEMOLITION, 60, 0, 60),
    ]
    # scene "z" survives the split listing but produced no region at all.
    s = summarize_regions(records, scenes_by_split={"train": ["a"], "val": ["b", "z"]})
    assert s["overall"]["scenes"] == 3
    assert s["per_split"]["train"]["scenes"] == 1
    assert s["per_split"]["val"]["scenes"] == 2
    assert s["per_split"]["val"][DEMOLITION] == 1
    assert s["per_scene"]["scenes_with_no_regions"] == 1


def test_summary_area_pixel_and_per_scene_distribution():
    records = [
        _rec("a", "train", CONSTRUCTION, 100, 100, 0),
        _rec("a", "train", DEMOLITION, 300, 0, 300),
        _rec("b", "train", CONSTRUCTION, 200, 200, 0),
    ]
    s = summarize_regions(records)
    assert s["area_px"][CONSTRUCTION]["min"] == 100
    assert s["area_px"][CONSTRUCTION]["max"] == 200
    assert s["area_px"][CONSTRUCTION]["median"] == 150
    assert s["area_px"][MIXED] is None            # no mixed regions present
    assert s["pixels"]["construction_pixels"] == 300
    assert s["pixels"]["demolition_pixels"] == 300
    assert s["pixels"]["total_changed_pixels"] == 600
    # scene "a" holds one construction and one demolition region
    assert s["per_scene"]["scenes_with_both_directions"] == 1
    assert s["per_scene"]["regions_per_scene"]["max"] == 2


def test_demolition_gate_flags_underrepresentation():
    records = [_rec(f"s{i}", "train", CONSTRUCTION, 100, 100, 0) for i in range(99)]
    records.append(_rec("s99", "train", DEMOLITION, 100, 0, 100))
    s = summarize_regions(records)
    assert s["gate"]["demolition_pct"] == 1.0
    assert s["gate"]["demolition_sufficiently_represented"] is False

    balanced = [_rec(f"s{i}", "train", CONSTRUCTION, 100, 100, 0) for i in range(90)]
    balanced += [_rec(f"d{i}", "train", DEMOLITION, 100, 0, 100) for i in range(10)]
    assert summarize_regions(balanced)["gate"]["demolition_sufficiently_represented"] is True


# ------------------------------------------------------- split integrity
def test_split_integrity_accepts_disjoint_splits():
    report = split_integrity({"train": ["a", "b"], "val": ["c"], "test": ["d"]})
    assert report["ok"] is True
    assert report["unique_scene_ids"] == 4
    assert report["scene_counts"] == {"train": 2, "val": 1, "test": 1}


def test_split_integrity_detects_scene_in_two_splits():
    report = split_integrity({"train": ["a", "b"], "test": ["b"]})
    assert report["ok"] is False
    assert "b" in report["cross_split_duplicates"]


def test_split_integrity_detects_duplicate_within_split():
    report = split_integrity({"train": ["a", "a"], "test": ["b"]})
    assert report["ok"] is False
    assert report["within_split_duplicates"]["train"] == 1


# ------------------------------------------------- ratified temporal ordering
def test_ratified_temporal_ordering_is_pinned():
    """The project decision lives in these four constants.

    Flipping any of them silently inverts every construction/demolition target
    while all statistics still look healthy, so the decision is pinned here.
    See docs/S2LOOKING_TEMPORAL_SEMANTICS.md.
    """
    assert BEFORE_IMAGE_DIR == "Image2"
    assert AFTER_IMAGE_DIR == "Image1"
    assert CONSTRUCTION_LABEL_DIR == "label1"
    assert DEMOLITION_LABEL_DIR == "label2"


def test_temporal_ordering_metadata_is_self_describing():
    assert TEMPORAL_ORDERING["before_image_dir"] == BEFORE_IMAGE_DIR
    assert TEMPORAL_ORDERING["after_image_dir"] == AFTER_IMAGE_DIR
    assert TEMPORAL_ORDERING["construction_label_dir"] == CONSTRUCTION_LABEL_DIR
    assert TEMPORAL_ORDERING["demolition_label_dir"] == DEMOLITION_LABEL_DIR


def test_temporal_ordering_is_not_claimed_as_author_documented():
    """It is a project decision, not something the authors state anywhere."""
    assert TEMPORAL_ORDERING["documented_by_authors"] is False
    assert "project decision" in TEMPORAL_ORDERING["basis"].lower()


def test_before_and_after_are_distinct_sources():
    assert BEFORE_IMAGE_DIR != AFTER_IMAGE_DIR
    assert CONSTRUCTION_LABEL_DIR != DEMOLITION_LABEL_DIR


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
