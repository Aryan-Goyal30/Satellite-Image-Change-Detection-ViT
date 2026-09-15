"""Focused tests for Stage 3B-3 Step 1: the shared crop/preprocessing protocol.

pytest is not a project dependency, so these are plain functions that assert.
Run either way:

    python tests/test_stage3b3.py
    pytest tests/test_stage3b3.py

Scope is deliberately narrow: geometry, channel ordering, normalisation, and
**parity** with the implementation the existing training crops were built with.
Nothing here loads a model, and the 3.77 GB crop array is only ever touched
through a memmap, a handful of rows at a time.
"""
import csv
import os
import shutil
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.common.preprocessing import IMAGENET_MEAN, IMAGENET_STD
from src.domains.built_environment.data.s2looking import (
    AFTER_IMAGE_DIR, BEFORE_IMAGE_DIR)
from src.domains.built_environment.direction.crops import (
    AFTER_SLICE, BEFORE_SLICE, CONTEXT_FRACTION, CROP_SIZE, MIN_SIDE,
    N_CHANNELS, PROTOCOL, crop_box, crop_image, crop_pair, region_model_input,
    to_model_input)


# ---------------------------------------------------------------- legacy ref
# Verbatim copy of the geometry that built the existing crops, kept HERE (test
# only) so parity is checked against the original arithmetic rather than
# against the module under test restating itself.
LEGACY_CROP_SIZE = 128
LEGACY_CONTEXT_FRACTION = 0.25
LEGACY_MIN_SIDE = 16


def legacy_crop_box(x, y, w, h, img_w, img_h):
    mx, my = LEGACY_CONTEXT_FRACTION * w, LEGACY_CONTEXT_FRACTION * h
    x0, x1 = x - mx, x + w + mx
    y0, y1 = y - my, y + h + my
    side = max(x1 - x0, y1 - y0, LEGACY_MIN_SIDE)
    side = min(side, img_w, img_h)
    cx, cy = x + w / 2.0, y + h / 2.0
    x0 = cx - side / 2.0
    y0 = cy - side / 2.0
    x0 = min(max(0.0, x0), img_w - side)
    y0 = min(max(0.0, y0), img_h - side)
    return int(round(x0)), int(round(y0)), int(round(side))


def legacy_extract(img, box):
    x0, y0, side = box
    return np.asarray(
        img.crop((x0, y0, x0 + side, y0 + side))
           .resize((LEGACY_CROP_SIZE, LEGACY_CROP_SIZE), Image.BILINEAR),
        dtype=np.uint8)


def legacy_normalise(crop_hwc6):
    """Verbatim RegionCrops.__getitem__ arithmetic with augmentation off."""
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)
    m6 = np.concatenate([mean, mean]).reshape(6, 1, 1)
    s6 = np.concatenate([std, std]).reshape(6, 1, 1)
    arr = np.ascontiguousarray(crop_hwc6.transpose(2, 0, 1), dtype=np.float32) / 255.0
    arr = (arr - m6) / s6
    return torch.from_numpy(arr)


def rgb(seed, w=256, h=256):
    a = np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(a)


# ------------------------------------------------------------- constants
def test_protocol_constants_match_the_trained_checkpoint():
    assert CROP_SIZE == 128
    assert CONTEXT_FRACTION == 0.25
    assert MIN_SIDE == 16
    assert N_CHANNELS == 6
    assert BEFORE_SLICE == slice(0, 3)
    assert AFTER_SLICE == slice(3, 6)
    assert PROTOCOL["crop_size"] == 128
    assert PROTOCOL["context_fraction"] == 0.25
    assert PROTOCOL["min_side_px"] == 16


# -------------------------------------------------------------- geometry
def test_crop_box_is_bit_identical_to_legacy_geometry():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        img_w = int(rng.integers(20, 1200))
        img_h = int(rng.integers(20, 1200))
        w = int(rng.integers(1, max(2, img_w)))
        h = int(rng.integers(1, max(2, img_h)))
        x = int(rng.integers(0, max(1, img_w - w + 1)))
        y = int(rng.integers(0, max(1, img_h - h + 1)))
        assert crop_box(x, y, w, h, img_w, img_h) == \
            legacy_crop_box(x, y, w, h, img_w, img_h), (x, y, w, h, img_w, img_h)


def test_context_expansion_is_25_percent_each_side():
    # 100x100 bbox -> 25 px added on every side -> side 150
    x0, y0, side = crop_box(400, 400, 100, 100, 1024, 1024)
    assert side == 150
    # centred on the bbox centre (450, 450)
    assert x0 == 375 and y0 == 375


def test_box_is_squared_about_the_bbox_centre():
    # 200 wide x 40 tall: expansion adds 0.25*w on EACH side, so the expanded
    # width is 1.5 * 200 = 300, and that dominates the 1.5 * 40 = 60 height.
    x0, y0, side = crop_box(100, 500, 200, 40, 1024, 1024)
    assert side == 300
    assert x0 + side / 2 == 100 + 200 / 2      # centred horizontally
    assert y0 + side / 2 == 500 + 40 / 2       # centred vertically


def test_boundary_clipping_shifts_rather_than_shrinks():
    # bbox hard against the top-left corner
    x0, y0, side = crop_box(0, 0, 100, 100, 1024, 1024)
    assert side == 150, "side must not shrink at the border"
    assert (x0, y0) == (0, 0), "box is shifted inside the scene"
    # and against the bottom-right corner
    x0, y0, side = crop_box(924, 924, 100, 100, 1024, 1024)
    assert side == 150
    assert x0 + side <= 1024 and y0 + side <= 1024
    assert (x0, y0) == (874, 874)


def test_minimum_side_floor():
    # a 4x4 bbox expands to 6, which is below the floor
    _x0, _y0, side = crop_box(500, 500, 4, 4, 1024, 1024)
    assert side == MIN_SIDE


def test_side_cannot_exceed_the_scene():
    _x0, _y0, side = crop_box(0, 0, 900, 900, 64, 48)
    assert side == 48, "side is capped by the smaller image dimension"


def test_box_overrun_is_bounded_by_one_pixel():
    """The geometry's true contract, not an idealised one.

    `x0` and `side` are rounded to int INDEPENDENTLY, so when the box is
    clamped hard against an edge the pair can overshoot by a single pixel. This
    is pre-existing behaviour of the implementation the crops were built with -
    it is reproduced here deliberately, not fixed, because changing it would
    invalidate the existing training data. PIL's .crop() zero-pads such a
    request, so the effect is at most a one-pixel border.
    """
    rng = np.random.default_rng(7)
    for _ in range(2000):
        img_w = int(rng.integers(16, 300))
        img_h = int(rng.integers(16, 300))
        w = int(rng.integers(1, img_w + 1))
        h = int(rng.integers(1, img_h + 1))
        x = int(rng.integers(0, img_w - w + 1))
        y = int(rng.integers(0, img_h - h + 1))
        x0, y0, side = crop_box(x, y, w, h, img_w, img_h)
        assert x0 >= 0 and y0 >= 0
        assert x0 + side <= img_w + 1, (x, y, w, h, img_w, img_h)
        assert y0 + side <= img_h + 1, (x, y, w, h, img_w, img_h)


def test_real_region_sizes_stay_strictly_inside_a_scene():
    """For S2Looking-shaped inputs the box is strictly inside the image.

    The one-pixel overrun above needs `side` to approach the image dimension,
    which region-sized bboxes in a 1024x1024 scene never do. Verified against
    all 40,221 generated regions at the time of writing; re-checked here on a
    representative sweep so a future geometry change cannot silently start
    clipping real crops.
    """
    rng = np.random.default_rng(21)
    for _ in range(3000):
        w = int(rng.integers(1, 400))
        h = int(rng.integers(1, 400))
        x = int(rng.integers(0, 1024 - w + 1))
        y = int(rng.integers(0, 1024 - h + 1))
        x0, y0, side = crop_box(x, y, w, h, 1024, 1024)
        assert x0 >= 0 and y0 >= 0
        assert x0 + side <= 1024 and y0 + side <= 1024, (x, y, w, h)


# --------------------------------------------------------------- cropping
def test_crop_image_matches_legacy_extract_bit_for_bit():
    img = rgb(1, 300, 200)
    for bbox in [(10, 10, 40, 40), (0, 0, 5, 5), (250, 150, 60, 60), (100, 90, 7, 130)]:
        box = crop_box(*bbox, img.width, img.height)
        assert np.array_equal(crop_image(img, box), legacy_extract(img, box))


def test_crop_pair_shape_dtype_and_channel_ordering():
    before = Image.fromarray(np.full((200, 200, 3), 10, np.uint8))
    after = Image.fromarray(np.full((200, 200, 3), 200, np.uint8))
    out = crop_pair(before, after, (50, 50, 60, 60))
    assert out.shape == (CROP_SIZE, CROP_SIZE, N_CHANNELS)
    assert out.dtype == np.uint8
    assert (out[:, :, BEFORE_SLICE] == 10).all(), "channels 0:3 must be BEFORE"
    assert (out[:, :, AFTER_SLICE] == 200).all(), "channels 3:6 must be AFTER"


def test_crop_pair_uses_the_same_box_for_both_dates():
    before, after = rgb(2, 256, 256), rgb(3, 256, 256)
    out = crop_pair(before, after, (30, 40, 50, 20))
    box = crop_box(30, 40, 50, 20, 256, 256)
    assert np.array_equal(out[:, :, BEFORE_SLICE], legacy_extract(before, box))
    assert np.array_equal(out[:, :, AFTER_SLICE], legacy_extract(after, box))


def test_crop_pair_rejects_mismatched_image_sizes():
    try:
        crop_pair(rgb(4, 256, 256), rgb(5, 128, 128), (0, 0, 10, 10))
    except ValueError as e:
        assert "same size" in str(e) or "differ in size" in str(e)
    else:
        raise AssertionError("mismatched date sizes must raise")


def test_source_image_is_not_resized_before_cropping():
    """A region in a large scene must be read from the source pixel grid."""
    img = rgb(6, 1024, 1024)
    bbox = (700, 700, 80, 80)
    box = crop_box(*bbox, 1024, 1024)
    direct = np.asarray(
        img.crop((box[0], box[1], box[0] + box[2], box[1] + box[2]))
           .resize((CROP_SIZE, CROP_SIZE), Image.BILINEAR), dtype=np.uint8)
    assert np.array_equal(crop_image(img, box), direct)


def test_accepts_numpy_and_pil_identically():
    arr_b = np.random.default_rng(8).integers(0, 256, (200, 200, 3), dtype=np.uint8)
    arr_a = np.random.default_rng(9).integers(0, 256, (200, 200, 3), dtype=np.uint8)
    bbox = (20, 30, 44, 51)
    from_np = crop_pair(arr_b, arr_a, bbox)
    from_pil = crop_pair(Image.fromarray(arr_b), Image.fromarray(arr_a), bbox)
    assert np.array_equal(from_np, from_pil)


def test_rejects_non_uint8_source():
    bad = np.zeros((64, 64, 3), np.float32)
    try:
        crop_pair(bad, bad, (0, 0, 10, 10))
    except TypeError as e:
        assert "uint8" in str(e)
    else:
        raise AssertionError("a float source must raise rather than be silently cast")


# ---------------------------------------------------------- normalisation
def test_to_model_input_shape_dtype_batched_and_unbatched():
    crop = np.random.default_rng(10).integers(0, 256, (128, 128, 6), dtype=np.uint8)
    b = to_model_input(crop)
    u = to_model_input(crop, batched=False)
    assert b.shape == (1, 6, 128, 128) and b.dtype == torch.float32
    assert u.shape == (6, 128, 128) and u.dtype == torch.float32
    assert torch.equal(b[0], u)


def test_to_model_input_is_bit_identical_to_legacy_normalisation():
    rng = np.random.default_rng(11)
    for _ in range(20):
        crop = rng.integers(0, 256, (128, 128, 6), dtype=np.uint8)
        assert torch.equal(to_model_input(crop, batched=False), legacy_normalise(crop))


def test_normalisation_applies_imagenet_stats_per_date():
    crop = np.zeros((128, 128, 6), np.uint8)          # all zeros -> -mean/std
    t = to_model_input(crop, batched=False)
    for c in range(3):
        expected = np.float32(0.0 - IMAGENET_MEAN[c]) / np.float32(IMAGENET_STD[c])
        assert torch.allclose(t[c], torch.tensor(expected))      # BEFORE
        assert torch.allclose(t[c + 3], torch.tensor(expected))  # AFTER, same stats


def test_to_model_input_rejects_wrong_shape_and_dtype():
    for bad, exc in (((np.zeros((128, 128, 3), np.uint8)), ValueError),
                     ((np.zeros((128, 128, 6), np.float32)), TypeError)):
        try:
            to_model_input(bad)
        except exc:
            continue
        raise AssertionError(f"expected {exc.__name__}")


def test_region_model_input_is_crop_pair_then_normalise():
    before, after = rgb(12, 256, 256), rgb(13, 256, 256)
    bbox = (60, 70, 40, 90)
    assert torch.equal(region_model_input(before, after, bbox, batched=False),
                       to_model_input(crop_pair(before, after, bbox), batched=False))


def test_to_model_input_matches_the_training_dataset_path():
    """Bit-identical to RegionCrops.__getitem__ with augmentation off."""
    from src.domains.built_environment.direction.dataset import RegionCrops
    tmp = tempfile.mkdtemp()
    try:
        x = np.random.default_rng(14).integers(0, 256, (6, 128, 128, 6), dtype=np.uint8)
        np.save(os.path.join(tmp, "train_x.npy"), x)
        np.save(os.path.join(tmp, "train_y.npy"), np.zeros(6, np.int64))
        ds = RegionCrops(tmp, "train", mode="both", augment=False)
        for i in range(len(ds)):
            assert torch.equal(ds[i][0], to_model_input(x[i], batched=False))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------- parity with generated crops
def _bbox_index():
    path = os.path.join(config.S2LOOKING_META, "region_labels.csv")
    if not os.path.exists(path):
        return None
    idx = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            idx[(r["split"], r["scene_id"], r["region_id"])] = (
                int(r["bbox_x"]), int(r["bbox_y"]), int(r["bbox_w"]), int(r["bbox_h"]))
    return idx


def test_parity_with_existing_generated_training_crops():
    """Rebuild real crops with the shared module and compare to what is on disk.

    This is the strongest available check: it proves the extracted protocol
    reproduces the actual bytes the classifier was trained on. Only a handful of
    rows are read, through a memmap - the full array is never loaded.
    """
    crops_dir = os.path.join(config.S2LOOKING_RAW, "crops")
    scenes_dir = os.path.join(config.S2LOOKING_RAW, "S2Looking")
    meta_path = os.path.join(crops_dir, "val_meta.csv")
    x_path = os.path.join(crops_dir, "val_x.npy")
    bboxes = _bbox_index()
    if not (os.path.exists(meta_path) and os.path.exists(x_path)
            and os.path.isdir(scenes_dir) and bboxes):
        print("      [skip] S2Looking crops/imagery not present on this machine")
        return

    with open(meta_path, newline="", encoding="utf-8") as f:
        meta = list(csv.DictReader(f))
    X = np.load(x_path, mmap_mode="r")
    assert len(meta) == X.shape[0]

    checked = 0
    for i in [0, 1, 2, len(meta) // 2, len(meta) - 2, len(meta) - 1]:
        row = meta[i]
        key = ("val", row["scene_id"], row["region_id"])
        if key not in bboxes:
            continue
        d = os.path.join(scenes_dir, "val")
        before = Image.open(os.path.join(d, BEFORE_IMAGE_DIR, row["scene_id"] + ".png"))
        after = Image.open(os.path.join(d, AFTER_IMAGE_DIR, row["scene_id"] + ".png"))
        rebuilt = crop_pair(before, after, bboxes[key])
        before.close()
        after.close()
        stored = np.asarray(X[i])
        assert rebuilt.shape == stored.shape
        assert rebuilt.dtype == stored.dtype
        assert np.array_equal(rebuilt, stored), (
            f"row {i} ({row['scene_id']}/r{row['region_id']}) differs from the "
            f"stored training crop")
        checked += 1
    assert checked >= 4, f"expected to verify at least 4 real crops, did {checked}"
    print(f"      verified {checked} real crops bit-identical to stored training data")


# ========================= Step 2: engine integration =====================
# Contract, classifier wrapper and engine wiring. Engine tests inject a FAKE
# classifier so they need neither a GPU nor the direction checkpoint.

class _FakeClassifier:
    """Stands in for DirectionClassifier without loading anything."""

    def __init__(self, predictions=None, fail_with=None):
        self.predictions = predictions
        self.fail_with = fail_with
        self.calls = 0
        self.seen = None

    def classify_regions(self, before, after, regions, **kw):
        self.calls += 1
        self.seen = (before, after, list(regions))
        if self.fail_with is not None:
            raise self.fail_with
        if self.predictions is not None:
            return self.predictions
        return [("construction", 0.99)] * len(regions)

    def provenance(self):
        return {"model": "fake", "threshold": 0.96}


# ---------------------------------------------------------- schema 1.2
def test_schema_version_is_1_2():
    from src.core.types import SCHEMA_VERSION
    assert SCHEMA_VERSION == "1.2"


def test_region_direction_fields_default_to_none():
    from src.core.types import Region
    r = Region(id=1, area_px=10, bbox_xywh=[0, 0, 2, 5], centroid_xy=[1.0, 2.5])
    assert r.direction is None
    assert r.direction_score is None


def test_to_dict_omits_direction_fields_when_none():
    from src.core.types import Region
    d = Region(id=1, area_px=10, bbox_xywh=[0, 0, 2, 5], centroid_xy=[1.0, 2.5]).to_dict()
    assert "direction" not in d and "direction_score" not in d
    assert set(d) == {"id", "area_px", "bbox_xywh", "centroid_xy"}


def test_to_dict_serialises_populated_direction_fields():
    from src.core.types import Region
    r = Region(id=3, area_px=99, bbox_xywh=[1, 2, 3, 4], centroid_xy=[2.0, 3.0],
               direction="demolition", direction_score=0.9731)
    d = r.to_dict()
    assert d["direction"] == "demolition"
    assert d["direction_score"] == 0.9731


def test_direction_score_is_independent_of_detector_confidence():
    """Two different models, two different fields - never conflated."""
    from src.core.types import Region
    r = Region(id=1, area_px=10, bbox_xywh=[0, 0, 1, 1], centroid_xy=[0.0, 0.0],
               confidence=0.5123, direction="construction", direction_score=0.9812)
    d = r.to_dict()
    assert d["confidence"] == 0.5123, "detector confidence must be untouched"
    assert d["direction_score"] == 0.9812
    assert d["confidence"] != d["direction_score"]


def test_existing_1_1_region_fields_are_unchanged():
    from src.core.types import Region
    fields = list(Region.__dataclass_fields__)
    assert fields[:8] == ["id", "area_px", "bbox_xywh", "centroid_xy",
                          "confidence", "layer", "area_m2", "centroid_crs_xy"]
    assert fields[8:] == ["direction", "direction_score"]


# ------------------------------------------------------------ abstention
def test_abstention_threshold_behaviour():
    from src.domains.built_environment.direction.classifier import label_for
    from src.domains.built_environment.direction import model_card as card
    assert card.DIRECTION_TAU == 0.960
    # at or above tau -> the predicted class
    assert label_for(0, 0.960)[0] == "construction"
    assert label_for(0, 0.9999)[0] == "construction"
    assert label_for(1, 0.97)[0] == "demolition"
    # below tau -> abstention
    assert label_for(0, 0.9599)[0] == "uncertain"
    assert label_for(1, 0.51)[0] == "uncertain"


def test_score_is_reported_even_when_abstaining():
    from src.domains.built_environment.direction.classifier import label_for
    label, score = label_for(1, 0.734)
    assert label == "uncertain"
    assert score == 0.734, "the score must still be visible when abstaining"


def test_uncertain_is_not_a_trained_class():
    from src.domains.built_environment.direction import model_card as card
    assert card.CLASS_NAMES == ("construction", "demolition")
    assert card.ABSTAIN_LABEL == "uncertain"
    assert card.ABSTAIN_LABEL not in card.CLASS_NAMES


# ------------------------------------------------------- model card facts
def test_model_card_reuses_the_shared_crop_protocol():
    from src.domains.built_environment.direction import model_card as card
    assert (card.CROP_SIZE, card.CONTEXT_FRACTION, card.MIN_SIDE) == \
        (CROP_SIZE, CONTEXT_FRACTION, MIN_SIDE)
    assert card.INPUT_CHANNELS == N_CHANNELS
    assert card.INPUT_SIZE == (128, 128)


def test_model_card_states_the_score_is_not_calibrated():
    from src.domains.built_environment.direction import model_card as card
    assert "NOT a calibrated probability" in card.NOT_CALIBRATED
    assert card.provenance()["score_is_calibrated"] is False


def test_model_card_temporal_ordering_is_not_author_documented():
    from src.domains.built_environment.direction import model_card as card
    assert card.TEMPORAL_ORDERING["documented_by_authors"] is False
    assert card.TEMPORAL_ORDERING["before_image_dir"] == "Image2"
    assert card.TEMPORAL_ORDERING["after_image_dir"] == "Image1"


def test_model_card_ordering_matches_the_data_module():
    """Guards against the card and the label pipeline drifting apart."""
    from src.domains.built_environment.data.s2looking import TEMPORAL_ORDERING as data_order
    from src.domains.built_environment.direction import model_card as card
    for key in ("before_image_dir", "after_image_dir",
                "construction_label_dir", "demolition_label_dir",
                "documented_by_authors"):
        assert card.TEMPORAL_ORDERING[key] == data_order[key], key


def test_provenance_carries_everything_needed_to_audit_a_direction_result():
    from src.domains.built_environment.direction import model_card as card
    p = card.provenance(weights_hash="deadbeef", checkpoint="checkpoints/x.pt")
    assert p["model"] and p["architecture"] == "ResNet-18"
    assert p["dataset"] == "S2Looking"
    assert p["weights_hash"] == "deadbeef"
    assert p["checkpoint"] == "checkpoints/x.pt"
    assert p["threshold"] == 0.960 and p["threshold_selected_on"] == "validation"
    assert p["crop_protocol"]["crop_size"] == 128
    assert p["crop_protocol"]["context_fraction"] == 0.25
    assert p["documented_by_authors"] is False
    assert p["score_is_calibrated"] is False
    assert "end to end" in p["evaluation_scope"].lower()
    assert "OUT OF ENVELOPE" in p["operating_envelope"]


# --------------------------------------------------- classifier wrapper
def test_importing_the_classifier_does_not_load_a_model():
    from src.domains.built_environment.direction.classifier import DirectionClassifier
    clf = DirectionClassifier(checkpoint="checkpoints/does_not_exist.pt")
    assert clf.is_loaded is False, "construction must not load weights"


def test_classifier_uses_the_shared_crops_module():
    from src.domains.built_environment.direction import classifier as clf_mod
    from src.domains.built_environment.direction import crops as crops_mod
    assert clf_mod.crops is crops_mod
    assert hasattr(clf_mod.crops, "region_model_input")


def test_missing_checkpoint_raises_direction_unavailable_not_a_traceback():
    from src.domains.built_environment.direction.classifier import (
        DirectionClassifier, DirectionUnavailable)
    clf = DirectionClassifier(checkpoint="checkpoints/definitely_missing.pt")
    try:
        clf.load()
    except DirectionUnavailable as e:
        assert e.user_message and "not present" in e.user_message
        assert "Traceback" not in e.user_message
    else:
        raise AssertionError("a missing checkpoint must raise DirectionUnavailable")


def test_classify_regions_on_empty_list_loads_nothing():
    from src.domains.built_environment.direction.classifier import DirectionClassifier
    clf = DirectionClassifier(checkpoint="checkpoints/definitely_missing.pt")
    assert clf.classify_regions(None, None, []) == []
    assert clf.is_loaded is False


def test_before_after_channel_ordering_through_the_shared_path():
    """BEFORE must land in channels 0:3 and AFTER in 3:6."""
    before = np.full((64, 64, 3), 7, np.uint8)
    after = np.full((64, 64, 3), 240, np.uint8)
    x = region_model_input(before, after, (10, 10, 20, 20), batched=False)
    assert x.shape == (6, 128, 128)
    # channel means must order the same way the raw values do
    assert float(x[0].mean()) < float(x[3].mean())


# ------------------------------------------------------- engine wiring
_ENGINE = []


def _engine_or_none():
    """The real engine, built once, or None when the detector checkpoint is absent."""
    if _ENGINE:
        return _ENGINE[0]
    if not os.path.exists(config.DEFAULT_CHECKPOINT):
        _ENGINE.append(None)
        return None
    from src.core import registry
    _ENGINE.append(registry.get("built_environment"))
    return _ENGINE[0]


def _demo_pair():
    """A real bundled example pair if present, else a synthetic one."""
    try:
        from src.common import examples as catalogue
        entries = catalogue.available(domain="built_environment")
        if entries:
            e = entries[0]
            return (np.array(Image.open(e.before_path).convert("RGB")),
                    np.array(Image.open(e.after_path).convert("RGB")))
    except Exception:
        pass
    rng = np.random.default_rng(0)
    return (rng.integers(0, 255, (128, 128, 3), dtype=np.uint8),
            rng.integers(0, 255, (128, 128, 3), dtype=np.uint8))


def test_default_analyze_does_not_touch_the_direction_model():
    eng = _engine_or_none()
    if eng is None:
        print("      [skip] detector checkpoint absent")
        return
    eng._direction_clf = None
    before, after = _demo_pair()
    out = eng.analyze(before, after)
    assert eng._direction_clf is None, "with_direction=False must not construct it"
    for r in out["change_result"].regions:
        assert r.direction is None and r.direction_score is None
    assert "direction" not in out["change_result"].params


def test_direction_is_strictly_additive_to_the_detector_output():
    eng = _engine_or_none()
    if eng is None:
        print("      [skip] detector checkpoint absent")
        return
    before, after = _demo_pair()
    eng._direction_clf = None
    base = eng.analyze(before, after)
    eng._direction_clf = _FakeClassifier()
    withdir = eng.analyze(before, after, with_direction=True)

    assert np.array_equal(base["mask"], withdir["mask"]), "mask changed"
    assert np.array_equal(base["probability"], withdir["probability"]), "probability changed"
    b, w = base["result"], withdir["result"]
    assert b["summary"] == w["summary"], "legacy summary changed"
    assert b["params"] == w["params"], "legacy params changed"
    assert b["regions"] == w["regions"], "legacy_regions changed"
    assert b["model"] == w["model"]
    # detector-owned region fields untouched
    for r0, r1 in zip(base["change_result"].regions, withdir["change_result"].regions):
        assert (r0.id, r0.area_px, list(r0.bbox_xywh), list(r0.centroid_xy)) == \
               (r1.id, r1.area_px, list(r1.bbox_xywh), list(r1.centroid_xy))
        assert r0.confidence == r1.confidence, "detector confidence changed"
        assert r0.direction is None, "baseline run must carry no direction"
        assert r1.direction is not None, "direction run must populate direction"


def test_legacy_regions_keys_are_exactly_the_frozen_four():
    eng = _engine_or_none()
    if eng is None:
        print("      [skip] detector checkpoint absent")
        return
    before, after = _demo_pair()
    eng._direction_clf = _FakeClassifier()
    out = eng.analyze(before, after, with_direction=True)
    for r in out["result"]["regions"]:
        assert set(r) == {"id", "area_px", "bbox_xywh", "centroid_xy"}


def test_direction_populates_region_fields_and_provenance():
    eng = _engine_or_none()
    if eng is None:
        print("      [skip] detector checkpoint absent")
        return
    before, after = _demo_pair()
    eng._direction_clf = _FakeClassifier()
    out = eng.analyze(before, after, with_direction=True)
    cr = out["change_result"]
    if not cr.regions:
        print("      [skip] no regions detected in the demo pair")
        return
    for r in cr.regions:
        assert r.direction in ("construction", "demolition", "uncertain")
        assert 0.0 <= r.direction_score <= 1.0
        assert "direction" in r.to_dict()
    assert cr.params["direction"]["threshold"] == 0.96
    assert cr.schema_version == "1.2"


def test_no_regions_means_the_direction_model_is_never_constructed():
    eng = _engine_or_none()
    if eng is None:
        print("      [skip] detector checkpoint absent")
        return
    flat = np.full((128, 128, 3), 128, np.uint8)
    eng._direction_clf = None
    out = eng.analyze(flat, flat.copy(), with_direction=True)
    if out["change_result"].regions:
        print("      [skip] identical pair still produced regions")
        return
    assert eng._direction_clf is None
    assert not any("Direction" in w for w in out["change_result"].warnings)


def test_direction_failure_warns_and_preserves_the_detector_result():
    eng = _engine_or_none()
    if eng is None:
        print("      [skip] detector checkpoint absent")
        return
    from src.domains.built_environment.direction.classifier import DirectionUnavailable
    before, after = _demo_pair()
    eng._direction_clf = None
    base = eng.analyze(before, after)
    if not base["change_result"].regions:
        print("      [skip] no regions detected in the demo pair")
        return
    eng._direction_clf = _FakeClassifier(
        fail_with=DirectionUnavailable("the direction model checkpoint is not present (x.pt)."))
    out = eng.analyze(before, after, with_direction=True)
    cr = out["change_result"]
    warned = [w for w in cr.warnings if "Direction analysis unavailable" in w]
    assert len(warned) == 1, f"expected exactly one direction warning, got {cr.warnings}"
    assert "not present" in warned[0]
    assert "Traceback" not in warned[0]
    for r in cr.regions:
        assert r.direction is None and r.direction_score is None
    assert "direction" not in cr.params
    assert np.array_equal(base["mask"], out["mask"])
    assert base["result"]["summary"] == out["result"]["summary"]


def test_unexpected_failure_is_reported_without_a_traceback():
    eng = _engine_or_none()
    if eng is None:
        print("      [skip] detector checkpoint absent")
        return
    before, after = _demo_pair()
    eng._direction_clf = None
    if not eng.analyze(before, after)["change_result"].regions:
        print("      [skip] no regions detected in the demo pair")
        return
    eng._direction_clf = _FakeClassifier(fail_with=ZeroDivisionError("internal"))
    cr = eng.analyze(before, after, with_direction=True)["change_result"]
    warned = [w for w in cr.warnings if "Direction analysis unavailable" in w]
    assert len(warned) == 1
    assert "internal" not in warned[0], "internal detail must not leak"


# ======================== Step 3: product / UI surface ====================
# Pure display helpers only - no Streamlit runtime, no GPU, no checkpoint.

def _region(id_=1, direction=None, direction_score=None, confidence=None):
    from src.core.types import Region
    return Region(id=id_, area_px=100, bbox_xywh=[0, 0, 10, 10],
                  centroid_xy=[5.0, 5.0], confidence=confidence,
                  direction=direction, direction_score=direction_score)


def test_product_analysis_explicitly_requests_direction():
    """The product must opt into direction - for the domain that has it.

    Revised in UI Phase 2. Before the product had a second domain, this asserted
    that ui/analysis.py passed a literal `with_direction=True`, which was the
    only way to prove opt-in for the single domain that existed. The product now
    dispatches over two domains and the environment engine's analyze() has no
    with_direction parameter at all, so a literal argument would be wrong rather
    than safe.

    The behavioural guarantee is unchanged and is now checked on BOTH sides:
    built_environment opts in with True, and environment does not request
    direction at all. The AST check below still proves ui/analysis.py actually
    applies that decision to the call, which a string grep would not.
    """
    import ast

    from ui.analysis import analyze_options

    class Engine:
        def __init__(self, supports):
            self.supports_direction = supports

    # built_environment: opts in, with True.
    assert analyze_options(Engine(True)) == {"with_direction": True}
    # environment: does not request direction at all - not even False, because
    # its analyze() does not accept the parameter.
    assert analyze_options(Engine(False)) == {}
    assert "with_direction" not in analyze_options(Engine(False))

    # The real engines declare the capability the decision is made on.
    from src.domains.built_environment.engine import ChangeEngine as BuiltEnv
    from src.domains.environment.engine import ChangeEngine as Environment
    assert BuiltEnv.supports_direction is True
    assert Environment.supports_direction is False

    # ...and the screen routes its analyze() call through that decision.
    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "ui", "analysis.py"), encoding="utf-8").read()
    calls = [n for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "analyze"]
    assert calls, "no engine.analyze(...) call found in ui/analysis.py"
    for call in calls:
        starred = [k.value for k in call.keywords if k.arg is None]
        assert any(isinstance(v, ast.Call)
                   and getattr(v.func, "id", None) == "analyze_options"
                   for v in starred), \
            "the product must apply analyze_options() to its analyze() call"


def test_engine_default_is_still_with_direction_false():
    import inspect
    from src.domains.built_environment.engine import ChangeEngine
    sig = inspect.signature(ChangeEngine.analyze)
    assert sig.parameters["with_direction"].default is False


def test_direction_labels_render_readably():
    from ui.results import direction_label
    assert direction_label("construction") == "Construction"
    assert direction_label("demolition") == "Demolition"
    assert direction_label("uncertain") == "Uncertain"


def test_missing_direction_renders_as_dash_not_a_class():
    from ui.results import NO_DIRECTION, direction_label
    assert direction_label(None) == NO_DIRECTION
    assert direction_label(None) not in ("Construction", "Demolition", "Uncertain")


def test_unknown_direction_value_does_not_invent_a_class():
    from ui.results import NO_DIRECTION, direction_label
    assert direction_label("redevelopment") == NO_DIRECTION
    assert direction_label("mixed") == NO_DIRECTION


def test_direction_counts_return_none_when_nothing_was_classified():
    """No misleading zero counts when direction did not run."""
    from ui.results import direction_counts
    assert direction_counts([_region(1), _region(2)]) is None
    assert direction_counts([]) is None


def test_direction_counts_keep_uncertain_separate():
    from ui.results import direction_counts
    regions = ([_region(i, "construction", 0.99) for i in range(8)]
               + [_region(i, "demolition", 0.98) for i in range(3)]
               + [_region(i, "uncertain", 0.61) for i in range(2)])
    counts = direction_counts(regions)
    assert counts == {"construction": 8, "demolition": 3, "uncertain": 2}
    # uncertain is never folded into either direction
    assert counts["construction"] + counts["demolition"] == 11


def test_region_table_separates_direction_score_from_detector_confidence():
    from ui.results import _region_table
    df = _region_table([_region(1, "construction", 0.9812, confidence=0.5123)])
    assert "Direction score" in df.columns
    assert "Detector confidence" in df.columns
    assert df["Direction score"].iloc[0] == 0.9812
    assert df["Detector confidence"].iloc[0] == 0.5123
    # never merged, never renamed into each other
    assert "Confidence" not in df.columns
    for banned in ("Probability", "Calibrated probability", "Accuracy"):
        assert banned not in df.columns


def test_region_table_omits_direction_columns_when_absent():
    from ui.results import _region_table
    df = _region_table([_region(1, confidence=0.4)])
    assert "Direction" not in df.columns
    assert "Direction score" not in df.columns
    assert "Area (px)" in df.columns and "Region" in df.columns


def test_region_table_renders_partial_direction_without_false_classification():
    from ui.results import NO_DIRECTION, _region_table
    df = _region_table([_region(1, "demolition", 0.97), _region(2)])
    assert list(df["Direction"]) == ["Demolition", NO_DIRECTION]
    assert df["Direction score"].iloc[0] == 0.97
    assert pd_isna(df["Direction score"].iloc[1])


def pd_isna(v):
    import pandas as pd
    return pd.isna(v)


def test_detector_columns_survive_direction_integration():
    from ui.results import _region_table
    df = _region_table([_region(1, "construction", 0.99, confidence=0.77)])
    for col in ("Region", "Area (px)", "X", "Y", "Width", "Height",
                "Centroid X", "Centroid Y"):
        assert col in df.columns, col


def test_direction_caveat_states_scope_and_non_calibration():
    from ui.results import DIRECTION_CAVEAT, DIRECTION_DOMAIN_GAP
    low = DIRECTION_CAVEAT.lower()
    assert "separate model" in low
    assert "s2looking" in low
    assert "not calibrated" in low or "not</strong> calibrated" in low
    assert "end-to-end" in low or "end to end" in low
    assert "uncertain" in low
    gap = DIRECTION_DOMAIN_GAP.lower()
    assert "s2looking" in gap and "levir" in gap
    assert "not been established" in gap


def test_direction_unavailable_warning_is_detected():
    from ui.results import has_direction_warning
    from src.domains.built_environment.engine import _direction_warning
    produced = _direction_warning(Exception("boom"))
    assert has_direction_warning([produced]), "UI must recognise the engine's warning"
    assert not has_direction_warning([])
    assert not has_direction_warning(["opencv not installed - region counting disabled"])


def test_ui_reuses_direction_model_card_wording():
    """Caveats trace back to the model card rather than being re-invented."""
    from ui import results as ui_results
    from src.domains.built_environment.direction import model_card as card
    assert ui_results.direction_card is card
    assert "NOT a calibrated probability" in card.NOT_CALIBRATED
    assert card.TEMPORAL_ORDERING["documented_by_authors"] is False


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
