"""UI Phase 4: the curated environmental demo bundle.

These check the BUNDLE and the catalogue, not the model. They assert that the
tracked demo artifacts are what they claim to be - six bands, the frozen order,
reflectance rather than normalised values - and that the catalogue carries
enough for an application to read them without sniffing or guessing.

The bundle is tracked, so these run on any checkout. Tests that compare against
the frozen v24 archive skip when that git-ignored archive is absent.
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config                                           # noqa: E402
from src.common import examples as cat                           # noqa: E402
from src.domains.environment.data import sentinel2 as s2         # noqa: E402
from src.domains.environment import normalization                # noqa: E402

DATASET_DIR = os.path.join(config.DATA_DIR, "environment", "dataset_v24")
HAS_ARCHIVE = os.path.exists(os.path.join(DATASET_DIR, "manifest.json"))

EXPECTED_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
EXPECTED_IDS = {"env_amazon_large_clearing", "env_seasia_small_clearing",
                "env_amazon_intact_forest"}


@pytest.fixture(scope="module")
def catalogue():
    with open(config.EXAMPLES_CATALOGUE, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def environment_examples():
    return [e for e in cat.load_catalogue() if e.domain == "environment"]


# ------------------------------------------------------------ 1. schema
def test_catalogue_remains_valid_json_and_loads(catalogue):
    assert catalogue["schema_version"] == "1.1"
    assert isinstance(catalogue["examples"], list) and catalogue["examples"]
    # Every entry must construct an Example; an unknown field would raise.
    loaded = cat.load_catalogue()
    assert len(loaded) == len(catalogue["examples"])


def test_catalogue_schema_accepts_environmental_examples(environment_examples):
    assert len(environment_examples) == 3
    assert {e.id for e in environment_examples} == EXPECTED_IDS
    for ex in environment_examples:
        assert ex.domain == "environment"
        assert ex.kind == cat.KIND_SENTINEL2_SIX_BAND
        assert ex.ground_truth_kind == cat.GT_NPY_MASK
        assert ex.bands == EXPECTED_BANDS or list(ex.bands) == EXPECTED_BANDS
        assert ex.gsd_m == 10.0
        assert ex.crs and ex.crs.startswith("EPSG:")
        assert ex.source_sample_id
        assert ex.name and ex.description and ex.source


def test_examples_round_trip_through_to_dict(environment_examples):
    """services.list_examples() -> example_from_dict() must lose nothing."""
    for ex in environment_examples:
        assert cat.Example(**ex.to_dict()) == ex


# ------------------------------------------------------- 2. domain filtering
def test_environmental_examples_are_filtered_by_domain():
    env = cat.available(domain="environment")
    built = cat.available(domain="built_environment")
    assert {e.id for e in env} == EXPECTED_IDS
    assert env and built
    assert not ({e.id for e in env} & {e.id for e in built})
    for e in built:
        assert e.domain == "built_environment"


def test_built_environment_examples_continue_to_work():
    """The bundled RGB pairs must be untouched by the schema change."""
    built = cat.available(domain="built_environment")
    assert len(built) >= 3
    for ex in built:
        assert ex.kind == cat.KIND_RGB_PAIR          # the 1.0 default
        assert ex.bands is None and ex.gsd_m is None and ex.crs is None
        assert ex.georef is None                      # no invented scale
        assert os.path.exists(ex.before_path) and os.path.exists(ex.after_path)


# -------------------------------------------------------------- 3. artifacts
def test_example_paths_exist(environment_examples):
    for ex in environment_examples:
        assert ex.exists()
        assert os.path.exists(ex.before_path), ex.before
        assert os.path.exists(ex.after_path), ex.after
        assert ex.has_ground_truth
        assert os.path.exists(ex.ground_truth_path), ex.ground_truth


def test_no_environment_example_points_into_the_frozen_archive(environment_examples):
    """The demo must be self-contained: data/ is git-ignored and 500 MB."""
    for ex in environment_examples:
        for path in (ex.before, ex.after, ex.ground_truth):
            normalised = path.replace("\\", "/")
            assert not normalised.startswith("data/"), path
            assert "dataset_v24" not in normalised, path
            assert normalised.startswith("examples/environment/"), path


def test_band_count_and_order_are_correct(environment_examples):
    for ex in environment_examples:
        assert list(ex.bands) == list(s2.BANDS)
        for path in (ex.before_path, ex.after_path):
            array = np.load(path)
            assert array.ndim == 3
            assert array.shape[-1] == 6, f"{path} has {array.shape[-1]} bands"
            assert array.shape == (256, 256, 6)
            assert array.dtype == np.float32


def test_demo_arrays_are_not_rgb(environment_examples):
    """An RGB array here would be silently wrong, so assert it cannot be."""
    for ex in environment_examples:
        for path in (ex.before_path, ex.after_path):
            array = np.load(path)
            assert array.shape[-1] != 3
            assert array.dtype != np.uint8
            # Distinct bands: a duplicated or dropped band would collapse these.
            assert len({array[:, :, i].tobytes() for i in range(6)}) == 6


def test_arrays_are_reflectance_not_normalised(environment_examples):
    """Normalised input would be centred on zero with negative values."""
    mean = np.asarray(normalization.MEAN, dtype=np.float32)
    for ex in environment_examples:
        array = np.load(ex.before_path)
        assert float(array.min()) >= 0.0, "negative values suggest normalisation"
        assert float(array.max()) < 2.0, "values too large to be reflectance"
        # Close to the TRAIN means, so no second conversion was applied.
        assert np.all(np.abs(array.reshape(-1, 6).mean(0) - mean) < 0.35)


def test_ground_truth_loads_as_a_binary_mask(environment_examples):
    from ui.analysis import _load_ground_truth

    for ex in environment_examples:
        mask = _load_ground_truth((ex.ground_truth_path, ex.ground_truth_kind))
        assert mask is not None
        assert mask.dtype == np.bool_
        assert mask.shape == (256, 256)
        raw = np.load(ex.ground_truth_path)
        assert raw.ndim == 2 and set(np.unique(raw)) <= {0, 1}
        # The one negative example genuinely has no labelled loss.
        if ex.id == "env_amazon_intact_forest":
            assert mask.sum() == 0
        else:
            assert mask.sum() > 0


def test_declared_ground_truth_kind_is_what_drives_loading():
    """The declared kind is authoritative, not the file extension."""
    from ui.analysis import _load_ground_truth

    assert _load_ground_truth(None) is None
    assert _load_ground_truth((None, cat.GT_NPY_MASK)) is None
    # A png_mask claim against a .npy file must fail closed, not silently work.
    ex = next(e for e in cat.available(domain="environment"))
    assert _load_ground_truth((ex.ground_truth_path, cat.GT_PNG_MASK)) is None
    assert _load_ground_truth((ex.ground_truth_path, cat.GT_NPY_MASK)) is not None


# --------------------------------------------------- 4. against the archive
@pytest.mark.skipif(not HAS_ARCHIVE, reason="frozen v24 archive not present")
def test_artifacts_are_bit_identical_to_the_frozen_source(environment_examples):
    with open(os.path.join(DATASET_DIR, "manifest.json"), encoding="utf-8") as fh:
        manifest = {r["sample_id"]: r for r in json.load(fh)["samples"]}

    for ex in environment_examples:
        record = manifest[ex.source_sample_id]
        assert record["split"] == "test", "examples must come from TEST only"
        for side, path in (("before", ex.before_path), ("after", ex.after_path)):
            source = s2.to_reflectance(
                np.load(os.path.join(DATASET_DIR,
                                     f"{ex.source_sample_id}_{side}.npy")),
                record[f"{side}_offset_applied"],
                record[f"{side}_baseline"]).astype(np.float32)
            assert np.array_equal(np.load(path), source), f"{ex.id} {side} drifted"
        label = np.load(os.path.join(DATASET_DIR,
                                     f"{ex.source_sample_id}_label.npy"))
        assert np.array_equal(np.load(ex.ground_truth_path), label)
        assert ex.gsd_m == record["gsd_m"] and ex.crs == record["crs"]


# ------------------------------------------------------ 5. application path
def test_examples_prepare_through_the_domain_adapter(environment_examples):
    """The bundle must satisfy the six-band contract with no UI conversion."""
    from src.common import pair_input

    for ex in environment_examples:
        pair = pair_input.prepare("environment", ex.before_path, ex.after_path)
        assert pair.ok, pair.errors
        assert pair.before.shape == (256, 256, 6)
        assert pair.preview_before.shape == (256, 256, 3)
        assert pair.preview_before.dtype == np.uint8


def test_example_georef_is_declared_not_invented(environment_examples):
    for ex in environment_examples:
        georef = ex.georef
        assert georef is not None and georef.has_scale
        assert georef.gsd_m == 10.0
        assert georef.units == "metre"
        assert georef.crs == ex.crs
        # No geotransform is claimed, so CRS centroids stay unavailable.
        assert georef.transform is None


def test_catalogue_exposes_no_training_internals(catalogue):
    """Descriptions are for users; hashes and paths are not."""
    blob = json.dumps(catalogue).lower()
    for leak in ("sha256", "326e8a27", "23721c09", "727ad32f", "checkpoint",
                 "threshold", "pos_weight", "epoch", "outputs/"):
        assert leak not in blob, f"catalogue leaks {leak!r}"
