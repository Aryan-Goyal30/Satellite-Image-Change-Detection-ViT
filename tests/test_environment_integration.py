"""Environmental Domain Integration, Step 1: architecture integration tests.

These test the INTEGRATION, not the model. No metric is re-measured here and no
training runs: the frozen Phase 4B E2 checkpoint, the frozen v24 dataset hashes
and the frozen operating threshold are treated as given, and what is checked is
that the engine serves them faithfully through the Earth Guardian contract.

Tests that need the E2 checkpoint or opencv skip cleanly when those are absent,
so a fresh checkout still runs the suite green. The contract tests - band
refusal, normalisation, registry laziness - need neither and always run.
"""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config                                             # noqa: E402
from src.core import registry                                      # noqa: E402
from src.core.types import ChangeResult, GeoRef                    # noqa: E402
from src.domains.environment import inputs, model_card             # noqa: E402
from src.domains.environment import normalization                  # noqa: E402
from src.domains.environment.inputs import BandContractError       # noqa: E402

# Frozen identities, from the Phase 4B manifests. Restated here so a silent
# substitution of the checkpoint or the dataset fails a test rather than
# quietly changing what the engine serves.
E2_CHECKPOINT_SHA256 = "326e8a272c6c4536c3a07ccaabc3feb8ad6905ac0db7e7e67b88ceb98096cc45"
E2_PARAMETERS = 25152209
E2_THRESHOLD = 0.91
E2_NORMALISATION_SUBSET_SHA256 = \
    "727ad32f98b05db5b654f8bb8b0f67031cd385bd320bb3e23dfd0a59523d8279"
V24_MANIFEST_SHA256 = "23721c09c864297a7857a3576e4398f700e642c3ab1e4531462eb603a2f827d8"

#: Built-environment regression: probability map over a fixed random pair,
#: captured before the sliding-window tiler was extracted into src/common.
BUILT_ENV_PROBABILITY_SHA256 = \
    "5a6c8e49e42d56dfb663bd5b39689267e52c19ba442536da2aed3b753a043122"

HAS_E2 = os.path.exists(config.ENVIRONMENT_CHECKPOINT)
HAS_BUILT_ENV = os.path.exists(config.DEFAULT_CHECKPOINT)
needs_e2 = pytest.mark.skipif(not HAS_E2, reason="frozen E2 checkpoint not present")


def _has_cv2():
    try:
        import cv2  # noqa: F401
        return True
    except ImportError:
        return False


def _reflectance(seed, size=256, bands=6, scale=0.3):
    """A plausible reflectance array. Values only, no claim to be real imagery."""
    rng = np.random.default_rng(seed)
    return (rng.random((size, size, bands)) * scale).astype(np.float32)


_ENGINE = []


def environment_engine():
    """The engine, built once: loading the checkpoint is the slow part."""
    if not _ENGINE:
        _ENGINE.append(registry.get("environment"))
    return _ENGINE[0]


# --------------------------------------------------------------- 1. registry
def test_environment_is_registered_and_the_registry_stays_torch_free():
    """The domain is reachable by name, and asking does not import torch."""
    assert "environment" in registry.available()
    assert "built_environment" in registry.available()

    # A subprocess, because torch may already be imported by another test.
    probe = (
        "import sys; "
        "sys.path.insert(0, %r); "
        "from src.core import registry; "
        "names = registry.available(); "
        "print(('environment' in names, 'torch' in sys.modules))"
        % config.ROOT)
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                         text=True, cwd=config.ROOT)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "(True, False)", out.stdout


# ------------------------------------------------------- 2. checkpoint identity
@needs_e2
def test_engine_serves_the_frozen_e2_checkpoint():
    """Identity, band set and parameter count match the frozen experiment."""
    engine = environment_engine()
    assert engine.domain == "environment"
    assert engine.model_name == "environment-siamese-unet-6band"
    assert engine.bands == ("B02", "B03", "B04", "B08", "B11", "B12")
    assert engine.model.n_params == E2_PARAMETERS
    assert engine.weights_hash == E2_CHECKPOINT_SHA256
    assert engine.metadata.provenance.weights_hash == E2_CHECKPOINT_SHA256


# ---------------------------------------------------------- 3. threshold source
@needs_e2
def test_threshold_is_read_from_the_checkpoint_not_hard_coded():
    """The operating point comes from the weights file, so it cannot drift."""
    from src.common.model_loader import read_checkpoint
    ck = read_checkpoint(config.ENVIRONMENT_CHECKPOINT, "cpu")
    assert float(ck["threshold_hint"]) == E2_THRESHOLD

    engine = environment_engine()
    assert engine.threshold == float(ck["threshold_hint"])
    assert engine.metadata.provenance.threshold == E2_THRESHOLD
    # The card's copy is documentation; it must agree but is not the source.
    assert model_card.VALIDATED_THRESHOLD == E2_THRESHOLD


# --------------------------------------------------------- 4. RGB is refused
@pytest.mark.parametrize("array,label", [
    (np.zeros((16, 16, 3), np.float32), "RGB"),
    (np.zeros((16, 16, 4), np.float32), "RGB+NIR"),
    (np.zeros((16, 16), np.float32), "single band"),
    (np.zeros((16, 16, 12), np.float32), "twelve bands"),
])
def test_wrong_band_count_is_refused_with_the_published_message(array, label):
    """No padding, no duplication, no synthesis - and one exact sentence."""
    with pytest.raises(BandContractError) as excinfo:
        inputs.as_six_band(array)
    assert excinfo.value.user_message == (
        "Environmental forest-loss analysis requires six-band Sentinel-2 "
        "imagery: B02, B03, B04, B08, B11 and B12."), label
    assert excinfo.value.user_message == inputs.REQUIRED_BANDS_MESSAGE


@needs_e2
def test_engine_analyze_refuses_rgb_before_running_the_model():
    """The refusal is enforced at the engine boundary, not only in the helper."""
    engine = environment_engine()
    rgb = _reflectance(1, size=64, bands=3)
    with pytest.raises(BandContractError) as excinfo:
        engine.analyze(rgb, rgb)
    assert excinfo.value.user_message == inputs.REQUIRED_BANDS_MESSAGE


# --------------------------------------------------- 5. missing bands are named
def test_a_mapping_missing_swir_names_the_missing_bands():
    """The user gets the contract sentence; the log gets which bands were absent."""
    partial = {b: np.zeros((8, 8), np.float32)
               for b in ("B02", "B03", "B04", "B08")}
    with pytest.raises(BandContractError) as excinfo:
        inputs.as_six_band(partial, "before")
    assert excinfo.value.user_message == inputs.REQUIRED_BANDS_MESSAGE
    detail = str(excinfo.value)
    assert "B11" in detail and "B12" in detail and "before" in detail

    complete = dict(partial, B11=np.zeros((8, 8), np.float32),
                    B12=np.zeros((8, 8), np.float32))
    assert inputs.as_six_band(complete).shape == (8, 8, 6)


# ------------------------------------------------- 6. digital numbers refused
def test_digital_numbers_are_refused_rather_than_rescaled():
    """Getting the BOA offset wrong shifts every band, so it is never guessed."""
    for dn in (np.full((8, 8, 6), 1200, np.uint16),
               np.full((8, 8, 6), 1200.0, np.float32)):
        with pytest.raises(BandContractError) as excinfo:
            inputs.as_six_band(dn)
        assert "digital numbers" in excinfo.value.user_message
        assert "reflectance_from_dn" in str(excinfo.value)

    # The documented conversion is the frozen acquisition rule, not a new one.
    from src.domains.environment.data import sentinel2 as s2
    dn = np.full((4, 4, 6), 1200, np.uint16)
    assert np.array_equal(inputs.reflectance_from_dn(dn, False, "02.14"),
                          s2.to_reflectance(dn, False, "02.14"))
    assert inputs.as_six_band(
        inputs.reflectance_from_dn(dn, False, "02.14")).shape == (4, 4, 6)


# ------------------------------------------------------- 7. normalisation
def test_normalisation_is_the_frozen_train_only_statistics():
    """In-code constants reproduce the file the experiment trained against."""
    stats = normalization.statistics()
    assert stats["bands"] == list(inputs.REQUIRED_BANDS)
    assert stats["subset_sha256"] == E2_NORMALISATION_SUBSET_SHA256
    assert stats["sha256"] == normalization.SOURCE_FILE_SHA256
    assert "TRAIN" in stats["source"]

    # ImageNet statistics describe 8-bit sRGB photographs and must never be
    # substituted for surface reflectance.
    from src.common.preprocessing import IMAGENET_MEAN
    assert not np.allclose(stats["mean"][:3], IMAGENET_MEAN)

    report = normalization.verify_against_file()
    if report is None:
        pytest.skip("characterization file absent (data/ is git-ignored)")
    assert report["agrees"], report["mismatches"]
    assert report["file_sha256"] == normalization.SOURCE_FILE_SHA256


# ------------------------------------------------------ 8. the v1 contract
@needs_e2
def test_analyze_returns_a_change_result_with_environment_provenance():
    """One real forward pass; the shape of the published contract is checked."""
    engine = environment_engine()
    before, after = _reflectance(11), _reflectance(12)
    out = engine.analyze(before, after)

    cr = out["change_result"]
    assert isinstance(cr, ChangeResult)
    assert out["probability"].shape == (256, 256)
    assert out["mask"].shape == (256, 256) and out["mask"].dtype == np.bool_
    assert float(out["probability"].min()) >= 0.0
    assert float(out["probability"].max()) <= 1.0

    layer = cr.primary_layer
    assert layer.name == "forest_loss"
    assert layer.threshold == E2_THRESHOLD
    # The layer mask is the thresholded map with sub-min_area components
    # removed, so it is a subset of it and agrees with the regions reported.
    thresholded = out["probability"] >= E2_THRESHOLD
    assert not np.any(layer.mask & ~thresholded)
    assert int(layer.mask.sum()) == sum(r.area_px for r in cr.regions)
    assert cr.quantities.changed_pixels == int(layer.mask.sum())

    prov = cr.provenance.to_dict()
    assert prov["model"] == "environment-siamese-unet-6band"
    assert prov["dataset"] == "TMF + Sentinel-2"
    assert prov["dataset_version"] == "v24"
    assert prov["threshold"] == E2_THRESHOLD
    assert prov["weights_hash"] == E2_CHECKPOINT_SHA256
    assert prov["score_calibrated"] is False
    assert prov["operating_envelope"]["gsd_m_range"] == [10.0, 10.0]

    assert cr.params["normalisation_subset_sha256"] == E2_NORMALISATION_SUBSET_SHA256
    assert cr.params["dataset_manifest_sha256"] == V24_MANIFEST_SHA256
    assert cr.input_info["bands"] == list(inputs.REQUIRED_BANDS)
    # The whole result must survive a JSON round trip: it is a published payload.
    json.dumps(cr.to_dict())


# ---------------------------------------------- 9. geography is never invented
@needs_e2
def test_area_appears_only_when_the_caller_supplies_a_scale():
    """Sentinel-2 is a 10 m grid, but the engine never assumes that of an array."""
    engine = environment_engine()
    before, after = _reflectance(21), _reflectance(22)

    plain = engine.analyze(before, after)["change_result"]
    assert plain.quantities.area_m2 is None
    assert plain.georef is None
    assert plain.input_info["gsd_m"] is None

    scaled = engine.analyze(before, after,
                            gsd_m=inputs.SENTINEL2_GSD_M)["change_result"]
    assert scaled.georef.gsd_m == 10.0
    assert scaled.input_info["gsd_m"] == 10.0
    expected = round(scaled.quantities.changed_pixels * 100.0, 1)
    assert scaled.quantities.area_m2 == expected
    # Adding a scale must not change the detection itself.
    assert plain.quantities.changed_pixels == scaled.quantities.changed_pixels

    if not _has_cv2():
        pytest.skip("opencv not installed - region geography not exercised")
    assert all(r.area_m2 is not None for r in scaled.regions)
    assert all(r.area_m2 is None for r in plain.regions)
    if scaled.regions:
        assert any("region F1" in w for w in scaled.warnings)


# -------------------------------------------- 10. built environment unchanged
@pytest.mark.skipif(not HAS_BUILT_ENV, reason="built-environment checkpoint absent")
def test_built_environment_is_bit_identical_after_the_shared_tiler():
    """Extracting the sliding window must not have moved a single value.

    Pinned to CPU deliberately: BUILT_ENV_PROBABILITY_SHA256 was captured from
    a CPU forward pass, and GPU (cuDNN) convolution kernels are not bit
    reproducible with CPU regardless of TF32 / autocast / determinism flags -
    that is a hardware-backend property, not something this test's arithmetic
    controls. Auto-selecting CUDA here would make the assertion pass or fail
    based on which device happens to be available, not on whether the
    shared tiler actually reproduces the pre-extraction output.
    """
    engine = registry.get("built_environment", device="cpu")
    rng = np.random.default_rng(7)
    a = rng.integers(0, 256, (300, 260, 3), dtype=np.uint8)
    b = rng.integers(0, 256, (300, 260, 3), dtype=np.uint8)

    prob = engine.probability_map(a, b)
    digest = hashlib.sha256(
        np.ascontiguousarray(prob, dtype=np.float32).tobytes()).hexdigest()
    assert digest == BUILT_ENV_PROBABILITY_SHA256

    # Its provenance payload must be unchanged: the two new Provenance fields
    # are omitted for an engine that does not declare them.
    prov = engine.analyze(a, b)["change_result"].provenance.to_dict()
    assert "dataset_version" not in prov
    assert "score_calibrated" not in prov
    assert prov["dataset"] == "LEVIR-CD"
    assert prov["weights_hash"] == \
        "3596be6b50f10dc063c4b0e74df60a0daf050618824da4d3507c36b2363221c9"


def test_shared_tiler_reassembles_an_exact_signal():
    """A predict() that returns a known field must come back unchanged."""
    from src.common.tiling import sliding_window_probability

    field = np.linspace(0, 1, 300 * 260, dtype=np.float32).reshape(300, 260)
    padded = np.pad(field[..., None], ((0, 212), (0, 252), (0, 0)), mode="symmetric")

    def predict(tiles):
        return tiles[0][..., 0]

    out = sliding_window_probability([padded[:300, :260]], predict,
                                     tile=256, overlap=64)
    assert out.shape == (300, 260)
    assert np.allclose(out, field, atol=1e-5)
