"""Step 2: application/inference dispatch across two domains.

These test the APPLICATION BOUNDARY - that the high-level path reads each
domain's input the way that domain requires, hands the engine what it accepts,
and returns the same ChangeResult contract either way. No model behaviour is
re-measured here.

Tests needing a checkpoint skip cleanly when it is absent. The dispatch and
refusal tests need no model at all and always run.
"""
import hashlib
import os
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config                                            # noqa: E402
from src.common import pair_input                                 # noqa: E402
from src.core import registry                                     # noqa: E402
from src.core.types import ChangeResult                           # noqa: E402
from src.domains.environment import inputs as env_inputs          # noqa: E402

BUILT_ENV_PROBABILITY_SHA256 = \
    "5a6c8e49e42d56dfb663bd5b39689267e52c19ba442536da2aed3b753a043122"

HAS_E2 = os.path.exists(config.ENVIRONMENT_CHECKPOINT)
HAS_BUILT_ENV = os.path.exists(config.DEFAULT_CHECKPOINT)
needs_e2 = pytest.mark.skipif(not HAS_E2, reason="frozen E2 checkpoint not present")
needs_built_env = pytest.mark.skipif(not HAS_BUILT_ENV,
                                     reason="built-environment checkpoint absent")


@pytest.fixture(scope="module")
def rgb_pair(tmp_path_factory):
    """A small RGB pair on disk, with real variation so it is not 'blank'."""
    directory = tmp_path_factory.mktemp("rgb")
    rng = np.random.default_rng(5)
    paths = []
    for name in ("before.png", "after.png"):
        path = directory / name
        Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)).save(path)
        paths.append(str(path))
    return tuple(paths)


@pytest.fixture(scope="module")
def six_band_pair(tmp_path_factory):
    """A small six-band reflectance pair on disk, as .npy."""
    directory = tmp_path_factory.mktemp("s2")
    rng = np.random.default_rng(6)
    paths = []
    for name in ("before.npy", "after.npy"):
        path = directory / name
        np.save(path, (rng.random((256, 256, 6)) * 0.3).astype(np.float32))
        paths.append(str(path))
    return tuple(paths)


# ------------------------------------------------------ 1. registry integration
def test_registry_and_adapters_cover_the_same_domains():
    """Every engine the app can request also has a way to read its input."""
    assert set(registry.available()) == {"built_environment", "environment"}
    assert set(pair_input.domains()) == {"built_environment", "environment"}

    from ui import services
    assert set(services.available_domains()) == {"built_environment", "environment"}
    assert services.DEFAULT_DOMAIN == "built_environment"


def test_an_unknown_domain_is_refused_rather_than_assumed_to_be_rgb():
    """There is no default adapter: assuming RGB is what this layer prevents."""
    with pytest.raises(KeyError) as excinfo:
        pair_input.get_adapter("disaster")
    assert "no input adapter" in str(excinfo.value)


# ------------------------------------------------------ 2. built_env dispatch
def test_built_environment_dispatch_reads_rgb(rgb_pair):
    """The RGB path returns model input and a preview, plus the georef verdict."""
    pair = pair_input.prepare("built_environment", *rgb_pair)
    assert pair.ok
    assert pair.domain == "built_environment"
    assert pair.before.shape == (256, 256, 3)
    assert pair.before.dtype == np.uint8
    # For an RGB domain the model input and the display image are one array.
    assert pair.preview_before is pair.before
    assert pair.georef is None            # PNGs carry no georeferencing
    assert "pixels" in pair.notes


@needs_built_env
def test_built_environment_reaches_a_change_result_through_the_app_path(rgb_pair):
    engine = registry.get("built_environment")
    pair = pair_input.prepare("built_environment", *rgb_pair,
                              engine.metadata.input_spec)
    assert pair.ok
    out = engine.analyze(pair.before, pair.after, georef=pair.georef)
    cr = out["change_result"]
    assert isinstance(cr, ChangeResult)
    assert cr.primary_layer.name == "structural_change"
    assert cr.provenance.dataset == "LEVIR-CD"
    assert "result" in out              # legacy shape still produced


# -------------------------------------------------------- 3. environment dispatch
def test_environment_dispatch_reads_six_bands(six_band_pair):
    """Six-band input is accepted and rendered for display without touching it."""
    pair = pair_input.prepare("environment", *six_band_pair)
    assert pair.ok
    assert pair.domain == "environment"
    assert pair.before.shape == (256, 256, 6)
    assert pair.before.dtype == np.float32
    # The preview is a separate true-colour composite, never the model input.
    assert pair.preview_before.shape == (256, 256, 3)
    assert pair.preview_before.dtype == np.uint8
    assert pair.preview_before is not pair.before
    assert pair.georef is None
    assert "Six-band" in pair.notes


@needs_e2
def test_environment_reaches_a_change_result_through_the_app_path(six_band_pair):
    engine = registry.get("environment")
    pair = pair_input.prepare("environment", *six_band_pair,
                              engine.metadata.input_spec)
    assert pair.ok
    out = engine.analyze(pair.before, pair.after, georef=pair.georef)
    cr = out["change_result"]
    assert isinstance(cr, ChangeResult)
    assert cr.primary_layer.name == "forest_loss"

    # Provenance must survive the dispatch unchanged.
    prov = cr.provenance.to_dict()
    assert prov["model"] == "environment-siamese-unet-6band"
    assert prov["dataset"] == "TMF + Sentinel-2"
    assert prov["dataset_version"] == "v24"
    assert prov["threshold"] == 0.91
    assert prov["score_calibrated"] is False
    assert prov["weights_hash"] == \
        "326e8a272c6c4536c3a07ccaabc3feb8ad6905ac0db7e7e67b88ceb98096cc45"
    assert cr.input_info["bands"] == list(env_inputs.REQUIRED_BANDS)


# ------------------------------------------------------- 4. RGB rejected safely
def test_rgb_is_rejected_for_environment_at_the_application_boundary(rgb_pair):
    """A rejected pair is a value with the published sentence, not a traceback."""
    pair = pair_input.prepare("environment", *rgb_pair)
    assert not pair.ok
    assert pair.before is None and pair.after is None
    assert pair.errors == [env_inputs.REQUIRED_BANDS_MESSAGE]


def test_environment_never_synthesises_the_missing_bands(rgb_pair):
    """The refusal must not be reachable past - no padding, no duplication."""
    pair = pair_input.prepare("environment", *rgb_pair)
    assert not pair.ok
    # Nothing that could be fed to a model came back.
    assert all(getattr(pair, name) is None for name in
               ("before", "after", "preview_before", "preview_after"))


def test_six_band_input_is_rejected_for_built_environment(six_band_pair):
    """The dispatch is symmetric: the RGB domain refuses .npy just as clearly."""
    pair = pair_input.prepare("built_environment", *six_band_pair)
    assert not pair.ok
    assert pair.errors and "readable image" in pair.errors[0]


# ------------------------------------------- 5. no eager loading of other models
def test_preparing_input_loads_no_model_at_all(rgb_pair, six_band_pair):
    """Reading a pair must not import torch, let alone build a network."""
    probe = (
        "import sys;"
        "sys.path.insert(0, %r);"
        "from src.common import pair_input;"
        "a = pair_input.prepare('built_environment', %r, %r);"
        "b = pair_input.prepare('environment', %r, %r);"
        "print((a.ok, b.ok, 'torch' in sys.modules))"
        % (config.ROOT, rgb_pair[0], rgb_pair[1],
           six_band_pair[0], six_band_pair[1]))
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                         text=True, cwd=config.ROOT)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "(True, True, False)", out.stdout


@needs_built_env
def test_requesting_one_domain_does_not_construct_the_other():
    """Asking for built_environment must not load the environment checkpoint."""
    probe = (
        "import sys;"
        "sys.path.insert(0, %r);"
        "from src.core import registry;"
        "e = registry.get('built_environment');"
        "loaded = [m for m in sys.modules if 'domains.environment' in m];"
        "print((e.domain, loaded))" % config.ROOT)
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                         text=True, cwd=config.ROOT)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "('built_environment', [])", out.stdout


# --------------------------------------------------- 6. built_env regression
@needs_built_env
def test_built_environment_probability_map_is_unchanged():
    """The detector itself must be untouched by the dispatch work.

    Pinned to CPU deliberately: BUILT_ENV_PROBABILITY_SHA256 was captured from
    a CPU forward pass, and GPU (cuDNN) convolution kernels are not bit
    reproducible with CPU regardless of TF32 / autocast / determinism flags -
    that is a hardware-backend property, not something this test's arithmetic
    controls. Auto-selecting CUDA here would make the assertion pass or fail
    based on which device happens to be available, not on whether the
    detector's own output actually changed.
    """
    engine = registry.get("built_environment", device="cpu")
    rng = np.random.default_rng(7)
    a = rng.integers(0, 256, (300, 260, 3), dtype=np.uint8)
    b = rng.integers(0, 256, (300, 260, 3), dtype=np.uint8)
    prob = engine.probability_map(a, b)
    digest = hashlib.sha256(
        np.ascontiguousarray(prob, dtype=np.float32).tobytes()).hexdigest()
    assert digest == BUILT_ENV_PROBABILITY_SHA256


@needs_built_env
def test_direction_support_is_declared_and_still_works(rgb_pair):
    """Directionality must survive: the option is advertised and accepted."""
    engine = registry.get("built_environment")
    assert engine.supports_direction is True
    assert registry.get_factory("environment")      # resolvable, not constructed

    pair = pair_input.prepare("built_environment", *rgb_pair)
    out = engine.analyze(pair.before, pair.after, with_direction=True)
    assert isinstance(out["change_result"], ChangeResult)


@needs_e2
def test_environment_declares_no_direction_classifier():
    """A fact, not an AttributeError, so the UI can branch on it safely."""
    engine = registry.get("environment")
    assert engine.supports_direction is False
    with pytest.raises(TypeError):
        engine.analyze(None, None, with_direction=True)


# ------------------------------------------------------- 7. the preview render
def test_preview_is_display_only_and_never_reaches_the_model(six_band_pair):
    """The composite uses a FIXED scale, so two images stay comparable."""
    before = np.load(six_band_pair[0])
    preview = env_inputs.preview_rgb(before)
    assert preview.shape == (256, 256, 3) and preview.dtype == np.uint8

    # Fixed, not per-image: scaling the input halves the output, which a
    # percentile stretch would hide. This is the rule the Phase 4B contact
    # sheets were corrected to use.
    dimmer = env_inputs.preview_rgb(before * 0.5)
    assert dimmer.mean() < preview.mean()
    assert env_inputs.PREVIEW_MAX_REFLECTANCE == 0.30
