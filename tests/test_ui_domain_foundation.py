"""UI Phase 2: the domain-aware foundation.

These test the UI's FOUNDATION - domain state, domain selection, and the input
routing the screens perform. No Streamlit script runs here: the screens are
exercised through their pure helpers and through a fake session state, so the
tests stay fast and deterministic.

The central invariant under test is the model/display split:

    PreparedPair.before / after            -> the engine
    PreparedPair.preview_before / _after   -> the screen and the payload
"""
import ast
import io
import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config                                           # noqa: E402
from src.common import pair_input                                # noqa: E402
from src.core import registry                                    # noqa: E402
from src.domains.environment import inputs as env_inputs         # noqa: E402
from ui import domains as domain_profiles                        # noqa: E402

HAS_E2 = os.path.exists(config.ENVIRONMENT_CHECKPOINT)
UI_DIR = os.path.join(config.ROOT, "ui")


@pytest.fixture
def session(monkeypatch):
    """A plain dict standing in for st.session_state, for ui.state."""
    import streamlit as st
    from ui import state as ui_state

    store = {}
    monkeypatch.setattr(st, "session_state", store, raising=False)
    monkeypatch.setattr(ui_state.st, "session_state", store, raising=False)
    return store


@pytest.fixture(scope="module")
def rgb_pair(tmp_path_factory):
    directory = tmp_path_factory.mktemp("ui_rgb")
    rng = np.random.default_rng(11)
    paths = []
    for name in ("before.png", "after.png"):
        path = directory / name
        Image.fromarray(rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)).save(path)
        paths.append(str(path))
    return tuple(paths)


@pytest.fixture(scope="module")
def six_band_pair(tmp_path_factory):
    directory = tmp_path_factory.mktemp("ui_s2")
    rng = np.random.default_rng(12)
    paths = []
    for name in ("before.npy", "after.npy"):
        path = directory / name
        np.save(path, (rng.random((128, 128, 6)) * 0.3).astype(np.float32))
        paths.append(str(path))
    return tuple(paths)


# ------------------------------------------------------------ 1. domain state
def test_default_domain_is_built_environment(session):
    """Existing behaviour stays the default when nothing has been selected."""
    from ui import state as ui_state

    assert ui_state.domain() == "built_environment"
    ui_state.init()
    assert session[ui_state.DOMAIN] == "built_environment"
    assert ui_state.DEFAULT_DOMAIN == "built_environment"


def test_selecting_built_environment(session):
    from ui import state as ui_state

    ui_state.init()
    ui_state.set_domain("built_environment")
    assert ui_state.domain() == "built_environment"


def test_selecting_environment(session):
    from ui import state as ui_state

    ui_state.init()
    ui_state.set_domain("environment")
    assert ui_state.domain() == "environment"
    assert session[ui_state.DOMAIN] == "environment"


def test_changing_domain_clears_stale_analysis(session):
    """A result from one domain must never survive into another."""
    from ui import state as ui_state

    ui_state.init()
    ui_state.set_analysis({"domain": "built_environment", "mask": "stale"})
    session[ui_state.VIEW] = "Error map"
    session["eg_highlight"] = "Region 3"
    assert ui_state.analysis() is not None

    ui_state.set_domain("environment")
    assert ui_state.analysis() is None
    assert ui_state.VIEW not in session
    assert "eg_highlight" not in session


def test_reselecting_the_same_domain_keeps_the_analysis(session):
    """Only a CHANGE discards work; re-selecting must not throw a result away."""
    from ui import state as ui_state

    ui_state.init()
    ui_state.set_domain("environment")
    ui_state.set_analysis({"domain": "environment"})
    ui_state.set_domain("environment")
    assert ui_state.analysis() == {"domain": "environment"}


# ------------------------------------------------------- 2. home domain source
def test_home_exposes_registered_domains_without_a_second_list():
    """Home must read availability from the registry, not from its own list."""
    from ui import services

    assert set(services.available_domains()) == {"built_environment", "environment"}

    source = open(os.path.join(UI_DIR, "home.py"), encoding="utf-8").read()
    assert "available_domains" in source
    # No hardcoded implemented-domain names in the screen.
    assert '"built_environment"' not in source
    assert '"environment"' not in source


def test_planned_domains_never_contradict_the_registry():
    """A planned domain that gains an engine must stop being 'coming later'."""
    from ui import services

    available = set(services.available_domains())
    planned = {name for name, _title, _blurb in domain_profiles.PLANNED}
    assert not (planned & available), \
        "a domain cannot be both registered and listed as not implemented"
    assert "disaster" in planned


def test_domain_cards_come_from_model_cards_without_building_a_model():
    """Home must not construct two networks just to label two buttons."""
    from ui import services

    card = services.domain_card.__wrapped__("environment")
    assert card["display_name"] == "Environment"
    assert card["task"] == "Binary forest-loss change detection"

    card = services.domain_card.__wrapped__("built_environment")
    assert card["display_name"] == "Construction & Urban Change"

    # Unknown names degrade to a usable title rather than raising.
    assert services.domain_card.__wrapped__("nope")["display_name"] == "Nope"


def test_ui_profiles_are_not_a_registry_of_implemented_domains():
    """ui/domains.py supplies copy only; it must not gate availability."""
    assert domain_profiles.profile("not_a_domain") is domain_profiles.FALLBACK
    for name in ("built_environment", "environment"):
        assert domain_profiles.profile(name)["file_types"]


# ------------------------------------------------- 3. domain-aware preparation
def test_built_environment_still_uses_rgb(rgb_pair):
    pair = pair_input.prepare("built_environment", *rgb_pair)
    assert pair.ok
    assert pair.before.shape == (128, 128, 3)
    assert pair.preview_before is pair.before        # same array for RGB


def test_environment_analysis_uses_six_band_preparation(six_band_pair):
    pair = pair_input.prepare("environment", *six_band_pair)
    assert pair.ok
    assert pair.before.shape == (128, 128, 6)
    assert pair.before.dtype == np.float32


def test_environment_display_uses_preview_rgb_not_six_band_input(six_band_pair):
    """The displayed array must be renderable and must not be the model input."""
    pair = pair_input.prepare("environment", *six_band_pair)
    assert pair.ok
    assert pair.preview_before.shape == (128, 128, 3)
    assert pair.preview_before.dtype == np.uint8
    assert pair.preview_before is not pair.before
    assert pair.preview_after is not pair.after


def test_rgb_cannot_be_routed_into_environment(rgb_pair):
    pair = pair_input.prepare("environment", *rgb_pair)
    assert not pair.ok
    assert pair.errors == [env_inputs.REQUIRED_BANDS_MESSAGE]
    assert pair.before is None and pair.preview_before is None


def test_an_uploaded_rgb_file_object_is_refused(six_band_pair):
    """Uploads arrive as file objects, not paths - the contract must still hold."""
    buffer = io.BytesIO()
    Image.fromarray(np.zeros((32, 32, 3), np.uint8)).save(buffer, "PNG")
    buffer.seek(0)
    with pytest.raises(env_inputs.BandContractError) as excinfo:
        env_inputs.as_six_band(buffer)
    assert excinfo.value.user_message == env_inputs.REQUIRED_BANDS_MESSAGE


def test_an_uploaded_six_band_file_object_is_accepted():
    buffer = io.BytesIO()
    np.save(buffer, (np.random.default_rng(3).random((32, 32, 6)) * 0.3
                     ).astype(np.float32))
    buffer.seek(0)
    assert env_inputs.as_six_band(buffer).shape == (32, 32, 6)


def test_uploader_file_types_differ_per_domain():
    """The environment uploader must not offer PNG/JPG at all."""
    built = domain_profiles.profile("built_environment")["file_types"]
    env = domain_profiles.profile("environment")["file_types"]
    assert "png" in built and "jpg" in built
    assert env == ["npy"]
    assert "png" not in env and "jpg" not in env


# --------------------------------------------------------- 4. analysis payload
def _payload_for(domain, sources, monkeypatch, session):
    """Run ui.analysis._run_analysis with Streamlit side effects neutralised."""
    from ui import analysis as ui_analysis
    from ui import services
    from ui import state as ui_state

    engine = services.get_engine.__wrapped__(domain)
    pair = pair_input.prepare(domain, *sources, engine.metadata.input_spec)
    assert pair.ok

    class _Spinner:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(ui_analysis.st, "spinner", lambda *a, **k: _Spinner())
    monkeypatch.setattr(ui_analysis.st, "rerun", lambda *a, **k: None)
    monkeypatch.setattr(ui_analysis.st, "error", lambda *a, **k: None)
    monkeypatch.setattr(ui_analysis.state, "count_inference", lambda: None)

    captured = {}
    monkeypatch.setattr(ui_analysis.state, "set_analysis", captured.update)
    monkeypatch.setattr(ui_analysis.state, "go", lambda *a, **k: None)

    ui_analysis._run_analysis(engine, pair, None, "test source", None, 32)
    assert captured, "analysis produced no payload"
    return captured, pair


@pytest.mark.skipif(not os.path.exists(config.DEFAULT_CHECKPOINT),
                    reason="built-environment checkpoint absent")
def test_payload_contains_domain_and_display_arrays(rgb_pair, monkeypatch, session):
    payload, pair = _payload_for("built_environment", rgb_pair, monkeypatch, session)

    assert payload["domain"] == "built_environment"
    assert payload["display_before"] is pair.preview_before
    assert payload["display_after"] is pair.preview_after
    assert payload["change_result"].primary_layer.name == "structural_change"
    assert payload["mask"].shape == payload["probability"].shape


@pytest.mark.skipif(not HAS_E2, reason="frozen E2 checkpoint not present")
def test_environment_payload_stores_display_not_model_input(
        six_band_pair, monkeypatch, session):
    """The six-band model input must not be carried into the payload."""
    payload, pair = _payload_for("environment", six_band_pair, monkeypatch, session)

    assert payload["domain"] == "environment"
    assert payload["display_before"].shape[-1] == 3
    assert payload["display_before"].dtype == np.uint8
    # Nothing six-band reached the payload under any key.
    for value in payload.values():
        if isinstance(value, np.ndarray) and value.ndim == 3:
            assert value.shape[-1] != 6, "model input leaked into the payload"
    # The UI Phase 2 compatibility aliases were removed in Phase 3: the payload
    # now carries display arrays only, so no screen can render model input.
    assert "before" not in payload and "after" not in payload


@pytest.mark.skipif(not HAS_E2, reason="frozen E2 checkpoint not present")
def test_results_can_consume_the_payload_without_fetching_an_engine(
        six_band_pair, monkeypatch, session):
    """Results must describe the engine that RAN, not the default one."""
    payload, _pair = _payload_for("environment", six_band_pair, monkeypatch, session)

    md = payload["metadata"]
    assert md is not None
    assert md.domain == payload["domain"] == "environment"
    assert md.display_name == "Environment"
    # Everything _about() needs is present without touching the registry.
    prov = payload["change_result"].provenance
    assert prov.dataset == "TMF + Sentinel-2"
    assert prov.model == md.name

    # _about() accepts the metadata rather than fetching it. That render()
    # actually uses the payload's snapshot is proved in
    # tests/test_ui_results_domains.py, which renders the screen with
    # services.get_engine booby-trapped.
    from ui import results as ui_results
    assert ui_results._about.__code__.co_argcount == 2
    source = open(os.path.join(UI_DIR, "results.py"), encoding="utf-8").read()
    assert 'payload.get("metadata")' in source


# ------------------------------------------------- 5. no UI-side domain logic
def test_no_band_or_normalisation_logic_leaked_into_the_ui():
    """The screens dispatch; they must not know what a band is.

    `mean` and `std` are matched as whole words. They are normalisation
    IDENTIFIERS, and the substring form flagged ordinary prose - "meaningful",
    "standard" - which says nothing about whether the UI does band arithmetic.
    Every band name and every normalisation term is still matched literally.
    """
    import re

    literal = ("B02", "B03", "B04", "B08", "B11", "B12",
               "reflectance", "normali", "to_reflectance")
    whole_word = ("mean", "std")
    for name in ("analysis.py", "home.py", "state.py", "services.py"):
        source = open(os.path.join(UI_DIR, name), encoding="utf-8").read()
        code = "\n".join(
            line for line in source.splitlines()
            if not line.lstrip().startswith("#"))
        # Strip docstrings: prose may legitimately mention the contract.
        module = ast.parse(source)
        docstrings = {ast.get_docstring(n) for n in ast.walk(module)
                      if isinstance(n, (ast.Module, ast.FunctionDef,
                                        ast.ClassDef, ast.AsyncFunctionDef))}
        for doc in docstrings:
            if doc:
                code = code.replace(doc, "")
        for token in literal:
            assert token not in code, f"{name} contains domain logic token {token!r}"
        for token in whole_word:
            assert not re.search(rf"\b{token}\b", code), \
                f"{name} contains domain logic identifier {token!r}"


def test_analysis_screen_never_renders_model_input():
    """st.image must only ever receive preview/display arrays."""
    source = open(os.path.join(UI_DIR, "analysis.py"), encoding="utf-8").read()
    assert "pair.preview_before" in source and "pair.preview_after" in source
    assert "out[\"before\"]" not in source and "out['before']" not in source
    assert "to_rgb" not in source
    assert "validate_pair" not in source
    assert "services.prepare_pair" in source
