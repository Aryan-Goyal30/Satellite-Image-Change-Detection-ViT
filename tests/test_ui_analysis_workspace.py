"""UI Phase 5A / Step 3: the Analysis workspace hierarchy.

Renders ui.analysis.render() against the same stub Streamlit the Results tests
use, extended with the few primitives this screen needs. Ordering assertions are
made on what the screen actually emitted, not on source text, so "requirements
come before the input selector" is a fact about the page.

Validation itself is never stubbed: the real services.prepare_pair() and the
real six-band contract run, so a refusal here is the product's own refusal.
"""
import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config                                            # noqa: E402
from src.core.engine import EngineMetadata                        # noqa: E402
from src.core.types import InputSpec, Provenance                  # noqa: E402
from test_ui_results_domains import (                             # noqa: E402
    _Ctx, _Recorder, _StubStreamlit)
from ui import domains as domain_profiles                         # noqa: E402

HAS_E2 = os.path.exists(config.ENVIRONMENT_CHECKPOINT)

SIX_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")


class _AnalysisStub(_StubStreamlit):
    """The Results stub plus the three primitives the input screen uses."""

    def __init__(self, recorder, session, uploads=None):
        super().__init__(recorder, session)
        self._uploads = list(uploads or [])
        self.spinner_text = None

    def file_uploader(self, label, *a, **k):
        self._rec.text.append(str(label))
        return self._uploads.pop(0) if self._uploads else None

    def slider(self, label, minimum, maximum, value=None, *a, **k):
        self._rec.text.append(str(label))
        return value

    def spinner(self, text="", *a, **k):
        self.spinner_text = str(text)
        self._rec.text.append(str(text))
        return _Ctx()

    def rerun(self, *a, **k):
        # Unlike Results, this screen legitimately reruns after Analyze.
        self._rec.text.append("<rerun>")

    def button(self, label="", *a, **k):
        # Overridden (not changed on the shared stub) so button labels are
        # visible to ordering assertions on this screen only.
        self._rec.text.append(str(label))
        return False


class _FakeEngine:
    """Enough engine for the input screen. No model, no inference."""

    def __init__(self, domain, display_name, task, bands, threshold,
                 supports_direction):
        self.domain = domain
        self.threshold = threshold
        self.supports_direction = supports_direction
        self.analyzed = []
        spec = InputSpec(in_channels=len(bands), bands=tuple(bands),
                         channel_order=",".join(bands))
        self.metadata = EngineMetadata(
            name=f"{domain}-model", domain=domain, display_name=display_name,
            task=task, capabilities=(), input_spec=spec,
            provenance=Provenance(model="m", dataset="d", threshold=threshold))

    def analyze(self, before, after, **kwargs):
        self.analyzed.append((before, after, kwargs))
        raise AssertionError("inference must not run unless Analyze is pressed")


def built_engine():
    return _FakeEngine("built_environment", "Built Environment",
                       "Binary structural change detection", ("R", "G", "B"),
                       0.625, True)


def environment_engine():
    return _FakeEngine("environment", "Environment",
                       "Binary forest-loss change detection", SIX_BANDS,
                       0.91, False)


@pytest.fixture(scope="module")
def rgb_sources(tmp_path_factory):
    directory = tmp_path_factory.mktemp("aw_rgb")
    rng = np.random.default_rng(21)
    out = []
    for name in ("before.png", "after.png"):
        path = directory / name
        Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(path)
        out.append(str(path))
    return tuple(out)


@pytest.fixture(scope="module")
def six_band_sources(tmp_path_factory):
    directory = tmp_path_factory.mktemp("aw_s2")
    rng = np.random.default_rng(22)
    out = []
    for name in ("before.npy", "after.npy"):
        path = directory / name
        np.save(path, (rng.random((64, 64, 6)) * 0.3).astype(np.float32))
        out.append(str(path))
    return tuple(out)


def render(engine, sources, monkeypatch, mode="own", examples=()):
    """Render ui.analysis.render() for `engine`, returning (recorder, stub)."""
    from ui import analysis as ui_analysis
    from ui import services
    from ui import state as ui_state

    profile = domain_profiles.profile(engine.domain)
    # The stub's radio honours a pre-seeded session value, which is how a user
    # switching the input mode reaches the screen on the next rerun.
    session = {f"eg_mode_{engine.domain}":
               profile["mode_own"] if mode == "own" else profile["mode_example"]}
    recorder = _Recorder()
    stub = _AnalysisStub(recorder, session,
                         uploads=list(sources) if mode == "own" else [])

    monkeypatch.setattr(ui_analysis, "st", stub)
    monkeypatch.setattr(ui_analysis.state, "domain", lambda: engine.domain)
    monkeypatch.setattr(ui_analysis.state, "go", lambda *a, **k: None)
    monkeypatch.setattr(ui_analysis.services, "get_engine", lambda *a, **k: engine)
    monkeypatch.setattr(ui_analysis.services, "list_examples",
                        lambda domain: list(examples))

    def _text(body, *a, **k):
        stub._rec.text.append(str(body))

    monkeypatch.setattr(ui_analysis.theme, "note", _text)
    monkeypatch.setattr(ui_analysis.theme, "label", _text)
    monkeypatch.setattr(ui_analysis.theme, "requirements",
                        lambda body, note="", *a, **k:
                        stub._rec.text.append(f"REQUIRES: {body} {note}".strip()))

    if mode == "own":
        # The uploader returns paths; file handling itself is unchanged.
        monkeypatch.setattr(stub, "file_uploader",
                            _uploader(stub, list(sources)))
    ui_analysis.render()
    return recorder, stub


def _uploader(stub, sources):
    class _Named(str):
        @property
        def name(self):
            return os.path.basename(self)

    queue = [_Named(s) for s in sources]

    def uploader(label, *a, **k):
        stub._rec.text.append(str(label))
        return queue.pop(0) if queue else None
    return uploader


def _index(rec, fragment):
    low = fragment.lower()
    hit = next((i for i, t in enumerate(rec.text) if low in t.lower()), None)
    assert hit is not None, f"never rendered: {fragment!r}"
    return hit


# ------------------------------------------------------- 1. requirements strip
def test_requirements_appear_before_input_selection(rgb_sources, monkeypatch):
    rec, _ = render(built_engine(), rgb_sources, monkeypatch)
    assert _index(rec, "REQUIRES:") < _index(rec, "Input")
    # ...and before the uploader controls themselves.
    assert _index(rec, "REQUIRES:") < _index(rec, "Drag and drop the earlier")


def test_built_environment_requirements_name_rgb_and_two_dates(rgb_sources,
                                                               monkeypatch):
    rec, _ = render(built_engine(), rgb_sources, monkeypatch)
    line = rec.text[_index(rec, "REQUIRES:")]
    assert "Two RGB images of the same place, taken at different times." in line


def test_environment_requirements_name_all_six_bands(six_band_sources,
                                                     monkeypatch):
    rec, _ = render(environment_engine(), six_band_sources, monkeypatch)
    line = rec.text[_index(rec, "REQUIRES:")]
    assert "all six bands" in line
    for band in SIX_BANDS:
        assert band in line, f"{band} missing from the requirement"
    assert "Sentinel-2" in line


def test_environment_gives_a_plain_language_reason(six_band_sources, monkeypatch):
    rec, _ = render(environment_engine(), six_band_sources, monkeypatch)
    line = rec.text[_index(rec, "REQUIRES:")]
    assert ("Forest loss is detected using infrared light that ordinary RGB "
            "photos do not record.") in line
    # No architecture vocabulary anywhere on the screen.
    blob = "\n".join(rec.text).lower()
    for jargon in ("swir", "encoder", "channel", "stem", "normalis", "normaliz",
                   "reflectance", "architecture"):
        assert jargon not in blob, f"screen exposes {jargon!r}"


def test_built_environment_never_shows_the_six_band_requirement(rgb_sources,
                                                               monkeypatch):
    rec, _ = render(built_engine(), rgb_sources, monkeypatch)
    blob = "\n".join(rec.text)
    for band in SIX_BANDS:
        assert band not in blob
    assert "six bands" not in blob.lower()


def test_band_names_come_from_the_engine_input_spec():
    """Copy lives in ui/domains.py; the band list comes from the model."""
    spec = InputSpec(in_channels=6, bands=SIX_BANDS)
    text = domain_profiles.requirement_text("environment", spec)
    assert ", ".join(SIX_BANDS) in text

    # A different declared band set changes the sentence, with no edit here.
    other = InputSpec(in_channels=2, bands=("B08", "B12"))
    assert "B08, B12" in domain_profiles.requirement_text("environment", other)
    assert "B02" not in domain_profiles.requirement_text("environment", other)

    # No band literal is hardcoded in the presentation copy.
    source = open(os.path.join(config.ROOT, "ui", "domains.py"),
                  encoding="utf-8").read()
    assert "B02" not in source


# ------------------------------------------------------------- 2. input mode
def test_environment_offers_use_my_own_data_not_upload_images(six_band_sources,
                                                              monkeypatch):
    rec, _ = render(environment_engine(), six_band_sources, monkeypatch)
    assert "Use my own data" in rec.text
    assert "Upload your own images" not in rec.text
    assert "Try an example" in rec.text


def test_built_environment_may_keep_an_image_oriented_label(rgb_sources,
                                                            monkeypatch):
    rec, _ = render(built_engine(), rgb_sources, monkeypatch)
    assert "Upload my own images" in rec.text
    assert "Try an example" in rec.text


# ---------------------------------------------------------------- 3. ordering
def test_analyze_is_rendered_before_advanced(six_band_sources, monkeypatch):
    rec, _ = render(environment_engine(), six_band_sources, monkeypatch)
    assert _index(rec, "Analyze change") < _index(rec, "Advanced")


def test_full_hierarchy_order(six_band_sources, monkeypatch):
    rec, _ = render(environment_engine(), six_band_sources, monkeypatch)
    order = [_index(rec, f) for f in
             ("### Environment Monitor", "Find forest lost", "REQUIRES:",
              "Input", "Before acquisition", "Analyze change", "Advanced")]
    assert order == sorted(order), f"hierarchy out of order: {order}"


def test_preview_shows_two_images_before_the_analyze_button(six_band_sources,
                                                            monkeypatch):
    rec, _ = render(environment_engine(), six_band_sources, monkeypatch)
    assert len(rec.images) == 2
    for image in rec.images:
        assert np.asarray(image).shape[-1] == 3, "preview must be RGB"
        assert np.asarray(image).dtype == np.uint8


# ---------------------------------------------------- 4. refusals unchanged
def test_environment_still_refuses_rgb_input(rgb_sources, monkeypatch):
    from src.domains.environment import inputs as env_inputs

    rec, _ = render(environment_engine(), rgb_sources, monkeypatch)
    assert env_inputs.REQUIRED_BANDS_MESSAGE in rec.text
    # Nothing was previewed and no Analyze button was offered.
    assert not rec.images
    assert "Analyze change" not in rec.text


def test_environment_accepts_valid_six_band_input(six_band_sources, monkeypatch):
    rec, _ = render(environment_engine(), six_band_sources, monkeypatch)
    from src.domains.environment import inputs as env_inputs
    assert env_inputs.REQUIRED_BANDS_MESSAGE not in rec.text
    assert "Analyze change" in rec.text


def test_refusal_points_at_bundled_examples_without_touching_validation(
        rgb_sources, monkeypatch):
    from src.domains.environment import inputs as env_inputs

    examples = [{"name": "Amazon"}, {"name": "SE Asia"}]
    rec, _ = render(environment_engine(), rgb_sources, monkeypatch,
                    examples=examples)
    blob = "\n".join(rec.text)
    # The refusal sentence is unchanged and still first.
    assert env_inputs.REQUIRED_BANDS_MESSAGE in rec.text
    assert "2 bundled examples" in blob
    assert _index(rec, env_inputs.REQUIRED_BANDS_MESSAGE) < _index(rec, "bundled")


def test_no_example_pointer_when_there_are_no_examples(rgb_sources, monkeypatch):
    rec, _ = render(environment_engine(), rgb_sources, monkeypatch, examples=())
    assert "bundled examples" not in "\n".join(rec.text)


def test_wrong_band_count_and_dn_data_are_still_refused(tmp_path, monkeypatch):
    from src.common import pair_input
    from src.domains.environment import inputs as env_inputs

    four = tmp_path / "four.npy"
    np.save(four, (np.random.default_rng(1).random((16, 16, 4)) * 0.3
                   ).astype(np.float32))
    pair = pair_input.prepare("environment", str(four), str(four))
    assert not pair.ok and pair.errors == [env_inputs.REQUIRED_BANDS_MESSAGE]

    dn = tmp_path / "dn.npy"
    np.save(dn, np.full((16, 16, 6), 1200, np.uint16))
    pair = pair_input.prepare("environment", str(dn), str(dn))
    assert not pair.ok and "digital numbers" in pair.errors[0]


# -------------------------------------------------- 5. model/display separation
def test_model_receives_six_bands_while_display_stays_rgb(six_band_sources,
                                                          monkeypatch):
    from src.common import pair_input

    pair = pair_input.prepare("environment", *six_band_sources)
    assert pair.before.shape[-1] == 6 and pair.before.dtype == np.float32
    assert pair.preview_before.shape[-1] == 3
    assert pair.preview_before.dtype == np.uint8
    assert pair.preview_before is not pair.before

    rec, _ = render(environment_engine(), six_band_sources, monkeypatch)
    for image in rec.images:
        assert np.asarray(image).shape[-1] != 6, "six-band array reached display"


def test_analysis_source_still_never_renders_model_input():
    source = open(os.path.join(config.ROOT, "ui", "analysis.py"),
                  encoding="utf-8").read()
    assert "pair.preview_before" in source and "pair.preview_after" in source
    assert 'out["before"]' not in source and "to_rgb" not in source
    assert "services.prepare_pair" in source


# -------------------------------------------------------------- 6. spinner
def test_spinner_text_is_domain_specific():
    assert domain_profiles.profile("built_environment")["spinner"] == \
        "Analyzing structural change..."
    assert domain_profiles.profile("environment")["spinner"] == \
        "Analyzing forest loss..."
    source = open(os.path.join(config.ROOT, "ui", "analysis.py"),
                  encoding="utf-8").read()
    assert 'st.spinner("Analyzing change..."' not in source


def test_spinner_used_by_run_analysis_matches_the_domain(six_band_sources,
                                                         monkeypatch):
    from src.common import pair_input
    from ui import analysis as ui_analysis

    recorder = _Recorder()
    stub = _AnalysisStub(recorder, {})
    monkeypatch.setattr(ui_analysis, "st", stub)
    monkeypatch.setattr(ui_analysis.state, "count_inference", lambda: None)
    monkeypatch.setattr(ui_analysis.state, "set_analysis", lambda p: None)
    monkeypatch.setattr(ui_analysis.state, "go", lambda *a, **k: None)

    class _Engine(_FakeEngine):
        def analyze(self, before, after, **kwargs):
            return {"change_result": None, "mask": None, "probability": None}

    engine = _Engine("environment", "Environment", "t", SIX_BANDS, 0.91, False)
    pair = pair_input.prepare("environment", *six_band_sources)
    ui_analysis._run_analysis(engine, pair, None, "src", 0.91, 32)
    assert stub.spinner_text == "Analyzing forest loss..."


# ------------------------------------------------------- 7. nothing else moved
def test_examples_and_catalogue_are_untouched():
    from src.common import examples as cat

    env = cat.available(domain="environment")
    assert len(env) == 3
    assert {e.id for e in env} == {"env_amazon_large_clearing",
                                   "env_seasia_small_clearing",
                                   "env_amazon_intact_forest"}
    for ex in env:
        assert ex.kind == cat.KIND_SENTINEL2_SIX_BAND
        assert ex.exists() and ex.has_ground_truth
    assert len(cat.available(domain="built_environment")) >= 3


def test_inference_does_not_run_while_rendering(six_band_sources, monkeypatch):
    """Analyze is explicit: rendering the page must never call the engine."""
    engine = environment_engine()
    render(engine, six_band_sources, monkeypatch)
    assert engine.analyzed == []


@pytest.mark.skipif(not HAS_E2, reason="frozen E2 checkpoint not present")
def test_real_environment_example_still_flows_end_to_end():
    """The real catalogue example, real adapter, real frozen engine."""
    from src.common import examples as cat, pair_input
    from ui import analysis as ui_analysis
    from ui import services

    engine = services.get_engine.__wrapped__("environment")
    ex = next(e for e in cat.available(domain="environment")
              if e.id == "env_amazon_large_clearing")
    pair = pair_input.prepare("environment", ex.before_path, ex.after_path,
                              engine.metadata.input_spec)
    assert pair.ok
    gt = ui_analysis._load_ground_truth((ex.ground_truth_path,
                                         ex.ground_truth_kind))
    out = engine.analyze(pair.before, pair.after, min_area_px=32,
                         georef=pair.georef or ex.georef,
                         **ui_analysis.analyze_options(engine))
    cr = out["change_result"]
    assert cr.primary_layer.name == "forest_loss"
    assert cr.quantities.changed_pixels > 0
    assert gt is not None and gt.shape == out["mask"].shape
