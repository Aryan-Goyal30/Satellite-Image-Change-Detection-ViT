"""UI Phase 3: one Results screen, two domains.

These render ui.results.render() for real against a stub Streamlit, capturing
every string and every image it emits. That is what makes "no direction
vocabulary appears for an environmental result" a fact about the page rather
than a claim about the source code.

No model runs here: ChangeResults are constructed directly, so the tests are
fast and do not need a checkpoint.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.engine import EngineMetadata                       # noqa: E402
from src.core.types import (                                     # noqa: E402
    ChangeResult, GeoRef, InputSpec, Layer, OperatingEnvelope, Provenance,
    Quantities, Region)

SIZE = 16


# --------------------------------------------------------------- stub streamlit
class _Recorder:
    """Collects everything the screen renders."""

    def __init__(self):
        self.text = []
        self.images = []
        self.tables = []
        self.warnings = []

    @property
    def blob(self):
        return "\n".join(self.text).lower()


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Col(_Ctx):
    def __init__(self, stub):
        self._stub = stub

    @property
    def _rec(self):
        return self._stub._rec

    def metric(self, label, value, *a, **k):
        self._rec.text.append(f"{label}: {value}")

    def button(self, *a, **k):
        return False


class _Slot(_Ctx):
    """st.container(): content appears where the container was CREATED.

    Faithful to Streamlit on the one property these tests depend on - ordering.
    Content written inside the `with` block is spliced back into the recorder at
    the index the container occupied, so recorded order matches rendered order
    even though the calls happen later.
    """

    def __init__(self, stub):
        self._stub = stub
        parent = stub._rec
        self._parent = parent
        self._at = (len(parent.text), len(parent.images), len(parent.tables))

    def __enter__(self):
        self._buffer = _Recorder()
        self._saved = self._stub._rec
        self._stub._rec = self._buffer
        return self

    def __exit__(self, *exc):
        self._stub._rec = self._saved
        text_at, images_at, tables_at = self._at
        self._parent.text[text_at:text_at] = self._buffer.text
        self._parent.images[images_at:images_at] = self._buffer.images
        self._parent.tables[tables_at:tables_at] = self._buffer.tables
        self._parent.warnings.extend(self._buffer.warnings)
        return False


class _StubStreamlit:
    def __init__(self, recorder, session):
        self._rec = recorder
        self.session_state = session

    # text
    def markdown(self, body="", *a, **k):
        self._rec.text.append(str(body))

    def write(self, *a, **k):
        for item in a:
            if isinstance(item, str):
                self._rec.text.append(item)

    def warning(self, body, *a, **k):
        self._rec.text.append(str(body))
        self._rec.warnings.append(str(body))

    def error(self, body, *a, **k):
        self._rec.text.append(str(body))

    def caption(self, body="", *a, **k):
        self._rec.text.append(str(body))

    def divider(self, *a, **k):
        pass

    # layout
    def columns(self, spec, **k):
        n = spec if isinstance(spec, int) else len(spec)
        return [_Col(self) for _ in range(n)]

    def expander(self, label, *a, **k):
        self._rec.text.append(str(label))
        return _Ctx()

    def container(self, *a, **k):
        return _Slot(self)

    # widgets
    def radio(self, label, options, *a, **k):
        self._rec.text.append(str(label))
        self._rec.text.extend(str(o) for o in options)
        key = k.get("key")
        if key and self.session_state.get(key) in options:
            return self.session_state[key]
        return options[0]

    def selectbox(self, label, options, *a, **k):
        self._rec.text.append(str(label))
        self._rec.text.extend(str(o) for o in options)
        return options[0]

    def button(self, *a, **k):
        return False

    def download_button(self, label, *a, **k):
        self._rec.text.append(str(label))

    def metric(self, label, value, *a, **k):
        self._rec.text.append(f"{label}: {value}")

    # content
    def image(self, image, *a, **k):
        self._rec.images.append(image)
        if k.get("caption"):
            self._rec.text.append(str(k["caption"]))

    def dataframe(self, df, *a, **k):
        self._rec.tables.append(df)
        self._rec.text.extend(str(c) for c in df.columns)

    def table(self, df, *a, **k):
        self._rec.tables.append(df)
        self._rec.text.extend(str(v) for v in df.to_numpy().ravel())

    def rerun(self, *a, **k):
        raise AssertionError("results.render() must not rerun for a valid payload")


# ------------------------------------------------------------------ fixtures
def _regions(direction=None):
    return [Region(id=1, area_px=40, bbox_xywh=[1, 1, 4, 4], centroid_xy=[3.0, 3.0],
                   confidence=0.84, layer="x", direction=direction,
                   direction_score=0.91 if direction else None)]


def _change_result(*, layer_name, model, dataset, task, threshold,
                   dataset_version=None, score_calibrated=None, params=None,
                   regions=None, georef=None, warnings=()):
    rng = np.random.default_rng(4)
    prob = rng.random((SIZE, SIZE)).astype(np.float32)
    mask = prob >= 0.5
    provenance = Provenance(
        model=model, dataset=dataset, threshold=threshold, task=task,
        version="1.0.0", weights_hash="deadbeef",
        evaluation_protocol="held-out split", dataset_version=dataset_version,
        score_calibrated=score_calibrated,
        operating_envelope=OperatingEnvelope(
            gsd_m_range=(10.0, 10.0), regions_validated=("Amazon",),
            sensors_validated=("Sentinel-2",), notes="envelope notes"))
    return ChangeResult(
        layers=[Layer(name=layer_name, mask=mask, score_map=prob, threshold=threshold,
                      mean_confidence=0.77, description="a layer")],
        regions=regions if regions is not None else _regions(),
        quantities=Quantities.from_mask(mask, georef),
        provenance=provenance,
        input_info={"height": SIZE, "width": SIZE,
                    "bands": ["B02", "B03", "B04", "B08", "B11", "B12"]
                    if dataset_version else None},
        params=dict(params or {"threshold": threshold, "min_area_px": 32}),
        runtime_seconds=0.1, georef=georef, warnings=list(warnings))


def _metadata(domain, display_name, task, model, capabilities, limitations=()):
    return EngineMetadata(
        name=model, domain=domain, display_name=display_name, task=task,
        capabilities=capabilities, input_spec=InputSpec(),
        provenance=Provenance(model=model, dataset="d", threshold=0.5),
        description="an architecture", limitations=tuple(limitations))


def environment_result():
    cr = _change_result(
        layer_name="forest_loss", model="environment-siamese-unet-6band",
        dataset="TMF + Sentinel-2", task="Binary forest-loss change detection",
        threshold=0.91, dataset_version="v24", score_calibrated=False,
        params={"threshold": 0.91, "min_area_px": 32, "tile": 256,
                "overlap": 64, "precision": "float32",
                "checkpoint": "outputs/environment_baseline/x.pt",
                "normalisation_subset_sha256": "727ad32f98b05db5",
                "dataset_manifest_sha256": "23721c09c864297a"},
        warnings=["Regions indicate where forest loss is likely, not how many "
                  "distinct clearings occurred."])
    md = _metadata("environment", "Environment",
                   "Binary forest-loss change detection",
                   "environment-siamese-unet-6band", ("forest_loss",),
                   ("operating in Africa",))
    return cr, md


def built_environment_result(with_direction=True):
    cr = _change_result(
        layer_name="structural_change", model="siamese-unet-resnet34",
        dataset="LEVIR-CD", task="Binary structural change detection",
        threshold=0.625,
        regions=_regions("construction" if with_direction else None),
        params={"threshold": 0.625, "min_area_px": 32,
                **({"direction": {
                    "model": "direction-resnet18", "version": "1.0.0",
                    "architecture": "ResNet-18", "dataset": "S2Looking",
                    "threshold": 0.6, "threshold_selected_on": "val",
                    "classes": ["construction", "demolition"],
                    "abstain_label": "uncertain",
                    "crop_protocol": {"crop_size": 128, "context_fraction": 0.25},
                    "temporal_ordering": {"before_image_dir": "Image1",
                                          "after_image_dir": "Image2"},
                    "weights_hash": "abc123", "evaluation_scope": "scope.",
                    "notes": "notes.", "documented_by_authors": False}}
                   if with_direction else {})})
    md = _metadata("built_environment", "Built Environment",
                   "Binary structural change detection",
                   "siamese-unet-resnet34", ("structural_change",),
                   ("counting individual buildings",))
    return cr, md


def _payload(cr, md, domain, ground_truth=None):
    rng = np.random.default_rng(9)
    display = rng.integers(0, 256, (SIZE, SIZE, 3), dtype=np.uint8)
    return {
        "domain": domain, "change_result": cr, "mask": cr.primary_layer.mask,
        "probability": cr.primary_layer.score_map,
        "display_before": display, "display_after": display.copy(),
        "ground_truth": ground_truth, "metadata": md, "source": "unit test",
    }


def render(payload, monkeypatch, forbid_engine=True, view=None):
    """Render ui.results against the stub, returning the recorder.

    `view` pre-selects a view the way a user clicking the selector would, so a
    test can inspect the caption of a view other than the default.
    """
    from ui import results as ui_results
    from ui import services
    from ui import state as ui_state
    from ui.panels import direction as direction_panel

    recorder = _Recorder()
    session = {} if view is None else {ui_state.VIEW: view}
    stub = _StubStreamlit(recorder, session)
    monkeypatch.setattr(ui_results, "st", stub)
    monkeypatch.setattr(direction_panel, "st", stub)
    def _text(body, *a, **k):
        stub._rec.text.append(str(body))

    monkeypatch.setattr(ui_results.theme, "note", _text)
    monkeypatch.setattr(ui_results.theme, "label", _text)
    monkeypatch.setattr(ui_results.theme, "verdict", _text)
    monkeypatch.setattr(ui_results.theme, "hero_metric",
                        lambda value, label, *a, **k:
                        stub._rec.text.append(f"{label}: {value}"))
    monkeypatch.setattr(ui_results.state, "analysis", lambda: payload)
    monkeypatch.setattr(ui_results.state, "go", lambda *a, **k: None)

    if forbid_engine:
        def _boom(*a, **k):
            raise AssertionError(
                "Results must not fetch an engine; the payload is self-describing")
        monkeypatch.setattr(services, "get_engine", _boom)

    ui_results.render()
    return recorder


# ------------------------------------------------------------------- 1. display
def test_results_renders_display_arrays_only(monkeypatch):
    cr, md = environment_result()
    payload = _payload(cr, md, "environment")
    rec = render(payload, monkeypatch)

    assert rec.images, "nothing was rendered"
    for image in rec.images:
        array = np.asarray(image)
        assert array.ndim in (2, 3)
        if array.ndim == 3:
            assert array.shape[-1] in (3, 4), f"non-displayable array {array.shape}"
    assert any(np.array_equal(np.asarray(i), payload["display_before"])
               for i in rec.images)


def test_six_band_model_input_can_never_reach_the_display_path(monkeypatch):
    """The payload carries no six-band array, so none can be rendered."""
    cr, md = environment_result()
    payload = _payload(cr, md, "environment")

    assert "before" not in payload and "after" not in payload
    for value in payload.values():
        if isinstance(value, np.ndarray):
            assert value.ndim != 3 or value.shape[-1] != 6

    rec = render(payload, monkeypatch)
    for image in rec.images:
        assert np.asarray(image).shape[-1] != 6

    # And the screen reads the display keys, not the model-input ones.
    source = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "ui", "results.py"), encoding="utf-8").read()
    assert 'payload.get("display_before")' in source
    assert 'payload["before"]' not in source and 'payload["after"]' not in source


def test_a_six_band_payload_is_never_produced_by_analysis():
    """Guard the producer too: ui/analysis.py must store previews only."""
    source = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "ui", "analysis.py"), encoding="utf-8").read()
    assert '"display_before": display_before' in source
    assert '"before": ' not in source and '"after": ' not in source


# ------------------------------------------------------------------- 2. about
def test_environment_about_comes_from_stored_metadata_and_provenance(monkeypatch):
    cr, md = environment_result()
    rec = render(_payload(cr, md, "environment"), monkeypatch)

    assert "environment-siamese-unet-6band" in rec.blob
    assert "tmf + sentinel-2" in rec.blob
    assert "v24" in rec.blob                       # dataset version surfaced
    assert "b02, b03, b04, b08, b11, b12" in rec.blob
    assert "binary forest-loss change detection" in rec.blob
    assert "operating in africa" in rec.blob       # limitations from metadata
    # Never the other domain's identity.
    assert "levir" not in rec.blob
    assert "siamese-unet-resnet34" not in rec.blob


def test_results_never_fetches_the_default_engine(monkeypatch):
    """render() with services.get_engine() booby-trapped must still succeed."""
    for cr, md, domain in (environment_result() + ("environment",),
                           built_environment_result() + ("built_environment",)):
        rec = render(_payload(cr, md, domain), monkeypatch, forbid_engine=True)
        assert rec.text


def test_built_environment_about_information_is_preserved(monkeypatch):
    cr, md = built_environment_result()
    rec = render(_payload(cr, md, "built_environment"), monkeypatch)
    assert "siamese-unet-resnet34" in rec.blob
    assert "levir-cd" in rec.blob
    assert "binary structural change detection" in rec.blob
    assert "counting individual buildings" in rec.blob


# --------------------------------------------------------------- 3. direction
DIRECTION_WORDS = ("direction", "construction", "demolition", "uncertain",
                   "s2looking", "classifier")


def test_direction_section_appears_for_a_built_environment_result(monkeypatch):
    cr, md = built_environment_result(with_direction=True)
    rec = render(_payload(cr, md, "built_environment"), monkeypatch)
    blob = rec.blob
    assert "direction" in blob
    assert "construction" in blob and "demolition" in blob and "uncertain" in blob
    assert "s2looking" in blob
    assert "not calibrated" in blob or "not</strong> calibrated" in blob
    table = next(t for t in rec.tables if "Region" in getattr(t, "columns", []))
    assert "Direction" in table.columns and "Direction score" in table.columns


def test_no_direction_vocabulary_for_an_environmental_result(monkeypatch):
    """Not one direction word may appear anywhere on an environmental page."""
    cr, md = environment_result()
    rec = render(_payload(cr, md, "environment"), monkeypatch)
    blob = rec.blob
    for word in DIRECTION_WORDS:
        assert word not in blob, f"environmental Results emitted {word!r}"
    assert "did not run" not in blob

    table = next(t for t in rec.tables if "Region" in getattr(t, "columns", []))
    assert "Direction" not in table.columns
    assert "Direction score" not in table.columns


def test_direction_is_gated_on_result_data_not_domain_name():
    """Results must not branch on a domain name to hide direction."""
    from ui.panels import direction as direction_panel

    source = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "ui", "results.py"), encoding="utf-8").read()
    code = "\n".join(line for line in source.splitlines()
                     if not line.lstrip().startswith("#"))
    assert '"built_environment"' not in code
    assert '"environment"' not in code

    cr, _md = environment_result()
    assert direction_panel.applies(cr) is False
    cr_built, _ = built_environment_result(with_direction=True)
    assert direction_panel.applies(cr_built) is True


def test_direction_absent_when_the_engine_did_not_classify(monkeypatch):
    """A built-environment result with no direction data shows no direction."""
    cr, md = built_environment_result(with_direction=False)
    rec = render(_payload(cr, md, "built_environment"), monkeypatch)
    assert "direction" not in rec.blob


def test_direction_unavailable_warning_still_surfaces(monkeypatch):
    """A failed direction run must stay visible, not silently vanish."""
    cr, md = built_environment_result(with_direction=False)
    cr.warnings.append("Direction analysis unavailable: the classifier failed.")
    rec = render(_payload(cr, md, "built_environment"), monkeypatch)
    assert "direction analysis was not completed" in rec.blob


# ----------------------------------------------------------------- 4. common
def test_common_change_result_fields_render_for_environment(monkeypatch):
    cr, md = environment_result()
    rec = render(_payload(cr, md, "environment"), monkeypatch)
    blob = rec.blob

    assert "of the analyzed area changed" in blob     # hero metric
    assert "detected regions" in blob
    assert "overlay" in blob and "change only" in blob
    assert "highlight a detected region" in blob
    assert "forest loss is likely" in blob          # engine warning surfaced
    assert "download result (json)" in blob
    table = next(t for t in rec.tables if "Region" in getattr(t, "columns", []))
    for column in ("Region", "Area (px)", "X", "Y", "Width", "Height"):
        assert column in table.columns


def test_error_map_only_when_ground_truth_is_supplied(monkeypatch):
    cr, md = environment_result()

    without = render(_payload(cr, md, "environment"), monkeypatch)
    assert "vs ground truth" not in without.blob
    assert "f1" not in without.blob                 # no fabricated metric

    gt = np.zeros_like(cr.primary_layer.mask)
    gt[2:6, 2:6] = True
    with_gt = render(_payload(cr, md, "environment", ground_truth=gt), monkeypatch)
    assert "vs ground truth" in with_gt.blob
    assert "f1" in with_gt.blob
    # Selecting it, the image caption still explains the colour coding.
    selected = render(_payload(cr, md, "environment", ground_truth=gt),
                      monkeypatch, view="vs Ground truth")
    assert "green tp / red fp / blue fn" in selected.blob


def test_geospatial_area_only_with_a_real_scale(monkeypatch):
    cr, md = environment_result()
    plain = render(_payload(cr, md, "environment"), monkeypatch)
    assert "ground area in m&sup2; is not reported" in plain.blob

    georef = GeoRef(crs="EPSG:32750", gsd_m=10.0, units="metre")
    scaled_cr, _ = environment_result()
    scaled_cr.georef = georef
    scaled_cr.quantities = Quantities.from_mask(scaled_cr.primary_layer.mask, georef)
    scaled = render(_payload(scaled_cr, md, "environment"), monkeypatch)
    assert "changed ground area" in scaled.blob
    assert "epsg:32750" in scaled.blob


# ------------------------------------------------------------ 5. score wording
def test_environmental_scores_are_never_called_calibrated_probabilities(monkeypatch):
    cr, md = environment_result()
    rec = render(_payload(cr, md, "environment"), monkeypatch)
    blob = rec.blob

    assert "uncalibrated" in blob
    # The About table states the calibration claim explicitly.
    about = next(t for t in rec.tables if "Scores calibrated" in t.to_numpy())
    rows = dict(zip(about.to_numpy()[:, 0], about.to_numpy()[:, 1]))
    assert rows["Scores calibrated"] == "No"
    # The view LABEL is domain-independent; the CAPTION carries the claim.
    assert "signal strength" in blob
    # Demoted out of the headline metrics, still available in About with the
    # uncalibrated noun.
    assert "mean detector score" in blob
    assert "mean detector confidence" not in blob

    # Selecting that view, the CAPTION carries the uncalibrated wording.
    signal = render(_payload(cr, md, "environment"), monkeypatch,
                    view="Signal strength")
    assert "change score (brighter = higher score)" in signal.blob
    assert "change probability" not in signal.blob
    # The word "probability" may only appear while denying calibration.
    for line in rec.text:
        low = line.lower()
        if "probability" in low:
            assert ("not" in low and ("probabilit" in low)), \
                f"environmental Results implies calibration: {line!r}"


def test_built_environment_score_wording_is_unchanged(monkeypatch):
    """An engine that makes no calibration claim keeps its original wording."""
    cr, md = built_environment_result()
    assert cr.provenance.score_calibrated is None
    rec = render(_payload(cr, md, "built_environment"), monkeypatch)
    blob = rec.blob
    # Same view label as every domain; the caption still says "probability",
    # which is the wording an engine making no calibration claim always had.
    assert "signal strength" in blob
    assert "mean detector confidence" in blob
    assert "mean detector score" not in blob
    assert "scores calibrated" not in blob

    # Selecting that view, the caption still says "probability" - the wording
    # an engine making no calibration claim always had.
    signal = render(_payload(cr, md, "built_environment"), monkeypatch,
                    view="Signal strength")
    assert "change probability (brighter = higher probability)" in signal.blob
    assert "change score (brighter" not in signal.blob


def test_score_helpers_respect_the_provenance_claim():
    from ui import results as ui_results

    cr_env, _ = environment_result()
    cr_built, _ = built_environment_result()
    assert ui_results.scores_are_uncalibrated(cr_env) is True
    assert ui_results.scores_are_uncalibrated(cr_built) is False
    assert ui_results.score_noun(cr_env) == "score"
    assert ui_results.score_noun(cr_built) == "probability"


# ------------------------------------------------------------ 6. region table
def test_region_table_stays_capability_driven():
    from ui.results import _region_table

    plain = _region_table(_regions())
    assert "Direction" not in plain.columns
    assert "Area (m2)" not in plain.columns
    assert "Detector confidence" in plain.columns

    directed = _region_table(_regions("demolition"))
    assert "Direction" in directed.columns and "Direction score" in directed.columns

    geo = _regions()
    geo[0].area_m2 = 4000.0
    geo[0].centroid_crs_xy = (500000.0, 9000000.0)
    with_geo = _region_table(geo)
    assert "Area (m2)" in with_geo.columns
    assert "Centroid E" in with_geo.columns and "Centroid N" in with_geo.columns


def test_environmental_regions_get_no_invented_semantic_labels(monkeypatch):
    cr, md = environment_result()
    rec = render(_payload(cr, md, "environment"), monkeypatch)
    table = next(t for t in rec.tables if "Region" in getattr(t, "columns", []))
    for banned in ("Direction", "Type", "Class", "Category", "Label",
                   "Deforestation", "Cause"):
        assert banned not in table.columns
    assert "not an identified object" in rec.blob


# --------------------------------------------------- 7. no leaked internals
def test_implementation_details_stay_out_of_the_rendered_page(monkeypatch):
    """Checkpoint hashes, digests, paths and tiling are not user-facing."""
    cr, md = environment_result()
    rec = render(_payload(cr, md, "environment"), monkeypatch)
    blob = rec.blob
    for leak in ("727ad32f", "23721c09", "normalisation", "outputs/environment",
                 ".pt", "overlap", "float32"):
        assert leak not in blob, f"implementation detail leaked: {leak!r}"


def test_detector_score_column_label_follows_the_calibration_claim(monkeypatch):
    """Header and footnote must agree: 'confidence' only when calibration is claimed."""
    from ui import results as ui_results

    cr_env, md_env = environment_result()
    cr_built, md_built = built_environment_result()
    assert ui_results.detector_score_label(cr_env) == ui_results.DETECTOR_SCORE_LABEL
    assert ui_results.detector_score_label(cr_built) == \
        ui_results.DETECTOR_CONFIDENCE_LABEL

    env = render(_payload(cr_env, md_env, "environment"), monkeypatch)
    table = next(t for t in env.tables if "Region" in getattr(t, "columns", []))
    assert "Detector score" in table.columns
    assert "Detector confidence" not in table.columns

    built = render(_payload(cr_built, md_built, "built_environment"), monkeypatch)
    table = next(t for t in built.tables if "Region" in getattr(t, "columns", []))
    assert "Detector confidence" in table.columns
    assert "Detector score" not in table.columns


# ================================================================ UI Phase 5A
# The verdict: one sentence answering "what changed?", computed from the result.
def _empty(cr):
    """The same result with nothing detected, so the empty wording is exercised."""
    import numpy as np
    from src.core.types import Quantities
    layer = cr.primary_layer
    layer.mask = np.zeros_like(layer.mask)
    cr.regions = []
    cr.quantities = Quantities.from_mask(layer.mask, cr.georef)
    return cr


def test_built_environment_verdict_when_change_is_detected(monkeypatch):
    from ui import results as ui_results

    cr, md = built_environment_result()
    expected = (f"Structural change detected across "
                f"{cr.quantities.changed_percentage:.2f}% of the analyzed area.")
    assert ui_results.verdict(cr) == expected
    assert cr.quantities.changed_pixels > 0

    rec = render(_payload(cr, md, "built_environment"), monkeypatch)
    assert expected.lower() in rec.blob


def test_built_environment_verdict_when_nothing_is_detected(monkeypatch):
    from ui import results as ui_results

    cr, md = built_environment_result()
    cr = _empty(cr)
    assert ui_results.verdict(cr) == \
        "No structural change detected in the analyzed area."

    rec = render(_payload(cr, md, "built_environment"), monkeypatch)
    assert "no structural change detected in the analyzed area." in rec.blob
    assert "0.00%" not in rec.blob.split("detected")[0]


def test_environment_verdict_when_forest_loss_is_detected(monkeypatch):
    from ui import results as ui_results

    cr, md = environment_result()
    expected = (f"Forest loss detected across "
                f"{cr.quantities.changed_percentage:.2f}% of the analyzed area.")
    assert ui_results.verdict(cr) == expected

    rec = render(_payload(cr, md, "environment"), monkeypatch)
    assert expected.lower() in rec.blob


def test_environment_verdict_when_no_forest_loss_is_detected(monkeypatch):
    from ui import results as ui_results

    cr, md = environment_result()
    cr = _empty(cr)
    assert ui_results.verdict(cr) == \
        "No forest loss detected in the analyzed area."

    rec = render(_payload(cr, md, "environment"), monkeypatch)
    assert "no forest loss detected in the analyzed area." in rec.blob


def test_verdict_percentage_comes_from_the_result_not_a_literal():
    """Change the quantities and the sentence must follow."""
    import numpy as np
    from ui import results as ui_results
    from src.core.types import Quantities

    cr, _md = environment_result()
    mask = np.zeros_like(cr.primary_layer.mask)
    mask[:8, :] = True                      # exactly half of a 16x16 tile
    cr.primary_layer.mask = mask
    cr.quantities = Quantities.from_mask(mask, None)
    assert ui_results.verdict(cr) == \
        "Forest loss detected across 50.00% of the analyzed area."


def test_verdict_noun_is_derived_from_the_declared_layer_not_a_domain_name():
    """A future domain gets a correct sentence with no change to results.py."""
    from ui import results as ui_results

    cr, _ = built_environment_result()
    assert ui_results.change_noun(cr) == "Structural change"
    cr_env, _ = environment_result()
    assert ui_results.change_noun(cr_env) == "Forest loss"

    cr_env.primary_layer.name = "burn_scar"
    assert ui_results.change_noun(cr_env) == "Burn scar"
    assert ui_results.verdict(cr_env).startswith("Burn scar detected across ")

    # No domain name is consulted anywhere in the verdict path.
    source = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "ui", "results.py"), encoding="utf-8").read()
    code = "\n".join(line for line in source.splitlines()
                     if not line.lstrip().startswith("#"))
    assert '"built_environment"' not in code and '"environment"' not in code


def test_verdict_precedes_all_result_content(monkeypatch):
    """It must be the first thing said about the result."""
    from ui import results as ui_results

    cr, md = environment_result()
    rec = render(_payload(cr, md, "environment"), monkeypatch)
    sentence = ui_results.verdict(cr)
    index = rec.text.index(sentence)
    for later in ("Before", "After", "#### The two dates"):
        assert index < rec.text.index(later), f"verdict must precede {later!r}"


def test_view_selector_uses_the_approved_plain_language_labels(monkeypatch):
    from ui import results as ui_results

    assert ui_results.VIEW_OVERLAY == "Overlay"
    assert ui_results.VIEW_MASK == "Change only"
    assert ui_results.VIEW_SIGNAL == "Signal strength"
    assert ui_results.VIEW_ERROR == "vs Ground truth"

    cr, md = environment_result()
    gt = np.zeros_like(cr.primary_layer.mask)
    gt[2:6, 2:6] = True
    rec = render(_payload(cr, md, "environment", ground_truth=gt), monkeypatch)
    for label in ("Overlay", "Change only", "Signal strength", "vs Ground truth"):
        assert label in rec.text, f"missing view label {label!r}"
    # The old technical labels are gone from the selector.
    for retired in ("Change overlay", "Change mask", "Probability map",
                    "Score map", "Error map"):
        assert retired not in rec.text


def test_signal_strength_label_is_identical_for_both_domains(monkeypatch):
    """One label; the calibration claim lives in the caption, not the tab."""
    for factory, domain in ((environment_result, "environment"),
                            (built_environment_result, "built_environment")):
        cr, md = factory()
        rec = render(_payload(cr, md, domain), monkeypatch)
        assert "Signal strength" in rec.text


def test_underlying_score_data_is_untouched_by_the_relabelling(monkeypatch):
    """Renaming a label must not alter a single value."""
    from ui import results as ui_results

    cr, md = environment_result()
    before_prob = cr.primary_layer.score_map.copy()
    before_mask = cr.primary_layer.mask.copy()
    before_prov = cr.provenance.to_dict()

    render(_payload(cr, md, "environment"), monkeypatch)

    assert np.array_equal(cr.primary_layer.score_map, before_prob)
    assert np.array_equal(cr.primary_layer.mask, before_mask)
    assert cr.provenance.to_dict() == before_prov
    assert cr.provenance.score_calibrated is False
    assert ui_results.score_noun(cr) == "score"


# =============================================== UI Phase 5A / Step 2: hierarchy
def _order(rec, *fragments):
    """Index of the first recorded element containing each fragment."""
    low = [t.lower() for t in rec.text]
    out = []
    for fragment in fragments:
        hit = next((i for i, t in enumerate(low) if fragment.lower() in t), None)
        assert hit is not None, f"never rendered: {fragment!r}"
        out.append(hit)
    return out


def test_hierarchy_verdict_then_visualization_then_metric(monkeypatch):
    """The first screenful answers what, where, then how much - in that order."""
    from ui import results as ui_results

    cr, md = environment_result()
    payload = _payload(cr, md, "environment")
    rec = render(payload, monkeypatch)

    sentence = ui_results.verdict(cr)
    verdict_at, hero_at, dates_at = _order(
        rec, sentence, "of the analyzed area changed", "#### The two dates")
    assert verdict_at < hero_at < dates_at

    # The primary visualization is the FIRST image on the page, and it is not
    # one of the two display arrays - it is the change view built from them.
    primary = rec.images[0]
    assert not np.array_equal(np.asarray(primary), payload["display_before"])
    assert not np.array_equal(np.asarray(primary), payload["display_after"])


def test_primary_visualization_precedes_before_after(monkeypatch):
    """The change map is above the evidence pair, not below it."""
    cr, md = environment_result()
    payload = _payload(cr, md, "environment")
    rec = render(payload, monkeypatch)

    assert len(rec.images) == 3
    # images[0] is the change view; [1] and [2] are the before/after evidence.
    assert np.array_equal(np.asarray(rec.images[1]), payload["display_before"])
    assert np.array_equal(np.asarray(rec.images[2]), payload["display_after"])

    # Exact labels, not substrings: the change caption itself ends in
    # "...over the after image", which a substring search would match.
    before_at = rec.text.index("Before")
    after_at = rec.text.index("After")
    caption_at = next(i for i, t in enumerate(rec.text)
                      if t == "Detected change over the after image")
    assert caption_at < before_at < after_at


def test_changed_percentage_is_the_hero_metric(monkeypatch):
    cr, md = environment_result()
    rec = render(_payload(cr, md, "environment"), monkeypatch)
    expected = f"{cr.quantities.changed_percentage:.2f}%"
    assert any(expected in t and "of the analyzed area changed" in t
               for t in rec.text), "hero metric missing or not the changed %"


def test_changed_pixels_left_the_primary_metric_row(monkeypatch):
    """Demoted, not deleted: absent from the metric row, present in About."""
    cr, md = environment_result()
    rec = render(_payload(cr, md, "environment"), monkeypatch)

    metric_rows = [t for t in rec.text if t.startswith("Detected regions:")
                   or t.startswith("Changed pixels:")
                   or t.startswith("Changed ground area:")]
    assert any(t.startswith("Detected regions:") for t in metric_rows)
    assert not any(t.startswith("Changed pixels:") for t in metric_rows)

    about = next(t for t in rec.tables if "Changed pixels" in t.to_numpy())
    rows = dict(zip(about.to_numpy()[:, 0], about.to_numpy()[:, 1]))
    assert rows["Changed pixels"].startswith(f"{cr.quantities.changed_pixels:,}")
    # And the value itself is still in the exported JSON, untouched.
    assert cr.to_dict()["quantities"]["changed_pixels"] == cr.quantities.changed_pixels


def test_mean_score_left_the_primary_metric_row_but_stays_in_about(monkeypatch):
    for factory, domain, label in (
            (environment_result, "environment", "Mean detector score"),
            (built_environment_result, "built_environment",
             "Mean detector confidence")):
        cr, md = factory()
        rec = render(_payload(cr, md, domain), monkeypatch)

        assert not any(t.startswith("Mean detector") and ":" in t
                       for t in rec.text), "mean score must not be a headline metric"
        about = next(t for t in rec.tables if label in t.to_numpy())
        rows = dict(zip(about.to_numpy()[:, 0], about.to_numpy()[:, 1]))
        assert rows[label] == f"{cr.primary_layer.mean_confidence:.3f}"


def test_ground_area_is_a_secondary_metric_only_with_a_real_scale(monkeypatch):
    cr, md = environment_result()
    plain = render(_payload(cr, md, "environment"), monkeypatch)
    assert not any(t.startswith("Changed ground area:") for t in plain.text)
    assert not any(t.startswith("Pixel size:") for t in plain.text)
    assert "ground area in m&sup2; is not reported" in plain.blob

    georef = GeoRef(crs="EPSG:32750", gsd_m=10.0, units="metre")
    scaled, _ = environment_result()
    scaled.georef = georef
    scaled.quantities = Quantities.from_mask(scaled.primary_layer.mask, georef)
    rec = render(_payload(scaled, md, "environment"), monkeypatch)
    assert any(t.startswith("Changed ground area:") for t in rec.text)
    assert any(t.startswith("Pixel size: 10 m") for t in rec.text)
    assert "epsg:32750" in rec.blob


def test_warnings_are_visible_and_never_inside_an_expander(monkeypatch):
    cr, md = environment_result()
    rec = render(_payload(cr, md, "environment"), monkeypatch)

    assert rec.warnings, "engine warnings must reach the page"
    assert any("forest loss is likely" in w.lower() for w in rec.warnings)
    # Rendered before the collapsed region section, so never nested in it.
    warning_at = _order(rec, "forest loss is likely")[0]
    inspect_at = _order(rec, "Inspect regions")[0]
    assert warning_at < inspect_at


def test_region_inspection_is_collapsed_but_intact(monkeypatch):
    cr, md = environment_result()
    rec = render(_payload(cr, md, "environment"), monkeypatch)

    assert any(t.startswith("Inspect regions (") for t in rec.text)
    table = next(t for t in rec.tables if "Region" in getattr(t, "columns", []))
    assert len(table) == len(cr.regions)
    for column in ("Region", "Area (px)", "X", "Y", "Width", "Height"):
        assert column in table.columns
    # Highlighting still available, and still next to the visualization.
    assert "highlight a detected region" in rec.blob


def test_direction_behaviour_survives_the_reorder(monkeypatch):
    cr, md = built_environment_result(with_direction=True)
    rec = render(_payload(cr, md, "built_environment"), monkeypatch)
    blob = rec.blob
    assert "construction" in blob and "demolition" in blob and "s2looking" in blob
    table = next(t for t in rec.tables if "Region" in getattr(t, "columns", []))
    assert "Direction" in table.columns and "Direction score" in table.columns


def test_environment_still_emits_no_direction_vocabulary_after_reorder(monkeypatch):
    cr, md = environment_result()
    rec = render(_payload(cr, md, "environment"), monkeypatch)
    for word in DIRECTION_WORDS:
        assert word not in rec.blob, f"environmental Results emitted {word!r}"


def test_ground_truth_comparison_survives_the_reorder(monkeypatch):
    cr, md = environment_result()
    gt = np.zeros_like(cr.primary_layer.mask)
    gt[2:6, 2:6] = True

    rec = render(_payload(cr, md, "environment", ground_truth=gt), monkeypatch)
    assert "vs ground truth" in rec.blob
    assert "f1" in rec.blob

    from src.common.visualization import compare_to_ground_truth
    expected = compare_to_ground_truth(cr, gt)
    assert f"f1 {expected.f1:.3f}" in rec.blob


def test_reordering_changed_no_result_values(monkeypatch):
    """Layout is presentation: every number must survive rendering untouched."""
    cr, md = environment_result()
    snapshot = cr.to_dict()
    prob = cr.primary_layer.score_map.copy()
    mask = cr.primary_layer.mask.copy()

    render(_payload(cr, md, "environment"), monkeypatch)

    assert cr.to_dict() == snapshot
    assert np.array_equal(cr.primary_layer.score_map, prob)
    assert np.array_equal(cr.primary_layer.mask, mask)
    assert cr.provenance.score_calibrated is False


def test_stub_container_places_content_where_it_was_created():
    """The stub's st.container() must order content like Streamlit does."""
    recorder = _Recorder()
    stub = _StubStreamlit(recorder, {})
    stub.markdown("first")
    slot = stub.container()
    stub.markdown("second")
    with slot:
        stub.markdown("INSIDE")
    stub.markdown("third")
    assert recorder.text == ["first", "INSIDE", "second", "third"]
