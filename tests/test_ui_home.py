"""UI Phase 5A / Step 4: the Home page domain cards.

Renders ui.home.render() against the shared stub Streamlit, so what a visitor
reads is asserted from what the screen actually emitted.

The rule Home is held to: no per-domain branching. Availability comes from the
registry, copy from the presentation profile, the input requirement from the
domain's model card, and the example count from the catalogue - so these tests
check the WIRING as much as the wording.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config                                            # noqa: E402
from test_ui_results_domains import _Recorder, _StubStreamlit     # noqa: E402
from ui import domains as domain_profiles                         # noqa: E402

SIX_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")
UI_DIR = os.path.join(config.ROOT, "ui")

# Captured once, before any test patches them: these are Streamlit-cached, and
# a second render() inside one test would otherwise try to unwrap a patch.
from ui import services as _services                              # noqa: E402
REAL_LIST_EXAMPLES = _services.list_examples.__wrapped__
REAL_DOMAIN_CARD = _services.domain_card.__wrapped__


class _HomeStub(_StubStreamlit):
    """Records button labels and which button was 'clicked'."""

    def __init__(self, recorder, session, click=None):
        super().__init__(recorder, session)
        self._click = click
        self.clicked = []

    def button(self, label="", *a, **k):
        self._rec.text.append(str(label))
        key = k.get("key")
        if self._click is not None and key == self._click:
            self.clicked.append(key)
            return True
        return False

    def rerun(self, *a, **k):
        self._rec.text.append("<rerun>")


def render(monkeypatch, click=None, examples=None):
    """Render ui.home.render(), returning (recorder, stub, navigation)."""
    from ui import home as ui_home

    recorder = _Recorder()
    stub = _HomeStub(recorder, {}, click=click)
    monkeypatch.setattr(ui_home, "st", stub)
    monkeypatch.setattr(ui_home.theme, "note",
                        lambda body, *a, **k: recorder.text.append(str(body)))

    if examples is not None:
        monkeypatch.setattr(ui_home.services, "list_examples",
                            lambda domain: list(examples.get(domain, [])))
    else:
        monkeypatch.setattr(ui_home.services, "list_examples",
                            REAL_LIST_EXAMPLES)
    monkeypatch.setattr(ui_home.services, "domain_card", REAL_DOMAIN_CARD)

    nav = {}
    monkeypatch.setattr(ui_home.state, "set_domain",
                        lambda name: nav.__setitem__("domain", name))
    monkeypatch.setattr(ui_home.state, "go",
                        lambda screen: nav.__setitem__("screen", screen))
    ui_home.render()
    return recorder, stub, nav


def blob(rec):
    return "\n".join(rec.text)


# ------------------------------------------------------------------- 1. hero
def test_home_renders_the_earth_guardian_identity(monkeypatch):
    rec, _, _ = render(monkeypatch)
    text = blob(rec)
    assert "Earth Guardian" in text
    assert "Understand what changed on Earth." in text
    assert "Compare the same place at two different times" in text


def test_hero_makes_no_accuracy_or_marketing_claims(monkeypatch):
    rec, _, _ = render(monkeypatch)
    low = blob(rec).lower()
    for claim in ("f1", "accuracy", "state of the art", "state-of-the-art",
                  "ai-powered", "%", "precision", "recall"):
        assert claim not in low, f"Home makes a claim it should not: {claim!r}"


# -------------------------------------------------------------- 2. the cards
def test_both_available_domains_are_presented(monkeypatch):
    from ui import services

    rec, _, _ = render(monkeypatch)
    text = blob(rec)
    for name in services.available_domains():
        card = REAL_DOMAIN_CARD(name)
        assert card["display_name"] in text
        assert domain_profiles.profile(name)["card_finds"] in text
    assert text.count("Available now") == len(services.available_domains())


def test_built_environment_card_states_its_input_requirement(monkeypatch):
    rec, _, _ = render(monkeypatch)
    text = blob(rec)
    assert ("Two ordinary RGB satellite images of the same place, taken at "
            "different times.") in text
    # ...and does not mention bands it does not need.
    assert "B02" not in text.split("Forest lost")[0]


def test_built_environment_card_advertises_direction(monkeypatch):
    rec, _, _ = render(monkeypatch)
    text = blob(rec)
    assert "Construction" in text and "Demolition" in text and "Uncertain" in text


def test_environment_card_names_sentinel2_and_all_six_bands(monkeypatch):
    rec, _, _ = render(monkeypatch)
    text = blob(rec)
    assert "Sentinel-2" in text
    assert "all six bands" in text
    for band in SIX_BANDS:
        assert band in text, f"{band} missing from the Home card"


def test_environment_card_makes_rgb_insufficiency_obvious(monkeypatch):
    rec, _, _ = render(monkeypatch)
    assert "Ordinary RGB photos will not work." in blob(rec)


def test_cards_explain_no_model_architecture(monkeypatch):
    rec, _, _ = render(monkeypatch)
    low = blob(rec).lower()
    for jargon in ("resnet", "u-net", "unet", "siamese", "encoder", "swir",
                   "checkpoint", "normalis", "normaliz", "threshold"):
        assert jargon not in low, f"Home exposes {jargon!r}"


# ------------------------------------------------------- 3. data-driven copy
def test_band_list_comes_from_the_model_card_not_from_home():
    """Home states a requirement it does not author."""
    from ui import services

    card = REAL_DOMAIN_CARD("environment")
    assert card["bands"] == list(SIX_BANDS)
    text = domain_profiles.card_needs_text("environment", card["bands"])
    assert ", ".join(SIX_BANDS) in text

    # A different declared band set changes the card with no edit to the copy.
    assert "B08, B12" in domain_profiles.card_needs_text("environment",
                                                         ("B08", "B12"))
    # And no band name is written into either presentation module.
    for name in ("home.py", "domains.py"):
        source = open(os.path.join(UI_DIR, name), encoding="utf-8").read()
        assert "B02" not in source, f"{name} hardcodes a band name"


def test_home_has_no_hardcoded_domain_names():
    source = open(os.path.join(UI_DIR, "home.py"), encoding="utf-8").read()
    assert "available_domains" in source
    assert '"built_environment"' not in source
    assert '"environment"' not in source
    assert '"disaster"' not in source


def test_home_builds_cards_without_constructing_an_engine(monkeypatch):
    """Drawing two cards must not load two neural networks."""
    from ui import home as ui_home

    def _boom(*a, **k):
        raise AssertionError("Home must not construct an engine")

    monkeypatch.setattr(ui_home.services, "get_engine", _boom)
    rec, _, _ = render(monkeypatch)
    assert "Earth Guardian" in blob(rec)


# ---------------------------------------------------------- 4. example count
def test_example_count_is_derived_from_the_catalogue(monkeypatch):
    rec, _, _ = render(monkeypatch, examples={
        "environment": [{}, {}, {}], "built_environment": [{}] * 27})
    text = blob(rec)
    assert "3 examples included" in text
    assert "27 examples included" in text


def test_example_count_follows_the_catalogue_not_a_literal(monkeypatch):
    rec, _, _ = render(monkeypatch, examples={"environment": [{}, {}]})
    text = blob(rec)
    assert "2 examples included" in text
    assert "3 examples included" not in text

    rec, _, _ = render(monkeypatch, examples={"environment": [{}]})
    assert "1 example included" in blob(rec)      # singular, not "1 examples"


def test_no_example_line_when_the_catalogue_is_empty(monkeypatch):
    rec, _, _ = render(monkeypatch, examples={})
    assert "examples included" not in blob(rec)


def test_the_real_catalogue_reports_three_environment_examples(monkeypatch):
    from src.common import examples as cat

    assert len(cat.available(domain="environment")) == 3
    rec, _, _ = render(monkeypatch)
    assert "3 examples included" in blob(rec)


# --------------------------------------------------------------- 5. disaster
def test_disaster_is_shown_but_unavailable(monkeypatch):
    rec, _, _ = render(monkeypatch)
    text = blob(rec)
    assert "Disaster Monitor" in text
    assert "Coming later" in text
    assert "Not implemented" in text


def test_disaster_is_not_a_registered_domain():
    from src.core import registry
    from ui import services

    assert "disaster" not in registry.available()
    assert "disaster" not in services.available_domains()
    planned = {name for name, _t, _b in domain_profiles.PLANNED}
    assert "disaster" in planned
    assert not (planned & set(services.available_domains()))


def test_disaster_has_no_start_action(monkeypatch):
    from ui import services

    rec, stub, _ = render(monkeypatch)
    starts = [t for t in rec.text if t == "Start analysis"]
    assert len(starts) == len(services.available_domains())
    assert len(starts) == 2, "one CTA per available monitor, and no more"


def test_a_planned_domain_that_gains_an_engine_stops_being_planned(monkeypatch):
    """The unavailable row can never contradict the registry."""
    from ui import home as ui_home

    monkeypatch.setattr(ui_home.domain_profiles, "PLANNED",
                        [("environment", "Environment", "not implemented")])
    rec, _, _ = render(monkeypatch)
    assert "Coming later" not in blob(rec)


# ------------------------------------------------------------------- 6. CTA
def test_starting_a_domain_sets_state_and_navigates(monkeypatch):
    from ui import services, state as ui_state

    for name in services.available_domains():
        _rec, stub, nav = render(monkeypatch, click=f"eg_start_{name}")
        assert stub.clicked == [f"eg_start_{name}"]
        assert nav["domain"] == name
        assert nav["screen"] == ui_state.WORKSPACE


def test_no_navigation_happens_without_a_click(monkeypatch):
    _rec, _stub, nav = render(monkeypatch)
    assert nav == {}


def test_there_is_no_separate_global_cta(monkeypatch):
    rec, _, _ = render(monkeypatch)
    text = blob(rec).lower()
    for label in ("get started", "launch", "try it now", "sign up"):
        assert label not in text
