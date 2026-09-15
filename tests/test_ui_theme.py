"""UI Phase 5A / Step 5: the theme contract.

These assert what the stylesheet PROMISES, not what a browser paints: that the
classes the screens render are defined, that the shared semantic colours still
match the figures, and that the restraints the product committed to - no
gradients, no animation, no downloads - still hold.

Deliberately not pixel or screenshot tests. They would lock in layout details
that are meant to stay adjustable.
"""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config                                            # noqa: E402
from src.common import visualization                              # noqa: E402
from ui import theme                                              # noqa: E402

CSS = theme._CSS


def rule_at(selector):
    """Character offset of a rule, or -1. Matches the rule, not a comment."""
    match = re.search(re.escape(selector) + r"\s*\{", CSS)
    return match.start() if match else -1


def defines(selector):
    return rule_at(selector) >= 0


# --------------------------------------------------------------- 1. constants
def test_theme_constants_remain_available():
    for name in ("INK", "MUTED", "GROUND", "PANEL", "LINE", "ACCENT",
                 "WARN", "DANGER", "TP_COLOR", "FP_COLOR", "FN_COLOR"):
        value = getattr(theme, name)
        assert isinstance(value, str) and re.fullmatch(r"#[0-9A-Fa-f]{6}", value)


def test_palette_values_are_unchanged():
    assert theme.INK == "#E6ECE6"
    assert theme.MUTED == "#8B9E96"
    assert theme.GROUND == "#0D1117"
    assert theme.PANEL == "#141A21"
    assert theme.LINE == "#263038"
    assert theme.ACCENT == "#63C8BF"
    assert theme.WARN == "#D9A85C"
    assert theme.DANGER == "#E2807F"


def test_helpers_remain_available():
    for name in ("apply", "label", "note", "verdict", "requirements",
                 "hero_metric"):
        assert callable(getattr(theme, name))


# ------------------------------------------------------- 2-6. emitted classes
@pytest.mark.parametrize("selector", [
    ".eg-verdict",
    ".eg-hero-metric", ".eg-hero-value", ".eg-hero-label",
    ".eg-requires", ".eg-requires-label", ".eg-requires-body",
    ".eg-requires-note",
    ".eg-card", ".eg-card h3", ".eg-card p", ".eg-card.eg-soon",
    ".eg-card-needs, .eg-card-extra", ".eg-card-key", ".eg-card-meta",
    ".eg-badge", ".eg-badge.eg-live",
    ".eg-hero h1", ".eg-tagline", ".eg-tagline-sub",
    ".eg-label", ".eg-note",
])
def test_class_is_defined(selector):
    assert defines(selector), f"{selector} is rendered by a screen but undefined"


def test_every_class_the_screens_render_is_styled():
    """Whatever the UI emits as class='eg-...' must exist in the stylesheet."""
    used = set()
    ui_dir = os.path.join(config.ROOT, "ui")
    for root, _dirs, files in os.walk(ui_dir):
        for name in files:
            if not name.endswith(".py") or name == "theme.py":
                continue
            source = open(os.path.join(root, name), encoding="utf-8").read()
            used.update(re.findall(r"class='(eg-[a-z-]+)'", source))
            used.update(re.findall(r'class="(eg-[a-z-]+)"', source))
    assert used, "no eg- classes found in the UI - the scan is broken"
    for name in sorted(used):
        assert f".{name}" in CSS, f"{name} is rendered but not styled"


def test_verdict_is_a_statement_not_a_headline():
    body = CSS[rule_at(".eg-verdict"):][:220]
    size = float(re.search(r"font-size:\s*([\d.]+)rem", body).group(1))
    assert 1.25 <= size <= 1.45, f"verdict at {size}rem is outside the agreed range"
    assert theme.INK in body
    assert "font-weight: 600" in body


def test_hero_metric_dominates_the_secondary_metrics():
    hero = CSS[rule_at(".eg-hero-value"):][:200]
    hero_size = float(re.search(r"font-size:\s*([\d.]+)rem", hero).group(1))
    secondary = CSS[rule_at('[data-testid="stMetricValue"]'):][:120]
    secondary_size = float(re.search(r"font-size:\s*([\d.]+)rem",
                                     secondary).group(1))
    assert hero_size > secondary_size, "the headline number must dominate"


def test_requirements_strip_is_informational_not_an_alert():
    body = CSS[rule_at(".eg-requires"):][:260]
    assert theme.PANEL in body, "should sit on the ordinary panel ground"
    assert theme.ACCENT in body, "marked by the accent, not by a warning colour"
    for alarming in (theme.DANGER, theme.WARN, theme.FP_COLOR):
        assert alarming not in body, "must not read as an error box"


def test_image_framing_is_defined():
    assert defines('[data-testid="stImage"] img')
    body = CSS[rule_at('[data-testid="stImage"] img'):][:160]
    assert "border: 1px solid" in body and theme.LINE in body
    assert "border-radius" in body


# ------------------------------------------------- 7. shared semantic colours
def test_tp_fp_fn_colours_are_unchanged():
    assert theme.TP_COLOR == "#3CC85A"
    assert theme.FP_COLOR == "#E64646"
    assert theme.FN_COLOR == "#4682EB"


def test_theme_and_figure_colours_still_agree():
    """The product and the report must not drift apart."""
    def to_rgb(value):
        return tuple(int(value[i:i + 2], 16) for i in (1, 3, 5))

    assert to_rgb(theme.TP_COLOR) == visualization.TP_COLOR
    assert to_rgb(theme.FP_COLOR) == visualization.FP_COLOR
    assert to_rgb(theme.FN_COLOR) == visualization.FN_COLOR


def test_overlay_colour_is_unchanged():
    import inspect
    signature = inspect.signature(visualization.overlay)
    assert signature.parameters["color"].default == (255, 40, 40)
    assert signature.parameters["alpha"].default == 0.45


# ---------------------------------------------------------- 8. the restraints
def test_no_prohibited_styling_is_introduced():
    low = CSS.lower()
    for banned in ("gradient", "@keyframes", "animation", "transition:",
                   "@import", "url(", "http://", "https://", "<script",
                   "box-shadow", "text-shadow"):
        assert banned not in low, f"theme introduces {banned!r}"


def test_theme_needs_no_network_and_no_javascript():
    assert "<style>" in CSS and "</style>" in CSS
    assert "@font-face" not in CSS
    assert "javascript:" not in CSS.lower()
    # One stylesheet, injected once.
    assert CSS.count("<style>") == 1


def test_layout_width_is_unchanged():
    body = CSS[rule_at(".block-container"):][:160]
    assert "max-width: 1280px" in body


# ------------------------------------- base colours over the forced dark ground
def test_text_colour_is_stated_not_inherited_from_streamlit():
    """The ground is forced dark, so the text must be forced too.

    There is no .streamlit/config.toml, so a default light base would otherwise
    paint dark text on this dark ground for every unstyled element.
    """
    assert not os.path.exists(os.path.join(config.ROOT, ".streamlit",
                                           "config.toml"))
    assert defines(".stApp, .stMarkdown, .stMarkdown p, .stMarkdown li")
    for selector in ('[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p',
                     '[data-testid="stExpander"] summary'):
        assert defines(selector), f"{selector} would inherit an unknown colour"


def test_headings_do_not_outweigh_the_verdict():
    section = CSS[rule_at(".block-container h3"):][:200]
    heading_size = float(re.search(r"font-size:\s*([\d.]+)rem",
                                   section).group(1))
    verdict = CSS[rule_at(".eg-verdict"):][:220]
    verdict_size = float(re.search(r"font-size:\s*([\d.]+)rem",
                                   verdict).group(1))
    assert heading_size < verdict_size, \
        "a section title must not outweigh the answer beneath it"


def test_hero_and_card_headings_keep_their_own_sizes():
    """Generic heading rules must not override the components that follow."""
    assert rule_at(".eg-hero h1") > rule_at(".block-container h1")
    assert rule_at(".eg-card h3") > rule_at(".block-container h3")
