"""Screen routing, selected domain, and analysis state.

Inference is expensive, so it runs only when the user presses Analyze. The
result is held in session state and every later interaction - switching a
visualisation, expanding a section, highlighting a region - reads from it.

`INFER_COUNT` counts how many times the engine has actually been invoked, which
makes "changing a view does not re-run the network" a testable property.

Domain
------
The selected domain is stored here, as a plain name. This module does NOT decide
which domains exist - services.available_domains() does, reading the engine
registry - so there is exactly one source of truth for that. Switching domain
discards the stored analysis, because a ChangeResult from one domain must never
be rendered under another one's heading or metadata.
"""
import streamlit as st

SCREEN = "eg_screen"
ANALYSIS = "eg_analysis"
INFER_COUNT = "eg_infer_count"
VIEW = "eg_view"
DOMAIN = "eg_domain"

HOME = "home"
WORKSPACE = "workspace"
RESULTS = "results"

#: The domain the product opens on, and the one assumed when nothing has been
#: selected yet. Keeps existing built-environment behaviour as the default.
DEFAULT_DOMAIN = "built_environment"


def init():
    st.session_state.setdefault(SCREEN, HOME)
    st.session_state.setdefault(ANALYSIS, None)
    st.session_state.setdefault(INFER_COUNT, 0)
    # Must match a view name in ui/results.py, which owns them; results.py
    # resets this key if the stored value is not among the available views.
    st.session_state.setdefault(VIEW, "Overlay")
    st.session_state.setdefault(DOMAIN, DEFAULT_DOMAIN)


def screen():
    return st.session_state.get(SCREEN, HOME)


def go(name):
    st.session_state[SCREEN] = name


def domain():
    """The selected domain name. Never None."""
    return st.session_state.get(DOMAIN) or DEFAULT_DOMAIN


def set_domain(name):
    """Select the active domain.

    Switching domain clears the stored analysis and the chosen view: a result
    belongs to the domain that produced it, and the available views differ, so
    carrying either across would render one domain's output under another's
    identity.
    """
    if name != domain():
        clear_analysis()
        st.session_state.pop(VIEW, None)
        st.session_state.pop("eg_highlight", None)
    st.session_state[DOMAIN] = name


def set_analysis(payload):
    st.session_state[ANALYSIS] = payload


def analysis():
    return st.session_state.get(ANALYSIS)


def clear_analysis():
    st.session_state[ANALYSIS] = None


def count_inference():
    st.session_state[INFER_COUNT] = st.session_state.get(INFER_COUNT, 0) + 1


def inference_count():
    return st.session_state.get(INFER_COUNT, 0)
