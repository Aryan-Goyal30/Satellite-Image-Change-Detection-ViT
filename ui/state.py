"""Screen routing and analysis state.

Inference is expensive, so it runs only when the user presses Analyze. The
result is held in session state and every later interaction - switching a
visualisation, expanding a section, highlighting a region - reads from it.

`INFER_COUNT` counts how many times the engine has actually been invoked, which
makes "changing a view does not re-run the network" a testable property.
"""
import streamlit as st

SCREEN = "eg_screen"
ANALYSIS = "eg_analysis"
INFER_COUNT = "eg_infer_count"
VIEW = "eg_view"

HOME = "home"
WORKSPACE = "workspace"
RESULTS = "results"


def init():
    st.session_state.setdefault(SCREEN, HOME)
    st.session_state.setdefault(ANALYSIS, None)
    st.session_state.setdefault(INFER_COUNT, 0)
    st.session_state.setdefault(VIEW, "Change overlay")


def screen():
    return st.session_state.get(SCREEN, HOME)


def go(name):
    st.session_state[SCREEN] = name


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
