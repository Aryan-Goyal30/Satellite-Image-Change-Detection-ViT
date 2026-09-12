"""Earth Guardian - Product V1.

    streamlit run app.py

A thin presentation layer over the Earth Guardian contracts:

    example catalogue / upload
        -> registry.get("built_environment")
        -> engine.analyze(before, after)
        -> ChangeResult
        -> screens

No screen imports a model, a dataset module, or a domain package. Inference runs
only when the user presses Analyze; every other interaction reads the stored
ChangeResult.
"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="Earth Guardian", page_icon="\U0001F30D",
                   layout="wide", initial_sidebar_state="collapsed")

from ui import state, theme          # noqa: E402
from ui.analysis import render as render_workspace   # noqa: E402
from ui.home import render as render_home            # noqa: E402
from ui.results import render as render_results      # noqa: E402

theme.apply()
state.init()

screen = state.screen()
if screen == state.WORKSPACE:
    render_workspace()
elif screen == state.RESULTS:
    render_results()
else:
    render_home()
