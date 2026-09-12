"""Screen 1 - Home.

A short landing screen: what Earth Guardian is, the one module that works
today, and an honest note about the modules that do not exist yet.
"""
import streamlit as st

from ui import services, state, theme


def render():
    st.markdown(
        "<div class='eg-hero'><h1>&#127757; Earth Guardian</h1></div>"
        "<div class='eg-tagline'>Understand what changed on Earth.</div>",
        unsafe_allow_html=True)

    live, env, dis = st.columns(3, gap="medium")

    with live:
        st.markdown(
            "<div class='eg-card'>"
            "<div class='eg-badge eg-live'>Available now</div>"
            "<h3>Built Environment Monitor</h3>"
            "<p>Detect and visualize structural changes between two satellite "
            "images of the same place.</p></div>",
            unsafe_allow_html=True)
        st.write("")
        if st.button("Start Analysis", type="primary", use_container_width=True):
            state.go(state.WORKSPACE)
            st.rerun()

    with env:
        st.markdown(
            "<div class='eg-card eg-soon'>"
            "<div class='eg-badge'>Coming later</div>"
            "<h3>Environmental Monitor</h3>"
            "<p>Vegetation, land cover and water change. Not implemented.</p>"
            "</div>", unsafe_allow_html=True)

    with dis:
        st.markdown(
            "<div class='eg-card eg-soon'>"
            "<div class='eg-badge'>Coming later</div>"
            "<h3>Disaster Monitor</h3>"
            "<p>Flood, burn scar and post-disaster damage. Not implemented.</p>"
            "</div>", unsafe_allow_html=True)

    st.write("")
    st.divider()

    try:
        md = services.get_engine().metadata
        theme.note(
            f"<strong>What this version does.</strong> You supply two images of "
            f"the same area taken at different times. The {md.display_name} "
            f"engine returns a pixel-level map of where structural change "
            f"occurred, how much changed, and how confident it is. "
            f"Selecting a location and dates, and having imagery fetched for "
            f"you, is a later stage of the roadmap.")
    except Exception:
        theme.note(
            "<strong>What this version does.</strong> You supply two images of "
            "the same area taken at different times; the engine returns a "
            "pixel-level map of where structural change occurred.")
