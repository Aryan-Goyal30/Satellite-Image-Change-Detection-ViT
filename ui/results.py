"""Screen 3 - Change analysis results.

Everything here reads from the ChangeResult produced by the Analyze action.
Switching views, highlighting a region or opening a section never re-runs the
network: the result already lives in session state.

Visual hierarchy: before/after, then detected change, then the summary, then
region details, then model information.
"""
import json

import numpy as np
import pandas as pd
import streamlit as st

from src.common.visualization import compare_to_ground_truth, draw_bboxes, overlay
from ui import services, state, theme

NO_HIGHLIGHT = "None"


def render():
    payload = state.analysis()
    if not payload:
        state.go(state.WORKSPACE)
        st.rerun()
        return

    cr = payload["change_result"]
    mask = payload["mask"]
    prob = payload["probability"]
    before, after = payload["before"], payload["after"]
    gt = payload.get("ground_truth")

    nav = st.columns([1, 1, 5])
    with nav[0]:
        if st.button("< New analysis", use_container_width=True):
            state.go(state.WORKSPACE)
            st.rerun()
    with nav[1]:
        if st.button("Home", use_container_width=True):
            state.go(state.HOME)
            st.rerun()

    st.markdown("### Change analysis")
    if payload.get("source"):
        theme.note(payload["source"])
    st.write("")

    # ---------------------------------------------------- before / after
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        theme.label("Before")
        st.image(before, use_container_width=True)
    with c2:
        theme.label("After")
        st.image(after, use_container_width=True)

    st.write("")
    st.divider()

    # ------------------------------------------------------ detected change
    st.markdown("#### Detected change")

    views = ["Change overlay", "Change mask", "Probability map"]
    comparison = None
    if gt is not None and gt.shape == mask.shape:
        comparison = compare_to_ground_truth(cr, gt)
        views.append("Error map")

    if state.VIEW in st.session_state and st.session_state[state.VIEW] not in views:
        st.session_state[state.VIEW] = views[0]
    view = st.radio("View", views, horizontal=True, key=state.VIEW,
                    label_visibility="collapsed")

    regions = cr.regions
    highlight = NO_HIGHLIGHT
    if regions:
        options = [NO_HIGHLIGHT] + [f"Region {r.id}" for r in regions]
        highlight = st.selectbox("Highlight a detected region", options,
                                 key="eg_highlight")

    image, caption = _view_image(view, before, after, mask, prob, comparison)
    if highlight != NO_HIGHLIGHT:
        rid = int(highlight.split()[1])
        box = next((r.bbox_xywh for r in regions if r.id == rid), None)
        if box is not None:
            image = draw_bboxes(image, [box])
            caption = f"{caption} - region {rid} highlighted"

    st.image(image, caption=caption, use_container_width=True)

    st.write("")
    st.divider()

    # ------------------------------------------------------------- summary
    st.markdown("#### Summary")
    q = cr.quantities
    layer = cr.primary_layer
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Changed area", f"{q.changed_percentage:.2f} %")
    m2.metric("Changed pixels", f"{q.changed_pixels:,}")
    m3.metric("Detected regions", len(regions))
    m4.metric("Mean confidence", f"{layer.mean_confidence:.3f}")

    if q.area_m2 is not None:
        st.metric("Changed area (m2)", f"{q.area_m2:,.0f}")
    else:
        theme.note(
            "Ground area in m&sup2; is not reported: this imagery carries no "
            "georeferencing or ground sample distance, so any figure would be "
            "invented.")

    if comparison is not None:
        theme.note(
            f"Ground truth is available for this example. Against it the "
            f"prediction scores <strong>F1 {comparison.f1:.3f}</strong> "
            f"({comparison.tp:,} correct, {comparison.fp:,} false positive, "
            f"{comparison.fn:,} missed pixels).")

    for w in cr.warnings:
        st.warning(w)

    # ------------------------------------------------------- region details
    if regions:
        st.write("")
        st.markdown("#### Region details")
        rows = [{"Region": r.id,
                 "Area (px)": r.area_px,
                 "X": r.bbox_xywh[0], "Y": r.bbox_xywh[1],
                 "Width": r.bbox_xywh[2], "Height": r.bbox_xywh[3],
                 "Centroid X": r.centroid_xy[0], "Centroid Y": r.centroid_xy[1]}
                for r in regions]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        theme.note(
            "Regions are connected components of the change mask. A region is "
            "an area of detected change, not an identified building.")

    # --------------------------------------------------------- model / about
    st.write("")
    with st.expander("About this analysis"):
        _about(cr)

    st.download_button(
        "Download result (JSON)", json.dumps(cr.to_dict(), indent=2),
        file_name="earth_guardian_result.json", mime="application/json")


def _view_image(view, before, after, mask, prob, comparison):
    if view == "Change mask":
        return (mask * 255).astype(np.uint8), "Predicted change mask"
    if view == "Probability map":
        return (prob * 255).astype(np.uint8), "Change probability (brighter = more likely)"
    if view == "Error map" and comparison is not None:
        return comparison.error_map, comparison.caption
    return overlay(after, mask), "Detected change over the after image"


def _about(cr):
    md = services.get_engine().metadata
    prov = cr.provenance

    st.markdown(f"**{md.display_name}** - {md.task}")
    rows = {
        "Model": f"{md.name} v{md.version}",
        "Architecture": md.description,
        "Training dataset": prov.dataset,
        "Decision threshold": prov.threshold,
        "Detects": ", ".join(md.capabilities),
        "Result schema": f"v{cr.schema_version}",
        "Analysis time": f"{cr.runtime_seconds}s",
    }
    st.table(pd.DataFrame({"": list(rows), " ": [str(v) for v in rows.values()]}))

    if md.limitations:
        st.markdown("**This analysis does not do:**")
        st.markdown("\n".join(f"- {item}" for item in md.limitations))

    env = prov.operating_envelope
    if env:
        bits = []
        if env.gsd_m_range:
            bits.append(f"validated at {env.gsd_m_range[0]}-{env.gsd_m_range[1]} m/px")
        if env.regions_validated:
            bits.append("validated on " + ", ".join(env.regions_validated))
        if bits:
            theme.note("Operating envelope: " + "; ".join(bits) + ".")
        if env.notes:
            theme.note(env.notes)
    if prov.evaluation_protocol:
        theme.note(f"Evaluation protocol: {prov.evaluation_protocol}")
