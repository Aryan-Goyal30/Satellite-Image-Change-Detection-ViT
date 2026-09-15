"""Screen 3 - Change analysis results.

Everything here reads from the ChangeResult produced by the Analyze action.
Switching views, highlighting a region or opening a section never re-runs the
network: the result already lives in session state.

Visual hierarchy: before/after, then detected change, then the summary, then
region details, then model information.

One page, two domains
---------------------
There is deliberately no per-domain Results page. The structure is driven by the
ChangeResult, and optional sections are mounted by asking the RESULT what it
carries - a ground-truth mask, direction provenance, a metric georeference, a
calibration claim - never by testing a domain name. The only place a domain name
appears on this screen is as a label the payload supplied.

Display arrays
--------------
The screen renders `display_before` / `display_after` from the payload and
nothing else. For the environment domain the model input is a six-band
reflectance array that cannot be displayed at all, so rendering anything else
would be a bug rather than a style choice.

Self-describing
---------------
Domain and engine metadata come from the stored payload. Results never asks the
registry which engine ran - it would get the DEFAULT one, and describe a
forest-loss result as a building detector.
"""
import json

import numpy as np
import pandas as pd
import streamlit as st

from src.common.visualization import compare_to_ground_truth, draw_bboxes, overlay
from ui import state, theme
from ui.panels import direction as direction_panel

NO_HIGHLIGHT = "None"

# Direction rendering lives in ui/panels/direction.py. These names are
# re-exported because they are part of this module's established surface.
from ui.panels.direction import (  # noqa: E402,F401
    DIRECTION_CAVEAT,
    DIRECTION_DOMAIN_GAP,
    DIRECTION_LABELS,
    DIRECTION_WARNING_PREFIX,
    NO_DIRECTION,
    direction_card,
    direction_counts,
    direction_label,
    has_direction_warning,
)

#: View names, in plain language. "Signal strength" is deliberately the same
#: label for every domain: it describes what the map shows without asserting
#: that the values are probabilities. Whether they may be read as probabilities
#: is stated in the caption and in About, driven by provenance - see
#: score_noun() and UNCALIBRATED_NOTE.
VIEW_OVERLAY = "Overlay"
VIEW_MASK = "Change only"
VIEW_SIGNAL = "Signal strength"
VIEW_ERROR = "vs Ground truth"

UNCALIBRATED_NOTE = (
    "Scores on this page are <strong>uncalibrated model scores</strong>, not "
    "probabilities: a score of 0.9 does not mean a 90% chance of change. The "
    "decision threshold is an operating point chosen on a validation split.")


def scores_are_uncalibrated(cr):
    """True only when the engine explicitly declares its scores uncalibrated.

    None means the engine made no claim either way, and is treated as it was
    before the field existed - so an engine that never declared calibration
    keeps its original wording.
    """
    return cr.provenance.score_calibrated is False


def score_noun(cr):
    """What to call the model's 0-1 output for this result."""
    return "score" if scores_are_uncalibrated(cr) else "probability"


# ------------------------------------------------------------------ verdict
def change_noun(cr):
    """Plain-language name for what this result detected.

    Taken from the LAYER the domain declared - model_card.LAYER_NAME, which
    already travels inside every ChangeResult - rather than from a domain name:

        "structural_change"  ->  "Structural change"
        "forest_loss"        ->  "Forest loss"

    So Results needs no domain check, and a domain added later gets a correct
    sentence without this module changing at all.
    """
    name = (cr.primary_layer.name or "").replace("_", " ").strip()
    return (name[:1].upper() + name[1:]) if name else "Change"


def verdict(cr):
    """One sentence answering "what changed?", computed from the result.

    The percentage is the result's own changed_percentage, formatted exactly as
    the Changed-area metric formats it, so the two can never appear to disagree.
    An empty mask says so plainly instead of reporting 0%.
    """
    noun = change_noun(cr)
    quantities = cr.quantities
    if quantities.changed_pixels == 0:
        return f"No {noun.lower()} detected in the analyzed area."
    return (f"{noun} detected across {quantities.changed_percentage:.2f}% "
            f"of the analyzed area.")


def render():
    payload = state.analysis()
    if not payload:
        state.go(state.WORKSPACE)
        st.rerun()
        return

    cr = payload["change_result"]
    mask = payload["mask"]
    prob = payload["probability"]
    # Display arrays only. `before`/`after` are NOT read: the environment
    # domain's model input is six-band and is not renderable.
    before = payload.get("display_before")
    after = payload.get("display_after")
    gt = payload.get("ground_truth")
    md = payload.get("metadata")

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
    # The answer first: one sentence, computed from this result's own
    # quantities, before any image or metric.
    theme.verdict(verdict(cr))
    # The result says which monitor produced it, from the stored payload.
    caption = " &nbsp;&middot;&nbsp; ".join(
        part for part in ((f"{md.display_name} Monitor" if md else None),
                          payload.get("source")) if part)
    if caption:
        theme.note(caption)

    # ------------------------------------------- where: primary visualisation
    # Reserved now and filled below, so the change map sits directly under the
    # verdict while the controls that drive it stay beneath the numbers. The
    # image itself is produced by the existing _view_image()/overlay path,
    # unchanged.
    image_slot = st.container()

    # ------------------------------------------------------- how much: metrics
    q = cr.quantities
    regions = cr.regions

    theme.hero_metric(f"{q.changed_percentage:.2f}%", "of the analyzed area changed")

    secondary = [("Detected regions", f"{len(regions):,}")]
    if q.area_m2 is not None:
        secondary.append(("Changed ground area", f"{q.area_m2:,.0f} m2"))
        if cr.georef is not None and cr.georef.gsd_m:
            secondary.append(("Pixel size", f"{cr.georef.gsd_m:g} m"))
    columns = st.columns(max(len(secondary), 2))
    for column, (label_text, value) in zip(columns, secondary):
        column.metric(label_text, value)

    _geospatial_note(cr, q)
    if scores_are_uncalibrated(cr):
        theme.note(UNCALIBRATED_NOTE)

    # ------------------------------------------------------------- controls
    views = [VIEW_OVERLAY, VIEW_MASK, VIEW_SIGNAL]
    comparison = None
    if gt is not None and gt.shape == mask.shape:
        comparison = compare_to_ground_truth(cr, gt)
        views.append(VIEW_ERROR)

    if state.VIEW in st.session_state and st.session_state[state.VIEW] not in views:
        st.session_state[state.VIEW] = views[0]
    view = st.radio("View", views, horizontal=True, key=state.VIEW,
                    label_visibility="collapsed")

    highlight = NO_HIGHLIGHT
    if regions:
        options = [NO_HIGHLIGHT] + [f"Region {r.id}" for r in regions]
        highlight = st.selectbox("Highlight a detected region", options,
                                 key="eg_highlight")

    image, image_caption = _view_image(view, cr, before, after, mask, prob,
                                       comparison)
    if highlight != NO_HIGHLIGHT:
        rid = int(highlight.split()[1])
        box = next((r.bbox_xywh for r in regions if r.id == rid), None)
        if box is not None:
            image = draw_bboxes(image, [box])
            image_caption = f"{image_caption} - region {rid} highlighted"
    with image_slot:
        st.image(image, caption=image_caption, use_container_width=True)

    st.write("")
    st.divider()

    # ------------------------------------------------ evidence: before / after
    st.markdown("#### The two dates")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        theme.label("Before")
        st.image(before, use_container_width=True)
    with c2:
        theme.label("After")
        st.image(after, use_container_width=True)

    # -------------------------------------------------------------- warnings
    # Never collapsed: these are the honesty mechanism, not an aside.
    for w in cr.warnings:
        st.warning(w)

    # ------------------------------------------------------- region details
    if regions:
        st.write("")
        with st.expander(f"Inspect regions ({len(regions):,})"):
            st.dataframe(_region_table(regions, detector_score_label(cr)),
                         use_container_width=True, hide_index=True)
            theme.note(_region_note(cr, regions))

    # Optional, and mounted by what the RESULT carries - not by domain.
    direction_panel.render_summary(cr)

    if comparison is not None:
        theme.note(
            f"Ground truth is available for this example. Against it the "
            f"prediction scores <strong>F1 {comparison.f1:.3f}</strong> "
            f"({comparison.tp:,} correct, {comparison.fp:,} false positive, "
            f"{comparison.fn:,} missed pixels).")

    # --------------------------------------------------------- model / about
    st.write("")
    with st.expander("About this analysis"):
        _about(cr, md)

    st.download_button(
        "Download result (JSON)", json.dumps(cr.to_dict(), indent=2),
        file_name="earth_guardian_result.json", mime="application/json")


def _region_note(cr, regions):
    """Footnote under the region table, worded for this result's claims."""
    note = ("Regions are connected components of the change mask. A region is "
            "an area of detected change, not an identified object.")
    if any(r.confidence is not None for r in regions):
        if scores_are_uncalibrated(cr):
            note += (f" <strong>{DETECTOR_SCORE_LABEL}</strong> is the mean model score "
                     "over that region's pixels. It is an uncalibrated score, "
                     "not a probability that the region is really a change.")
        else:
            note += (f" <strong>{DETECTOR_CONFIDENCE_LABEL}</strong> is the mean predicted "
                     "change probability over that region's pixels - a summary of "
                     "model score, not a calibrated per-region probability.")
    if direction_panel.has_direction_columns(regions):
        note += direction_panel.region_note()
    return note


#: Default header for the detector's per-region score. An engine that declares
#: its scores uncalibrated gets DETECTOR_SCORE_LABEL instead, so the column
#: header and the footnote under the table always say the same thing.
DETECTOR_CONFIDENCE_LABEL = "Detector confidence"
DETECTOR_SCORE_LABEL = "Detector score"


def detector_score_label(cr):
    return (DETECTOR_SCORE_LABEL if scores_are_uncalibrated(cr)
            else DETECTOR_CONFIDENCE_LABEL)


def _region_table(regions, score_label=DETECTOR_CONFIDENCE_LABEL):
    """Region table; optional columns appear only when the data supports them.

    "Direction score" and the detector's own score are deliberately separate
    columns from two different models, and are never merged or renamed into one
    another.
    """
    has_conf = any(r.confidence is not None for r in regions)
    has_area = any(r.area_m2 is not None for r in regions)
    has_crs = any(r.centroid_crs_xy is not None for r in regions)
    has_direction = direction_panel.has_direction_columns(regions)

    rows = []
    for r in regions:
        row = {"Region": r.id}
        if has_direction:
            row["Direction"] = direction_label(r.direction)
            row["Direction score"] = (r.direction_score
                                      if r.direction_score is not None
                                      else None)
        if has_conf:
            row[score_label] = r.confidence
        row["Area (px)"] = r.area_px
        if has_area:
            row["Area (m2)"] = r.area_m2
        row.update({"X": r.bbox_xywh[0], "Y": r.bbox_xywh[1],
                    "Width": r.bbox_xywh[2], "Height": r.bbox_xywh[3],
                    "Centroid X": r.centroid_xy[0], "Centroid Y": r.centroid_xy[1]})
        if has_crs and r.centroid_crs_xy is not None:
            row["Centroid E"] = r.centroid_crs_xy[0]
            row["Centroid N"] = r.centroid_crs_xy[1]
        rows.append(row)
    return pd.DataFrame(rows)


def _geospatial_note(cr, q):
    """What the geography does or does not support. The ground-area FIGURE is a
    secondary metric in render(); this explains it, or explains its absence."""
    georef = cr.georef
    if q.area_m2 is not None:
        if georef is not None and georef.crs:
            theme.note(f"Georeferenced input: {georef.crs}. Ground area is "
                       f"pixel count x pixel area; it is not a surveyed "
                       f"measurement and assumes the supplied geotransform.")
        return

    if georef is not None and georef.crs:
        theme.note(
            f"Georeferenced input ({georef.crs}), but ground area in m&sup2; is "
            f"not available: "
            f"{'the CRS is not metric' if georef.units == 'degrees' else 'no usable metric pixel size was found'}. "
            f"Results are reported in pixels.")
    else:
        theme.note(
            "Ground area in m&sup2; is not reported: this imagery carries no "
            "georeferencing or ground sample distance, so any figure would be "
            "invented. Supply a georeferenced GeoTIFF or a ground sample "
            "distance to obtain m&sup2;.")


def _view_image(view, cr, before, after, mask, prob, comparison):
    if view == VIEW_MASK:
        return (mask * 255).astype(np.uint8), "Predicted change mask"
    if view == VIEW_SIGNAL:
        noun = score_noun(cr)
        return ((prob * 255).astype(np.uint8),
                f"Change {noun} (brighter = higher {noun})")
    if view == VIEW_ERROR and comparison is not None:
        return comparison.error_map, comparison.caption
    return overlay(after, mask), "Detected change over the after image"


def _about(cr, md=None):
    """Identity of the model that produced THIS result.

    Reads the payload's metadata snapshot and the result's own provenance. It
    must not ask the registry for an engine: that returns the default one, which
    for another domain's result would describe the wrong model entirely.
    """
    prov = cr.provenance

    if md is not None:
        st.markdown(f"**{md.display_name}** - {md.task}")
    elif prov.task:
        st.markdown(f"**{prov.task}**")

    rows = {
        "Model": f"{prov.model} v{prov.version}" if prov.version else prov.model,
    }
    if md is not None and md.description:
        rows["Architecture"] = md.description
    rows["Training dataset"] = prov.dataset
    if prov.dataset_version:
        rows["Dataset version"] = prov.dataset_version
    bands = (cr.input_info or {}).get("bands")
    if bands:
        rows["Input bands"] = ", ".join(bands)
    rows["Decision threshold"] = prov.threshold
    if prov.score_calibrated is not None:
        rows["Scores calibrated"] = "Yes" if prov.score_calibrated else "No"
    rows["Minimum region size"] = f"{cr.params.get('min_area_px')} px"
    # Demoted from the headline metrics in UI Phase 5A/2, kept here for
    # technical inspection. Still present in the JSON download unchanged.
    rows["Changed pixels"] = (f"{cr.quantities.changed_pixels:,} of "
                              f"{cr.quantities.total_pixels:,}")
    # Same vocabulary as the region-table column: "confidence" for an engine
    # that makes no uncalibrated claim, "score" for one that does.
    rows[f"Mean {detector_score_label(cr).lower()}"] = (
        f"{cr.primary_layer.mean_confidence:.3f}")
    if md is not None and md.capabilities:
        rows["Detects"] = ", ".join(md.capabilities)
    rows["Result schema"] = f"v{cr.schema_version}"
    rows["Analysis time"] = f"{cr.runtime_seconds}s"

    st.table(pd.DataFrame({"": list(rows), " ": [str(v) for v in rows.values()]}))

    if scores_are_uncalibrated(cr):
        theme.note(UNCALIBRATED_NOTE)

    if md is not None and md.limitations:
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

    # Optional, and mounted by what the RESULT carries - not by domain.
    direction_panel.render_about(cr)
