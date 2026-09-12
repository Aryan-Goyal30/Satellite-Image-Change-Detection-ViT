"""Screen 2 - Input and analysis setup.

Two input paths - a bundled example or your own pair - then one explicit
Analyze action. Inputs are validated before the engine is ever called, and the
engine is the only thing that performs inference.

GeoTIFF pairs are read for metadata only (CRS, geotransform, pixel size). When
that metadata supports it, the analysis gains ground area in m2. Nothing is
reprojected, resampled or registered.
"""
import numpy as np
import streamlit as st
from PIL import Image

from src import config
from src.common.image_input import describe_georef, open_image, to_rgb, validate_pair
from ui import services, state, theme


def _preview(before, after):
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        theme.label("Before")
        st.image(before, use_container_width=True)
    with c2:
        theme.label("After")
        st.image(after, use_container_width=True)


def _example_input(md):
    catalogue = services.list_examples(md.domain)
    if not catalogue:
        st.warning(
            "No bundled examples are available on this machine. Run "
            "`python scripts/build_example_catalogue.py`, or upload your own pair.")
        return None, None, None, None

    names = [e["name"] for e in catalogue]
    picked = st.selectbox("Example", names, label_visibility="collapsed")
    entry = catalogue[names.index(picked)]
    ex = services.example_from_dict(entry)

    theme.note(f"{ex.description} &nbsp;&middot;&nbsp; source: {ex.source}")
    st.write("")

    try:
        before = open_image(ex.before_path)
        after = open_image(ex.after_path)
    except ValueError as e:
        st.error(str(e))
        return None, None, None, None

    gt_path = ex.ground_truth_path if ex.has_ground_truth else None
    return before, after, gt_path, ex.name


def _upload_input():
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        theme.label("Before image")
        up_before = st.file_uploader(
            "Drag and drop the earlier image", type=["png", "jpg", "jpeg", "tif", "tiff"],
            key="eg_up_before", label_visibility="collapsed")
    with c2:
        theme.label("After image")
        up_after = st.file_uploader(
            "Drag and drop the later image", type=["png", "jpg", "jpeg", "tif", "tiff"],
            key="eg_up_after", label_visibility="collapsed")

    if not (up_before and up_after):
        theme.note("Upload both images to continue.")
        return None, None, None, None

    try:
        before = open_image(up_before)
        after = open_image(up_after)
    except ValueError as e:
        st.error(str(e))
        return None, None, None, None

    label = f"{up_before.name} / {up_after.name}"
    return before, after, None, label


def render():
    md = services.get_engine().metadata if _engine_ok() else None
    if md is None:
        return

    engine = services.get_engine()

    top = st.columns([1, 6])
    with top[0]:
        if st.button("< Home", use_container_width=True):
            state.go(state.HOME)
            st.rerun()

    st.markdown(f"### {md.display_name} Monitor")
    theme.note(f"{md.task}. Supply two images of the same area at different dates.")
    st.write("")

    mode = st.radio("Input", ["Try an example", "Upload your own images"],
                    horizontal=True, label_visibility="collapsed")

    if mode == "Try an example":
        before, after, gt_path, source = _example_input(md)
    else:
        before, after, gt_path, source = _upload_input()

    spec = md.input_spec
    theme.note(
        f"This engine expects paired optical {spec.channel_order} imagery, both "
        f"images covering the same area at the same size. Best results require "
        f"images that are reasonably aligned and comparable. Multispectral and "
        f"SAR products are not supported. Georeferenced GeoTIFFs additionally "
        f"report ground area in m&sup2;.")

    if before is None or after is None:
        return

    st.write("")
    _preview(before, after)
    st.write("")

    # ---- validation before the engine is ever called ----
    report = validate_pair(before, after, spec)
    for issue in report.errors:
        st.error(issue.message)
    for issue in report.warnings:
        st.warning(issue.message)
    theme.note(describe_georef(report))

    with st.expander("Advanced"):
        threshold = st.slider(
            "Decision threshold", 0.05, 0.95, float(engine.threshold), 0.01,
            help="Default is the value selected on the validation split.")
        min_area = st.slider(
            "Minimum region size (pixels)", 0, 512, config.DEFAULT_MIN_AREA_PX, 8,
            help="Connected components smaller than this are discarded as noise.")

    st.write("")
    if st.button("Analyze change", type="primary", use_container_width=True,
                 disabled=not report.ok):
        _run_analysis(engine, before, after, gt_path, source, threshold, min_area,
                      report.georef)


def _engine_ok():
    try:
        services.get_engine()
        return True
    except FileNotFoundError as e:
        # The engine reports which checkpoint it looked for; the UI does not
        # hold a filename of its own.
        st.error(f"The analysis engine could not load its model. {e}")
    except Exception as e:
        st.error(f"The analysis engine could not be loaded: {e}")
    return False


def _run_analysis(engine, before, after, gt_path, source, threshold, min_area,
                  georef=None):
    """The one place inference is triggered."""
    try:
        with st.spinner("Analyzing change..."):
            out = engine.analyze(to_rgb(before), to_rgb(after),
                                 threshold=threshold, min_area_px=min_area,
                                 georef=georef)
        state.count_inference()
    except ValueError as e:
        st.error(str(e))
        return
    except MemoryError:
        st.error("The images are too large to analyze on this machine. "
                 "Try a smaller crop.")
        return
    except Exception:
        st.error("The analysis could not be completed. Check that both images "
                 "are valid, the same size, and cover the same area.")
        return

    gt_mask = None
    if gt_path:
        try:
            gt_mask = np.array(Image.open(gt_path).convert("L")) > 127
        except Exception:
            gt_mask = None

    state.set_analysis({
        "change_result": out["change_result"],
        "mask": out["mask"],
        "probability": out["probability"],
        "before": out["before"],
        "after": out["after"],
        "ground_truth": gt_mask,
        "source": source,
    })
    state.go(state.RESULTS)
    st.rerun()
