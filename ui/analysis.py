"""Screen 2 - Input and analysis setup.

Two input paths - a bundled example or your own pair - then one explicit
Analyze action. Inputs are validated before the engine is ever called, and the
engine is the only thing that performs inference.

Domain awareness
----------------
The screen reads the selected domain from state and asks the services layer for
that engine. Reading the pair is delegated to services.prepare_pair(), which
dispatches on the domain: RGB images for built_environment, six-band Sentinel-2
surface reflectance for environment. No band handling, normalisation or
model-specific logic lives here.

Model input vs display
----------------------
Two different things, deliberately never mixed:

    PreparedPair.before / after            what the engine reads
    PreparedPair.preview_before / _after   what a person sees

Only the preview arrays are rendered or stored. For an RGB domain the two are
the same array; for the environment domain the preview is a true-colour
composite the domain itself renders from the six bands.

GeoTIFF pairs are read for metadata only (CRS, geotransform, pixel size). When
that metadata supports it, the analysis gains ground area in m2. Nothing is
reprojected, resampled or registered.
"""
import numpy as np
import streamlit as st
from PIL import Image

from src import config
from src.common import examples as example_catalogue
from ui import domains as domain_profiles
from ui import services, state, theme


def analyze_options(engine):
    """Domain-aware keyword arguments for engine.analyze().

    Direction classification is an optional, built-environment-only capability.
    The engine declares whether it has one (`supports_direction`), so the
    product opts in for the domain that does and never passes the argument to a
    domain that does not - whose analyze() has no such parameter at all.
    """
    if getattr(engine, "supports_direction", False):
        return {"with_direction": True}
    return {}


def _preview(before, after, labels):
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        theme.label(labels[0])
        st.image(before, use_container_width=True)
    with c2:
        theme.label(labels[1])
        st.image(after, use_container_width=True)


def _example_input(md):
    catalogue = services.list_examples(md.domain)
    if not catalogue:
        st.warning(
            "No bundled examples are available on this machine for this "
            "monitor. Run `python scripts/build_example_catalogue.py`, or "
            "upload your own pair.")
        return None, None, None, None, None

    names = [e["name"] for e in catalogue]
    picked = st.selectbox("Example", names, label_visibility="collapsed",
                          key=f"eg_example_{md.domain}")
    entry = catalogue[names.index(picked)]
    ex = services.example_from_dict(entry)

    theme.note(f"{ex.description} &nbsp;&middot;&nbsp; source: {ex.source}")
    st.write("")

    ground_truth = ((ex.ground_truth_path, ex.ground_truth_kind)
                    if ex.has_ground_truth else None)
    # Sources, not opened images: the domain's adapter decides how to read them.
    # `ex.georef` is None unless the example itself declares a real metric
    # scale, so no ground area is ever invented for a demo pair.
    return ex.before_path, ex.after_path, ground_truth, ex.name, ex.georef


def _upload_input(domain):
    profile = domain_profiles.profile(domain)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        theme.label(profile["upload_label_before"])
        up_before = st.file_uploader(
            profile["upload_before"], type=profile["file_types"],
            key=f"eg_up_before_{domain}", label_visibility="collapsed")
    with c2:
        theme.label(profile["upload_label_after"])
        up_after = st.file_uploader(
            profile["upload_after"], type=profile["file_types"],
            key=f"eg_up_after_{domain}", label_visibility="collapsed")

    if not (up_before and up_after):
        theme.note(profile["upload_hint"])
        return None, None, None, None, None

    return (up_before, up_after, None,
            f"{up_before.name} / {up_after.name}", None)


def render():
    domain = state.domain()
    if not _engine_ok(domain):
        return

    engine = services.get_engine(domain)
    md = engine.metadata
    profile = domain_profiles.profile(domain)

    top = st.columns([1, 6])
    with top[0]:
        if st.button("< Home", use_container_width=True):
            state.go(state.HOME)
            st.rerun()

    st.markdown(f"### {md.display_name} Monitor")

    # ---- what this monitor needs, BEFORE any input is chosen ----------------
    # The sentence is this module's copy; the band list inside it comes from the
    # engine's own declared InputSpec, so the two cannot drift apart.
    # `profile["purpose"]` (what this monitor finds) is deliberately not
    # repeated here - the Home card the user just clicked already said it, and
    # the requirements strip below is what they need to act on next.
    theme.requirements(
        domain_profiles.requirement_text(domain, md.input_spec),
        profile["requires_note"])

    # ---- choose an input ----------------------------------------------------
    modes = [profile["mode_example"], profile["mode_own"]]
    mode = st.radio("Input", modes, horizontal=True,
                    label_visibility="collapsed", key=f"eg_mode_{domain}")

    if mode == profile["mode_example"]:
        before, after, ground_truth, source, example_georef = _example_input(md)
    else:
        before, after, ground_truth, source, example_georef = _upload_input(domain)

    if before is None or after is None:
        return

    # ---- domain-aware preparation, before the engine is ever called ----------
    # Unchanged: the adapter decides, and a refusal keeps its exact wording.
    pair = services.prepare_pair(domain, before, after, md.input_spec)
    for message in pair.errors:
        st.error(message)
    for message in pair.warnings:
        st.warning(message)

    if not pair.ok:
        _offer_examples(md.domain)
        return

    # ---- the evidence -------------------------------------------------------
    # Only ever the preview arrays; the model input is not displayable in
    # general and is never rendered.
    _preview(pair.preview_before, pair.preview_after,
             (profile["upload_label_before"], profile["upload_label_after"]))
    st.write("")

    # ---- the primary action -------------------------------------------------
    # Reserved here and filled below, so Analyze sits directly under the preview
    # while the optional controls it reads stay beneath it.
    analyze_slot = st.container()

    with st.expander("Advanced"):
        threshold = st.slider(
            "Decision threshold", 0.05, 0.95, float(engine.threshold), 0.01,
            key=f"eg_threshold_{domain}",
            help="Default is the value selected on the validation split.")
        min_area = st.slider(
            "Minimum region size (pixels)", 0, 512, config.DEFAULT_MIN_AREA_PX, 8,
            key=f"eg_minarea_{domain}",
            help="Connected components smaller than this are discarded as noise.")
        theme.note(profile["details"])
        if pair.notes:
            theme.note(pair.notes)

    with analyze_slot:
        if st.button("Analyze change", type="primary", use_container_width=True):
            # A georef the EXAMPLE declares wins only where the adapter found
            # none: .npy arrays carry no georeferencing of their own, and the
            # catalogue records the pixel size the frozen source states.
            _run_analysis(engine, pair, ground_truth, source, threshold,
                          min_area, pair.georef or example_georef)


def _offer_examples(domain):
    """After a refusal, point at the bundled examples - if there are any.

    Purely additive: the refusal message above is produced by the domain and is
    not altered, and nothing here touches validation.
    """
    try:
        available = services.list_examples(domain)
    except Exception:                                            # noqa: BLE001
        return
    if available:
        theme.note(
            f"You can try one of the {len(available)} bundled examples for this "
            f"monitor instead - switch the selector above to "
            f"&ldquo;{domain_profiles.profile(domain)['mode_example']}&rdquo;.")


def _engine_ok(domain):
    try:
        services.get_engine(domain)
        return True
    except FileNotFoundError as e:
        # The engine reports which checkpoint it looked for; the UI does not
        # hold a filename of its own.
        st.error(f"The analysis engine could not load its model. {e}")
    except Exception as e:
        st.error(f"The analysis engine could not be loaded: {e}")
    return False


def _load_ground_truth(ground_truth):
    """Optional ground-truth mask for an example, as a boolean array.

    `ground_truth` is (path, kind) from the catalogue. The declared kind is
    authoritative; the file extension is only a fallback for a schema 1.0 entry
    written before the field existed.
    """
    if not ground_truth:
        return None
    path, kind = ground_truth
    if not path:
        return None
    if kind is None:
        kind = (example_catalogue.GT_NPY_MASK
                if str(path).lower().endswith(".npy")
                else example_catalogue.GT_PNG_MASK)
    try:
        if kind == example_catalogue.GT_NPY_MASK:
            return np.load(path) > 0.5
        return np.array(Image.open(path).convert("L")) > 127
    except Exception:                                            # noqa: BLE001
        return None


def _run_analysis(engine, pair, ground_truth, source, threshold, min_area,
                  georef=None):
    """The one place inference is triggered."""
    try:
        with st.spinner(domain_profiles.profile(engine.domain)["spinner"]):
            # The model reads pair.before / pair.after - never the preview.
            # The engine default stays with_direction=False, so every other
            # consumer (CLI, tests) is unaffected; see analyze_options().
            out = engine.analyze(pair.before, pair.after,
                                 threshold=threshold, min_area_px=min_area,
                                 georef=georef if georef is not None else pair.georef,
                                 **analyze_options(engine))
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

    display_before, display_after = pair.preview_before, pair.preview_after
    state.set_analysis({
        "domain": engine.domain,
        "change_result": out["change_result"],
        "mask": out["mask"],
        "probability": out["probability"],
        # The ONLY arrays a screen may render. The model input - six-band and
        # not displayable for the environment domain - is deliberately absent
        # from the payload, so there is nothing for a screen to render wrongly.
        "display_before": display_before,
        "display_after": display_after,
        "ground_truth": _load_ground_truth(ground_truth),
        # Metadata snapshot so Results never has to ask the registry which
        # engine ran - it would otherwise fetch the DEFAULT one.
        "metadata": engine.metadata,
        "source": source,
    })
    state.go(state.RESULTS)
    st.rerun()
