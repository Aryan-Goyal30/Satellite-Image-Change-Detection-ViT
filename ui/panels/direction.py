"""Direction panel - construction / demolition, when a result carries it.

Moved out of ui/results.py in UI Phase 3. Previously the direction section and
its "Direction classifier" About block rendered unconditionally, so a forest-loss
result displayed a direction heading and the line "Direction classification did
not run for this result." Direction is a built-environment capability and that
vocabulary has no meaning for another domain.

Mounting rule
-------------
`applies(cr)` is the single gate, and it asks the RESULT, never the domain: a
panel appears when the result carries direction provenance, a classified region,
or the engine's direction-unavailable warning. No domain name appears in this
module or in the code that mounts it.

Everything here concerns a SEPARATE model from the change detector - different
architecture, different dataset, different evaluation - and the wording is taken
from that model's card rather than restated.
"""
import pandas as pd
import streamlit as st

from src.domains.built_environment.direction import model_card as direction_card
from ui import theme

#: Region.direction value -> label shown to a user. Nothing else is displayable;
#: an unknown value falls through to the em dash rather than being invented.
DIRECTION_LABELS = {
    "construction": "Construction",
    "demolition": "Demolition",
    "uncertain": "Uncertain",
}
NO_DIRECTION = "—"          # em dash: "not classified", never a class

#: Always visible next to the direction results. Short on purpose - the fuller
#: text lives in the expander below it.
DIRECTION_CAVEAT = (
    "Direction is a <strong>separate model</strong> evaluated on S2Looking "
    "ground-truth regions. Its benchmark figures do <strong>not</strong> "
    "represent end-to-end performance on regions detected from this imagery. "
    "Direction scores are <strong>not calibrated probabilities</strong>, and "
    "mixed or ambiguous changes may be reported as Uncertain."
)

#: The domain gap, stated plainly.
DIRECTION_DOMAIN_GAP = (
    "S2Looking and LEVIR-CD differ in scene and domain characteristics; transfer "
    "performance of the direction classifier to LEVIR-style imagery has not been "
    "established."
)

DIRECTION_WARNING_PREFIX = "Direction analysis unavailable"


def direction_label(value):
    """Readable label for a Region.direction value."""
    if value is None:
        return NO_DIRECTION
    return DIRECTION_LABELS.get(value, NO_DIRECTION)


def direction_counts(regions):
    """Counts per direction, or None when no region carries one.

    Returns None rather than a dict of zeros so the UI can stay silent instead
    of implying the classifier ran and found nothing.
    """
    classified = [r for r in regions if r.direction is not None]
    if not classified:
        return None
    return {key: sum(1 for r in classified if r.direction == key)
            for key in DIRECTION_LABELS}


def has_direction_warning(warnings):
    """True when direction analysis was requested but could not be completed."""
    return any(str(w).startswith(DIRECTION_WARNING_PREFIX) for w in warnings)


def applies(cr):
    """Does this result carry direction information of any kind?

    The ONLY gate for every direction element on the page. A result without
    direction provenance, without a classified region and without the
    unavailable warning renders no direction vocabulary at all.
    """
    if (cr.params or {}).get("direction"):
        return True
    if any(r.direction is not None for r in cr.regions):
        return True
    return has_direction_warning(cr.warnings)


def has_direction_columns(regions):
    """True when a region table should carry direction columns."""
    return any(r.direction is not None for r in regions)


def region_note():
    """The region-table footnote for direction columns."""
    return (" <strong>Direction score</strong> is a different model's "
            "decision score for the predicted direction. It is "
            "<strong>not a calibrated probability</strong>, and it is "
            "unrelated to detector confidence. "
            f"{NO_DIRECTION} means the region was not classified.")


def render_summary(cr):
    """Compact direction counts plus the required caveat.

    Shows nothing at all when no region carries a direction, so a failed or
    disabled run never renders as zero counts.
    """
    if not applies(cr):
        return

    if has_direction_warning(cr.warnings):
        st.warning(
            "Direction analysis was not completed for this image pair, so no "
            "construction/demolition result is shown below. The change "
            "detection results above are unaffected.")
        return

    counts = direction_counts(cr.regions)
    if not counts:
        return

    st.markdown(
        "**Direction** &nbsp; "
        f"{counts['construction']} Construction &nbsp;&middot;&nbsp; "
        f"{counts['demolition']} Demolition &nbsp;&middot;&nbsp; "
        f"{counts['uncertain']} Uncertain",
        unsafe_allow_html=True)
    theme.note(DIRECTION_CAVEAT)

    with st.expander("About direction classification"):
        st.markdown(f"- {direction_card.EVALUATION_SCOPE}")
        st.markdown(f"- {direction_card.NOT_CALIBRATED}")
        st.markdown(f"- {DIRECTION_DOMAIN_GAP}")
        st.markdown(f"- {direction_card.ABSTENTION}")
        st.markdown(
            f"- Temporal convention: **{direction_card.TEMPORAL_ORDERING['before_image_dir']} "
            f"= BEFORE**, **{direction_card.TEMPORAL_ORDERING['after_image_dir']} = AFTER** "
            "in the training data. This ordering is a project-level assumption, "
            "**not documented by the dataset authors**.")


def render_about(cr):
    """Direction classifier identity, kept visibly separate from the detector.

    Reads the provenance the engine actually recorded for this result, so the
    checkpoint hash and threshold shown are the ones that ran.
    """
    if not applies(cr):
        return

    prov = (cr.params or {}).get("direction")
    st.write("")
    st.markdown("**Direction classifier** - separate model, separate dataset")

    if not prov:
        theme.note("Direction analysis was not completed for this result.")
        return

    crop = prov.get("crop_protocol", {})
    ordering = prov.get("temporal_ordering", {})
    rows = {
        "Model": f"{prov.get('model')} v{prov.get('version')}",
        "Architecture": prov.get("architecture"),
        "Training dataset": prov.get("dataset"),
        "Decision threshold": f"{prov.get('threshold')} "
                              f"(selected on {prov.get('threshold_selected_on')})",
        "Classes": ", ".join(prov.get("classes", []))
                   + f" (+ {prov.get('abstain_label')} below threshold)",
        "Region crop": f"{crop.get('crop_size')}px, "
                       f"{crop.get('context_fraction')} context",
        "Temporal convention": f"{ordering.get('before_image_dir')} = BEFORE, "
                               f"{ordering.get('after_image_dir')} = AFTER",
        "Weights hash": (prov.get("weights_hash") or "-")[:16] + "...",
    }
    st.table(pd.DataFrame({"": list(rows), " ": [str(v) for v in rows.values()]}))

    theme.note(
        "<strong>Evaluation scope:</strong> " + prov.get("evaluation_scope", "")
        + " Benchmark figures for this model are measured on S2Looking "
          "ground-truth regions and are <strong>not</strong> this product's "
          "end-to-end accuracy on the imagery above.")
    theme.note("<strong>Direction score is not a calibrated probability.</strong> "
               + prov.get("notes", ""))
    theme.note(DIRECTION_DOMAIN_GAP)
    theme.note(
        "The temporal ordering above is a <strong>project-level assumption</strong>, "
        "not documented by the dataset authors "
        f"(documented_by_authors = {prov.get('documented_by_authors')}).")
