"""Screen 1 - Home.

A short landing screen: what Earth Guardian is, which monitors work today, and
an honest note about the ones that do not exist yet.

Which monitors are selectable comes from services.available_domains(), which
reads the engine registry. This screen holds no list of implemented domains of
its own and no per-domain branching: a monitor appears here because an engine is
registered for it, its copy comes from the presentation profile, its input
requirement from the model card it declares, and its example count from the
catalogue. A planned monitor disappears from the unavailable row the moment an
engine exists for it.
"""
import streamlit as st

from ui import domains as domain_profiles
from ui import services, state, theme


def _examples_line(name):
    """"N examples included", counted from the catalogue. Silent when there are none."""
    try:
        count = len(services.list_examples(name))
    except Exception:                                            # noqa: BLE001
        return ""
    if not count:
        return ""
    return f"{count} example{'s' if count != 1 else ''} included"


def _available_card(name):
    """One card for a registered monitor: finds, needs, extra, then the CTA."""
    card = services.domain_card(name)
    profile = domain_profiles.profile(name)

    parts = [
        "<div class='eg-card'>",
        "<div class='eg-badge eg-live'>Available now</div>",
        f"<h3>{card['display_name']}</h3>",
        f"<p>{profile['card_finds']}</p>",
        "<div class='eg-card-needs'><span class='eg-card-key'>Needs</span>",
        f"{domain_profiles.card_needs_text(name, card.get('bands'))}</div>",
    ]
    if profile["card_extra"]:
        parts.append(f"<div class='eg-card-extra'>{profile['card_extra']}</div>")
    examples = _examples_line(name)
    if examples:
        parts.append(f"<div class='eg-card-meta'>{examples}</div>")
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def _planned_card(title, blurb):
    """A monitor with no engine: visibly subdued, and with no action at all."""
    st.markdown(
        "<div class='eg-card eg-soon'>"
        "<div class='eg-badge'>Coming later</div>"
        f"<h3>{title}</h3><p>{blurb}</p></div>",
        unsafe_allow_html=True)


def render():
    st.markdown(
        "<div class='eg-hero'><h1>&#127757; Earth Guardian</h1></div>"
        "<div class='eg-tagline'>Understand what changed on Earth.<br>"
        "<span class='eg-tagline-sub'>Compare the same place at two different "
        "times to find meaningful change.</span></div>",
        unsafe_allow_html=True)

    available = services.available_domains()
    # Planned monitors are only shown while they genuinely have no engine.
    planned = [p for p in domain_profiles.PLANNED if p[0] not in available]

    columns = st.columns(max(len(available) + len(planned), 1), gap="medium")
    index = 0

    for name in available:
        with columns[index]:
            _available_card(name)
            st.write("")
            if st.button("Start analysis", type="primary",
                         use_container_width=True, key=f"eg_start_{name}"):
                state.set_domain(name)
                state.go(state.WORKSPACE)
                st.rerun()
        index += 1

    for _name, title, blurb in planned:
        with columns[index]:
            _planned_card(title, blurb)
        index += 1

    st.write("")
    st.divider()

    theme.note(
        "<strong>What this version does.</strong> You choose a monitor, supply "
        "two images of the same area taken at different times, and the engine "
        "returns a pixel-level map of where change occurred, how much changed, "
        "and how confident it is. Each monitor states the imagery it requires. "
        "Selecting a location and dates, and having imagery fetched for you, is "
        "a later stage of the roadmap.")
