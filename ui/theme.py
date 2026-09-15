"""Earth Guardian visual theme.

One restrained stylesheet: a dark Earth-observation ground, a single teal
accent, and typography that stays out of the way of the imagery. No gradients,
no animation, no decorative charts - the images are the product.
"""
import streamlit as st

INK = "#E6ECE6"
MUTED = "#8B9E96"
GROUND = "#0D1117"
PANEL = "#141A21"
LINE = "#263038"
ACCENT = "#63C8BF"
WARN = "#D9A85C"
DANGER = "#E2807F"

# Semantic colours shared with the figures so the product and the report agree.
TP_COLOR = "#3CC85A"
FP_COLOR = "#E64646"
FN_COLOR = "#4682EB"

_CSS = f"""
<style>
  .stApp {{ background: {GROUND}; }}
  .block-container {{ padding-top: 2.2rem; max-width: 1280px; }}

  /* Base text colour. The ground above is forced dark, so the text must be
     stated too rather than left to Streamlit's own light/dark detection - the
     project ships no .streamlit/config.toml, and a light base would render
     dark text on this dark ground. Every eg- class below still sets its own. */
  .stApp, .stMarkdown, .stMarkdown p, .stMarkdown li {{ color: {INK}; }}
  [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p {{ color: {INK}; }}
  [data-testid="stExpander"] summary {{ color: {INK}; }}
  [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{
      color: {MUTED};
  }}

  /* Headings. Streamlit's defaults are larger than this layout needs: an
     unstyled "###" section title would outweigh the verdict beneath it and
     inverts the hierarchy the result pages depend on. Declared BEFORE
     .eg-hero h1 so the hero keeps its own larger size. */
  .block-container h1 {{ font-size: 1.8rem; font-weight: 700; color: {INK}; }}
  .block-container h2 {{ font-size: 1.4rem; font-weight: 600; color: {INK}; }}
  .block-container h3 {{
      font-size: 1.15rem; font-weight: 600; color: {INK};
      letter-spacing: -0.01em; margin: 0 0 .25rem 0;
  }}
  .block-container h4 {{
      font-size: .95rem; font-weight: 600; color: {INK};
      margin: 1.1rem 0 .5rem 0;
  }}

  .eg-hero h1 {{
      font-size: 2.6rem; font-weight: 700; letter-spacing: -0.02em;
      color: {INK}; margin: 0 0 .35rem 0;
  }}
  .eg-tagline {{ font-size: 1.15rem; color: {MUTED}; margin-bottom: 2rem; }}
  .eg-tagline-sub {{ font-size: .95rem; color: {MUTED}; opacity: .85; }}

  .eg-card {{
      background: {PANEL}; border: 1px solid {LINE}; border-radius: 6px;
      padding: 1.25rem 1.4rem; height: 100%;
  }}
  .eg-card h3 {{
      color: {INK}; font-size: 1.15rem; font-weight: 700; margin: 0 0 .55rem 0;
  }}
  .eg-card p {{ color: {MUTED}; font-size: .92rem; margin: 0; line-height: 1.55; }}
  .eg-card.eg-soon {{ opacity: .55; }}

  /* Home card detail lines: what the monitor needs, what else it offers, and
     how many examples ship with it. Information does the visual work. Needs
     reads as INK (a requirement, not a remark); extra stays MUTED and a touch
     smaller, secondary to it - same tokens, ordered by the hierarchy the card
     already communicates. */
  .eg-card-needs, .eg-card-extra {{
      color: {MUTED}; font-size: .85rem; line-height: 1.55; margin-top: .75rem;
  }}
  .eg-card-needs {{ color: {INK}; }}
  .eg-card-extra {{ font-size: .82rem; opacity: .85; }}
  .eg-card-key {{
      display: block; color: {ACCENT}; font-size: .66rem; font-weight: 600;
      letter-spacing: .12em; text-transform: uppercase; margin-bottom: .2rem;
  }}
  .eg-card-meta {{
      color: {MUTED}; font-size: .72rem; letter-spacing: .06em;
      margin-top: .8rem;
  }}

  /* Equal-height cards with the CTA anchored to the same bottom edge. The
     card's element-container is the flex item that grows; the card fills it
     via its own height: 100% above, and the spacer + button that follow sit
     at their natural size, pushed to the bottom of the row. */
  [data-testid="stHorizontalBlock"] {{ align-items: stretch; }}
  [data-testid="stColumn"] {{ display: flex; }}
  [data-testid="stColumn"] > [data-testid="stVerticalBlock"] {{
      width: 100%; display: flex; flex-direction: column;
  }}
  [data-testid="stColumn"] [data-testid="stElementContainer"]:has(.eg-card) {{
      flex: 1 0 auto;
  }}

  /* The one CTA colour in the product: the existing accent, not a second one.
     Streamlit's own primary-button red is overridden everywhere a
     type="primary" button appears (Home's Start analysis, Analysis's Analyze
     change) so the app never shows two different "go" colours. */
  [data-testid="stBaseButton-primary"] {{
      background-color: {ACCENT}; border-color: {ACCENT}; color: {GROUND};
  }}
  [data-testid="stBaseButton-primary"]:hover,
  [data-testid="stBaseButton-primary"]:focus:not(:active) {{
      background-color: {ACCENT}; border-color: {ACCENT}; color: {GROUND};
  }}

  .eg-badge {{
      display: inline-block; font-size: .7rem; letter-spacing: .08em;
      text-transform: uppercase; padding: .18rem .5rem; border-radius: 3px;
      border: 1px solid {LINE}; color: {MUTED}; margin-bottom: .6rem;
  }}
  .eg-badge.eg-live {{ color: {ACCENT}; border-color: {ACCENT}; }}

  .eg-label {{
      font-size: .72rem; letter-spacing: .12em; text-transform: uppercase;
      color: {MUTED}; margin-bottom: .35rem;
  }}
  .eg-note {{ color: {MUTED}; font-size: .85rem; line-height: 1.55; }}

  /* The one-sentence answer at the top of a result. Reads as a statement, not
     a heading: full ink, larger than body, no decoration. */
  .eg-verdict {{
      color: {INK}; font-size: 1.3rem; font-weight: 600;
      letter-spacing: -0.01em; line-height: 1.4; margin: .35rem 0 .5rem 0;
  }}

  /* The headline number. Deliberately larger than the secondary metrics so
     the page has one obvious answer rather than four equal ones. */
  .eg-hero-metric {{ margin: .1rem 0 1.1rem 0; }}
  .eg-hero-value {{
      color: {INK}; font-size: 3rem; font-weight: 700;
      line-height: 1; letter-spacing: -0.02em;
  }}
  .eg-hero-label {{
      color: {MUTED}; font-size: .72rem; letter-spacing: .12em;
      text-transform: uppercase; margin-top: .4rem;
  }}

  [data-testid="stMetricValue"] {{ font-size: 1.6rem; }}
  [data-testid="stMetricLabel"] {{ color: {MUTED}; }}

  /* Result imagery reads as a data panel, not a floating picture. */
  [data-testid="stImage"] img {{
      border: 1px solid {LINE}; border-radius: 6px;
  }}

  /* What this monitor needs, stated before the user chooses an input. The
     accent edge marks it as a precondition rather than a remark. */
  .eg-requires {{
      background: {PANEL}; border: 1px solid {LINE};
      border-left: 2px solid {ACCENT}; border-radius: 4px;
      padding: .7rem .9rem; margin: .2rem 0 1.1rem 0;
  }}
  .eg-requires-label {{
      font-size: .68rem; letter-spacing: .12em; text-transform: uppercase;
      color: {ACCENT}; margin-bottom: .3rem;
  }}
  .eg-requires-body {{ color: {INK}; font-size: .95rem; line-height: 1.5; }}
  .eg-requires-note {{
      color: {MUTED}; font-size: .85rem; line-height: 1.5; margin-top: .35rem;
  }}

  hr {{ border-color: {LINE}; }}
</style>
"""


def apply():
    st.markdown(_CSS, unsafe_allow_html=True)


def label(text):
    st.markdown(f"<div class='eg-label'>{text}</div>", unsafe_allow_html=True)


def note(text):
    st.markdown(f"<div class='eg-note'>{text}</div>", unsafe_allow_html=True)


def verdict(text):
    """The one-sentence answer at the top of a result."""
    st.markdown(f"<div class='eg-verdict'>{text}</div>", unsafe_allow_html=True)


def requirements(body, note=""):
    """What a monitor needs, shown before the user chooses an input."""
    extra = f"<div class='eg-requires-note'>{note}</div>" if note else ""
    st.markdown(
        f"<div class='eg-requires'><div class='eg-requires-label'>Requires</div>"
        f"<div class='eg-requires-body'>{body}</div>{extra}</div>",
        unsafe_allow_html=True)


def hero_metric(value, label):
    """The single headline number of a result, above the secondary metrics."""
    st.markdown(
        f"<div class='eg-hero-metric'><div class='eg-hero-value'>{value}</div>"
        f"<div class='eg-hero-label'>{label}</div></div>",
        unsafe_allow_html=True)
