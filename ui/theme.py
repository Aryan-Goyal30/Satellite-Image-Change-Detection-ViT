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

  .eg-hero h1 {{
      font-size: 2.6rem; font-weight: 700; letter-spacing: -0.02em;
      color: {INK}; margin: 0 0 .35rem 0;
  }}
  .eg-tagline {{ font-size: 1.15rem; color: {MUTED}; margin-bottom: 2rem; }}

  .eg-card {{
      background: {PANEL}; border: 1px solid {LINE}; border-radius: 6px;
      padding: 1.1rem 1.25rem; height: 100%;
  }}
  .eg-card h3 {{ color: {INK}; font-size: 1.05rem; margin: 0 0 .4rem 0; }}
  .eg-card p {{ color: {MUTED}; font-size: .92rem; margin: 0; line-height: 1.5; }}
  .eg-card.eg-soon {{ opacity: .55; }}

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

  [data-testid="stMetricValue"] {{ font-size: 1.6rem; }}
  [data-testid="stMetricLabel"] {{ color: {MUTED}; }}

  hr {{ border-color: {LINE}; }}
</style>
"""


def apply():
    st.markdown(_CSS, unsafe_allow_html=True)


def label(text):
    st.markdown(f"<div class='eg-label'>{text}</div>", unsafe_allow_html=True)


def note(text):
    st.markdown(f"<div class='eg-note'>{text}</div>", unsafe_allow_html=True)
