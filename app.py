"""Earth Guardian - minimal demo UI.

    streamlit run app.py

A THIN presentation layer:

    example catalogue  ->  engine registry  ->  domain engine  ->  ChangeResult

It asks the registry for a domain engine, resolves inputs through the example
catalogue, and renders a ChangeResult. It never imports a model and never needs
to know how a dataset is organised on disk, so a future imagery provider or a
second domain can be swapped in without changing this file.

All user-facing capability text comes from the engine's metadata / model card.
"""
import json
import os
import sys

import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import config
from src.common import examples as example_catalogue
from src.core import registry
from src.domains.built_environment.engine import overlay

st.set_page_config(page_title="Earth Guardian", page_icon="EG", layout="wide")

ENGINE_NAME = "built_environment"


@st.cache_resource
def get_engine(name, ckpt):
    return registry.get(name, checkpoint=ckpt)


@st.cache_data
def list_examples(domain):
    """Selectable inputs, resolved through the catalogue (not the dataset)."""
    return [e.to_dict() for e in example_catalogue.available(domain=domain)]


st.title("Earth Guardian")

with st.sidebar:
    st.header("Model")
    ckpt = st.text_input("Checkpoint", config.DEFAULT_CHECKPOINT_REL)
    try:
        engine = get_engine(ENGINE_NAME, ckpt)
    except Exception as e:
        st.error(str(e))
        st.stop()

    md = engine.metadata
    prov = md.provenance
    st.success(f"{md.name} v{md.version}")
    st.metric("Validation F1", f"{engine.val_f1:.4f}")
    st.caption(f"Trained on {prov.dataset}")

    st.header("Parameters")
    thr = st.slider("Threshold", 0.05, 0.95, float(engine.threshold), 0.01,
                    help="Default is the value selected on the validation split.")
    min_area = st.slider("Min region size (px)", 0, 512, 32, 8)

    st.divider()
    # Scope text is taken from the model card, not written here, so the UI can
    # never claim more than the engine declares.
    env = prov.operating_envelope
    scope = [f"**Scope.** {md.display_name} - {md.task}.",
             f"Detects: {', '.join(md.capabilities)}."]
    if env:
        if env.gsd_m_range:
            scope.append(f"Validated at {env.gsd_m_range[0]}-{env.gsd_m_range[1]} m/px.")
        if env.regions_validated:
            scope.append(f"Validated on: {', '.join(env.regions_validated)}.")
        if env.notes:
            scope.append(env.notes)
    st.caption(" ".join(scope))

# ---------------------------------------------------------------- inputs
st.caption(f"{md.display_name} - {md.task}")

src = st.radio("Input", ["Bundled example", "Upload a pair"], horizontal=True)
before = after = None
gt_path = None

if src == "Bundled example":
    catalogue = list_examples(md.domain)
    if not catalogue:
        st.warning(
            "No bundled examples resolve on this machine. Run "
            "`python scripts/build_example_catalogue.py` after preparing the data.")
        st.stop()
    labels = [e["name"] for e in catalogue]
    pick = st.selectbox("Select example", labels)
    chosen = catalogue[labels.index(pick)]
    ex = example_catalogue.Example(**chosen)
    before, after = ex.before_path, ex.after_path
    gt_path = ex.ground_truth_path if ex.has_ground_truth else None
    st.caption(f"{ex.description}  |  source: {ex.source}")
else:
    c1, c2 = st.columns(2)
    ub = c1.file_uploader("Before image", type=["png", "jpg", "jpeg", "tif"])
    ua = c2.file_uploader("After image", type=["png", "jpg", "jpeg", "tif"])
    if ub and ua:
        before, after = Image.open(ub), Image.open(ua)

if before is None or after is None:
    st.info("Select or upload a co-located image pair to analyze.")
    st.stop()

# ---------------------------------------------------------------- preview
c1, c2 = st.columns(2)
c1.image(before, caption="BEFORE", use_container_width=True)
c2.image(after, caption="AFTER", use_container_width=True)

if not st.button("Analyze", type="primary", use_container_width=True):
    st.stop()

with st.spinner("Running change detection..."):
    out = engine.analyze(before, after, threshold=thr, min_area_px=min_area)

# ChangeResult is the representation the UI works with.
cr = out["change_result"]
mask, prob = out["mask"], out["probability"]
q = cr.quantities
layer = cr.primary_layer

# ---------------------------------------------------------------- results
m1, m2, m3, m4 = st.columns(4)
m1.metric("Changed area", f"{q.changed_percentage:.2f} %")
m2.metric("Changed pixels", f"{q.changed_pixels:,}")
m3.metric("Regions detected", len(cr.regions))
m4.metric("Mean confidence", f"{layer.mean_confidence:.3f}")

cols = st.columns(4 if gt_path else 3)
cols[0].image(overlay(out["after"], mask), caption="Change overlay", use_container_width=True)
cols[1].image((mask * 255).astype(np.uint8), caption="Predicted mask", use_container_width=True)
cols[2].image((prob * 255).astype(np.uint8), caption="Probability map", use_container_width=True)
if gt_path:
    gt = np.array(Image.open(gt_path).convert("L")) > 127
    h, w = mask.shape
    tp = mask & gt; fp = mask & ~gt; fn = ~mask & gt
    err = np.full((h, w, 3), 25, dtype=np.uint8)
    err[tp] = (60, 200, 90); err[fp] = (230, 70, 70); err[fn] = (70, 130, 235)
    inter = float(tp.sum()); f1 = 2 * inter / (2 * inter + fp.sum() + fn.sum() + 1e-9)
    cols[3].image(err, caption=f"Error map - green TP / red FP / blue FN (F1 {f1:.3f})",
                  use_container_width=True)

st.caption(f"Inference {cr.runtime_seconds}s  |  threshold {cr.params['threshold']}  |  "
           f"{md.name} v{md.version}  |  result schema v{cr.schema_version}")

for w in cr.warnings:
    st.warning(w)

if q.area_m2 is None:
    st.info("Ground area in m2 is not reported: this imagery carries no "
            "georeferencing or documented GSD, so it would be fabricated.")

with st.expander("Structured result (ChangeResult, schema v1 - the product's interface)"):
    st.json(cr.to_dict())
st.download_button("Download result JSON", json.dumps(cr.to_dict(), indent=2),
                   file_name="earth_guardian_result.json", mime="application/json")
