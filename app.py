"""Earth Guardian - minimal demo UI.

    streamlit run app.py

This is a THIN presentation layer. Every computation happens in
src/inference/engine.py, so the same engine can later sit behind a REST API
without touching this file.
"""
import io
import json
import os
import sys

import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.inference.engine import ChangeEngine, overlay

ROOT = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Earth Guardian", page_icon="EG", layout="wide")


@st.cache_resource
def get_engine(ckpt):
    return ChangeEngine(ckpt)


@st.cache_data
def list_tiles(label_dir, only_with_change, min_frac=0.05, limit=300):
    """Test tiles for the demo picker.

    The filter looks at the GROUND-TRUTH label only (>= 5% change pixels), never
    at model output, so it selects tiles where there is something to detect -
    not tiles where the model happens to do well.
    """
    names = sorted(os.listdir(label_dir))
    if not only_with_change:
        return names[:limit]
    out = []
    for n in names:
        m = np.array(Image.open(os.path.join(label_dir, n)).convert("L")) > 127
        if m.mean() >= min_frac:
            out.append(n)
            if len(out) >= limit:
                break
    return out


st.title("Earth Guardian")
st.caption("Built-Environment Change Monitor - structural / building change detection")

with st.sidebar:
    st.header("Model")
    ckpt = st.text_input("Checkpoint", "checkpoints/siamese_unet_r34_best.pt")
    try:
        engine = get_engine(ckpt)
    except Exception as e:
        st.error(str(e))
        st.stop()
    st.success(f"{engine.model_name} v{engine.version}")
    st.metric("Validation F1", f"{engine.val_f1:.4f}")
    st.caption(f"Trained on {engine.trained_on}")

    st.header("Parameters")
    thr = st.slider("Threshold", 0.05, 0.95, float(engine.threshold), 0.01,
                    help="Default is the value selected on the validation split.")
    min_area = st.slider("Min region size (px)", 0, 512, 32, 8)

    st.divider()
    st.caption(
        "**Scope.** This model detects structural (building) change in "
        "co-located high-resolution optical imagery. It was trained on "
        "LEVIR-CD (Texas, USA, 0.5 m/px). It does **not** detect floods, "
        "fires, deforestation or snow change."
    )

# ---------------------------------------------------------------- inputs
src = st.radio("Input", ["Bundled example", "Upload a pair"], horizontal=True)
before = after = None

if src == "Bundled example":
    tiles = os.path.join(ROOT, "data", "levir_cd_tiles", "test", "A")
    demo = os.path.join(ROOT, "images")
    if os.path.isdir(tiles) and os.listdir(tiles):
        only_change = st.checkbox(
            "Only tiles whose ground truth contains change (>= 5% of pixels)", value=True,
            help="Filters on the ground-truth label, not on model output.")
        names = list_tiles(os.path.join(ROOT, "data", "levir_cd_tiles", "test", "label"),
                           only_change)
        pick = st.selectbox("LEVIR-CD test tile (held-out test split)", names)
        before = os.path.join(tiles, pick)
        after = os.path.join(ROOT, "data", "levir_cd_tiles", "test", "B", pick)
        gt_path = os.path.join(ROOT, "data", "levir_cd_tiles", "test", "label", pick)
    else:
        pick = st.selectbox("Sample scene", ["before1.png / after1.png",
                                             "before.png / after.png",
                                             "before2.png / after2.png"])
        b, a = pick.split(" / ")
        before, after = os.path.join(demo, b), os.path.join(demo, a)
        gt_path = None
else:
    c1, c2 = st.columns(2)
    ub = c1.file_uploader("Before image", type=["png", "jpg", "jpeg", "tif"])
    ua = c2.file_uploader("After image", type=["png", "jpg", "jpeg", "tif"])
    gt_path = None
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

res, mask, prob = out["result"], out["mask"], out["probability"]
s = res["summary"]

# ---------------------------------------------------------------- results
m1, m2, m3, m4 = st.columns(4)
m1.metric("Changed area", f"{s['changed_area_pct']:.2f} %")
m2.metric("Changed pixels", f"{s['changed_pixels']:,}")
m3.metric("Regions detected", s["n_regions"] if s["n_regions"] is not None else "n/a")
m4.metric("Mean confidence", f"{s['mean_confidence']:.3f}")

cols = st.columns(4 if gt_path and os.path.exists(gt_path) else 3)
cols[0].image(overlay(out["after"], mask), caption="Change overlay", use_container_width=True)
cols[1].image((mask * 255).astype(np.uint8), caption="Predicted mask", use_container_width=True)
cols[2].image((prob * 255).astype(np.uint8), caption="Probability map", use_container_width=True)
if gt_path and os.path.exists(gt_path):
    gt = np.array(Image.open(gt_path).convert("L")) > 127
    h, w = mask.shape
    tp = mask & gt; fp = mask & ~gt; fn = ~mask & gt
    err = np.full((h, w, 3), 25, dtype=np.uint8)
    err[tp] = (60, 200, 90); err[fp] = (230, 70, 70); err[fn] = (70, 130, 235)
    inter = float(tp.sum()); f1 = 2 * inter / (2 * inter + fp.sum() + fn.sum() + 1e-9)
    cols[3].image(err, caption=f"Error map - green TP / red FP / blue FN (F1 {f1:.3f})",
                  use_container_width=True)

st.caption(f"Inference {res['runtime_seconds']}s  |  threshold {res['params']['threshold']}  |  "
           f"{res['model']['name']} v{res['model']['version']}")

if s["changed_area_m2"] is None:
    st.info("Ground area in m2 is not reported: LEVIR-CD imagery carries no "
            "georeferencing or documented GSD, so it would be fabricated. "
            "Supply a real GSD to enable it.")

with st.expander("Structured result (this JSON is the product's actual interface)"):
    st.json(res)
st.download_button("Download result JSON", json.dumps(res, indent=2),
                   file_name="earth_guardian_result.json", mime="application/json")
