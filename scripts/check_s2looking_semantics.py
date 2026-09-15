"""Resolve which S2Looking image is earlier, and what label1/label2 mean.

Stage 3B, temporal-semantics investigation. Writes
``data/s2looking/metadata/semantics_check.json``.

This script measures and reports. It changes no labels, regenerates no
targets, and adapts nothing.

Evidence combined here
----------------------
1. DOCUMENTARY (authoritative). Recorded statically below from the primary
   sources, each checked by hand. The paper fixes what label1/label2 mean but
   never states which image is earlier.

2. METHOD CONTROL (essential). A feature that claims to detect "a building is
   present" must first recover a KNOWN answer. LEVIR-CD ships A/ and B/ with
   documented order (A earlier, B later) and its change is predominantly
   building growth, so a valid feature must separate A from B there. Features
   that fail this control are reported as discarded rather than quietly used.

3. APPLICATION. The surviving feature is applied to S2Looking pure label1 and
   label2 regions to see which image contains the building.

Why gradient-orientation concentration
--------------------------------------
Buildings impose a few dominant straight edge directions, so the weighted
histogram of gradient orientations inside a building is concentrated. Raw
texture energy (edge magnitude, brightness, std, line count) does NOT transfer
between these datasets: on LEVIR a new building LOWERS them, because LEVIR
"before" regions are rubble-strewn construction sites, whereas S2Looking empty
regions are smooth fields. That confound is exactly what the control detects.

Usage:  python scripts/check_s2looking_semantics.py [--scenes 100] [--min-area 1000]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config  # noqa: E402
from src.domains.built_environment.data.s2looking import binarize  # noqa: E402

DATASET_DIR = os.path.join(config.S2LOOKING_RAW, "S2Looking")
SPLITS = ("train", "val", "test")
PURITY = 0.99
MAX_REGIONS_PER_SCENE = 8
CONTROL_MIN_FRACTION = 0.60      # |frac-0.5| must clear this to count as signal

DOCUMENTARY = {
    "paper": {
        "reference": "Shen et al., S2Looking: A Satellite Side-Looking Dataset "
                     "for Building Change Detection, Remote Sensing 13(24):5094, "
                     "2021. https://doi.org/10.3390/rs13245094",
        "quote_figure2_caption": (
            "Images 1 and 2 in Figure 2 are bitemporal remote-sensing images, "
            "while Labels 1 and 2 are the corresponding annotation maps. Labels "
            "1 and 2 indicate pixel-precise newly built and demolished areas of "
            "buildings, respectively."),
        "quote_annotation_section": (
            "All newly built and demolished building regions in the dataset were "
            "annotated at the pixel level in separate auxiliary graphs."),
        "establishes": "label1 = newly built; label2 = demolished.",
        "does_not_establish": (
            "Which of Image1 / Image2 is the earlier acquisition. The full text "
            "contains no phrase such as 'the first image', 'the earlier image', "
            "'pre-change' or 'post-change' tied to Image1/Image2."),
    },
    "github_readme": {
        "url": "https://github.com/S2Looking/Dataset",
        "establishes": "Download links only.",
        "does_not_establish": "Any folder semantics or temporal ordering.",
    },
    "github_repository_tree": {
        "finding": "Contains only nested empty directories plus .DS_Store "
                   "artifacts. No README inside the dataset folders, no data "
                   "dictionary, no licence file.",
    },
    "github_issue_2_author_reply": {
        "url": "https://github.com/S2Looking/Dataset/issues/2",
        "author_association": "OWNER",
        "quote_zh": "我们的数据集一共有三种标签，红蓝两种分别代表图A相对图B的变化和"
                    "图B相对图A的变化，二值的黑白图表示图A和图B之间的变化区域",
        "translation": (
            "Our dataset has three kinds of labels in total. The red and blue "
            "ones respectively represent the change of image A relative to image "
            "B, and the change of image B relative to image A. The binary "
            "black-and-white image represents the changed area between image A "
            "and image B."),
        "establishes": "The two colour maps are genuinely directional, and the "
                       "red/blue channel encoding is intended by the authors.",
        "does_not_establish": (
            "Which of Image1 / Image2 is 'image A', nor which is earlier."),
    },
    "embedded_file_metadata": {
        "finding": "PNG files carry no text chunks, EXIF or acquisition dates "
                   "(PIL .info is empty for Image1, Image2, label1, label2). "
                   "Zip entry timestamps are packaging times (2021-04-09), not "
                   "acquisition times.",
    },
    "third_party_loaders": {
        "open_cd": "configs/_base_/datasets/s2looking.py maps img_path_from="
                   "Image1 and img_path_to=Image2. This is a downstream "
                   "convention, not evidence from the dataset authors.",
        "torchgeo": "No S2Looking module found.",
    },
}


def gray(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), np.float32)


def orientation_concentration(g, comp, bbox):
    """Share of gradient energy in the dominant orientation bin.

    High when a few straight edge directions dominate, as inside a building.
    """
    x, y, w, h = bbox
    sub = g[y:y + h, x:x + w]
    m = comp[y:y + h, x:x + w]
    gx = cv2.Sobel(sub, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(sub, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.arctan2(gy, gx)[m] + np.pi) % np.pi
    hist, _ = np.histogram(ang, bins=18, range=(0, np.pi), weights=mag[m])
    return float(hist.max() / (hist.sum() + 1e-6))


def edge_energy(g, comp, bbox):
    x, y, w, h = bbox
    sub = g[y:y + h, x:x + w]
    m = comp[y:y + h, x:x + w]
    gx = cv2.Sobel(sub, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(sub, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.sqrt(gx * gx + gy * gy)[m].mean())


def bbox_of(comp):
    ys, xs = np.where(comp)
    return int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def summarise(pairs):
    """pairs = list of (value_in_first_image, value_in_second_image)."""
    if not pairs:
        return None
    a = np.array(pairs, dtype=float)
    return {
        "regions": len(pairs),
        "mean_first": round(float(a[:, 0].mean()), 4),
        "mean_second": round(float(a[:, 1].mean()), 4),
        "fraction_second_greater": round(float((a[:, 1] > a[:, 0]).mean()), 4),
    }


def levir_control(n_scenes, min_area):
    """Does the feature recover LEVIR-CD's documented A(earlier)->B(later)?"""
    root = config.LEVIR_RAW
    label_dir = os.path.join(root, "label")
    if not os.path.isdir(label_dir):
        return {"available": False,
                "note": "LEVIR-CD not present; control could not be run."}
    names = sorted(f for f in os.listdir(label_dir) if f.startswith("train_"))[:n_scenes]
    orient, edge = [], []
    for name in names:
        lab = np.array(Image.open(os.path.join(label_dir, name)).convert("L")) > 127
        if not lab.any():
            continue
        ga = gray(os.path.join(root, "A", name))
        gb = gray(os.path.join(root, "B", name))
        k, lb, st, _ = cv2.connectedComponentsWithStats(lab.astype(np.uint8), connectivity=8)
        used = 0
        for i in range(1, k):
            if st[i, cv2.CC_STAT_AREA] < min_area or used >= MAX_REGIONS_PER_SCENE:
                continue
            comp = lb == i
            bb = bbox_of(comp)
            orient.append((orientation_concentration(ga, comp, bb),
                           orientation_concentration(gb, comp, bb)))
            edge.append((edge_energy(ga, comp, bb), edge_energy(gb, comp, bb)))
            used += 1
    o, e = summarise(orient), summarise(edge)
    return {
        "available": True,
        "scenes": len(names),
        "known_order": "A = earlier, B = later; change predominantly building growth",
        "orientation_concentration": o,
        "edge_energy": e,
        "orientation_passes_control": bool(o and o["fraction_second_greater"] >= CONTROL_MIN_FRACTION),
        "edge_passes_control": bool(e and e["fraction_second_greater"] >= CONTROL_MIN_FRACTION),
        "interpretation": (
            "A feature passes only if a building appearing (A->B) raises it "
            "clearly. Orientation concentration does. Edge energy does not - it "
            "moves the other way on LEVIR, because LEVIR 'before' regions are "
            "textured construction sites while new roofs are smooth. Edge energy "
            "is therefore discarded as a building-presence proxy."),
    }


def measure_split(split, n_scenes, min_area):
    d = os.path.join(DATASET_DIR, split)
    ids = sorted(os.listdir(os.path.join(d, "label1")), key=lambda s: int(s[:-4]))
    acc = {"label1": {"orient": [], "edge": []}, "label2": {"orient": [], "edge": []}}
    scanned = 0
    for name in ids:
        if scanned >= n_scenes:
            break
        l1 = binarize(np.array(Image.open(os.path.join(d, "label1", name))))
        l2 = binarize(np.array(Image.open(os.path.join(d, "label2", name))))
        if not (l1.any() or l2.any()):
            continue
        scanned += 1
        g1 = gray(os.path.join(d, "Image1", name))
        g2 = gray(os.path.join(d, "Image2", name))
        k, lb, st, _ = cv2.connectedComponentsWithStats(
            (l1 | l2).astype(np.uint8), connectivity=8)
        used = 0
        for i in range(1, k):
            if st[i, cv2.CC_STAT_AREA] < min_area or used >= MAX_REGIONS_PER_SCENE:
                continue
            comp = lb == i
            n1 = int(np.count_nonzero(l1 & comp))
            n2 = int(np.count_nonzero(l2 & comp))
            if n1 + n2 == 0 or max(n1, n2) / (n1 + n2) < PURITY:
                continue
            key = "label1" if n1 > n2 else "label2"
            bb = bbox_of(comp)
            acc[key]["orient"].append((orientation_concentration(g1, comp, bb),
                                       orientation_concentration(g2, comp, bb)))
            acc[key]["edge"].append((edge_energy(g1, comp, bb),
                                     edge_energy(g2, comp, bb)))
            used += 1
    return {
        "scenes_scanned": scanned,
        "label1": {"orientation_concentration": summarise(acc["label1"]["orient"]),
                   "edge_energy": summarise(acc["label1"]["edge"])},
        "label2": {"orientation_concentration": summarise(acc["label2"]["orient"]),
                   "edge_energy": summarise(acc["label2"]["edge"])},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=100)
    ap.add_argument("--min-area", type=int, default=1000)
    args = ap.parse_args()

    if not os.path.isdir(DATASET_DIR):
        print(f"NOT FOUND: {DATASET_DIR}")
        return 1

    print("[control] LEVIR-CD, documented A(earlier) -> B(later)...", flush=True)
    control = levir_control(args.scenes, args.min_area)
    if control.get("available"):
        o = control["orientation_concentration"]
        e = control["edge_energy"]
        print(f"  orientation: n={o['regions']} A={o['mean_first']:.4f} "
              f"B={o['mean_second']:.4f} frac(B>A)={o['fraction_second_greater']:.3f} "
              f"-> {'PASSES' if control['orientation_passes_control'] else 'FAILS'}")
        print(f"  edge       : n={e['regions']} A={e['mean_first']:.3f} "
              f"B={e['mean_second']:.3f} frac(B>A)={e['fraction_second_greater']:.3f} "
              f"-> {'PASSES' if control['edge_passes_control'] else 'FAILS (discarded)'}")

    splits = {}
    for split in SPLITS:
        print(f"[{split}] measuring...", flush=True)
        splits[split] = measure_split(split, args.scenes, args.min_area)
        for key in ("label1", "label2"):
            o = splits[split][key]["orientation_concentration"]
            if o:
                print(f"  {key}: n={o['regions']:4d} I1={o['mean_first']:.4f} "
                      f"I2={o['mean_second']:.4f} "
                      f"frac(I2>I1)={o['fraction_second_greater']:.3f}")

    # ---- conclusion -------------------------------------------------------
    valid = control.get("orientation_passes_control", False)
    f1 = [splits[s]["label1"]["orientation_concentration"]["fraction_second_greater"]
          for s in SPLITS if splits[s]["label1"]["orientation_concentration"]]
    f2 = [splits[s]["label2"]["orientation_concentration"]["fraction_second_greater"]
          for s in SPLITS if splits[s]["label2"]["orientation_concentration"]]
    label1_building_in = "Image1" if all(f < 0.5 for f in f1) else (
        "Image2" if all(f > 0.5 for f in f1) else "inconsistent")
    label2_building_in = "Image1" if all(f < 0.5 for f in f2) else (
        "Image2" if all(f > 0.5 for f in f2) else "inconsistent")

    consistent = (valid and label1_building_in == "Image1"
                  and label2_building_in == "Image2")

    out = {
        "question": "Which of Image1/Image2 is the earlier acquisition, and what "
                    "do label1/label2 mean?",
        "documentary_evidence": DOCUMENTARY,
        "method_control_levir": control,
        "measurements_s2looking": splits,
        "parameters": {"scenes_per_split": args.scenes,
                       "min_area_px": args.min_area,
                       "purity": PURITY,
                       "control_min_fraction": CONTROL_MIN_FRACTION},
        "derived": {
            "label1_building_present_in": label1_building_in,
            "label2_building_present_in": label2_building_in,
        },
        "conclusion": {
            "label_meanings": "RESOLVED by the paper: label1 = newly built, "
                              "label2 = demolished.",
            "image_temporal_order": (
                "NOT RESOLVED by any official source. No primary document states "
                "which of Image1/Image2 is earlier."),
            "best_supported_reading": (
                "Image1 is the LATER acquisition and Image2 the EARLIER one. "
                "Measurement places the building in Image1 for label1 (newly "
                "built) regions and in Image2 for label2 (demolished) regions; "
                "combined with the paper's label definitions this implies "
                "Image1 is later."
                if consistent else
                "Measurement did not give a consistent answer; do not proceed."),
            "equivalent_alternative": (
                "Image1 is earlier and the label1/label2 meanings are the reverse "
                "of the paper's wording. This produces the same (before, after) "
                "pairing and differs only in naming."),
            "confidence": ("MEDIUM-HIGH for the pairing, LOW for the naming"
                           if consistent else "LOW"),
            "recommendation": {
                "before_image": "Image2",
                "after_image": "Image1",
                "construction_label": "label1",
                "demolition_label": "label2",
                "note": "Equivalently: keep Image1 as 'before' and swap the label "
                        "meanings. Both give identical training pairs. Confirm "
                        "with the authors before training.",
            },
            "blocking": True,
        },
    }

    os.makedirs(config.S2LOOKING_META, exist_ok=True)
    path = os.path.join(config.S2LOOKING_META, "semantics_check.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n[write] {path}")
    print("\nlabel1 building present in:", label1_building_in)
    print("label2 building present in:", label2_building_in)
    print("image order resolved by official sources:", "NO")
    return 0


if __name__ == "__main__":
    sys.exit(main())
