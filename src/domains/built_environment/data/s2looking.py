"""Region-level construction/demolition targets from S2Looking annotations.

DATA PREPARATION ONLY. Nothing here is imported by the engine, the CLI or the
UI, and no production behaviour depends on it. It turns S2Looking's two
pixel-precise annotation maps into region-level ground truth for the Stage 3B
directionality study.

What the two annotation maps mean (S2Looking, Remote Sensing 13(24):5094):
    label1 = newly built building areas   -> construction
    label2 = demolished building areas    -> demolition

Both are 1024x1024 RGB PNGs with values in {0, 255}. Verified on the released
files: label1 carries its annotation in the BLUE channel and label2 in the RED
channel, and ``label1 OR label2`` reproduces the bundled combined ``label``
mask exactly. Decode with binarize(), which treats any channel above the
threshold as annotated.

Temporal ordering - RATIFIED PROJECT DECISION, not a documented fact
-------------------------------------------------------------------
No official S2Looking source states which of Image1/Image2 is the earlier
acquisition (see docs/S2LOOKING_TEMPORAL_SEMANTICS.md). Based on the available
evidence the project has ratified:

    BEFORE = Image2      AFTER = Image1
    CONSTRUCTION = label1    DEMOLITION = label2

These constants are the single place that decision is expressed. Anything that
reads the dataset must go through them rather than hard-coding a folder name.

Region definition
-----------------
A region is a connected component (8-connected) of ``label1 OR label2``,
filtered by ``min_area_px``. This is deliberately the same notion of a region
that production uses in src/common/regions.py, so a Stage 3B target describes
the same kind of object the product already shows. tests/test_stage3b.py
asserts that the geometry produced here matches ``extract_regions`` exactly.

Targets
-------
    purity = max(n_construction, n_demolition) / (n_construction + n_demolition)

    purity >= PURITY_THRESHOLD -> construction or demolition, whichever dominates
    otherwise                  -> mixed

MIXED is a real property of the ground truth: a redevelopment site where
demolition and construction occur inside one connected component. It is NOT a
model-uncertainty class, and nothing here emits "uncertain".

The purity threshold is a dataset-preparation parameter. It is fixed here,
recorded in the generated metadata, and never tuned against the test split.
"""
from __future__ import annotations

import numpy as np

from src import config

CONSTRUCTION = "construction"
DEMOLITION = "demolition"
MIXED = "mixed"
TARGETS = (CONSTRUCTION, DEMOLITION, MIXED)

#: Dataset-preparation parameter, fixed in advance and recorded in metadata.
PURITY_THRESHOLD = 0.95

# --- ratified temporal ordering (project decision, see docs) ----------------
#: Folder holding the EARLIER acquisition of each pair.
BEFORE_IMAGE_DIR = "Image2"
#: Folder holding the LATER acquisition of each pair.
AFTER_IMAGE_DIR = "Image1"
#: Annotation map whose regions are construction (buildings present only after).
CONSTRUCTION_LABEL_DIR = "label1"
#: Annotation map whose regions are demolition (buildings present only before).
DEMOLITION_LABEL_DIR = "label2"

#: Recorded verbatim in generated metadata so downstream artifacts carry the
#: provenance of this decision rather than an implied fact.
TEMPORAL_ORDERING = {
    "before_image_dir": BEFORE_IMAGE_DIR,
    "after_image_dir": AFTER_IMAGE_DIR,
    "construction_label_dir": CONSTRUCTION_LABEL_DIR,
    "demolition_label_dir": DEMOLITION_LABEL_DIR,
    "basis": "Ratified project decision. Official S2Looking documentation does "
             "NOT state which of Image1/Image2 is earlier; the paper fixes only "
             "that label1 = newly built and label2 = demolished. See "
             "docs/S2LOOKING_TEMPORAL_SEMANTICS.md.",
    "documented_by_authors": False,
}

DEFAULT_MIN_AREA_PX = config.DEFAULT_MIN_AREA_PX


def binarize(array: np.ndarray, positive_threshold: int = 127) -> np.ndarray:
    """Boolean mask from a label raster, collapsing a colour axis if present.

    The encoding is verified on the real files before this is used; the
    threshold only guards against edge antialiasing, exactly as the LEVIR
    tiler does.
    """
    a = np.asarray(array)
    if a.ndim == 3:
        # VERIFIED ON THE REAL FILES: S2Looking stores label1 in the BLUE
        # channel and label2 in the RED channel, both as 0/255 RGB PNGs.
        # Collapsing to channel 0 would silently return an empty mask for
        # every construction annotation, so any channel above the threshold
        # counts as annotated.
        if a.dtype == bool:
            return a.any(axis=2)
        return (a > positive_threshold).any(axis=2)
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D label raster, got shape {a.shape}")
    if a.dtype == bool:
        return a
    return a > positive_threshold


def _as_mask(array: np.ndarray, name: str) -> np.ndarray:
    """Accept a bool mask or a strict 0/1 mask; reject anything else.

    Refusing unexpected values is deliberate: a label raster that is not what
    we verified must fail loudly rather than be silently reinterpreted.
    """
    a = np.asarray(array)
    if a.dtype == bool:
        return a
    values = np.unique(a)
    if not np.isin(values, (0, 1)).all():
        raise ValueError(
            f"{name}: expected a boolean or 0/1 mask, found values "
            f"{values[:8].tolist()}. Binarize explicitly with binarize().")
    return a.astype(bool)


def _components(mask: np.ndarray):
    """8-connected labelling - the same call production region extraction uses."""
    import cv2
    return cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)


def region_targets(label1, label2,
                   min_area_px: int = DEFAULT_MIN_AREA_PX,
                   purity_threshold: float = PURITY_THRESHOLD):
    """Region-level targets for one scene.

    ``label1`` is the newly-built mask, ``label2`` the demolished mask; both
    boolean (or strict 0/1) and the same shape.

    Returns a list of dicts ordered by descending area, with ``region_id``
    numbered from 1. IDs are deterministic: components are sorted by descending
    pixel area with ties broken by labelling order, matching
    src/common/regions.extract_regions.
    """
    l1 = _as_mask(label1, "label1")
    l2 = _as_mask(label2, "label2")
    if l1.shape != l2.shape:
        raise ValueError(f"label1 shape {l1.shape} != label2 shape {l2.shape}")
    if l1.ndim != 2:
        raise ValueError(f"expected 2-D label masks, got {l1.shape}")
    if not (0.0 < purity_threshold <= 1.0):
        raise ValueError(f"purity_threshold must be in (0, 1], got {purity_threshold}")

    change = l1 | l2
    n, labels, stats, centroids = _components(change)

    import cv2
    kept = []
    for i in range(1, n):                      # label 0 is background
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue
        kept.append((
            area,
            (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
             int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])),
            (round(float(centroids[i][0]), 1), round(float(centroids[i][1]), 1)),
            i,
        ))
    kept.sort(key=lambda r: -r[0])              # stable: ties keep label order

    out = []
    for region_id, (area, bbox, centroid, label_value) in enumerate(kept, 1):
        component = labels == label_value
        n1 = int(np.count_nonzero(l1 & component))
        n2 = int(np.count_nonzero(l2 & component))
        overlap = int(np.count_nonzero(l1 & l2 & component))
        denominator = n1 + n2                   # > 0: every pixel is in l1 or l2
        purity = max(n1, n2) / denominator

        if purity >= purity_threshold and n1 != n2:
            target = CONSTRUCTION if n1 > n2 else DEMOLITION
        else:
            target = MIXED

        out.append({
            "region_id": region_id,
            "target": target,
            "area_px": area,
            "bbox_x": bbox[0], "bbox_y": bbox[1],
            "bbox_w": bbox[2], "bbox_h": bbox[3],
            "centroid_x": centroid[0], "centroid_y": centroid[1],
            "construction_pixels": n1,
            "demolition_pixels": n2,
            "overlap_pixels": overlap,
            "purity": round(purity, 6),
        })
    return out


# ------------------------------------------------------------- statistics
def _describe(values):
    """min / median / mean / max for a list of numbers, or None when empty."""
    if not values:
        return None
    a = np.asarray(values, dtype=float)
    return {
        "count": int(a.size),
        "min": float(a.min()),
        "median": float(np.median(a)),
        "mean": round(float(a.mean()), 2),
        "max": float(a.max()),
    }


def _counts(records):
    """Target counts and percentages for a collection of region records."""
    total = len(records)
    out = {"regions": total}
    for target in TARGETS:
        n = sum(1 for r in records if r["target"] == target)
        out[target] = n
        out[target + "_pct"] = round(100.0 * n / total, 3) if total else 0.0
    return out


def summarize_regions(records, scenes_by_split=None, demolition_gate_pct=5.0):
    """Aggregate region records into the Stage 3B-1 statistics report.

    ``records`` are the dicts produced by region_targets, each additionally
    carrying ``scene_id`` and ``split``. ``scenes_by_split`` maps a split name
    to the full list of scene ids in it, so scenes that produced no region at
    all are still counted.

    ``demolition_gate_pct`` is the previously agreed engineering gate, not a
    scientific threshold. It is reported, never enforced silently.
    """
    splits = sorted({r["split"] for r in records})
    if scenes_by_split:
        splits = sorted(set(splits) | set(scenes_by_split))

    overall = _counts(records)
    overall["scenes"] = (sum(len(v) for v in scenes_by_split.values())
                         if scenes_by_split else len({r["scene_id"] for r in records}))

    per_split = {}
    for split in splits:
        subset = [r for r in records if r["split"] == split]
        entry = _counts(subset)
        scene_ids = (list(scenes_by_split.get(split, []))
                     if scenes_by_split else sorted({r["scene_id"] for r in subset}))
        entry["scenes"] = len(scene_ids)
        n_c, n_d = entry[CONSTRUCTION], entry[DEMOLITION]
        entry["construction_to_demolition_ratio"] = (round(n_c / n_d, 3) if n_d else None)
        per_split[split] = entry

    area = {t: _describe([r["area_px"] for r in records if r["target"] == t])
            for t in TARGETS}
    area["all"] = _describe([r["area_px"] for r in records])

    pixels = {
        "construction_pixels": sum(r["construction_pixels"] for r in records),
        "demolition_pixels": sum(r["demolition_pixels"] for r in records),
        "overlap_pixels": sum(r.get("overlap_pixels", 0) for r in records),
        "total_changed_pixels": sum(r["area_px"] for r in records),
    }

    # Per-scene distribution. Scenes with no surviving region count as zero,
    # which is why the full scene list matters.
    by_scene = {}
    for r in records:
        entry = by_scene.setdefault(r["scene_id"], {t: 0 for t in TARGETS})
        entry[r["target"]] += 1
    if scenes_by_split:
        for ids in scenes_by_split.values():
            for scene_id in ids:
                by_scene.setdefault(scene_id, {t: 0 for t in TARGETS})

    totals = [sum(v.values()) for v in by_scene.values()]
    both = sum(1 for v in by_scene.values() if v[CONSTRUCTION] and v[DEMOLITION])
    per_scene = {
        "regions_per_scene": _describe(totals),
        "construction_per_scene": _describe([v[CONSTRUCTION] for v in by_scene.values()]),
        "demolition_per_scene": _describe([v[DEMOLITION] for v in by_scene.values()]),
        "mixed_per_scene": _describe([v[MIXED] for v in by_scene.values()]),
        "scenes_with_both_directions": both,
        "scenes_with_no_regions": sum(1 for t in totals if t == 0),
    }

    demolition_pct = overall[DEMOLITION + "_pct"]
    return {
        "parameters": {
            "purity_threshold": PURITY_THRESHOLD,
            "min_area_px": DEFAULT_MIN_AREA_PX,
            "connectivity": 8,
            "demolition_gate_pct": demolition_gate_pct,
        },
        "overall": overall,
        "per_split": per_split,
        "area_px": area,
        "pixels": pixels,
        "per_scene": per_scene,
        "gate": {
            "demolition_pct": demolition_pct,
            "gate_pct": demolition_gate_pct,
            "demolition_sufficiently_represented": demolition_pct >= demolition_gate_pct,
            "note": "Engineering gate agreed in the Stage 3B audit, not a "
                    "scientific requirement. Below it, stop and review.",
        },
    }


def split_integrity(scenes_by_split):
    """Check that no scene id appears in more than one split.

    The scene is the atomic split unit: a scene in two splits would leak
    regions between training and evaluation.
    """
    seen, duplicates = {}, {}
    for split, ids in scenes_by_split.items():
        for scene_id in ids:
            if scene_id in seen:
                duplicates.setdefault(scene_id, [seen[scene_id]]).append(split)
            else:
                seen[scene_id] = split
    per_split_dupes = {s: len(ids) - len(set(ids)) for s, ids in scenes_by_split.items()}
    return {
        "scene_counts": {s: len(ids) for s, ids in scenes_by_split.items()},
        "unique_scene_ids": len(seen),
        "cross_split_duplicates": duplicates,
        "within_split_duplicates": per_split_dupes,
        "ok": not duplicates and not any(per_split_dupes.values()),
    }
