"""Audit the Phase 4A environmental dataset - Milestone 2.

Reads the manifest produced by build_environment_dataset.py and answers the
question the milestone actually asks: is this dataset scientifically defensible,
or is it pathological?

Produces two artifacts:

    audit.json          machine-readable statistics
    contact_sheet.png   BEFORE / AFTER / LABEL for representative samples

The representativeness section exists because Milestone 1 deliberately chose a
window that was 61.9% positive to make the visual check decisive. A dataset of
windows like that would be trivially separable and completely unrepresentative
of real forest-loss detection, so the positive-fraction distribution is reported
in full rather than as a single average.

Usage:  python scripts/audit_environment_dataset.py
"""
import argparse
import datetime
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config                                       # noqa: E402
from src.domains.environment.data import dataset as ds       # noqa: E402
from src.domains.environment.data import events as ev        # noqa: E402
from src.domains.environment.data import sentinel2 as s2     # noqa: E402

OUT_DIR = os.path.join(config.DATA_DIR, "environment", "dataset")
MANIFEST = os.path.join(OUT_DIR, "manifest.json")
AUDIT = os.path.join(OUT_DIR, "audit.json")
SHEET = os.path.join(OUT_DIR, "contact_sheet.png")

#: Manifest of the pre-2.1 protocol, kept so the temporal correction can be
#: measured against what it replaced rather than merely asserted.
V1_MANIFEST = os.path.join(config.DATA_DIR, "environment", "dataset",
                           "manifest_v1_protocol.json")

#: Contact sheets use a FIXED reflectance mapping, not a per-sample percentile
#: stretch. An independent 2-98% stretch rescales every window to full contrast,
#: which repeatedly made faint brightness variation look like thick cloud and
#: produced two incorrect contamination findings in earlier milestones. A fixed
#: ceiling with a fixed gamma keeps brightness comparable between samples; the
#: gamma only lifts dark tropical canopy into a visible range, identically for
#: every image.
FIXED_SCALE_MAX = 0.30
FIXED_SCALE_GAMMA = 0.6


def _doy(record, side):
    """Day of year for one side of a pair, computed if the manifest predates it."""
    key = f"{side}_doy"
    if record.get(key) is not None:
        return int(record[key])
    stamp = record[f"{side}_datetime"][:10]
    return datetime.datetime.strptime(stamp, "%Y-%m-%d").timetuple().tm_yday


def _seasonal_direction(samples):
    """How one-directional is the seasonal offset between BEFORE and AFTER?

    This is the statistic the Milestone 2.1 correction targets. The old rule
    put BEFORE later in the seasonal cycle than AFTER in 42 of 43 samples - a
    bias always pointing the same way, which a model can learn instead of
    forest loss. A corrected protocol should land near 50/50, with the residual
    being scatter rather than direction.
    """
    later = sum(1 for r in samples if _doy(r, "before") > _doy(r, "after"))
    earlier = sum(1 for r in samples if _doy(r, "before") < _doy(r, "after"))
    same = len(samples) - later - earlier
    total = max(len(samples), 1)
    return {"before_later_in_season": later,
            "before_earlier_in_season": earlier,
            "same_doy": same,
            "before_later_pct": round(100 * later / total, 1),
            "before_earlier_pct": round(100 * earlier / total, 1),
            "directional_imbalance_pct": round(
                100 * abs(later - earlier) / total, 1),
            "mean_signed_doy_offset": round(
                sum(_doy(r, "after") - _doy(r, "before") for r in samples) / total, 2)}


def compare_protocols(current, previous_path):
    """Old protocol versus new, when a previous manifest was kept."""
    if not os.path.exists(previous_path):
        return {"available": False,
                "note": f"no previous manifest at {os.path.basename(previous_path)}"}
    with open(previous_path, encoding="utf-8") as fh:
        old = json.load(fh)
    old_samples = old.get("samples", [])
    if not old_samples:
        return {"available": False, "note": "previous manifest has no samples"}
    return {
        "available": True,
        "previous_script_version": old.get("script_version"),
        "previous_rule": old.get("protocol", {}).get("pairing_rule"),
        "current_rule": current["protocol"].get("selection_rule"),
        "previous": {"n": len(old_samples),
                     "seasonal_direction": _seasonal_direction(old_samples),
                     "abs_doy_difference": describe(
                         [abs(_doy(r, "before") - _doy(r, "after"))
                          for r in old_samples], 1)},
        "current": {"n": len(current["samples"]),
                    "seasonal_direction": _seasonal_direction(current["samples"]),
                    "abs_doy_difference": describe(
                        [abs(_doy(r, "before") - _doy(r, "after"))
                         for r in current["samples"]], 1)},
    }


def describe(values, digits=4):
    """Min/median/mean/max plus percentiles for a numeric column."""
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    out = {"n": len(values),
           "min": round(float(array.min()), digits),
           "max": round(float(array.max()), digits),
           "mean": round(float(array.mean()), digits),
           "median": round(float(np.median(array)), digits)}
    for p in (5, 10, 25, 75, 90, 95):
        out[f"p{p}"] = round(float(np.percentile(array, p)), digits)
    return out


def load_arrays(sample_id):
    def get(name):
        return np.load(os.path.join(OUT_DIR, f"{sample_id}_{name}.npy"),
                       mmap_mode="r")
    return get("before"), get("after"), get("label")


def dnbr_map(record, before, after):
    """Per-pixel change in Normalised Burn Ratio, a standard forest-loss index.

    Two things this has to get right. Reflectance conversion is per scene, using
    that scene's own baseline and offset flag, because a dataset spanning
    baselines 00.01 to 05.09 cannot share one rule. And the denominator is
    guarded: NBR divides by (NIR + SWIR), which nodata and near-zero pixels can
    drive through zero - unguarded, that produced a class mean of -1993 on a
    first attempt. Such pixels become NaN and are excluded, never averaged in.
    """
    def nbr(stack, applied, baseline):
        ref = s2.to_reflectance(np.asarray(stack), applied, baseline)
        nir = ref[:, :, s2.BAND_INDEX["B08"]]
        swir = ref[:, :, s2.BAND_INDEX["B12"]]
        denom = nir + swir
        safe = np.abs(denom) > 0.01
        return np.where(safe, (nir - swir) / np.where(safe, denom, 1.0), np.nan)

    return (nbr(after, record["after_offset_applied"], record["after_baseline"]) -
            nbr(before, record["before_offset_applied"], record["before_baseline"]))


# --------------------------------------------------------------------- audit
def build_audit(manifest):
    samples = manifest["samples"]
    positives = [r for r in samples if r["sample_type"] == "positive"]
    negatives = [r for r in samples if r["sample_type"] == "negative"]
    rejections = manifest["rejections"]

    pos_fractions = [r["positive_fraction"] for r in positives]
    all_fractions = [r["positive_fraction"] for r in samples]
    buckets = Counter(ds.positive_bucket(f) for f in pos_fractions)

    lons = [r["bbox_lonlat"][0] for r in samples] + [r["bbox_lonlat"][2] for r in samples]
    lats = [r["bbox_lonlat"][1] for r in samples] + [r["bbox_lonlat"][3] for r in samples]

    split_tiles = manifest["split_tiles"]
    centres = {}
    for record in samples:
        lon = (record["bbox_lonlat"][0] + record["bbox_lonlat"][2]) / 2
        lat = (record["bbox_lonlat"][1] + record["bbox_lonlat"][3]) / 2
        centres.setdefault(record["mgrs_tile"], []).append((lon, lat))
    tile_centres = {t: (float(np.mean([p[0] for p in v])),
                        float(np.mean([p[1] for p in v])))
                    for t, v in centres.items()}

    split_counts = Counter(r["split"] for r in samples)
    split_types = {s: Counter(r["sample_type"] for r in samples if r["split"] == s)
                   for s in ds.SPLITS}

    return {
        "totals": {
            "candidate_events_discovered": manifest.get("events_discovered"),
            "candidates_attempted": manifest["candidates_attempted"],
            "candidate_pairs_attempted": manifest["candidates_attempted"],
            "accepted_samples": len(samples),
            "accepted_positive": len(positives),
            "accepted_negative": len(negatives),
            "rejected_candidates": rejections["total"],
            "acceptance_rate": round(len(samples) / max(manifest["candidates_attempted"], 1), 4),
        },
        "rejections": {"by_reason": rejections["counts"], "total": rejections["total"]},
        "geography": {
            "tmf_tiles": sorted({r["tmf_tile"] for r in samples}),
            "n_tmf_tiles": len({r["tmf_tile"] for r in samples}),
            "mgrs_tiles": sorted({r["mgrs_tile"] for r in samples}),
            "n_mgrs_tiles": len({r["mgrs_tile"] for r in samples}),
            "n_aois": len({r["aoi_id"] for r in samples}),
            "utm_zones": sorted({r["utm_zone"] for r in samples}),
            "crs": sorted({r["crs"] for r in samples}),
            "extent_lonlat": [round(min(lons), 4), round(min(lats), 4),
                              round(max(lons), 4), round(max(lats), 4)],
            "samples_per_mgrs_tile": dict(Counter(r["mgrs_tile"] for r in samples)),
            "samples_per_tmf_tile": dict(Counter(r["tmf_tile"] for r in samples)),
            "forest_contexts": dict(Counter(
                r.get("forest_context", "unrecorded") for r in samples)),
            "n_forest_contexts": len({r.get("forest_context", "unrecorded")
                                      for r in samples}),
            "events_per_tmf_tile": {
                t: len({r["event_id"] for r in samples
                        if r["tmf_tile"] == t and r["event_id"]})
                for t in sorted({r["tmf_tile"] for r in samples})},
        },
        "temporal": {
            "before_doy": describe([_doy(r, "before") for r in samples], 1),
            "after_doy": describe([_doy(r, "after") for r in samples], 1),
            "abs_doy_difference": describe([abs(_doy(r, "before") - _doy(r, "after"))
                                            for r in samples], 1),
            "seasonal_direction": _seasonal_direction(samples),
            "selection_rule": manifest["protocol"].get("selection_rule"),
            "tmf_years": dict(sorted(Counter(r["tmf_year"] for r in samples).items())),
            "before_dates": dict(sorted(Counter(r["before_datetime"][:10] for r in samples).items())),
            "after_dates": dict(sorted(Counter(r["after_datetime"][:10] for r in samples).items())),
            "date_gap_days": describe([r["date_gap_days"] for r in samples], 1),
            "doy_gap_days": describe([r["doy_gap_days"] for r in samples], 1),
            "seasonal_window": manifest["protocol"]["seasonal_window"],
            "max_doy_gap_observed": max((r["doy_gap_days"] for r in samples), default=None),
            "processing_baselines": dict(sorted(Counter(
                [r["before_baseline"] for r in samples] +
                [r["after_baseline"] for r in samples]).items())),
        },
        "label": {
            "positive_fraction_all_samples": describe(all_fractions, 6),
            "positive_fraction_positive_samples": describe(pos_fractions, 6),
            "positive_fraction_buckets": {b: buckets.get(b, 0) for b in ds.POSITIVE_BUCKETS},
            "positive_fraction_buckets_pct": {
                b: round(100 * buckets.get(b, 0) / max(len(positives), 1), 1)
                for b in ds.POSITIVE_BUCKETS},
            "labelled_px": describe([r["positive_px"] for r in samples], 1),
            "labelled_px_total": int(sum(r["positive_px"] for r in samples)),
            "event_size_distribution": dict(Counter(
                r["event_size"] for r in positives if r["event_size"])),
            "distinct_events_represented": len({r["event_id"] for r in positives
                                                if r["event_id"]}),
            "training_eligibility": dict(Counter(
                r.get("training_eligibility", "unrecorded") for r in samples)),
            "training_eligibility_positives": dict(Counter(
                r.get("training_eligibility", "unrecorded") for r in positives)),
            "min_positive_pixels": manifest["protocol"].get("min_positive_pixels"),
            "positive_px_distribution": describe(
                [r["positive_px"] for r in positives], 1),
            "event_positive_px": describe([r["event_positive_px"] for r in positives], 1),
        },
        "quality": {
            "before_cloud_pct": describe([r["before_cloud_pct"] for r in samples]),
            "after_cloud_pct": describe([r["after_cloud_pct"] for r in samples]),
            "invalid_fraction": describe([r["invalid_fraction"] for r in samples], 6),
            "invalid_threshold": ds.INVALID_MAX_FRACTION,
            "cloud_threshold": manifest["protocol"]["cloud_max_pct"],
            "residual_cloud_fraction": describe(
                [r["residual_cloud_fraction"] for r in samples
                 if r.get("residual_cloud_fraction") is not None], 6),
            "residual_cloud_threshold": manifest["protocol"].get("residual_cloud_threshold"),
            "residual_cloud_method": manifest["protocol"].get("residual_cloud_method"),
            "rejected_by_residual_cloud": rejections["counts"].get("residual_cloud", 0),
        },
        "splits": {
            "counts": {s: split_counts.get(s, 0) for s in ds.SPLITS},
            "by_type": {s: dict(split_types[s]) for s in ds.SPLITS},
            "mgrs_tiles": split_tiles,
            "integrity": ds.split_integrity(split_tiles),
            "separation_km": ds.min_split_separation_km(split_tiles, tile_centres),
            "min_separation_target_km": ds.MIN_SPLIT_SEPARATION_KM,
            "split_unit": ds.SPLIT_UNIT,
        },
        "negatives": {
            "by_subtype": dict(Counter(
                r.get("negative_subtype") or r.get("negative_source")
                for r in negatives)),
            "later_deforestation_count": sum(
                1 for r in negatives
                if (r.get("negative_subtype") or r.get("negative_source"))
                == "later_deforestation"),
            "by_sampling_pool": dict(Counter(r.get("negative_pool") for r in negatives)),
            "pool_shares_requested": manifest["protocol"].get("negative_pool_shares"),
            "negative_to_positive_ratio": (
                round(len(negatives) / len(positives), 3) if positives else None),
            "subtype_by_tmf_tile": {
                t: dict(Counter((r.get("negative_subtype") or r.get("negative_source"))
                                for r in negatives if r["tmf_tile"] == t))
                for t in sorted({r["tmf_tile"] for r in negatives})},
            "subtype_by_split": {
                s: dict(Counter((r.get("negative_subtype") or r.get("negative_source"))
                                for r in negatives if r["split"] == s))
                for s in ds.SPLITS},
            "subtype_vocabulary": list(ev.NEGATIVE_SUBTYPES),
            "note": ("Subtype comes from the TMF year composition of the window, "
                     "not from the imagery. No claim is made that these are hard "
                     "negatives beyond what the dNBR separation below supports."),
        },
    }


def separation_check(samples):
    """Does dNBR actually separate labelled forest loss from everything else?

    Two measurements, answering different questions:

    * whole-window means per class - these overlap by construction, since a
      positive window is usually only a few per cent deforested;
    * inside-label versus outside-label WITHIN each positive sample - a paired
      comparison in which both regions come from the same two acquisitions, so
      atmosphere, sun angle and any seasonal offset cancel.

    The paired figure is the one that carries evidential weight. The unpaired
    class means are reported for completeness, not as proof.
    """
    pos, neg, inside, outside = [], [], [], []
    correct = 0
    for record in samples:
        before, after, label = load_arrays(record["sample_id"])
        invalid = np.load(os.path.join(OUT_DIR, f"{record['sample_id']}_invalid.npy"),
                          mmap_mode="r")
        delta = dnbr_map(record, before, after)
        usable = (~np.asarray(invalid).astype(bool)) & np.isfinite(delta)
        if usable.sum() < 100:
            continue
        value = float(delta[usable].mean())
        (pos if record["sample_type"] == "positive" else neg).append(value)

        mask = np.asarray(label).astype(bool)
        hit, miss = usable & mask, usable & ~mask
        if record["sample_type"] == "positive" and hit.sum() >= 200 and miss.sum() >= 200:
            a, b = float(delta[hit].mean()), float(delta[miss].mean())
            inside.append(a)
            outside.append(b)
            correct += int(a < b)

    return {"positive_samples_mean_dnbr": describe(pos, 4),
            "negative_samples_mean_dnbr": describe(neg, 4),
            "inside_label_dnbr": describe(inside, 4),
            "outside_label_dnbr": describe(outside, 4),
            "paired_separation": describe([i - o for i, o in zip(inside, outside)], 4),
            "samples_in_correct_direction": f"{correct}/{len(inside)}",
            "interpretation": (
                "dNBR falls where vegetation is lost. The paired inside-vs-"
                "outside comparison is the load-bearing evidence: both regions "
                "share one scene pair, so per-scene atmosphere and sun angle "
                "cancel. Whole-window class means overlap by construction.")}


# ------------------------------------------------------------- contact sheet
def choose_examples(samples):
    """Pick representative samples spanning the cases a reviewer must see.

    Coverage is by CATEGORY, not by extremes: one median example from each
    event-size class and one from each negative subtype. Picking the largest
    and smallest instead would show the tails and hide the dataset.
    """
    positives = [r for r in samples if r["sample_type"] == "positive"]
    negatives = [r for r in samples if r["sample_type"] == "negative"]
    chosen, used = [], set()

    def take(record, caption):
        if record and record["sample_id"] not in used:
            used.add(record["sample_id"])
            chosen.append((record, caption))

    eligible = [r for r in positives
                if r.get("training_eligibility") == "training_eligible"]
    for size in ("small", "medium", "large", "very_large"):
        match = sorted((r for r in eligible if r.get("event_size") == size),
                       key=lambda r: r["positive_fraction"])
        take(match[len(match) // 2] if match else None,
             f"positive: {size} event")

    small = [r for r in positives
             if r.get("training_eligibility") == "small_event"]
    take(small[0] if small else None,
         "positive: small_event (below eligibility, kept not trained)")

    for subtype in ("stable_forest", "prior_deforestation",
                    "later_deforestation", "mixed_history"):
        match = [r for r in negatives
                 if (r.get("negative_subtype") or r.get("negative_source")) == subtype]
        take(match[0] if match else None, f"negative: {subtype}")
    return chosen[:10]


def render_split_map(samples, path):
    """Where each split lives geographically. Auditability, not cartography.

    Plain longitude/latitude scatter - no basemap and no mapping dependency,
    which would be a new package for a diagnostic figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#d62728"}
    fig, ax = plt.subplots(figsize=(11, 6))
    for split in ds.SPLITS:
        pts = [r for r in samples if r["split"] == split]
        if not pts:
            continue
        lons = [(r["bbox_lonlat"][0] + r["bbox_lonlat"][2]) / 2 for r in pts]
        lats = [(r["bbox_lonlat"][1] + r["bbox_lonlat"][3]) / 2 for r in pts]
        marks = ["o" if r["sample_type"] == "positive" else "^" for r in pts]
        for lon, lat, mark in zip(lons, lats, marks):
            ax.scatter(lon, lat, c=colours[split], marker=mark, s=34,
                       edgecolors="black", linewidths=0.3, alpha=0.85)
        ax.scatter([], [], c=colours[split], label=f"{split} (n={len(pts)})")
    ax.scatter([], [], c="grey", marker="o", label="positive")
    ax.scatter([], [], c="grey", marker="^", label="negative")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Sample locations by split - MGRS tile is the atomic split unit")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def render(examples, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    rows = len(examples)
    fig, axes = plt.subplots(rows, 3, figsize=(11.5, 3.6 * rows))
    if rows == 1:
        axes = np.array([axes])
    rgb = [s2.BAND_INDEX[b] for b in ("B04", "B03", "B02")]

    for row, (record, caption) in enumerate(examples):
        before, after, label = load_arrays(record["sample_id"])
        ref_b = s2.to_reflectance(np.asarray(before), record["before_offset_applied"],
                                  record["before_baseline"])
        ref_a = s2.to_reflectance(np.asarray(after), record["after_offset_applied"],
                                  record["after_baseline"])
        # Fixed mapping, identical for every sample and both dates, so apparent
        # brightness differences are real rather than an artefact of rescaling.
        def composite(ref):
            scaled = np.stack([ref[:, :, i] for i in rgb], -1) / FIXED_SCALE_MAX
            return np.clip(scaled, 0, 1) ** FIXED_SCALE_GAMMA

        axes[row, 0].imshow(composite(ref_b))
        axes[row, 0].set_title(f"BEFORE {record['before_datetime'][:10]}", fontsize=9)
        axes[row, 1].imshow(composite(ref_a))
        axes[row, 1].set_title(f"AFTER {record['after_datetime'][:10]}", fontsize=9)
        axes[row, 2].imshow(np.asarray(label),
                            cmap=ListedColormap(["#111111", "#ff2d2d"]),
                            vmin=0, vmax=1)
        axes[row, 2].set_title(
            f"TMF {record['tmf_year']}  pos={100*record['positive_fraction']:.2f}%",
            fontsize=9)
        axes[row, 0].set_ylabel(
            f"{record['sample_id']}\n{record['mgrs_tile']} [{record['split']}]\n{caption}",
            fontsize=7.5, rotation=0, ha="right", va="center", labelpad=68)
        for col in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    fig.suptitle("Earth Guardian Phase 4A - environmental dataset contact sheet\n"
                 f"FIXED reflectance scale 0-{FIXED_SCALE_MAX}, gamma "
                 f"{FIXED_SCALE_GAMMA}, identical for every sample; "
                 "label is 30 m-quantised", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(path, dpi=105, bbox_inches="tight")
    plt.close(fig)


def main():
    global OUT_DIR, MANIFEST, AUDIT, SHEET
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=OUT_DIR,
                        help="dataset directory to audit")
    parser.add_argument("--compare", default=V1_MANIFEST,
                        help="manifest of a previous protocol, for the "
                             "old-versus-new temporal comparison")
    args = parser.parse_args()
    OUT_DIR = args.dir
    MANIFEST = os.path.join(OUT_DIR, "manifest.json")
    AUDIT = os.path.join(OUT_DIR, "audit.json")
    SHEET = os.path.join(OUT_DIR, "contact_sheet.png")

    if not os.path.exists(MANIFEST):
        print(f"NOT FOUND: {MANIFEST} - run build_environment_dataset.py first")
        return 1
    with open(MANIFEST, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if not manifest["samples"]:
        print("manifest contains no accepted samples; nothing to audit")
        return 1

    audit = build_audit(manifest)
    audit["separation"] = separation_check(manifest["samples"])
    audit["protocol_comparison"] = compare_protocols(manifest, args.compare)
    audit["reproducibility"] = {
        "seed": manifest["seed"], "script_version": manifest["script_version"],
        "audit_script_version": "phase4a-m2-audit-v1",
        "generated_utc": manifest["generated_utc"],
        "protocol": manifest["protocol"],
        "product_ids": sorted({r["before_product_id"] for r in manifest["samples"]} |
                              {r["after_product_id"] for r in manifest["samples"]}),
    }
    with open(AUDIT, "w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2)

    examples = choose_examples(manifest["samples"])
    render(examples, SHEET)
    split_map = os.path.join(OUT_DIR, "split_map.png")
    render_split_map(manifest["samples"], split_map)

    total = sum(os.path.getsize(os.path.join(OUT_DIR, f))
                for f in os.listdir(OUT_DIR))
    audit["storage_bytes"] = total
    with open(AUDIT, "w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2)

    print(json.dumps({k: audit[k] for k in
                      ("totals", "rejections", "geography", "splits", "negatives")},
                     indent=2))
    print("\nlabel:", json.dumps(audit["label"]["positive_fraction_buckets_pct"]))
    print("positive fraction (positives only):",
          json.dumps(audit["label"]["positive_fraction_positive_samples"]))
    print("separation:", json.dumps(audit["separation"], indent=2))
    print(f"\nstorage {total/1e6:.1f} MB")
    print("audit         ", AUDIT)
    print("contact sheet ", SHEET, f"({len(examples)} examples)")
    print("split map     ", split_map)
    return 0


if __name__ == "__main__":
    sys.exit(main())
