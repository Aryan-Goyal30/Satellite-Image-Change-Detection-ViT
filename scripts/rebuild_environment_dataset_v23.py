"""Rebuild the environmental dataset as v2.3 - Milestone 2.3 correction pass.

v2.3 is v2.2 corrected, not v2.2 rebuilt. Three defects found by the v2.2 audit
are addressed, and everything else is carried across byte-identically:

  1. BOA offset. One baseline-04.00 window reported boa_offset_applied=False
     while its stored DN were already corrected; obeying the flag drove 66.6%
     of it negative. The stored arrays are RAW DN and were never wrong - the
     fault was in the conversion - so no pixel needs reprocessing. What does
     need recomputing is every value DERIVED from the conversion, namely the
     residual-cloud fraction.

  2. Water. 13 of 260 samples were majority open water, all negatives. TMF
     records no deforestation over sea, so ocean is a perfect
     DeforestationYear == 0 negative. They are dropped and replaced with land
     negatives.

  3. Spatial holdout. Clustering at 100 km instead of 50 km costs nothing on
     this sample set and doubles the achieved separation.

The frozen temporal-selection protocol is untouched: BEFORE = Y-1, AFTER = Y+1,
both inside 1 Aug - 15 Sep, minimising |DOY difference|, ties broken by summed
cloud then STAC id.

Usage:  python scripts/rebuild_environment_dataset_v23.py [--target 260]
"""
import argparse
import datetime
import importlib.util
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config                                       # noqa: E402
from src.domains.environment.data import dataset as ds       # noqa: E402
from src.domains.environment.data import events as ev        # noqa: E402
from src.domains.environment.data import sentinel2 as s2     # noqa: E402
from src.domains.environment.data import tmf                 # noqa: E402

SCRIPT_VERSION = "phase4a-milestone2.3-v1"
SRC_DIR = os.path.join(config.DATA_DIR, "environment", "dataset_v22")
OUT_DIR = os.path.join(config.DATA_DIR, "environment", "dataset_v23")
ARRAYS = ("before", "after", "label", "invalid")

#: Replacement negatives are drawn with a seed offset from the original so the
#: draw is different but still deterministic.
REPLACEMENT_SEED_OFFSET = 1000


def load_builder():
    """The v2.2 builder, reused for acquiring replacement negatives."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "build_environment_dataset.py")
    spec = importlib.util.spec_from_file_location("builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def nbr(ref):
    nir = ref[:, :, s2.BAND_INDEX["B08"]]
    swir = ref[:, :, s2.BAND_INDEX["B12"]]
    denom = nir + swir
    safe = np.abs(denom) > 0.01
    return np.where(safe, (nir - swir) / np.where(safe, denom, 1.0), np.nan)


def reassess(record, src_dir):
    """Recompute everything derived from the conversion. Local, no network.

    The arrays are raw DN and are trusted as-is; only values COMPUTED from them
    are recomputed, under the corrected, data-driven offset rule.
    """
    sid = record["sample_id"]
    before = np.load(os.path.join(src_dir, f"{sid}_before.npy"))
    after = np.load(os.path.join(src_dir, f"{sid}_after.npy"))
    label = np.load(os.path.join(src_dir, f"{sid}_label.npy"))
    invalid = np.load(os.path.join(src_dir, f"{sid}_invalid.npy")).astype(bool)
    usable = ~invalid

    db = s2.offset_decision(before, record["before_offset_applied"],
                            record["before_baseline"])
    da = s2.offset_decision(after, record["after_offset_applied"],
                            record["after_baseline"])
    ref_b = s2.to_reflectance(before, record["before_offset_applied"],
                              record["before_baseline"])
    ref_a = s2.to_reflectance(after, record["after_offset_applied"],
                              record["after_baseline"])

    residual = max(s2.residual_cloud_fraction(ref_b, usable),
                   s2.residual_cloud_fraction(ref_a, usable))
    water = max(s2.water_fraction(ref_b, usable), s2.water_fraction(ref_a, usable))

    reversed_dnbr = None
    if record["sample_type"] == "positive" and record["positive_px"] >= 200:
        delta = nbr(ref_a) - nbr(ref_b)
        ok = usable & np.isfinite(delta)
        mask = label.astype(bool)
        hit, miss = ok & mask, ok & ~mask
        if hit.sum() >= 50 and miss.sum() >= 50:
            reversed_dnbr = bool(delta[hit].mean() >= delta[miss].mean())

    return {
        "residual_cloud_fraction": round(residual, 6),
        "water_fraction": round(water, 6),
        "water_threshold": s2.WATER_MAX_FRACTION,
        "water_method": ("max over both dates of the share of valid pixels with "
                         "NDWI = (B03-B08)/(B03+B08) > 0"),
        "offset_metadata_says": [db["metadata_says"], da["metadata_says"]],
        "offset_applied_actual": [db["applied"], da["applied"]],
        "offset_overridden": bool(db["overridden"] or da["overridden"]),
        "offset_negative_fraction": [round(db["negative_fraction"], 6),
                                     round(da["negative_fraction"], 6)],
        "dnbr_reversed": reversed_dnbr,
    }


def copy_arrays(sid, src_dir, dst_dir):
    """Byte-identical carry-across. copy2 preserves content and mtime."""
    for name in ARRAYS:
        shutil.copy2(os.path.join(src_dir, f"{sid}_{name}.npy"),
                     os.path.join(dst_dir, f"{sid}_{name}.npy"))


def acquire_replacements(builder, carried, wanted, ledger, stamp, attempts_cap):
    """NETWORK. Draw fresh land negatives to restore the count.

    Seeded differently from the original draw so the blocks are new, and the
    builder's own duplicate guards (event id, AOI) stop it re-sampling anything
    already carried. The water gate now runs inside `process`, so a replacement
    cannot itself be water.
    """
    replacements, attempted = [], 0
    for tile in sorted(builder.TMF_TILES):
        usable, why = builder.tile_is_usable(tile)
        if not usable:
            continue
        scan = builder.load_scan(tile)
        counts = {y: scan[f"y{y}"] for y in builder.SCAN_YEARS}
        for year in builder.TARGET_YEARS:
            if len(replacements) >= wanted or attempted >= attempts_cap:
                break
            blocks = ev.select_negative_blocks(
                counts, year, scan["any_def"], 4,
                seed=builder.SEED + REPLACEMENT_SEED_OFFSET + year)
            grid = builder.tmf_grid(tile)
            for index, block in enumerate(blocks):
                if len(replacements) >= wanted or attempted >= attempts_cap:
                    break
                lon, lat = ev.block_to_lonlat(*block["centroid_block"], grid=grid)
                candidate = {"tmf_tile": tile, "year": year, "lon": lon, "lat": lat,
                             "sample_type": "negative", "event_id": None,
                             "event_positive_px": 0, "event_size": None,
                             "negative_source": block["pool"], "n_events_found": 0,
                             "neg_index": 900 + index}
                attempted += 1
                record = builder.process(candidate, ledger,
                                         carried + replacements, stamp)
                if record:
                    replacements.append(record)
                    print(f"    + {record['sample_id']} {record['mgrs_tile']} "
                          f"{record['negative_subtype']} water="
                          f"{record.get('water_fraction', 0):.3f}", flush=True)
    return replacements, attempted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=260)
    parser.add_argument("--attempts", type=int, default=90)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(os.path.join(SRC_DIR, "manifest.json"), encoding="utf-8") as fh:
        source = json.load(fh)

    print(f"[v22] {len(source['samples'])} samples")
    ledger = ds.RejectionLedger()

    # --- phase 1: reassess every carried sample, locally --------------------
    carried, dropped = [], []
    for record in source["samples"]:
        updated = dict(record)
        updated.update(reassess(record, SRC_DIR))
        if s2.water_dominated(updated["water_fraction"]):
            ledger.reject(record["sample_id"], "water_dominated",
                          f"{updated['water_fraction']:.4f}")
            dropped.append({"sample_id": record["sample_id"],
                            "reason": "water_dominated",
                            "water_fraction": updated["water_fraction"],
                            "sample_type": record["sample_type"],
                            "negative_subtype": record.get("negative_subtype"),
                            "tmf_tile": record["tmf_tile"]})
            continue
        if s2.residual_cloud_suspect(updated["residual_cloud_fraction"]):
            ledger.reject(record["sample_id"], "residual_cloud",
                          f"{updated['residual_cloud_fraction']:.4f}")
            dropped.append({"sample_id": record["sample_id"],
                            "reason": "residual_cloud_on_recompute",
                            "residual_cloud_fraction": updated["residual_cloud_fraction"],
                            "sample_type": record["sample_type"],
                            "tmf_tile": record["tmf_tile"]})
            continue
        updated["provenance"] = {
            "source": "carried_from_v22", "origin_sample_id": record["sample_id"],
            "arrays": "byte-identical", "recomputed": [
                "residual_cloud_fraction", "water_fraction", "offset_*", "dnbr_reversed"],
            "offset_overridden": updated["offset_overridden"]}
        carried.append(updated)

    overridden = [r["sample_id"] for r in carried if r["offset_overridden"]]
    reversed_ids = [r["sample_id"] for r in carried if r.get("dnbr_reversed")]
    print(f"[carry] {len(carried)} kept, {len(dropped)} dropped "
          f"({sum(1 for d in dropped if d['reason'] == 'water_dominated')} water)")
    print(f"[offset] overridden by the data-driven guard: {len(overridden)} {overridden}")
    print(f"[dnbr] reversed inside-vs-outside: {len(reversed_ids)}")

    for record in carried:
        copy_arrays(record["sample_id"], SRC_DIR, OUT_DIR)

    # --- phase 2: replace the dropped negatives ----------------------------
    wanted = max(0, args.target - len(carried))
    print(f"[replace] seeking {wanted} land negatives (cap {args.attempts} attempts)",
          flush=True)
    builder = load_builder()
    builder.OUT_DIR = OUT_DIR
    replacements, attempted = ([], 0)
    if wanted:
        replacements, attempted = acquire_replacements(
            builder, carried, wanted, ledger, stamp, args.attempts)

    # Give replacements unmistakable ids so they can never collide with a
    # carried sample and are visible as replacements in the manifest.
    dropped_ids = [d["sample_id"] for d in dropped]
    for index, record in enumerate(replacements):
        old = record["sample_id"]
        new = f"{old.rsplit('_', 1)[0]}_r{index + 1:03d}"
        for name in ARRAYS:
            src = os.path.join(OUT_DIR, f"{old}_{name}.npy")
            if os.path.exists(src):
                os.replace(src, os.path.join(OUT_DIR, f"{new}_{name}.npy"))
        record["sample_id"] = new
        record["provenance"] = {
            "source": "new_in_v23", "reason": "replaces a water-dominated negative",
            "replaces": dropped_ids[index] if index < len(dropped_ids) else None,
            "seed": builder.SEED + REPLACEMENT_SEED_OFFSET}
        record.setdefault("dnbr_reversed", None)

    samples = carried + replacements
    print(f"[total] {len(samples)} samples "
          f"({sum(1 for r in samples if r['sample_type'] == 'positive')} positive, "
          f"{sum(1 for r in samples if r['sample_type'] == 'negative')} negative)")

    # --- phase 3: splits at the stronger clustering distance ----------------
    centres = {}
    for record in samples:
        lon = (record["bbox_lonlat"][0] + record["bbox_lonlat"][2]) / 2
        lat = (record["bbox_lonlat"][1] + record["bbox_lonlat"][3]) / 2
        centres.setdefault(record["mgrs_tile"], []).append((lon, lat))
    tile_centres = {t: (float(np.mean([p[0] for p in v])),
                        float(np.mean([p[1] for p in v])))
                    for t, v in centres.items()}
    weights = {}
    for record in samples:
        weights[record["mgrs_tile"]] = weights.get(record["mgrs_tile"], 0) + 1

    blocks = ds.group_tiles_into_blocks(tile_centres, min_km=ds.SPLIT_CLUSTER_KM)
    split_tiles = ds.assign_splits(blocks, weights, ratios=(0.70, 0.15, 0.15))
    lookup = {t: s for s, tiles in split_tiles.items() for t in tiles}
    for record in samples:
        record["split"] = lookup[record["mgrs_tile"]]
    separation = ds.min_split_separation_km(split_tiles, tile_centres)
    print(f"[splits] clustered at {ds.SPLIT_CLUSTER_KM} km -> "
          f"{len(blocks)} blocks, minimum separation {separation['minimum']} km")

    protocol = dict(source["protocol"])
    protocol.update({
        "water_threshold": s2.WATER_MAX_FRACTION,
        "water_method": ("share of valid pixels with NDWI = (B03-B08)/(B03+B08) "
                         "> 0, max over both dates"),
        "max_negative_reflectance_fraction": s2.MAX_NEGATIVE_REFLECTANCE_FRACTION,
        "offset_rule": ("metadata rule (baseline >= 04.00 and not "
                        "boa_offset_applied), overridden when applying it would "
                        "drive more than 1% of pixels below zero reflectance"),
        "split_cluster_km": ds.SPLIT_CLUSTER_KM,
        "min_split_separation_km": ds.MIN_SPLIT_SEPARATION_KM,
    })

    manifest = {
        "dataset": "earth-guardian-environment-phase4a-m2.3",
        "script_version": SCRIPT_VERSION, "seed": source["seed"],
        "generated_utc": stamp, "derived_from": "dataset_v22",
        "protocol": protocol, "split_tiles": split_tiles, "samples": samples,
        "rejections": {"counts": ledger.counts, "total": ledger.total,
                       "records": ledger.records},
        "candidates_attempted": source["candidates_attempted"] + attempted,
        "correction_pass": {
            "dropped": dropped,
            "replacements_attempted": attempted,
            "replacements_accepted": len(replacements),
            "offset_overridden_samples": overridden,
            "dnbr_reversed_samples": reversed_ids,
            "notes": ("v2.2 arrays are raw DN and were never wrong; the BOA fault "
                      "was in the conversion, so carried arrays are byte-identical "
                      "and only derived values were recomputed."),
        },
    }
    path = os.path.join(OUT_DIR, "manifest.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\n[done] {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
