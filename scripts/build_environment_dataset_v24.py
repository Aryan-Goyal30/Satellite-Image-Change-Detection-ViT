"""Assemble dataset_v24 - Milestone 2.4 sampling and evaluation correction.

v2.4 is v2.3 with two changes and nothing else:

  1. Splits are reassigned with a CLASS-AWARE geographic optimiser. v2.3 packed
     geographic blocks by size alone and was blind to class, leaving the test
     split with 13 positives against 26 negatives - too few positives to
     measure a segmentation model on. The atomic unit is unchanged, so MGRS
     disjointness and the 100 km separation are preserved exactly.

  2. Additional Asian LAND samples are appended from a targeted top-up, drawn
     under the frozen protocol with every gate intact.

Every v2.3 sample is carried byte-identically. No label is touched, no QC gate
is relaxed, and the temporal rule is untouched: BEFORE = Y-1, AFTER = Y+1, both
inside 1 Aug - 15 Sep, minimising |DOY difference|, ties broken by summed cloud
then STAC id.

Usage:  python scripts/build_environment_dataset_v24.py
"""
import argparse
import datetime
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config                                       # noqa: E402
from src.domains.environment.data import dataset as ds       # noqa: E402
from src.domains.environment.data import sentinel2 as s2     # noqa: E402

SCRIPT_VERSION = "phase4a-milestone2.4-v1"
V23_DIR = os.path.join(config.DATA_DIR, "environment", "dataset_v23")
TOPUP_DIR = os.path.join(config.DATA_DIR, "environment", "asia_topup")
OUT_DIR = os.path.join(config.DATA_DIR, "environment", "dataset_v24")
ARRAYS = ("before", "after", "label", "invalid")
ASIA_TILES = {"N0_E110", "N10_E100"}

#: Seeds tried for the split optimiser. The lowest-cost assignment wins; the
#: search is deterministic per seed, so the winner is reproducible.
SPLIT_SEEDS = 24


def nbr(ref):
    nir = ref[:, :, s2.BAND_INDEX["B08"]]
    swir = ref[:, :, s2.BAND_INDEX["B12"]]
    denom = nir + swir
    safe = np.abs(denom) > 0.01
    return np.where(safe, (nir - swir) / np.where(safe, denom, 1.0), np.nan)


def dnbr_reversed(record, directory):
    """Whether inside-label dNBR fails to fall below outside-label dNBR."""
    if record["sample_type"] != "positive" or record["positive_px"] < 200:
        return None
    sid = record["sample_id"]
    before = np.load(os.path.join(directory, f"{sid}_before.npy"))
    after = np.load(os.path.join(directory, f"{sid}_after.npy"))
    label = np.load(os.path.join(directory, f"{sid}_label.npy")).astype(bool)
    invalid = np.load(os.path.join(directory, f"{sid}_invalid.npy")).astype(bool)
    ref_b = s2.to_reflectance(before, record["before_offset_applied"],
                              record["before_baseline"])
    ref_a = s2.to_reflectance(after, record["after_offset_applied"],
                              record["after_baseline"])
    delta = nbr(ref_a) - nbr(ref_b)
    ok = ~invalid & np.isfinite(delta)
    hit, miss = ok & label, ok & ~label
    if hit.sum() < 50 or miss.sum() < 50:
        return None
    return bool(delta[hit].mean() >= delta[miss].mean())


def copy_arrays(sid, src, dst, new_sid=None):
    for name in ARRAYS:
        shutil.copy2(os.path.join(src, f"{sid}_{name}.npy"),
                     os.path.join(dst, f"{new_sid or sid}_{name}.npy"))


def load_topup():
    """Asian samples acquired by the targeted top-up, if it produced any."""
    path = os.path.join(TOPUP_DIR, "manifest.json")
    if not os.path.exists(path):
        return [], {}
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("samples", []), data.get("rejections", {})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=SPLIT_SEEDS)
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    with open(os.path.join(V23_DIR, "manifest.json"), encoding="utf-8") as fh:
        v23 = json.load(fh)
    carried = []
    for record in v23["samples"]:
        updated = dict(record)
        prior = dict(record.get("provenance") or {})
        updated["provenance"] = {
            "source": "carried_from_v23", "origin_sample_id": record["sample_id"],
            "arrays": "byte-identical", "v23_provenance": prior,
            "sampling_reason": "existing v2.3 sample, split reassigned only"}
        carried.append(updated)
        copy_arrays(record["sample_id"], V23_DIR, OUT_DIR)
    print(f"[carry] {len(carried)} samples from v23, arrays byte-identical")

    # --- append the targeted Asian land top-up ------------------------------
    topup, topup_rejections = load_topup()
    added = []
    for index, record in enumerate(topup, 1):
        old = record["sample_id"]
        new = f"{old.rsplit('_', 1)[0]}_a{index:03d}"
        copy_arrays(old, TOPUP_DIR, OUT_DIR, new_sid=new)
        updated = dict(record)
        updated["sample_id"] = new
        updated["dnbr_reversed"] = dnbr_reversed(record, TOPUP_DIR)
        updated["provenance"] = {
            "source": "new_in_v24", "origin_sample_id": old,
            "acquisition": "targeted SE Asia land top-up",
            "sampling_reason": ("SE Asia fell to 4.2% of v2.3 after ocean "
                                "negatives were removed; drawn from land-"
                                "guaranteed negative pools under the frozen "
                                "protocol with every gate intact"),
            "replaces": None,
            "pool_shares": "LAND_POOL_SHARES (stable_forest reduced to 0.20)"}
        added.append(updated)
    print(f"[topup] {len(added)} Asian land samples appended")

    samples = carried + added
    positives = sum(1 for r in samples if r["sample_type"] == "positive")
    asia = sum(1 for r in samples if r["tmf_tile"] in ASIA_TILES)
    print(f"[total] {len(samples)} samples, {positives} positive, "
          f"{len(samples) - positives} negative, {asia} SE Asia "
          f"({100 * asia / len(samples):.1f}%)")

    # --- class-aware geographic split ---------------------------------------
    centres = {}
    for record in samples:
        lon = (record["bbox_lonlat"][0] + record["bbox_lonlat"][2]) / 2
        lat = (record["bbox_lonlat"][1] + record["bbox_lonlat"][3]) / 2
        centres.setdefault(record["mgrs_tile"], []).append((lon, lat))
    tile_centres = {t: (float(np.mean([p[0] for p in v])),
                        float(np.mean([p[1] for p in v])))
                    for t, v in centres.items()}
    stats = {}
    for record in samples:
        entry = stats.setdefault(record["mgrs_tile"], {"n": 0, "pos": 0})
        entry["n"] += 1
        entry["pos"] += int(record["sample_type"] == "positive")

    blocks = ds.group_tiles_into_blocks(tile_centres, min_km=ds.SPLIT_CLUSTER_KM)
    best, best_cost, best_seed = None, None, None
    for seed in range(args.seeds):
        candidate = ds.assign_splits_balanced(blocks, stats, seed=seed)
        cost = ds.split_cost(candidate, stats, (0.70, 0.15, 0.15), len(samples))
        if best_cost is None or cost < best_cost:
            best, best_cost, best_seed = candidate, cost, seed

    separation = ds.min_split_separation_km(best, tile_centres)
    integrity = ds.split_integrity(best)
    if separation["minimum"] is not None and separation["minimum"] < ds.MIN_SPLIT_SEPARATION_KM:
        print(f"ABORT: separation {separation['minimum']} km below the "
              f"{ds.MIN_SPLIT_SEPARATION_KM} km requirement")
        return 1
    if not integrity["disjoint"]:
        print("ABORT: MGRS sets are not disjoint")
        return 1

    lookup = {t: s for s, tiles in best.items() for t in tiles}
    for record in samples:
        record["split"] = lookup[record["mgrs_tile"]]
    print(f"[split] {len(blocks)} blocks, seed {best_seed}, cost {best_cost:.4f}, "
          f"minimum separation {separation['minimum']} km")
    for split in ds.SPLITS:
        rows = [r for r in samples if r["split"] == split]
        pos = sum(1 for r in rows if r["sample_type"] == "positive")
        print(f"    {split:<6} n={len(rows):<4} pos={pos:<4} neg={len(rows) - pos:<4} "
              f"asia={sum(1 for r in rows if r['tmf_tile'] in ASIA_TILES)}")

    protocol = dict(v23["protocol"])
    protocol.update({
        "split_assignment": ("class-aware geographic optimiser over whole "
                             "blocks: size-greedy start, then hill-climbing "
                             "over single-block moves and two-block swaps"),
        "split_cost_class_weight": ds.CLASS_BALANCE_WEIGHT,
        "split_seeds_tried": args.seeds,
        "land_pool_shares": "stable_forest reduced to 0.20 for SE Asia top-up",
    })

    manifest = {
        "dataset": "earth-guardian-environment-phase4a-m2.4",
        "script_version": SCRIPT_VERSION, "seed": v23["seed"],
        "generated_utc": stamp, "derived_from": "dataset_v23",
        "protocol": protocol, "split_tiles": best, "samples": samples,
        "rejections": v23["rejections"],
        "topup_rejections": topup_rejections,
        "candidates_attempted": v23["candidates_attempted"] + len(topup),
        "correction_pass": {
            "split_optimiser_cost": round(best_cost, 6),
            "split_optimiser_seed": best_seed,
            "split_separation_km": separation,
            "split_integrity": integrity,
            "asia_samples_added": len(added),
            "carried_from_v23": len(carried),
            "notes": ("v2.3 samples carried byte-identically; only the split "
                      "assignment changed. Asian additions use the frozen "
                      "temporal rule and every existing QC gate."),
        },
    }
    path = os.path.join(OUT_DIR, "manifest.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\n[done] {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
