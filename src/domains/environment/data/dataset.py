"""Environmental dataset assembly: gates, splits and manifest (Milestone 2).

Everything here is the bookkeeping that makes the dataset defensible rather
than merely large: the quality gates a candidate must pass, the rejection ledger
that records the ones that did not, the leakage-controlled split, and the
per-sample manifest record.

Nothing in this module touches the network or the model.
"""
from __future__ import annotations

import math

TILE_SIZE = 256
GSD_M = 10.0

#: Union of BEFORE and AFTER unusable pixels may not exceed this fraction.
INVALID_MAX_FRACTION = 0.05

#: Engineering eligibility threshold, NOT a scientific optimum and not claimed
#: to be one. The first build produced positives as small as 9 pixels out of
#: 65,536, which carries too little signal to learn from. 200 px is about 0.31%
#: of a 256x256 window, or roughly 2 hectares on the ground at 10 m.
#:
#: Samples below it are kept, not deleted: they are classified `small_event` and
#: remain available for robustness evaluation, where sensitivity to tiny
#: clearings is exactly what one would want to measure.
MIN_POSITIVE_PIXELS = 200

TRAINING_ELIGIBLE = "training_eligible"
SMALL_EVENT = "small_event"
ELIGIBILITY_CLASSES = (TRAINING_ELIGIBLE, SMALL_EVENT)


def training_eligibility(sample_type: str, positive_px: int,
                         minimum: int = MIN_POSITIVE_PIXELS) -> str:
    """Classify a sample for training use without discarding anything.

    Negatives are always eligible: carrying zero positive pixels is the point
    of a negative, not a deficiency in it.
    """
    if sample_type == "negative":
        return TRAINING_ELIGIBLE
    return TRAINING_ELIGIBLE if positive_px >= minimum else SMALL_EVENT

#: Atomic geographic unit for splitting. Crops from one MGRS tile never straddle
#: two splits.
SPLIT_UNIT = "mgrs_tile"

#: The REQUIREMENT: split blocks must be at least this far apart. Reported,
#: never assumed.
MIN_SPLIT_SEPARATION_KM = 50.0

#: The distance actually used to cluster tiles into blocks. Held above the
#: requirement because, measured on the v2.2 sample set, raising it from 50 km
#: to 100 km cost nothing at all: identical 182/39/39 split counts, identical
#: 70/15/15 proportions, and a slightly BETTER positive balance (92/18/17
#: against 90/15/22), while doubling the achieved separation to 100.3 km.
#: Beyond this it does bite - at 150 km the validation and test positives
#: collapse to 6 and 11 - so this is the last free step, not a maximum.
SPLIT_CLUSTER_KM = 100.0

SPLITS = ("train", "val", "test")

#: Every way a candidate can fail. A candidate is never dropped silently: it
#: lands in exactly one of these buckets and is written to the audit.
REJECT_REASONS = (
    "no_s2_pair",             # no acceptable acquisition in Y-1 and/or Y+1
    "seasonal_window",        # pair outside the seasonal-consistency bound
    "cloud",                  # scene-level eo:cloud_cover >= 20 per cent
    "scl_invalid",            # union invalid fraction above 5 per cent
    "residual_cloud",         # blue-band haze/smoke diagnostic above threshold
    "water_dominated",        # majority open water: not a forest window at all
    "missing_bands",          # an asset was absent or unreadable
    "crs_grid",               # BEFORE and AFTER disagree on CRS or grid
    "utm_zone",               # window would straddle UTM zones
    "class50",                # ongoing 2023-2025 disturbance ambiguity
    "insufficient_imagery",   # window not fully inside the raster
    "invalid_label",          # TMF window unreadable or degenerate
    "insufficient_positive_px",  # positive candidate whose label is empty
    "duplicate_event",        # event already represented by another sample
    "other",
)

#: Rejection reasons renamed by the Milestone 2.2 specification, mapped to the
#: names this project already used. Kept as a mapping rather than a rename so
#: existing manifests, tests and audits stay readable.
REJECT_REASON_ALIASES = {
    "class_50": "class50",
    "missing_band": "missing_bands",
    "grid_mismatch": "crs_grid",
    "unsupported_crs": "crs_grid",
    "outside_utm_constraint": "utm_zone",
    "insufficient_positive_pixels": "insufficient_positive_px",
}


class RejectionLedger:
    """Counts and records every rejected candidate, by reason."""

    def __init__(self):
        self.counts = {reason: 0 for reason in REJECT_REASONS}
        self.records: list = []

    def reject(self, candidate_id: str, reason: str, detail: str = "") -> None:
        if reason not in self.counts:
            reason = "other"
        self.counts[reason] += 1
        self.records.append({"candidate_id": candidate_id, "reason": reason,
                             "detail": detail})

    @property
    def total(self) -> int:
        return sum(self.counts.values())


# ---------------------------------------------------------------------- gates
def class50_ambiguous(year_histogram: dict, target_year: int) -> bool:
    """True when the label of a window cannot be attributed cleanly.

    The ongoing-disturbance class 50 of TMF lives in the transition map, not in
    DeforestationYear, so it should never appear here - its presence means the
    raster is not the layer we think it is. Target years from 2023 onward are
    ambiguous for a second reason: those disturbances are not yet attributed to
    degradation or deforestation, so equality on the year does not imply forest
    loss.
    """
    if any(int(value) == 50 for value in year_histogram):
        return True
    return target_year >= 2023


def invalid_gate(invalid_fraction: float) -> bool:
    return invalid_fraction <= INVALID_MAX_FRACTION


def positive_bucket(fraction: float) -> str:
    """Representativeness bucket for the positive fraction of a sample."""
    if fraction < 0.01:
        return "<1%"
    if fraction < 0.05:
        return "1-5%"
    if fraction < 0.25:
        return "5-25%"
    if fraction < 0.50:
        return "25-50%"
    return ">50%"


POSITIVE_BUCKETS = ("<1%", "1-5%", "5-25%", "25-50%", ">50%")


# --------------------------------------------------------------------- splits
def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in kilometres."""
    radius = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def group_tiles_into_blocks(tile_centres: dict,
                            min_km: float = MIN_SPLIT_SEPARATION_KM) -> list:
    """Cluster MGRS tiles into geographic blocks separated by at least `min_km`.

    Single-linkage: tiles closer than `min_km` are transitively merged, so any
    two resulting blocks are guaranteed to be at least `min_km` apart. Splitting
    on blocks rather than on individual tiles is what stops a train tile sitting
    directly against a test tile.
    """
    names = sorted(tile_centres)
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            lon1, lat1 = tile_centres[a]
            lon2, lat2 = tile_centres[b]
            if haversine_km(lon1, lat1, lon2, lat2) < min_km:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb

    blocks: dict = {}
    for name in names:
        blocks.setdefault(find(name), []).append(name)
    return [sorted(v) for _, v in sorted(blocks.items())]


def assign_splits(blocks: list, weights: dict, ratios=(0.6, 0.2, 0.2)) -> dict:
    """Assign whole geographic blocks to train, val and test.

    Blocks are placed largest-first into whichever split is furthest below its
    target share. The result is deterministic, and it keeps every crop from one
    MGRS tile - indeed from one geographic block - inside a single split.
    """
    order = sorted(blocks, key=lambda b: (-sum(weights.get(t, 0) for t in b), b[0]))
    total = sum(weights.values()) or 1
    targets = dict(zip(SPLITS, (r * total for r in ratios)))
    assigned = {s: [] for s in SPLITS}
    current = {s: 0.0 for s in SPLITS}

    for block in order:
        size = sum(weights.get(t, 0) for t in block)
        split = min(SPLITS, key=lambda s: (current[s] - targets[s], s))
        assigned[split].extend(block)
        current[split] += size
    return {s: sorted(v) for s, v in assigned.items()}


#: How much harder class balance is pushed than size balance. Size is already
#: satisfiable exactly by the greedy packer, so with equal weights the search
#: sits in a local minimum where no single move can pay for itself.
CLASS_BALANCE_WEIGHT = 3.0


def split_cost(assigned: dict, stats: dict, ratios, total: int) -> float:
    """How far an assignment is from the ideal, on SIZE and on CLASS together.

    * size - how far each split is from its target share of samples;
    * class - how far each split's positive count is from the rate the WHOLE
      dataset has. Mirroring the dataset is the right target, not a flat 50/50:
      a split that is 48.8% positive in a dataset that is 48.8% positive is
      balanced, and forcing it to 50 would be arbitrary.

    Geography is not scored, because it is not negotiable. Candidates are whole
    geographic blocks already clustered beyond the separation requirement, so
    every assignment considered is geographically valid by construction and
    balance is optimised strictly INSIDE that constraint.
    """
    rate = (sum(s["pos"] for s in stats.values()) / total) if total else 0.0
    target = dict(zip(SPLITS, ratios))
    size = 0.0
    klass = 0.0
    for split in SPLITS:
        n = sum(stats[t]["n"] for t in assigned[split])
        pos = sum(stats[t]["pos"] for t in assigned[split])
        size += abs(n - target[split] * total)
        klass += abs(pos - rate * n)
    return (size + CLASS_BALANCE_WEIGHT * klass) / max(total, 1)


def assign_splits_balanced(blocks: list, stats: dict, ratios=(0.70, 0.15, 0.15),
                           seed: int = 0, iterations: int = 60000) -> dict:
    """Assign whole geographic blocks to splits, balancing size AND class.

    `assign_splits` packs blocks by size alone and is class-blind, which in the
    v2.3 dataset left the test split with 13 positives against 26 negatives -
    too few positives to measure a segmentation model on.

    The atomic unit is unchanged (a geographic block of MGRS tiles), so
    disjointness and separation are preserved exactly and this cannot express a
    geographically invalid split. What changes is the search: a size-greedy
    start, then hill-climbing over BOTH single-block moves and two-block SWAPS
    between different splits.

    Swaps are what make it work. The greedy start already hits the size targets
    exactly, so any lone move unbalances sizes by about as much as it could
    improve class balance and pure move-descent stalls immediately. Exchanging
    two similarly sized blocks leaves the size term almost untouched while
    changing class composition, which is the move the problem actually needs.

    Deterministic for a given seed.
    """
    import numpy as np

    total = sum(s["n"] for s in stats.values())
    weights = {t: stats[t]["n"] for t in stats}
    seeded = assign_splits(blocks, weights, ratios)
    home = {t: s for s, tiles in seeded.items() for t in tiles}

    by_block = [tuple(b) for b in blocks]
    placed = {b: home[b[0]] for b in by_block}

    def materialise():
        return {s: [t for b in by_block if placed[b] == s for t in b]
                for s in SPLITS}

    best = materialise()
    best_cost = split_cost(best, stats, ratios, total)
    rng = np.random.default_rng(seed)

    for step in range(iterations):
        a = by_block[int(rng.integers(len(by_block)))]
        if step % 2:                                   # single-block move
            origin, target = placed[a], SPLITS[int(rng.integers(len(SPLITS)))]
            if target == origin:
                continue
            undo = [(a, origin)]
            placed[a] = target
        else:                                          # two-block swap
            b = by_block[int(rng.integers(len(by_block)))]
            if placed[a] == placed[b]:
                continue
            undo = [(a, placed[a]), (b, placed[b])]
            placed[a], placed[b] = placed[b], placed[a]

        candidate = materialise()
        if any(not candidate[s] for s in SPLITS):       # never empty a split
            for block, split in undo:
                placed[block] = split
            continue
        cost = split_cost(candidate, stats, ratios, total)
        if cost < best_cost:
            best, best_cost = candidate, cost
        else:
            for block, split in undo:
                placed[block] = split
    return {s: sorted(v) for s, v in best.items()}


def split_integrity(split_tiles: dict) -> dict:
    """Verify that the three MGRS tile sets are pairwise disjoint."""
    sets = {s: set(split_tiles.get(s, ())) for s in SPLITS}
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    intersections = {f"{a}&{b}": sorted(sets[a] & sets[b]) for a, b in pairs}
    return {"intersections": intersections,
            "disjoint": all(not v for v in intersections.values()),
            "counts": {s: len(sets[s]) for s in SPLITS}}


def min_split_separation_km(split_tiles: dict, tile_centres: dict) -> dict:
    """Smallest centre-to-centre distance between tiles of different splits."""
    out = {}
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    for a, b in pairs:
        best = None
        for ta in split_tiles.get(a, ()):
            for tb in split_tiles.get(b, ()):
                if ta not in tile_centres or tb not in tile_centres:
                    continue
                lon1, lat1 = tile_centres[ta]
                lon2, lat2 = tile_centres[tb]
                d = haversine_km(lon1, lat1, lon2, lat2)
                best = d if best is None else min(best, d)
        out[f"{a}-{b}"] = None if best is None else round(best, 2)
    known = [v for v in out.values() if v is not None]
    out["minimum"] = min(known) if known else None
    return out


# ------------------------------------------------------------------- manifest
MANIFEST_FIELDS = (
    "sample_id", "split", "sample_type", "negative_source",
    "tmf_tile", "tmf_year", "tmf_layer", "tmf_version", "event_id",
    "mgrs_tile", "aoi_id",
    "before_product_id", "after_product_id",
    "before_stac_id", "after_stac_id",
    "before_datetime", "after_datetime",
    "before_cloud_pct", "after_cloud_pct",
    "before_baseline", "after_baseline",
    "before_offset_applied", "after_offset_applied", "boa_add_offset",
    "date_gap_days", "doy_gap_days", "seasonal_window",
    # --- Milestone 2.1 protocol correction -------------------------------
    "before_doy", "after_doy", "doy_difference",
    "selection_rule", "selection_seed", "selection_provenance",
    "training_eligibility", "min_positive_pixels",
    "residual_cloud_fraction", "residual_cloud_threshold",
    "residual_cloud_method", "residual_cloud_before", "residual_cloud_after",
    "negative_subtype",
    # --- Milestone 2.3 correction pass -----------------------------------
    "water_fraction", "water_threshold", "water_method",
    "offset_metadata_says", "offset_applied_actual", "offset_overridden",
    "dnbr_reversed", "provenance",
    "bbox", "bbox_lonlat", "crs", "utm_zone", "gsd_m", "transform",
    "tile_size", "bands", "band_resample",
    "invalid_fraction", "positive_fraction", "positive_px",
    "label_source", "label_resample", "tmf_year_histogram",
    "generated_utc", "script_version",
)


def validate_record(record: dict) -> list:
    """Names of required manifest fields that are missing from `record`."""
    return [f for f in MANIFEST_FIELDS if f not in record]
