"""TMF deforestation event discovery and deterministic sampling (Milestone 2).

A dataset needs to be built from *events* - distinct deforestation regions -
rather than from arbitrary windows, so that samples are spatially independent
and one large clearing cannot supply hundreds of near-duplicate crops.

Working resolution
------------------
A TMF tile is ~37k x 37k px (1.4 billion), far too large to label connected
components on directly in memory. A scan pass reduces it to a grid of
BLOCK x BLOCK counts, where BLOCK is chosen so one block is approximately one
256 x 256 @ 10 m sample window. Components are then labelled on that grid: an
"event" is a connected cluster of blocks containing deforestation in year Y.

This costs boundary precision - two clearings less than one block apart merge
into a single event - which is acceptable because the block grid is only used to
*choose* sample locations. Every label written to disk is still sampled from the
full-resolution TMF raster.

scipy is not a project dependency, so connected-component labelling is
implemented here with union-find over the sparse set of positive cells.
"""
from __future__ import annotations

import numpy as np

#: TMF pixels per block edge. 85 px x ~30 m is about 2.55 km, which matches one
#: 256 px x 10 m sample window.
BLOCK = 85

#: A block must hold at least this many positive pixels to seed an event, which
#: suppresses isolated TMF speckle.
MIN_BLOCK_POSITIVES = 25

#: An event must hold at least this many positive pixels to be a candidate.
MIN_EVENT_POSITIVES = 200


def label_components(mask: np.ndarray) -> np.ndarray:
    """4-connected component labels (1..n, with 0 as background).

    Two-pass union-find. Only True cells are visited, so the cost scales with
    the positive area rather than with the size of the grid.
    """
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    labels = np.zeros((height, width), np.int32)
    parent = [0]

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for row in range(height):
        for col in np.flatnonzero(mask[row]):
            up = labels[row - 1, col] if row else 0
            left = labels[row, col - 1] if col else 0
            if up and left:
                labels[row, col] = min(up, left)
                union(up, left)
            elif up or left:
                labels[row, col] = up or left
            else:
                parent.append(len(parent))
                labels[row, col] = len(parent) - 1

    roots = np.array([find(i) for i in range(len(parent))], np.int32)
    remap = np.zeros(len(parent), np.int32)
    for new, old in enumerate(sorted(set(roots[1:].tolist())), start=1):
        remap[roots == old] = new
    return remap[labels]


def extract_events(counts: np.ndarray, year: int, tmf_tile: str,
                   min_block: int = MIN_BLOCK_POSITIVES,
                   min_event: int = MIN_EVENT_POSITIVES) -> list:
    """Connected deforestation events from a block-count grid.

    `counts[r, c]` is the number of DeforestationYear == `year` pixels inside
    block (r, c). The centroid is positive-weighted, so a sample centred on it
    sits on the clearing rather than on the geometric middle of the event.
    """
    counts = np.asarray(counts)
    labels = label_components(counts >= min_block)
    events = []
    for index in range(1, int(labels.max()) + 1):
        rows, cols = np.nonzero(labels == index)
        weights = counts[rows, cols].astype(np.float64)
        total = int(weights.sum())
        if total < min_event:
            continue
        peak = int(np.argmax(weights))
        events.append({
            "event_id": f"{tmf_tile}_{year}_{index:04d}",
            "tmf_tile": tmf_tile,
            "year": year,
            "n_blocks": int(rows.size),
            "positive_px": total,
            "centroid_block": (float((rows * weights).sum() / weights.sum()),
                               float((cols * weights).sum() / weights.sum())),
            # Densest block of the event. A sprawling or ring-shaped event can
            # have a centroid that falls in a hole, so this is the fallback that
            # guarantees a window actually containing deforestation.
            "peak_block": (float(rows[peak]), float(cols[peak])),
            "bbox_block": (int(rows.min()), int(cols.min()),
                           int(rows.max()), int(cols.max())),
        })
    events.sort(key=lambda e: e["event_id"])
    return events


def size_bucket(positive_px: int) -> str:
    """Event-size class, counted in TMF pixels of about 900 m2 each."""
    if positive_px < 1000:
        return "small"
    if positive_px < 10000:
        return "medium"
    if positive_px < 100000:
        return "large"
    return "very_large"


BUCKETS = ("small", "medium", "large", "very_large")


def sample_events(events: list, n: int, seed: int,
                  min_separation_blocks: int = 8) -> list:
    """Deterministically choose up to `n` events, spread out and size-mixed.

    Three properties the audit depends on:

    * **Deterministic** - the same events and seed give the same selection, so
      the dataset can be regenerated without recording a manual choice.
    * **Size-stratified** - drawing round-robin across size buckets stops the
      dataset collapsing onto the few huge clearings, which are both the easiest
      to detect and the least representative.
    * **Spatially separated** - events whose centroids are closer than
      `min_separation_blocks` are skipped, so sample windows do not overlap.
    """
    by_bucket: dict = {b: [] for b in BUCKETS}
    for event in sorted(events, key=lambda e: e["event_id"]):
        by_bucket[size_bucket(event["positive_px"])].append(event)

    rng = np.random.default_rng(seed)
    for bucket in BUCKETS:
        order = rng.permutation(len(by_bucket[bucket]))
        by_bucket[bucket] = [by_bucket[bucket][i] for i in order]

    chosen: list = []
    cursors = {b: 0 for b in BUCKETS}
    while len(chosen) < n:
        progressed = False
        for bucket in BUCKETS:
            if len(chosen) >= n:
                break
            items = by_bucket[bucket]
            while cursors[bucket] < len(items):
                candidate = items[cursors[bucket]]
                cursors[bucket] += 1
                progressed = True
                if _far_enough(candidate, chosen, min_separation_blocks):
                    chosen.append(candidate)
                    break
        if not progressed:
            break
    return sorted(chosen, key=lambda e: e["event_id"])


def _far_enough(candidate: dict, chosen: list, min_blocks: int) -> bool:
    row, col = candidate["centroid_block"]
    for other in chosen:
        if other.get("tmf_tile") != candidate.get("tmf_tile"):
            continue
        orow, ocol = other["centroid_block"]
        if abs(row - orow) < min_blocks and abs(col - ocol) < min_blocks:
            return False
    return True


def block_to_lonlat(block_row: float, block_col: float, grid: dict,
                    block: int = BLOCK) -> tuple:
    """Block-grid coordinates -> (lon, lat) of the centre of that block."""
    px = (block_col + 0.5) * block
    py = (block_row + 0.5) * block
    return (grid["origin_lon"] + px * grid["pixel_deg"],
            grid["origin_lat"] - py * grid["pixel_deg"])


# ------------------------------------------------------------------ negatives
#: Negative subtypes, derived from the TMF year composition of the window.
#: These describe what the LABEL source says, not what the imagery shows - the
#: spectral character is recorded separately and never used to rename a class.
NEGATIVE_SUBTYPES = ("stable_forest", "prior_deforestation",
                     "later_deforestation", "mixed_history")


def negative_subtype(year_histogram: dict, target_year: int) -> str:
    """Classify a negative window from its TMF DeforestationYear histogram.

    `later_deforestation` is the interesting case: the window *is* deforested,
    just not in the target year, so it carries genuine clearing signal under a
    negative label for this pair. `prior_deforestation` supplies existing roads,
    clearings and regrowth. Neither is asserted to be a "hard negative" - that
    claim needs evidence from the imagery, which the audit reports separately.
    """
    years = [int(y) for y, count in year_histogram.items()
             if int(y) != 0 and count > 0]
    if not years:
        return "stable_forest"
    earlier = any(y < target_year for y in years)
    later = any(y > target_year for y in years)
    if earlier and later:
        return "mixed_history"
    return "later_deforestation" if later else "prior_deforestation"


#: Desired mix of negative sampling pools. `later_deforestation` is deliberately
#: over-weighted: it is the hardest and scarcest case - genuine clearing carried
#: under a NEGATIVE label for this particular pair - and the first build yielded
#: exactly one of them out of thirteen negatives.
#:
#: These are sampling INTENTS derived from block-level year counts. The subtype
#: recorded in the manifest always comes from the full TMF year histogram of the
#: finished window via `negative_subtype`, which may disagree; the intent is
#: provenance, the histogram is authority.
NEGATIVE_POOL_SHARES = (("later_deforestation", 0.40),
                        ("prior_deforestation", 0.30),
                        ("stable_forest", 0.30))

#: Mix for regions where the stable_forest pool is unsafe. That pool is defined
#: as any_def == 0, and open ocean satisfies it perfectly - 9 of the 13
#: water-dominated samples in the v2.2 build came from it, all in the insular
#: and coastal Asian tiles. The disturbed pools require prior deforestation and
#: are therefore land by construction.
#:
#: stable_forest is reduced rather than removed: undisturbed forest is a class
#: worth having, and the water gate still filters the ocean draws. This changes
#: only WHERE candidates are drawn, never which of them are accepted.
LAND_POOL_SHARES = (("later_deforestation", 0.40),
                    ("prior_deforestation", 0.40),
                    ("stable_forest", 0.20))


def negative_pools(counts_by_year: dict, target_year: int,
                   any_def: np.ndarray) -> dict:
    """Boolean masks of blocks eligible for each negative sampling pool.

    A candidate must be clean of the target year AND of its immediate
    neighbours, then eroded by one block, so that 30 m-quantised label bleed
    from an adjoining clearing cannot contaminate it - a 256 px @ 10 m window
    spans 2.56 km while a block spans 2.55 km, so it always overlaps neighbours.
    """
    target = np.asarray(counts_by_year[target_year])
    clean = target == 0
    for neighbour in (target_year - 1, target_year + 1):
        if neighbour in counts_by_year:
            clean &= np.asarray(counts_by_year[neighbour]) == 0
    clean = _erode8(clean)

    later = np.zeros_like(target, dtype=np.int64)
    for year, counts in counts_by_year.items():
        if int(year) > target_year:
            later = later + np.asarray(counts)

    any_def = np.asarray(any_def)
    later_mask = clean & (later >= MIN_BLOCK_POSITIVES)
    return {
        "later_deforestation": later_mask,
        "prior_deforestation": clean & (any_def >= MIN_BLOCK_POSITIVES) & ~later_mask,
        "stable_forest": clean & (any_def == 0),
    }


def select_negative_blocks(counts_by_year: dict, target_year: int,
                           any_def: np.ndarray, n: int, seed: int,
                           exclude=(), min_separation_blocks: int = 8,
                           shares=NEGATIVE_POOL_SHARES) -> list:
    """Deterministically choose negative sample blocks near real events.

    Negatives are not random forest patches. The draw is stratified across the
    pools in `shares`, so the dataset deliberately contains old clearings, later
    clearings and untouched canopy rather than whatever happened to be nearby.

    A pool that cannot fill its quota does not waste it: the shortfall spills
    into the remaining pools in order, so the caller still receives `n` blocks
    where the tile can supply them.
    """
    pools = negative_pools(counts_by_year, target_year, any_def)
    rng = np.random.default_rng(seed)
    chosen: list = []

    quotas = [(name, int(round(n * share))) for name, share in shares]
    # Two passes: honour the quotas, then spill leftovers into any pool with
    # supply remaining.
    for wanted_only in (True, False):
        for name, quota in quotas:
            mask = pools.get(name)
            if mask is None or not mask.any():
                continue
            limit = quota if wanted_only else n
            taken = sum(1 for c in chosen if c["pool"] == name)
            rows, cols = np.nonzero(mask)
            for i in rng.permutation(rows.size):
                if len(chosen) >= n or (wanted_only and taken >= limit):
                    break
                candidate = {"centroid_block": (float(rows[i]), float(cols[i])),
                             "pool": name, "tmf_tile": None}
                if not _far_enough(candidate, chosen, min_separation_blocks):
                    continue
                if any(_same_cell(candidate, e) for e in exclude):
                    continue
                if any(_same_cell(candidate, e) for e in chosen):
                    continue
                chosen.append(candidate)
                taken += 1
            if len(chosen) >= n:
                break
        if len(chosen) >= n:
            break
    return chosen


def _erode8(mask: np.ndarray) -> bool:
    """True only where a cell and all eight of its neighbours are True.

    Borders are cleared rather than wrapped, so a block on the edge of the tile
    is never treated as clean on the strength of the opposite edge.
    """
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                out &= np.roll(np.roll(mask, dr, axis=0), dc, axis=1)
    out[0, :] = out[-1, :] = False
    out[:, 0] = out[:, -1] = False
    return out


def _same_cell(candidate: dict, other: dict) -> bool:
    row, col = candidate["centroid_block"]
    orow, ocol = other["centroid_block"]
    return abs(row - orow) < 1 and abs(col - ocol) < 1
