"""Build the Phase 4A environmental dataset - Milestone 2.

Applies the protocol validated in Milestone 1 to many locations automatically:

    TMF event discovery -> deterministic sampling -> Sentinel-2 pairing
    -> quality gates -> aligned 6-band pair + 30 m-quantised label -> manifest

Every candidate ends in exactly one of two places: an accepted sample, or the
rejection ledger with a reason. Nothing is discarded silently.

Cheap gates run before expensive ones. SCL (one small asset per date) decides
the invalid-fraction gate before any of the twelve reflectance windows are
fetched, so a cloudy candidate costs two small reads instead of fourteen.

Usage:  python scripts/build_environment_dataset.py [--limit N]
"""
import argparse
import datetime
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config                                          # noqa: E402
from src.domains.environment.data import dataset as ds          # noqa: E402
from src.domains.environment.data import events as ev           # noqa: E402
from src.domains.environment.data import geo                    # noqa: E402
from src.domains.environment.data import sentinel2 as s2        # noqa: E402
from src.domains.environment.data import stac                   # noqa: E402
from src.domains.environment.data import tmf                    # noqa: E402

Image.MAX_IMAGE_PIXELS = None

SCRIPT_VERSION = "phase4a-milestone2.2-v1"

#: Fixed seed. Re-running with this seed reproduces the same sample ids,
#: subject only to external data availability.
SEED = 20260913

#: Deforestation years used as positive targets.
TARGET_YEARS = (2019, 2020, 2021, 2022)

#: Years counted during the tile scan - TWO beyond the last target year.
#:
#: One is not enough, and the reason is an interaction rather than an oversight.
#: `negative_pools` excludes any block deforested in Y-1, Y or Y+1, so that
#: 30 m-quantised label bleed from an adjoining year cannot contaminate a
#: negative. For the last target year the only candidate "later" year is Y+1 -
#: which that very exclusion removes. Measured with SCAN_YEARS ending at 2023,
#: every tile yielded exactly zero later_deforestation blocks at Y=2022 while
#: 2019-2021 populated normally.
#:
#: Scanning 2023-2024 never makes them positive targets: `class50_ambiguous`
#: still refuses any target year from 2023 onward as unattributable. These years
#: only ever characterise a negative.
SCAN_YEARS = (2019, 2020, 2021, 2022, 2023, 2024)

#: TMF tiles in use. The geographic origin of each is read from its own GeoTIFF
#: tags rather than assumed: the tiles do NOT share a rounded origin (N0_W70
#: starts at -70.000141, N0_W60 at -60.000005) and they differ in width by a
#: pixel, so reusing one origin for the other would shift every label by about
#: half a TMF pixel.
#: Six tiles spanning three forest contexts: central and western Amazon
#: (N0_W60/N0_W70), the southern Amazon deforestation arc and south-western
#: Amazon (S10_W60/S10_W70), and insular plus mainland South-East Asia
#: (N0_E110 Borneo, N10_E100 Indochina). JRC's own continent codes differ
#: accordingly - SAM versus ASI.
#:
#: A tile whose raster is not on disk is skipped with a warning rather than
#: crashing the build, so the tile list and the download state stay decoupled.
TMF_TILES = ("N0_W60", "N0_W70", "S10_W60", "S10_W70", "N0_E110", "N10_E100")

#: Descriptive forest context per tile, for the biome-coverage audit. These
#: label the REGION a tile covers; they are not a biome classification derived
#: from the data, and nothing in the pipeline branches on them.
#:
#: The Congo Basin is absent because the JRC endpoint returns HTTP 500 for every
#: African longitude tried (E000-E030 across N10/N0/S10) while serving Asian
#: tiles normally. Recorded as an acquisition limitation, not as a biome that
#: was chosen against.
TILE_CONTEXT = {
    "N0_W60": "Amazon - central (Para / Amazonas, Brazil)",
    "N0_W70": "Amazon - western (Colombia / Peru / Brazil)",
    "S10_W60": "Amazon - southern deforestation arc (Rondonia / Mato Grosso)",
    "S10_W70": "Amazon - south-western (Acre / Peru / Bolivia)",
    "N0_E110": "South-East Asia - insular (Borneo)",
    "N10_E100": "South-East Asia - mainland (Indochina)",
}
TMF_VERSION = "v1 INT_1982_2025 ID30 (JRC_TMF_DeforestationYear)"

#: Candidates drawn per TMF tile per year. Deliberately more than needed: the
#: gates decide the final count, the sampler does not.
#: 6 tiles x 4 years x (10 + 10) = 480 candidates, for a 200-250 target.
#:
#: A uniform draw is deliberately over-provisioned because supply is wildly
#: uneven. Measured pair availability under the frozen 1 Aug - 15 Sep window:
#: Amazon probes 8/8 (100%), Borneo 1/12 (8%), Indochina 2/12 (17%) - the window
#: is the Amazon dry season and the SE Asian monsoon. Drawing uniformly lets the
#: four Amazon tiles carry the target while the Asian tiles still contribute the
#: genuine biome diversity they can.
#:
#: Over-drawing is cheap where it fails: a `no_s2_pair` rejection costs two STAC
#: queries and reads no imagery at all, so the wasted Asian candidates cost
#: minutes, not hours. No gate is relaxed - only the number of candidates
#: offered to the unchanged gates.
POSITIVES_PER_TILE_YEAR = 10
NEGATIVES_PER_TILE_YEAR = 10

#: Hard ceiling on accepted samples, so a permissive year cannot run away.
MAX_SAMPLES = 260

#: Split proportions. The atomic unit is the MGRS tile, so these are targets
#: that whole geographic blocks are packed against, never exact quotas.
SPLIT_RATIOS = (0.70, 0.15, 0.15)

TILE = ds.TILE_SIZE
OUT_DIR = os.path.join(config.DATA_DIR, "environment", "dataset")
SCAN_DIR = os.path.join(config.DATA_DIR, "environment", "scan")

#: Concurrency for one candidate's asset reads. A 256x256 sample touches 14
#: assets (six bands plus SCL, on two dates), each an independent HTTP range
#: read, so the build is latency-bound rather than CPU-bound - the first full
#: build averaged ~120 s per candidate almost entirely in waiting.
#:
#: This changes I/O SCHEDULING ONLY. The same bytes are fetched, the same gates
#: are applied in the same order, and the same samples are selected; nothing
#: about the frozen v2.1 protocol depends on the order the sockets return in.
ASSET_WORKERS = 6

#: Set by --land-pools-only for a targeted top-up in regions where the
#: stable_forest pool would draw ocean. None means the standard mix.
POOL_SHARES_OVERRIDE = None

_HEADER_CACHE = {}
_HEADER_LOCK = threading.Lock()
_TMF_IMAGES = {}
_TMF_GRIDS = {}
_TMF_LOCK = threading.Lock()


def header(url):
    """COG header, cached - scenes are reused across nearby candidates.

    A duplicate fetch under a race is harmless (the header is immutable), so
    the lock guards only the dictionary, never the network call.
    """
    with _HEADER_LOCK:
        cached = _HEADER_CACHE.get(url)
    if cached is None:
        cached = s2.read_header(url)
        with _HEADER_LOCK:
            _HEADER_CACHE[url] = cached
    return cached


def tmf_image(tile):
    if tile not in _TMF_IMAGES:
        path = os.path.join(config.DATA_DIR, "environment", "tmf",
                            f"DeforestationYear_{tile}.tif")
        _TMF_IMAGES[tile] = Image.open(path)
    return _TMF_IMAGES[tile]


def tmf_grid(tile):
    """Geographic grid of a TMF tile, read from the GeoTIFF tags of that tile."""
    if tile not in _TMF_GRIDS:
        image = tmf_image(tile)
        tiepoint = image.tag_v2[33922]
        scale = image.tag_v2[33550]
        _TMF_GRIDS[tile] = tmf.geotransform(tiepoint[3], tiepoint[4], scale[0])
    return _TMF_GRIDS[tile]


# ------------------------------------------------------------------ scanning
def tile_raster_path(tile):
    return os.path.join(config.DATA_DIR, "environment", "tmf",
                        f"DeforestationYear_{tile}.tif")


def tile_is_usable(tile):
    """Whether a tile raster is present AND fully decodable.

    Existence is not integrity: a partially downloaded raster sits on disk and
    passes an exists() check, then scans as garbage. TIFF writes its last tile
    row last, so decoding the bottom-right block is the cheap test that
    actually catches truncation.
    """
    path = tile_raster_path(tile)
    if not os.path.exists(path):
        return False, "raster not on disk"
    try:
        with Image.open(path) as image:
            width, height = image.size
            np.asarray(image.crop((width - 256, height - 256, width, height)))
    except Exception as exc:
        return False, f"raster unreadable ({type(exc).__name__}) - truncated?"
    return True, ""


def load_scan(tile):
    """Block-level TMF counts for one tile, scanning it if not already cached.

    The cache filename carries the scanned year set: widening SCAN_YEARS must
    not silently reuse a scan that never counted the new years.
    """
    tag = f"{min(SCAN_YEARS)}-{max(SCAN_YEARS)}"
    path = os.path.join(SCAN_DIR, f"scan_{tile}_{tag}.npz")
    if os.path.exists(path):
        return np.load(path)
    os.makedirs(SCAN_DIR, exist_ok=True)
    print(f"[scan] {tile} (first run)", flush=True)
    image = tmf_image(tile)
    width, height = image.size
    nbc, nbr = width // ev.BLOCK, height // ev.BLOCK
    counts = {y: np.zeros((nbr, nbc), np.int32) for y in SCAN_YEARS}
    any_def = np.zeros((nbr, nbc), np.int32)

    for br0 in range(0, nbr, 12):
        br1 = min(br0 + 12, nbr)
        arr = np.asarray(image.crop((0, br0 * ev.BLOCK, nbc * ev.BLOCK,
                                     br1 * ev.BLOCK)))
        nb = br1 - br0

        def blocks(mask):
            return mask.reshape(nb, ev.BLOCK, nbc, ev.BLOCK).sum(axis=(1, 3),
                                                                 dtype=np.int32)
        for year in SCAN_YEARS:
            counts[year][br0:br1] = blocks(arr == year)
        any_def[br0:br1] = blocks(arr > 0)

    np.savez_compressed(path, any_def=any_def,
                        **{f"y{y}": counts[y] for y in SCAN_YEARS})
    return np.load(path)


# ------------------------------------------------------------- window geometry
def window_for_point(scene, lon, lat):
    """Locate a TILE x TILE 10 m window centred on (lon, lat) in a scene.

    Returns (col0, row0, header10, None), or (None, None, None, reason). The
    window must lie wholly inside the raster: a partial window would be padded
    with nodata that the invalid gate cannot see, which is exactly the failure
    that produced the empty first candidate in Milestone 1.
    """
    head = header(scene.asset("B02"))
    origin_e, origin_n = head["origin"]
    gsd = head["resolution"][0]
    if abs(gsd - ds.GSD_M) > 1e-6:
        return None, None, None, "crs_grid"

    # Hemisphere comes from the scene's MGRS band, not from the sign of the
    # latitude: it must match the false northing the raster itself was written
    # with, and the two disagree for a point sitting just across the equator
    # from its own tile.
    easting, northing = geo.lonlat_to_utm(lon, lat, scene.utm_zone,
                                          south=scene.southern)
    col = (easting - origin_e) / gsd
    row = (origin_n - northing) / gsd
    # Even origin so the 20 m bands and SCL subsample exactly.
    col0 = int(round(col - TILE / 2)) & ~1
    row0 = int(round(row - TILE / 2)) & ~1
    if col0 < 0 or row0 < 0 or col0 + TILE > head["width"] or row0 + TILE > head["height"]:
        return None, None, None, "insufficient_imagery"
    return col0, row0, head, None


def read_scl(scene, col0, row0):
    url = scene.asset("SCL")
    head = header(url)
    arr = s2.read_window(url, head, col0 // 2, row0 // 2, TILE // 2, TILE // 2)
    return s2.upsample_nearest(arr, 2)


def read_bands(scene, col0, row0):
    """The six protocol bands on the 10 m grid, as a TILE x TILE x 6 stack.

    Bands are fetched concurrently; the stack is rebuilt in canonical
    s2.BANDS order regardless of completion order.
    """
    def one(band):
        url = scene.asset(band)
        head = header(url)
        if s2.BAND_RESOLUTION_M[band] == 10:
            return s2.read_window(url, head, col0, row0, TILE, TILE)
        return s2.upsample_nearest(
            s2.read_window(url, head, col0 // 2, row0 // 2,
                           TILE // 2, TILE // 2), 2)

    with ThreadPoolExecutor(max_workers=ASSET_WORKERS) as pool:
        planes = list(pool.map(one, s2.BANDS))
    return np.stack(planes, axis=-1)


def read_label(tile, scene, col0, row0, year):
    """TMF years and binary label on the sample grid."""
    head = header(scene.asset("B02"))
    origin_e, origin_n = head["origin"]
    # south=True was hardcoded here while window_for_point derived it from the
    # point. The two agree across the Amazon tiles, which are all southern, and
    # disagree for N10_E100 (Indochina, +10 to 0) - where every candidate would
    # have been rejected as invalid_label and the tile written off as having no
    # usable data.
    lons, lats = geo.grid_lonlat(origin_e, origin_n, col0, row0, TILE, TILE,
                                 ds.GSD_M, scene.utm_zone, south=scene.southern)
    cols, rows = tmf.lonlat_to_pixel(lons, lats, tmf_grid(tile))
    image = tmf_image(tile)
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    if c0 < 0 or r0 < 0 or c1 > image.size[0] or r1 > image.size[1]:
        return None, None, None, "invalid_label"
    block = np.asarray(image.crop((c0, r0, c1, r1)))
    years = block[rows - r0, cols - c0]
    return tmf.label_from_years(years, year), years, (lons, lats), None


# ------------------------------------------------------------------ candidates
def candidate_id(candidate):
    """Stable identifier for a candidate, used by the rejection ledger."""
    return (candidate["event_id"] or
            f"{candidate['tmf_tile']}_{candidate['year']}"
            f"_neg{candidate.get('neg_index', 0):02d}")


def sampling_block(event, counts):
    """Where to centre the sample window for an event.

    The positive-weighted centroid is preferred, because centring on the
    densest spot of every event would bias the dataset toward exactly the dense
    windows the representativeness check exists to prevent.

    But for a sprawling or ring-shaped event the centroid can land in a hole
    between clearings, which in the first build produced two "positive" samples
    whose labels were entirely empty. When the centroid block holds no
    deforestation, the densest block of that same event is used instead.
    """
    row, col = event["centroid_block"]
    r, c = int(round(row)), int(round(col))
    inside = 0 <= r < counts.shape[0] and 0 <= c < counts.shape[1]
    if inside and counts[r, c] >= ev.MIN_BLOCK_POSITIVES:
        return row, col
    return event["peak_block"]


def build_candidates():
    """Deterministic candidate list: positives from events, negatives nearby."""
    candidates = []
    for tile in sorted(TMF_TILES):
        usable, why = tile_is_usable(tile)
        if not usable:
            print(f"[skip] {tile}: {why}", flush=True)
            continue
        scan = load_scan(tile)
        counts = {y: scan[f"y{y}"] for y in SCAN_YEARS}
        for year in TARGET_YEARS:
            found = ev.extract_events(counts[year], year, tile)
            picked = ev.sample_events(found, POSITIVES_PER_TILE_YEAR,
                                      seed=SEED + year)
            for event in picked:
                lon, lat = ev.block_to_lonlat(*sampling_block(event, counts[year]),
                                              grid=tmf_grid(tile))
                candidates.append({"tmf_tile": tile, "year": year, "lon": lon,
                                   "lat": lat, "sample_type": "positive",
                                   "event_id": event["event_id"],
                                   "event_positive_px": event["positive_px"],
                                   "event_size": ev.size_bucket(event["positive_px"]),
                                   "negative_source": None,
                                   "n_events_found": len(found)})
            negatives = ev.select_negative_blocks(
                counts, year, scan["any_def"], NEGATIVES_PER_TILE_YEAR,
                seed=SEED + year, exclude=picked,
                shares=POOL_SHARES_OVERRIDE or ev.NEGATIVE_POOL_SHARES)
            for index, neg in enumerate(negatives):
                lon, lat = ev.block_to_lonlat(*neg["centroid_block"],
                                              grid=tmf_grid(tile))
                candidates.append({"tmf_tile": tile, "year": year, "lon": lon,
                                   "lat": lat, "sample_type": "negative",
                                   "event_id": None, "event_positive_px": 0,
                                   "event_size": None,
                                   "negative_source": neg["pool"],
                                   "n_events_found": len(found),
                                   "neg_index": index})
    return candidates


# -------------------------------------------------------------------- assembly
def process(candidate, ledger, accepted, stamp):
    """Run one candidate through the gates. Returns a manifest record or None."""
    year = candidate["year"]
    cid = candidate_id(candidate)

    # One sample per distinct event. Checked before any network work, so a
    # duplicate costs nothing.
    if candidate["event_id"] and any(r["event_id"] == candidate["event_id"]
                                     for r in accepted):
        ledger.reject(cid, "duplicate_event", candidate["event_id"])
        return None

    try:
        scenes = stac.search_scenes(candidate["lon"], candidate["lat"],
                                    (year - 1, year + 1))
    except Exception as exc:                       # network/service failure
        ledger.reject(cid, "other", f"stac: {type(exc).__name__}")
        return None
    if not scenes:
        ledger.reject(cid, "no_s2_pair", "no scenes returned")
        return None

    last_reason = "no_s2_pair"
    for mgrs in sorted(stac.group_by_tile(scenes)):
        group = stac.group_by_tile(scenes)[mgrs]
        t1, t2, reason, pair_info = stac.select_pair(group, year)
        if reason:
            last_reason = reason
            continue

        col0, row0, head, reason = window_for_point(t1, candidate["lon"],
                                                    candidate["lat"])
        if reason:
            last_reason = reason
            continue
        head2 = header(t2.asset("B02"))
        if head2["origin"] != head["origin"] or head2["width"] != head["width"]:
            last_reason = "crs_grid"
            continue

        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                scl1 = pool.submit(read_scl, t1, col0, row0)
                scl2 = pool.submit(read_scl, t2, col0, row0)
                invalid = (s2.invalid_from_scl(scl1.result()) |
                           s2.invalid_from_scl(scl2.result()))
        except Exception as exc:
            ledger.reject(cid, "missing_bands", f"scl: {type(exc).__name__}")
            return None
        invalid_fraction = float(invalid.mean())
        if not ds.invalid_gate(invalid_fraction):
            ledger.reject(cid, "scl_invalid", f"{invalid_fraction:.4f}")
            return None

        label, years, lonlat, reason = read_label(candidate["tmf_tile"], t1,
                                                  col0, row0, year)
        if reason:
            ledger.reject(cid, reason, "")
            return None
        histogram = tmf.describe_years(years)
        if ds.class50_ambiguous(histogram, year):
            ledger.reject(cid, "class50", f"year {year}")
            return None

        positive_fraction = tmf.positive_fraction(label)
        if candidate["sample_type"] == "negative" and positive_fraction > 0:
            ledger.reject(cid, "invalid_label",
                          f"negative carries {positive_fraction:.4f} positives")
            return None
        # A positive sample whose label is empty is simply mislabelled: the
        # window missed the clearing. It must never reach the dataset.
        if candidate["sample_type"] == "positive" and positive_fraction == 0:
            # Distinct from `small_event`: this window contains NO labelled
            # pixel at all, so it is mislabelled rather than merely small.
            # Small-but-real events are accepted and tagged, never rejected.
            ledger.reject(cid, "insufficient_positive_px",
                          "positive candidate has an empty label")
            return None

        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                fb = pool.submit(read_bands, t1, col0, row0)
                fa = pool.submit(read_bands, t2, col0, row0)
                before, after = fb.result(), fa.result()
        except Exception as exc:
            ledger.reject(cid, "missing_bands", f"{type(exc).__name__}")
            return None

        # Residual-cloud QC, a second gate independent of SCL. SCL passed two
        # haze-blanketed windows in the first build at an invalid fraction of
        # exactly zero. Quality control only: it never touches the label.
        usable = ~invalid
        ref_before = s2.to_reflectance(before, t1.offset_applied, t1.baseline)
        ref_after = s2.to_reflectance(after, t2.offset_applied, t2.baseline)
        decision_before = s2.offset_decision(before, t1.offset_applied, t1.baseline)
        decision_after = s2.offset_decision(after, t2.offset_applied, t2.baseline)
        cloud_before = s2.residual_cloud_fraction(ref_before, usable)
        cloud_after = s2.residual_cloud_fraction(ref_after, usable)
        residual_cloud = max(cloud_before, cloud_after)
        if s2.residual_cloud_suspect(residual_cloud):
            ledger.reject(cid, "residual_cloud",
                          f"{residual_cloud:.4f} > {s2.RESIDUAL_CLOUD_MAX_FRACTION}")
            return None

        # Water gate. TMF records no deforestation over open water, so sea and
        # lake windows qualify as pristine `DeforestationYear == 0` negatives -
        # 13 of 260 v2.2 samples were majority water. Not a forest window.
        water = max(s2.water_fraction(ref_before, usable),
                    s2.water_fraction(ref_after, usable))
        if s2.water_dominated(water):
            ledger.reject(cid, "water_dominated",
                          f"{water:.4f} > {s2.WATER_MAX_FRACTION}")
            return None

        aoi = f"{t1.mgrs_tile}_c{col0}_r{row0}"
        if any(r["aoi_id"] == aoi for r in accepted):
            ledger.reject(cid, "duplicate_event", f"window {aoi} already sampled")
            return None

        index = len(accepted) + 1
        sample_id = f"env_{candidate['tmf_tile']}_{year}_{index:04d}"
        for name, array in (("before", before), ("after", after),
                            ("label", label),
                            ("invalid", invalid.astype(np.uint8))):
            np.save(os.path.join(OUT_DIR, f"{sample_id}_{name}.npy"), array)

        origin_e, origin_n = head["origin"]
        lons, lats = lonlat
        subtype = (ev.negative_subtype(histogram, year)
                   if candidate["sample_type"] == "negative" else None)
        return {
            "sample_id": sample_id, "split": None,
            "sample_type": candidate["sample_type"],
            "negative_source": subtype,
            "negative_pool": candidate["negative_source"],
            "tmf_tile": candidate["tmf_tile"], "tmf_year": year,
            "tmf_layer": tmf.LAYER, "tmf_version": TMF_VERSION,
            "forest_context": TILE_CONTEXT.get(candidate["tmf_tile"], "unrecorded"),
            "event_id": candidate["event_id"],
            "event_positive_px": candidate["event_positive_px"],
            "event_size": candidate["event_size"],
            "mgrs_tile": t1.mgrs_tile,
            "aoi_id": f"{t1.mgrs_tile}_c{col0}_r{row0}",
            "before_product_id": t1.product_id, "after_product_id": t2.product_id,
            "before_stac_id": t1.stac_id, "after_stac_id": t2.stac_id,
            "before_datetime": t1.dt.isoformat() + "Z",
            "after_datetime": t2.dt.isoformat() + "Z",
            "before_cloud_pct": round(t1.cloud_pct, 4),
            "after_cloud_pct": round(t2.cloud_pct, 4),
            "before_nodata_pct": round(t1.nodata_pct, 4),
            "after_nodata_pct": round(t2.nodata_pct, 4),
            "before_baseline": t1.baseline, "after_baseline": t2.baseline,
            "before_offset_applied": t1.offset_applied,
            "after_offset_applied": t2.offset_applied,
            "boa_add_offset": s2.BOA_ADD_OFFSET,
            "date_gap_days": (t2.dt - t1.dt).days,
            "doy_gap_days": stac.doy_gap(t1, t2),
            "seasonal_window": f"{stac.SEASON_START} to {stac.SEASON_END}",
            # --- Milestone 2.1 protocol correction ------------------------
            "before_doy": t1.doy, "after_doy": t2.doy,
            "doy_difference": stac.doy_gap(t1, t2),
            "selection_rule": stac.SELECTION_RULE,
            "selection_seed": SEED,
            "selection_provenance": pair_info,
            "training_eligibility": ds.training_eligibility(
                candidate["sample_type"], int(label.sum())),
            "min_positive_pixels": ds.MIN_POSITIVE_PIXELS,
            "residual_cloud_fraction": round(residual_cloud, 6),
            "residual_cloud_before": round(cloud_before, 6),
            "residual_cloud_after": round(cloud_after, 6),
            "residual_cloud_threshold": s2.RESIDUAL_CLOUD_MAX_FRACTION,
            "residual_cloud_method": s2.RESIDUAL_CLOUD_METHOD,
            "negative_subtype": subtype,
            # --- Milestone 2.3 correction pass ----------------------------
            "water_fraction": round(water, 6),
            "water_threshold": s2.WATER_MAX_FRACTION,
            "water_method": "max over both dates of the share of valid pixels "
                            "with NDWI = (B03-B08)/(B03+B08) > 0",
            "offset_metadata_says": [decision_before["metadata_says"],
                                     decision_after["metadata_says"]],
            "offset_applied_actual": [decision_before["applied"],
                                      decision_after["applied"]],
            "offset_overridden": bool(decision_before["overridden"]
                                      or decision_after["overridden"]),
            "dnbr_reversed": None,
            "provenance": {"source": "built", "milestone": "2.3"},
            "bbox": [origin_e + col0 * ds.GSD_M,
                     origin_n - (row0 + TILE) * ds.GSD_M,
                     origin_e + (col0 + TILE) * ds.GSD_M,
                     origin_n - row0 * ds.GSD_M],
            "bbox_lonlat": [float(lons.min()), float(lats.min()),
                            float(lons.max()), float(lats.max())],
            "crs": f"EPSG:{t1.epsg}", "utm_zone": t1.utm_zone,
            "gsd_m": ds.GSD_M,
            "transform": [origin_e + col0 * ds.GSD_M, ds.GSD_M, 0.0,
                          origin_n - row0 * ds.GSD_M, 0.0, -ds.GSD_M],
            "tile_size": [TILE, TILE], "bands": list(s2.BANDS),
            "band_resample": "B11/B12 20 m -> 10 m nearest-neighbour",
            "invalid_fraction": round(invalid_fraction, 6),
            "positive_fraction": round(positive_fraction, 6),
            "positive_px": int(label.sum()),
            "label_source": f"JRC TMF DeforestationYear == {year}",
            "label_resample": tmf.LABEL_RESAMPLE,
            "tmf_year_histogram": {str(k): v for k, v in sorted(histogram.items())},
            "generated_utc": stamp, "script_version": SCRIPT_VERSION,
        }

    ledger.reject(cid, last_reason, "no usable MGRS tile")
    return None


def apply_overrides(args):
    """Apply CLI overrides for a targeted top-up acquisition.

    None of these touch a quality gate or the temporal rule: they change which
    tiles are visited, how many candidates are drawn, which seed drives the
    draw, and which negative pools it draws from.
    """
    global TMF_TILES, POSITIVES_PER_TILE_YEAR, NEGATIVES_PER_TILE_YEAR
    global SEED, POOL_SHARES_OVERRIDE
    if args.tiles:
        TMF_TILES = tuple(t.strip() for t in args.tiles.split(",") if t.strip())
    if args.positives:
        POSITIVES_PER_TILE_YEAR = args.positives
    if args.negatives:
        NEGATIVES_PER_TILE_YEAR = args.negatives
    if args.seed_offset:
        SEED = SEED + args.seed_offset
    if args.land_pools_only:
        POOL_SHARES_OVERRIDE = ev.LAND_POOL_SHARES


def load_exclusions(path):
    """Existing samples that a top-up must not re-sample.

    They seed the duplicate-event and AOI guards inside `process`, so a top-up
    cannot land on ground an earlier build already covered.
    """
    if not path or not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return json.load(fh).get("samples", [])


def main():
    global OUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=MAX_SAMPLES)
    parser.add_argument("--candidates", type=int, default=0,
                        help="cap candidates attempted (0 = all); used by the "
                             "Milestone 2.1 validation run")
    parser.add_argument("--out", default=OUT_DIR,
                        help="output directory; point at a fresh one to leave "
                             "a previous protocol's dataset intact")
    parser.add_argument("--tiles", default="",
                        help="comma-separated TMF tiles to restrict the build "
                             "to, for targeted top-up acquisition")
    parser.add_argument("--positives", type=int, default=0,
                        help="override positives drawn per tile-year")
    parser.add_argument("--negatives", type=int, default=0,
                        help="override negatives drawn per tile-year")
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="shift the sampling seed so a top-up draws blocks "
                             "the original build did not")
    parser.add_argument("--exclude", default="",
                        help="manifest whose samples must not be re-sampled; "
                             "its records seed the duplicate-event and AOI "
                             "guards without being rebuilt")
    parser.add_argument("--land-pools-only", action="store_true",
                        help="draw negatives only from pools that require prior "
                             "disturbance, which are land by construction - the "
                             "stable_forest pool (any_def == 0) also matches "
                             "open ocean")
    args = parser.parse_args()
    OUT_DIR = args.out
    apply_overrides(args)

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ledger = ds.RejectionLedger()

    candidates = build_candidates()
    if args.candidates and args.candidates < len(candidates):
        # Evenly spaced, not the first N. A prefix would cover only the first
        # TMF tile and the earliest years, which is exactly the spread a
        # temporal-protocol validation needs to exercise. Deterministic.
        step = len(candidates) / args.candidates
        candidates = [candidates[int(i * step)] for i in range(args.candidates)]
    print(f"[candidates] {len(candidates)} "
          f"({sum(1 for c in candidates if c['sample_type'] == 'positive')} positive, "
          f"{sum(1 for c in candidates if c['sample_type'] == 'negative')} negative)\n",
          flush=True)

    excluded = load_exclusions(args.exclude)
    if excluded:
        print(f"[exclude] {len(excluded)} existing samples guard the duplicate "
              f"checks; they are not rebuilt", flush=True)
    accepted = list(excluded)
    started = time.time()
    for i, candidate in enumerate(candidates, 1):
        if len(accepted) - len(excluded) >= args.limit:
            break
        # A transient network failure must cost one candidate, not the whole
        # run: the first build died at candidate 37 of 64 on a single empty
        # HTTP response and wrote no manifest at all.
        try:
            record = process(candidate, ledger, accepted, stamp)
        except Exception as exc:
            ledger.reject(candidate_id(candidate), "other",
                          f"{type(exc).__name__}: {exc}")
            record = None
        mark = "OK " if record else "-- "
        detail = (f"{record['mgrs_tile']} pos={record['positive_fraction']:.3f} "
                  f"inv={record['invalid_fraction']:.3f}" if record
                  else ledger.records[-1]["reason"])
        print(f"  [{i:3d}/{len(candidates)}] {mark} {candidate['tmf_tile']} "
              f"{candidate['year']} {candidate['sample_type'][:3]}  {detail}"
              f"   ({time.time()-started:.0f}s)", flush=True)
        if record:
            accepted.append(record)

    accepted = accepted[len(excluded):]        # drop the guard records again

    # --- splits: whole geographic blocks, never individual crops -------------
    centres = {}
    for record in accepted:
        lon = (record["bbox_lonlat"][0] + record["bbox_lonlat"][2]) / 2
        lat = (record["bbox_lonlat"][1] + record["bbox_lonlat"][3]) / 2
        centres.setdefault(record["mgrs_tile"], []).append((lon, lat))
    tile_centres = {t: (float(np.mean([p[0] for p in v])),
                        float(np.mean([p[1] for p in v])))
                    for t, v in centres.items()}
    weights = {}
    for record in accepted:
        weights[record["mgrs_tile"]] = weights.get(record["mgrs_tile"], 0) + 1

    blocks = ds.group_tiles_into_blocks(tile_centres, min_km=ds.SPLIT_CLUSTER_KM)
    split_tiles = ds.assign_splits(blocks, weights, ratios=SPLIT_RATIOS)
    lookup = {t: s for s, tiles in split_tiles.items() for t in tiles}
    for record in accepted:
        record["split"] = lookup[record["mgrs_tile"]]

    manifest = {
        "dataset": "earth-guardian-environment-phase4a-m2",
        "script_version": SCRIPT_VERSION, "seed": SEED, "generated_utc": stamp,
        "protocol": {
            "tmf_layer": tmf.LAYER, "tmf_version": TMF_VERSION,
            "tmf_tiles": sorted(TMF_TILES), "target_years": list(TARGET_YEARS),
            "scan_years": list(SCAN_YEARS),
            "forest_contexts": TILE_CONTEXT,
            "split_ratios": list(SPLIT_RATIOS),
            "asset_workers": ASSET_WORKERS,
            "seasonal_window": {"start": list(stac.SEASON_START),
                                "end": list(stac.SEASON_END),
                                "max_doy_gap": stac.SEASONAL_MAX_DOY_GAP},
            "cloud_max_pct": stac.CLOUD_MAX_PCT,
            "scl_invalid_classes": sorted(s2.SCL_INVALID),
            "invalid_max_fraction": ds.INVALID_MAX_FRACTION,
            "bands": list(s2.BANDS), "tile_size": [TILE, TILE],
            "gsd_m": ds.GSD_M,
            "label_resample": tmf.LABEL_RESAMPLE,
            "band_resample": "B11/B12 20 m -> 10 m nearest-neighbour",
            "pairing_rule": stac.SELECTION_RULE,
            "selection_rule": stac.SELECTION_RULE,
            "selection_seed": SEED,
            "min_positive_pixels": ds.MIN_POSITIVE_PIXELS,
            "residual_cloud_threshold": s2.RESIDUAL_CLOUD_MAX_FRACTION,
            "residual_cloud_blue": s2.RESIDUAL_CLOUD_BLUE,
            "residual_cloud_method": s2.RESIDUAL_CLOUD_METHOD,
            "negative_pool_shares": [list(x) for x in ev.NEGATIVE_POOL_SHARES],
            "excluded_classes": list(tmf.EXCLUDED_CLASSES),
            "positives_per_tile_year": POSITIVES_PER_TILE_YEAR,
            "negatives_per_tile_year": NEGATIVES_PER_TILE_YEAR,
            "split_unit": ds.SPLIT_UNIT,
            "min_split_separation_km": ds.MIN_SPLIT_SEPARATION_KM,
        },
        "split_tiles": split_tiles,
        "samples": accepted,
        "rejections": {"counts": ledger.counts, "total": ledger.total,
                       "records": ledger.records},
        "candidates_attempted": len(candidates),
    }
    path = os.path.join(OUT_DIR, "manifest.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\n[done] accepted {len(accepted)}, rejected {ledger.total}")
    print(f"       {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
