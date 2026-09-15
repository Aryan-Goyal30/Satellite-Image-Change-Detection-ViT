"""Build ONE inspectable environmental sample - Phase 4A Milestone 1.

TMF deforestation location -> Sentinel-2 T1 + T2 -> cloud/quality filtering ->
aligned 6-band pair -> aligned target mask -> manifest -> visualisation.

This is a VALIDATION milestone, not a dataset builder: it produces a single
sample so a human can answer one question by looking at the figure -

    does the TMF-labelled forest loss actually appear between T1 and T2?

Nothing here trains a model or touches the built-environment domain.

Sources
-------
Labels   JRC Tropical Moist Forest, DeforestationYear (BigTIFF, EPSG:4326, 30 m)
Imagery  Sentinel-2 L2A via the public AWS sentinel-cogs mirror (COG, EPSG:32721,
         10/20 m), read with HTTP range requests - no full-band downloads.

Why the COG mirror and not CDSE: CDSE STAC serves asset hrefs as `s3://eodata/...`
which need credentials. The mirror carries the same ESA products - its
`s2:product_uri` matches the CDSE product id exactly - over anonymous HTTPS.
Product identity in the manifest is therefore still the official CDSE id.

Usage:  python scripts/build_environment_sample.py
"""
import datetime
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config  # noqa: E402
from src.domains.environment.data import sentinel2 as s2  # noqa: E402
from src.domains.environment.data import tmf  # noqa: E402

# --------------------------------------------------------------- study site
# Chosen by scanning the TMF tile for the densest DeforestationYear==2020 block
# whose pixels are valid in BOTH acquisitions. See SELECTION_REASON.
TMF_TILE = "N0_W60"
TMF_PATH = os.path.join(config.DATA_DIR, "environment", "tmf",
                        f"DeforestationYear_{TMF_TILE}.tif")
TMF_ORIGIN_LON = -60.000004962325846
TMF_ORIGIN_LAT = 0.0
TMF_PIXEL_DEG = 0.00026949458523585647
TMF_YEAR = 2020

MGRS_TILE = "21MYS"
UTM_ZONE = 21
UTM_EPSG = "EPSG:32721"
GRID_ORIGIN_E = 699960.0
GRID_ORIGIN_N = 9700000.0
GSD_M = 10.0
WINDOW_COL0 = 10076
WINDOW_ROW0 = 5360
TILE = 256

COG_ROOT = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/21/M/YS"
SCENES = {
    "T1": {
        "base": f"{COG_ROOT}/2019/8/S2B_21MYS_20190810_1_L2A/",
        "stac_id": "S2B_21MYS_20190810_1_L2A",
        "product_id": "S2B_MSIL2A_20190810T140059_N0500_R067_T21MYS_20230505T191316",
        "datetime": "2019-08-10T14:03:35.241000Z",
        "cloud_pct": 0.84, "nodata_pct": 4.8,
        "baseline": "05.00", "offset_applied": True,
    },
    "T2": {
        "base": f"{COG_ROOT}/2021/8/S2B_21MYS_20210809_1_L2A/",
        "stac_id": "S2B_21MYS_20210809_1_L2A",
        "product_id": "S2B_MSIL2A_20210809T140059_N0500_R067_T21MYS_20230116T100735",
        "datetime": "2021-08-09T14:03:29.132000Z",
        "cloud_pct": 0.519779, "nodata_pct": 4.889436,
        "baseline": "05.00", "offset_applied": True,
    },
}

SELECTION_REASON = (
    "Densest TMF DeforestationYear==2020 block whose pixels are valid in BOTH "
    "acquisitions. The first T2 candidate (S2A_21MYS_20210807) was rejected: it "
    "is a 77.1% nodata partial swath and left the site empty. It was replaced by "
    "S2B_21MYS_20210809 (4.9% nodata) from the same August window - a quality "
    "substitution, not a widening of the temporal rule."
)
PAIRING_RULE = ("T1 in Y-1, T2 in Y+1; year Y is skipped because TMF records "
                "only the YEAR of first deforestation, not the date.")
SEASONAL_WINDOW = ("August; T1 2019-08-10 and T2 2021-08-09 are 1 day apart in "
                   "day-of-year and ~6 s apart in time-of-day - well inside "
                   "the +/-45 day rule.")
INVALID_THRESHOLD = 0.05
OUT_DIR = os.path.join(config.DATA_DIR, "environment", "samples")
SAMPLE_ID = "env_amazon_21MYS_2020_0001"
SCRIPT_VERSION = "phase4a-milestone1-v2"


def fetch_window(scene):
    """All six bands plus SCL for one date, on the 10 m sample grid."""
    out = {}
    for band in s2.BANDS + ("SCL",):
        url = scene["base"] + band + ".tif"
        header = s2.read_header(url)
        if s2.BAND_RESOLUTION_M.get(band, 20) == 10:
            arr = s2.read_window(url, header, WINDOW_COL0, WINDOW_ROW0, TILE, TILE)
        else:
            arr = s2.read_window(url, header, WINDOW_COL0 // 2, WINDOW_ROW0 // 2,
                                 TILE // 2, TILE // 2)
            arr = s2.upsample_nearest(arr, 2)
        out[band] = arr
        print(f"    {band}: {arr.shape} {arr.dtype} "
              f"min={arr.min()} max={arr.max()} mean={arr.mean():.1f}", flush=True)
    return out


def sample_grid_lonlat():
    """Longitude/latitude of every 10 m sample-pixel centre."""
    from src.domains.environment.data.geo import utm_to_lonlat
    lons = np.empty((TILE, TILE))
    lats = np.empty((TILE, TILE))
    for row in range(TILE):
        northing = GRID_ORIGIN_N - (WINDOW_ROW0 + row + 0.5) * GSD_M
        for col in range(TILE):
            easting = GRID_ORIGIN_E + (WINDOW_COL0 + col + 0.5) * GSD_M
            lon, lat = utm_to_lonlat(easting, northing, UTM_ZONE, south=True)
            lons[row, col] = lon
            lats[row, col] = lat
    return lons, lats


def build_label(lons, lats):
    """TMF DeforestationYear sampled onto the 10 m grid, nearest-neighbour."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    grid = tmf.geotransform(TMF_ORIGIN_LON, TMF_ORIGIN_LAT, TMF_PIXEL_DEG)
    cols, rows = tmf.lonlat_to_pixel(lons, lats, grid)
    with Image.open(TMF_PATH) as img:
        c0, c1 = int(cols.min()), int(cols.max()) + 1
        r0, r1 = int(rows.min()), int(rows.max()) + 1
        block = np.array(img.crop((c0, r0, c1, r1)))
    years = block[rows - r0, cols - c0]
    return tmf.label_from_years(years, TMF_YEAR), years


def render(before, after, label, path):
    """BEFORE | AFTER | LABEL, with a SWIR diagnostic row."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    ref_b = s2.to_reflectance(before, True)
    ref_a = s2.to_reflectance(after, True)
    rgb = [s2.BAND_INDEX[b] for b in ("B04", "B03", "B02")]
    swir = [s2.BAND_INDEX[b] for b in ("B12", "B08", "B04")]

    # One shared stretch across BOTH dates: stretching each separately could
    # manufacture or hide apparent change.
    stacked = np.concatenate([ref_b, ref_a], axis=0)
    r_lo, r_hi = np.percentile(stacked[:, :, rgb], (2, 98))
    s_lo, s_hi = np.percentile(stacked[:, :, swir], (2, 98))

    def composite(ref, idx, lo, hi):
        return np.clip((np.stack([ref[:, :, i] for i in idx], -1) - lo) / (hi - lo), 0, 1)

    fig, ax = plt.subplots(2, 3, figsize=(16, 10.6))
    ax[0, 0].imshow(composite(ref_b, rgb, r_lo, r_hi))
    ax[0, 0].set_title("BEFORE  T1  2019-08-10   RGB (B4/B3/B2)")
    ax[0, 1].imshow(composite(ref_a, rgb, r_lo, r_hi))
    ax[0, 1].set_title("AFTER   T2  2021-08-09   RGB (B4/B3/B2)")
    ax[0, 2].imshow(label, cmap=ListedColormap(["#111111", "#ff2d2d"]), vmin=0, vmax=1)
    ax[0, 2].set_title(f"TMF DeforestationYear == {TMF_YEAR}\n"
                       f"{int(label.sum()):,} px = {100*label.mean():.1f}% "
                       f"(30 m-quantised)")
    ax[1, 0].imshow(composite(ref_b, swir, s_lo, s_hi))
    ax[1, 0].set_title("BEFORE  SWIR2/NIR/Red (B12/B8/B4)")
    ax[1, 1].imshow(composite(ref_a, swir, s_lo, s_hi))
    ax[1, 1].set_title("AFTER   SWIR2/NIR/Red (B12/B8/B4)")
    overlay = composite(ref_a, rgb, r_lo, r_hi).copy()
    hit = label > 0
    overlay[hit] = 0.62 * overlay[hit] + 0.38 * np.array([1.0, 0.15, 0.15])
    ax[1, 2].imshow(overlay)
    ax[1, 2].set_title("TMF label over AFTER (red = labelled loss)")
    for a in ax.ravel():
        a.axis("off")
    fig.suptitle("Earth Guardian Phase 4A Milestone 1 - one environmental sample\n"
                 f"MGRS {MGRS_TILE} - {UTM_EPSG} - 10 m - {TILE}x{TILE} - "
                 "shared 2-98% stretch across both dates", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    if not os.path.exists(TMF_PATH):
        print(f"NOT FOUND: {TMF_PATH}")
        return 1
    os.makedirs(OUT_DIR, exist_ok=True)

    windows = {}
    for name, scene in SCENES.items():
        print(f"[{name}] {scene['stac_id']}", flush=True)
        windows[name] = fetch_window(scene)

    print("[label] sampling TMF onto the 10 m grid...", flush=True)
    lons, lats = sample_grid_lonlat()
    label, years = build_label(lons, lats)

    before = np.stack([windows["T1"][b] for b in s2.BANDS], axis=-1)
    after = np.stack([windows["T2"][b] for b in s2.BANDS], axis=-1)
    invalid = (s2.invalid_from_scl(windows["T1"]["SCL"])
               | s2.invalid_from_scl(windows["T2"]["SCL"]))

    invalid_fraction = float(invalid.mean())
    positive_fraction = tmf.positive_fraction(label)
    gate = "ACCEPT" if invalid_fraction <= INVALID_THRESHOLD else "REJECT"
    print(f"\n  invalid fraction  {invalid_fraction*100:.4f}%  -> {gate}")
    print(f"  positive fraction {positive_fraction*100:.3f}%")

    np.save(os.path.join(OUT_DIR, "sample_0001_before.npy"), before)
    np.save(os.path.join(OUT_DIR, "sample_0001_after.npy"), after)
    np.save(os.path.join(OUT_DIR, "sample_0001_label.npy"), label)
    np.save(os.path.join(OUT_DIR, "sample_0001_invalid.npy"), invalid.astype(np.uint8))
    visual = os.path.join(OUT_DIR, "sample_0001_visual.png")
    render(before, after, label, visual)

    manifest = {
        "sample_id": SAMPLE_ID, "milestone": "phase4a-m1",
        "tmf_year": TMF_YEAR, "tmf_tile": TMF_TILE, "tmf_layer": tmf.LAYER,
        "tmf_dataset": tmf.DATASET, "tmf_citation": tmf.CITATION,
        "tmf_licence": tmf.LICENCE, "tmf_native_res_m": tmf.NATIVE_RESOLUTION_M,
        "tmf_crs": "EPSG:4326", "tmf_pixel_deg": TMF_PIXEL_DEG,
        "mgrs_tile": MGRS_TILE,
        "aoi_id": f"amazon_{MGRS_TILE}_c{WINDOW_COL0}_r{WINDOW_ROW0}",
        "before_product_id": SCENES["T1"]["product_id"],
        "after_product_id": SCENES["T2"]["product_id"],
        "before_stac_id": SCENES["T1"]["stac_id"],
        "after_stac_id": SCENES["T2"]["stac_id"],
        "before_datetime": SCENES["T1"]["datetime"],
        "after_datetime": SCENES["T2"]["datetime"],
        "before_cloud_pct": SCENES["T1"]["cloud_pct"],
        "after_cloud_pct": SCENES["T2"]["cloud_pct"],
        "before_nodata_pct": SCENES["T1"]["nodata_pct"],
        "after_nodata_pct": SCENES["T2"]["nodata_pct"],
        "before_baseline": SCENES["T1"]["baseline"],
        "after_baseline": SCENES["T2"]["baseline"],
        "boa_add_offset": s2.BOA_ADD_OFFSET,
        "boa_offset_already_applied": True,
        "reflectance_scale": s2.QUANTIFICATION_VALUE,
        "crs": UTM_EPSG, "utm_zone": UTM_ZONE, "hemisphere": "south",
        "gsd_m": GSD_M,
        "transform": [GRID_ORIGIN_E + WINDOW_COL0 * GSD_M, GSD_M, 0.0,
                      GRID_ORIGIN_N - WINDOW_ROW0 * GSD_M, 0.0, -GSD_M],
        "window_col0": WINDOW_COL0, "window_row0": WINDOW_ROW0,
        "tile_size": [TILE, TILE],
        "bbox_utm": [GRID_ORIGIN_E + WINDOW_COL0 * GSD_M,
                     GRID_ORIGIN_N - (WINDOW_ROW0 + TILE) * GSD_M,
                     GRID_ORIGIN_E + (WINDOW_COL0 + TILE) * GSD_M,
                     GRID_ORIGIN_N - WINDOW_ROW0 * GSD_M],
        "bbox_lonlat": [float(lons.min()), float(lats.min()),
                        float(lons.max()), float(lats.max())],
        "bands": list(s2.BANDS),
        "band_resolution_m": {b: s2.BAND_RESOLUTION_M[b] for b in s2.BANDS},
        "band_resample": "B11/B12 20 m -> 10 m nearest-neighbour",
        "scl_invalid_classes": sorted(s2.SCL_INVALID),
        "invalid_fraction": round(invalid_fraction, 6),
        "invalid_threshold": INVALID_THRESHOLD, "invalid_gate": gate,
        "positive_fraction": round(positive_fraction, 6),
        "label_source": f"JRC TMF DeforestationYear == {TMF_YEAR}",
        "label_resample": tmf.LABEL_RESAMPLE,
        "excluded_classes": list(tmf.EXCLUDED_CLASSES),
        "seasonal_window": SEASONAL_WINDOW,
        "pairing_rule": PAIRING_RULE,
        "selection_reason": SELECTION_REASON,
        "imagery_source": "AWS sentinel-cogs public mirror of ESA Sentinel-2 L2A",
        "metadata_source": "CDSE STAC (product ids) + Earth Search STAC (assets, baseline, offset flag)",
        "acquisition_note": ("CDSE STAC assets are s3://eodata and require "
                             "credentials; the public COG mirror carries the same "
                             "products - product_uri matches the CDSE product id."),
        "tmf_year_histogram": {str(k): v for k, v in sorted(tmf.describe_years(years).items())},
        "artifacts": {"before": "sample_0001_before.npy",
                      "after": "sample_0001_after.npy",
                      "label": "sample_0001_label.npy",
                      "invalid": "sample_0001_invalid.npy",
                      "visual": "sample_0001_visual.png"},
        "generated_utc": datetime.datetime.now(datetime.timezone.utc)
                         .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "script_version": SCRIPT_VERSION,
    }
    path = os.path.join(OUT_DIR, "sample_0001_manifest.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\n[write] {OUT_DIR}")
    for name in sorted(os.listdir(OUT_DIR)):
        size = os.path.getsize(os.path.join(OUT_DIR, name))
        print(f"    {size/1024:9.0f} KB  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
