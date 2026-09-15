"""Sentinel-2 L2A access for the Environment domain (Phase 4A).

Windowed reads from Cloud-Optimized GeoTIFFs over HTTP range requests, decoded
with zlib and numpy only. No rasterio, no GDAL - consistent with
src/common/georef.py, which reads GeoTIFF metadata through Pillow for the same
reason.

Why windowed: a single 10 m band for one MGRS tile is ~210 MB. A 256x256 sample
touches at most a handful of the COG's internal 1024x1024 tiles, so one sample
costs a few hundred KB instead of gigabytes.

Two decoding details that are easy to get wrong and silently corrupt data:

* TIFF **Predictor 2** (horizontal differencing) is applied before Deflate on
  these products. Without reversing it the values are garbage that still
  "looks like" an array - reflectance came out spanning the full uint16 range
  instead of a plausible 0-0.4.
* **BitsPerSample differs per asset**: reflectance bands are 16-bit with
  1024x1024 internal tiles, while SCL is 8-bit with 512x512 tiles. Assuming
  uint16 everywhere fails on SCL.

Reflectance scaling and the baseline-04.00 offset are handled by
`to_reflectance`, which refuses to double-correct.
"""
from __future__ import annotations

import struct
import subprocess
import time
import zlib

import numpy as np

#: Model input bands, in the fixed order used by every artifact.
BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")

#: Native ground sample distance of each band, in metres.
BAND_RESOLUTION_M = {"B02": 10, "B03": 10, "B04": 10, "B08": 10,
                     "B11": 20, "B12": 20}

#: Index of each band within a stacked sample.
BAND_INDEX = {b: i for i, b in enumerate(BANDS)}

#: RGB composite bands (red, green, blue) for visual inspection only.
RGB_BANDS = ("B04", "B03", "B02")

TARGET_RESOLUTION_M = 10

#: Scene Classification Layer values treated as unusable.
#: 0 no-data, 1 saturated/defective, 3 cloud shadow, 8 cloud medium,
#: 9 cloud high, 10 thin cirrus, 11 snow/ice.
SCL_INVALID = frozenset({0, 1, 3, 8, 9, 10, 11})

SCL_MEANING = {
    0: "no data", 1: "saturated or defective", 2: "dark area",
    3: "cloud shadow", 4: "vegetation", 5: "bare soil", 6: "water",
    7: "cloud low probability / unclassified", 8: "cloud medium probability",
    9: "cloud high probability", 10: "thin cirrus", 11: "snow or ice",
}

#: L2A digital numbers are reflectance x 10000.
QUANTIFICATION_VALUE = 10000.0

#: Processing baseline 04.00 (from 2022-01-25) shifts the dynamic range.
BOA_ADD_OFFSET = -1000.0

_CURL_TIMEOUT = "120"

#: Transient empty responses from S3 are retried rather than raised. A single
#: one of them aborted a 40-minute dataset build at candidate 37 of 64, so the
#: cost of not retrying is far higher than the cost of a few seconds of backoff.
_RETRIES = 4
_RETRY_BACKOFF_S = 2.0


# ----------------------------------------------------------------- transport
def _range_get(url: str, start: int, end: int) -> bytes:
    """One HTTP range request, retried on a transient empty response."""
    stderr = ""
    for attempt in range(_RETRIES):
        proc = subprocess.run(
            ["curl", "-s", "--max-time", _CURL_TIMEOUT, "-r", f"{start}-{end}", url],
            capture_output=True)
        if proc.stdout:
            return proc.stdout
        stderr = proc.stderr.decode("utf-8", "replace")[:200]
        if attempt + 1 < _RETRIES:
            time.sleep(_RETRY_BACKOFF_S * (attempt + 1))
    raise OSError(f"empty response for bytes {start}-{end} of {url} after "
                  f"{_RETRIES} attempts: {stderr}")


# -------------------------------------------------------------- COG metadata
def read_header(url: str, probe_bytes: int = 131072) -> dict:
    """Parse the COG's first IFD: geometry, tiling, compression, tile index."""
    head = _range_get(url, 0, probe_bytes - 1)
    endian = "<"
    n_entries = struct.unpack(endian + "H", head[8:10])[0]

    tags = {}
    for i in range(n_entries):
        base = 10 + i * 12
        tag, typ, count = struct.unpack(endian + "HHI", head[base:base + 8])
        value = struct.unpack(endian + "I", head[base + 8:base + 12])[0]
        tags[tag] = (typ, count, value)

    def array(tag):
        typ, count, value = tags[tag]
        size = 4 if typ == 4 else 2
        need = count * size
        raw = (head[value:value + need] if value + need <= len(head)
               else _range_get(url, value, value + need - 1))
        return struct.unpack(endian + ("I" if typ == 4 else "H") * count, raw[:need])

    def doubles(tag):
        _typ, count, value = tags[tag]
        return struct.unpack(endian + "d" * count, head[value:value + count * 8])

    bits = tags[258][2]
    scale = doubles(33550)
    tiepoint = doubles(33922)
    return {
        "width": tags[256][2], "height": tags[257][2],
        "tile_width": tags[322][2], "tile_height": tags[323][2],
        "compression": tags[259][2], "predictor": tags.get(317, (0, 0, 1))[2],
        "bits": bits, "dtype": "<u1" if bits == 8 else "<u2",
        "tile_offsets": array(324), "tile_counts": array(325),
        "origin": (tiepoint[3], tiepoint[4]),
        "resolution": (scale[0], scale[1]),
    }


def read_window(url: str, header: dict, col: int, row: int,
                width: int, height: int) -> np.ndarray:
    """Read a pixel window, fetching only the internal tiles it overlaps."""
    tw, th = header["tile_width"], header["tile_height"]
    across = (header["width"] + tw - 1) // tw
    dtype = np.uint8 if header["bits"] == 8 else np.uint16
    out = np.zeros((height, width), dtype)

    for tr in range(row // th, (row + height - 1) // th + 1):
        for tc in range(col // tw, (col + width - 1) // tw + 1):
            index = tr * across + tc
            offset = header["tile_offsets"][index]
            count = header["tile_counts"][index]
            if count == 0:
                continue
            raw = zlib.decompress(_range_get(url, offset, offset + count - 1))
            tile = np.frombuffer(raw, dtype=header["dtype"]).reshape(th, tw).copy()
            if header["predictor"] == 2:
                tile = np.cumsum(tile, axis=1, dtype=tile.dtype)
            oy, ox = tr * th, tc * tw
            y0, y1 = max(row, oy), min(row + height, oy + th)
            x0, x1 = max(col, ox), min(col + width, ox + tw)
            out[y0 - row:y1 - row, x0 - col:x1 - col] = tile[y0 - oy:y1 - oy,
                                                             x0 - ox:x1 - ox]
    return out


# ------------------------------------------------------------- array helpers
def upsample_nearest(array: np.ndarray, factor: int = 2) -> np.ndarray:
    """Nearest-neighbour upsample. Used for 20 m -> 10 m band and SCL alignment.

    Nearest, not bilinear: SCL is categorical, and interpolating class codes
    would invent classes that do not exist. The 20 m reflectance bands use the
    same method so every layer of a sample shares one resampling rule.
    """
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    return np.repeat(np.repeat(array, factor, axis=0), factor, axis=1)


def invalid_from_scl(scl: np.ndarray) -> np.ndarray:
    """Boolean unusable-pixel mask from a Scene Classification Layer array."""
    return np.isin(scl, list(SCL_INVALID))


# ------------------------------------------------- residual-cloud diagnostic
#: SCL misses thin cloud, haze and smoke. In the first environmental build two
#: of 86 windows were blanketed by haze that SCL passed at an invalid fraction
#: of exactly zero - one of them pale across the whole frame, the other veiled
#: by burning-season smoke.
#:
#: The statistic is the fraction of valid pixels whose BLUE surface reflectance
#: exceeds RESIDUAL_CLOUD_BLUE. Blue is the discriminator because haze, smoke
#: and cloud scatter strongly at short wavelengths, while bright bare soil - the
#: thing that must NOT be flagged - is red- and SWIR-sloped and stays low in
#: blue.
#:
#: Calibrated on those 86 windows (two Amazon TMF tiles, 4 UTM zones):
#:     clean forest             0.0000
#:     cleared field grid       0.0004
#:     bright bare soil         0.0024   <- worst legitimate case
#:     haze / smoke             0.0959 - 1.0000
#: The rejection threshold sits about 20x above the worst legitimate case and
#: about 2x below the mildest genuine haze. It is an ENGINEERING threshold
#: calibrated in one biome, not a universal physical constant, which is why it
#: is a named, configurable parameter and why the fraction is recorded for every
#: sample whether or not it trips.
RESIDUAL_CLOUD_BLUE = 0.15
RESIDUAL_CLOUD_MAX_FRACTION = 0.05
RESIDUAL_CLOUD_METHOD = (
    "fraction of valid pixels with B02 surface reflectance > "
    f"{RESIDUAL_CLOUD_BLUE} (blue-band haze/smoke/thin-cloud diagnostic); "
    "quality control only, never used to decide the forest-loss target")


def residual_cloud_fraction(reflectance: np.ndarray, valid=None) -> float:
    """Fraction of valid pixels that look like haze, smoke or thin cloud.

    QUALITY CONTROL ONLY. This says nothing about whether a pixel is
    deforested - it reports whether the atmosphere makes the observation
    untrustworthy. It must never be fed into the label.

    `reflectance` is a HxWx6 stack in the canonical band order, already
    converted by `to_reflectance` so the per-scene baseline and offset have been
    handled. `valid` is an optional boolean mask of usable pixels.
    """
    blue = np.asarray(reflectance)[:, :, BAND_INDEX["B02"]]
    mask = (np.ones(blue.shape, bool) if valid is None
            else np.asarray(valid, dtype=bool))
    if not mask.any():
        return 1.0                      # nothing usable is maximally suspect
    return float((blue[mask] > RESIDUAL_CLOUD_BLUE).mean())


def residual_cloud_suspect(fraction: float,
                           threshold: float = RESIDUAL_CLOUD_MAX_FRACTION) -> bool:
    """Whether a window is too atmospherically contaminated to trust."""
    return fraction > threshold


def needs_offset(offset_applied: bool, baseline=None) -> bool:
    """Whether BOA_ADD_OFFSET still has to be subtracted from stored DN.

    The offset exists ONLY for products processed with baseline 04.00 or later
    (introduced 2022-01-25). For anything earlier there is no offset to undo,
    and `earthsearch:boa_offset_applied` reads False merely because the concept
    does not apply - not because a correction is outstanding.

    Keying on that flag alone is therefore wrong, and destructively so: tropical
    forest sits at 200-600 DN in the visible bands, so subtracting 1000 turns
    B02/B03/B04 negative, which is impossible for surface reflectance. Measured
    across this project's own archive, mean DN differs between the pre- and
    post-04.00 groups by -137 to +80 per band - nothing like the +1000 an
    outstanding offset would show.

    `baseline` is required whenever `offset_applied` is False, because that is
    exactly the case the flag cannot resolve on its own.
    """
    if offset_applied:
        return False
    if baseline is None:
        raise ValueError("baseline is required to decide the BOA offset when "
                         "offset_applied is False")
    return float(baseline) >= 4.0


#: Largest share of negative-reflectance pixels a correct conversion may yield.
#:
#: Surface reflectance cannot be negative, but atmospheric correction can leave
#: a thin negative tail over deep water and shadow, so the bound is empirical
#: rather than exactly zero. Measured across the 520 windows of the v2.2 build:
#: 519 produced EXACTLY zero negative pixels, and the single metadata failure
#: produced 66.6%. There is no overlap whatsoever, so 1% sits two orders of
#: magnitude below the failure and above everything legitimate.
MAX_NEGATIVE_REFLECTANCE_FRACTION = 0.01


def offset_decision(dn: np.ndarray, offset_applied: bool, baseline=None) -> dict:
    """Decide from the DATA whether the BOA offset is genuinely outstanding.

    `earthsearch:boa_offset_applied` is not always right. In the v2.2 build one
    baseline-04.00 product reported `False` while its stored DN were already
    corrected; subtracting 1000 drove 66.6% of that window negative, and its
    six band means matched its own partner date to within ~30 DN.

    So the metadata rule is applied provisionally and then CHECKED against the
    result: if it would push an implausible share of pixels below zero, it is
    overridden and the DN are taken as already corrected. This detects the
    catastrophic failure - a whole-image shift - without touching bright or
    specular pixels, because it only ever inspects the negative tail.

    Returns the full decision so callers can record provenance rather than
    silently diverging from the metadata.
    """
    metadata_says = needs_offset(offset_applied, baseline)
    decision = {"metadata_says": metadata_says, "applied": metadata_says,
                "overridden": False, "negative_fraction": 0.0}
    if not metadata_says:
        return decision

    shifted = (np.asarray(dn).astype(np.float32) + BOA_ADD_OFFSET) / QUANTIFICATION_VALUE
    negative = float((shifted < 0).mean())
    decision["negative_fraction"] = negative
    if negative > MAX_NEGATIVE_REFLECTANCE_FRACTION:
        decision["applied"] = False
        decision["overridden"] = True
    return decision


def to_reflectance(dn: np.ndarray, offset_applied: bool,
                   baseline=None, trust_metadata: bool = False) -> np.ndarray:
    """Digital numbers -> surface reflectance.

    `offset_applied` and `baseline` come from the product metadata
    (`earthsearch:boa_offset_applied`, `s2:processing_baseline`). Applying the
    offset when the distributor already did it shifts every value by 0.1
    reflectance; applying it to a pre-04.00 product shifts values that never
    carried it. See `needs_offset` for the metadata rule and `offset_decision`
    for the data-driven check that overrides it when the DN contradict it.

    `trust_metadata=True` skips that check and obeys the flag exactly - for
    tests that need the unguarded behaviour, not for production reads.
    """
    values = np.asarray(dn).astype(np.float32)
    apply = (needs_offset(offset_applied, baseline) if trust_metadata
             else offset_decision(dn, offset_applied, baseline)["applied"])
    if apply:
        values = values + BOA_ADD_OFFSET
    return values / QUANTIFICATION_VALUE


# ------------------------------------------------------------ water exclusion
#: Open water is not forest, and TMF records no deforestation over it, so a sea
#: or lake window is a perfect `DeforestationYear == 0` negative. In the v2.2
#: build 13 of 260 samples (5.0%) were more than half water, four of them
#: essentially entirely ocean - all of them negatives.
#:
#: NDWI = (green - NIR) / (green + NIR) is positive over water because water
#: absorbs NIR strongly while vegetation reflects it. Measured on v2.2, land
#: samples sat at a median NDWI>0 share of 0.0000 (p90 0.0090) while the water
#: samples ran 0.558-1.000, so the 0.50 cut falls in a wide empty gap.
WATER_NDWI_THRESHOLD = 0.0
WATER_MAX_FRACTION = 0.50


def water_fraction(reflectance: np.ndarray, valid=None) -> float:
    """Share of valid pixels that look like open water.

    Quality control only. Like the residual-cloud diagnostic this never touches
    the forest-loss target - it decides whether a WINDOW belongs in a forest
    dataset at all, not whether any pixel is deforested.
    """
    ref = np.asarray(reflectance)
    green = ref[:, :, BAND_INDEX["B03"]].astype(np.float32)
    nir = ref[:, :, BAND_INDEX["B08"]].astype(np.float32)
    ndwi = (green - nir) / (green + nir + 1e-6)
    mask = (np.ones(ndwi.shape, bool) if valid is None
            else np.asarray(valid, dtype=bool))
    if not mask.any():
        return 1.0
    return float((ndwi[mask] > WATER_NDWI_THRESHOLD).mean())


def water_dominated(fraction: float,
                    threshold: float = WATER_MAX_FRACTION) -> bool:
    """Whether a window is too much open water to belong in a forest dataset."""
    return fraction > threshold
