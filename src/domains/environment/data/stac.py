"""Sentinel-2 scene discovery and temporal pairing (Phase 4A Milestone 2).

Milestone 1 hardcoded one validated scene pair. A dataset needs the same
protocol applied automatically to many locations, so this module turns a
coordinate plus a TMF deforestation year into a BEFORE/AFTER pair - or into an
explicit, logged refusal.

Network and logic are deliberately separated: `search_scenes` is the only
function that touches the network, and every selection rule below it operates on
plain `Scene` records. That is what lets the pairing rules be tested offline.

Product identity
----------------
Assets are read from the public AWS COG mirror because CDSE serves
`s3://eodata` hrefs that need credentials. The mirror carries the same ESA
products, and `s2:product_uri` is the official CDSE product id - so identity in
every manifest stays official regardless of which host served the bytes.

The seasonal window
-------------------
T1 and T2 must be phenologically comparable or the model would learn seasonal
greenness instead of forest loss. The window is 1 August - 15 September: 46 days
wide, centred on the Amazon dry season, and containing the Milestone 1 pair
(10 Aug 2019 / 9 Aug 2021).

Within that window the pair is chosen to MINIMISE the day-of-year separation
(see `select_pair`). An earlier version instead took the last acceptable scene
in Y-1 and the first in Y+1; that bracketed year Y tightly in calendar terms
but drove T1 to the end of the window and T2 to its start, so 42 of 43 samples
in the first build had a September BEFORE and an August AFTER - a systematic,
one-directional seasonal bias that a model could learn in place of forest loss.
`SEASONAL_MAX_DOY_GAP` remains asserted, so a widened window can never silently
loosen the guarantee.

Baselines are mixed
-------------------
The archive spans processing baselines 02.xx to 05.00. Baseline >= 04.00 shifts
reflectance by BOA_ADD_OFFSET, and distributors differ on whether they have
already applied it. Every Scene therefore carries `offset_applied` straight from
the product metadata; it is never assumed from the date.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass

STAC_URL = "https://earth-search.aws.element84.com/v1/search"
COLLECTION = "sentinel-2-l2a"

#: Scene-level cloud gate, per the Milestone 1 protocol.
CLOUD_MAX_PCT = 20.0

#: Seasonal window as (month, day). Defined on the calendar, not on day-of-year,
#: so leap years do not shift it.
SEASON_START = (8, 1)
SEASON_END = (9, 15)

#: Upper bound on |DOY(T1) - DOY(T2)|. 46 = window width (+1 for leap wobble);
#: guaranteed by the window, asserted rather than relied upon.
SEASONAL_MAX_DOY_GAP = 46

MIRROR_HOST = "https://sentinel-cogs.s3.us-west-2.amazonaws.com"


@dataclass(frozen=True)
class Scene:
    """One Sentinel-2 L2A acquisition, as much as the protocol needs."""
    stac_id: str
    product_id: str          # official CDSE/ESA product identity
    dt: _dt.datetime
    cloud_pct: float
    nodata_pct: float
    baseline: str
    offset_applied: bool
    epsg: int
    utm_zone: int
    lat_band: str
    grid_square: str
    base_url: str

    @property
    def mgrs_tile(self) -> str:
        return f"{self.utm_zone}{self.lat_band}{self.grid_square}"

    @property
    def southern(self) -> bool:
        """Whether this scene's UTM grid uses the southern false northing.

        MGRS latitude bands run C-M south of the equator and N-X north of it.
        The two hemispheres differ by a 10,000,000 m false northing, so
        assuming the wrong one displaces a window by roughly 90 degrees of
        latitude - which silently turns every sample in a northern tile into a
        label lookup far outside the raster.
        """
        return self.lat_band < "N"

    @property
    def date(self) -> str:
        return self.dt.strftime("%Y-%m-%d")

    @property
    def doy(self) -> int:
        return self.dt.timetuple().tm_yday

    def asset(self, band: str) -> str:
        return f"{self.base_url}{band}.tif"


def parse_datetime(text: str) -> _dt.datetime:
    """STAC RFC3339 timestamp -> naive UTC datetime."""
    return _dt.datetime.strptime(text.replace("Z", "")[:26].ljust(26, "0"),
                                 "%Y-%m-%dT%H:%M:%S.%f")


def _https_base(item: dict) -> str:
    """Directory URL holding the COGs of a scene, taken from an asset href."""
    assets = item.get("assets", {})
    for key in ("blue", "red", "scl"):
        href = assets.get(key, {}).get("href")
        if href:
            return href.rsplit("/", 1)[0] + "/"
    s3 = item["properties"]["earthsearch:s3_path"]
    return MIRROR_HOST + s3.split("sentinel-cogs", 1)[1] + "/"


def scene_from_item(item: dict) -> Scene:
    """STAC feature -> Scene. Raises KeyError if protocol fields are absent."""
    p = item["properties"]
    return Scene(
        stac_id=item["id"],
        product_id=p["s2:product_uri"].replace(".SAFE", ""),
        dt=parse_datetime(p["datetime"]),
        cloud_pct=float(p["eo:cloud_cover"]),
        nodata_pct=float(p.get("s2:nodata_pixel_percentage", 0.0)),
        baseline=str(p["s2:processing_baseline"]),
        offset_applied=bool(p["earthsearch:boa_offset_applied"]),
        epsg=int(p["proj:epsg"]),
        utm_zone=int(p["mgrs:utm_zone"]),
        lat_band=str(p["mgrs:latitude_band"]),
        grid_square=str(p["mgrs:grid_square"]),
        base_url=_https_base(item),
    )


# ------------------------------------------------------------------ selection
def in_season(when) -> bool:
    """True when the acquisition falls inside the seasonal window."""
    key = (when.month, when.day)
    return SEASON_START <= key <= SEASON_END


def acceptable(scene: Scene) -> bool:
    """Scene-level gate: inside the seasonal window and below the cloud limit."""
    return in_season(scene.dt) and scene.cloud_pct < CLOUD_MAX_PCT


def doy_gap(a: Scene, b: Scene) -> int:
    """Day-of-year separation, ignoring the year."""
    return abs(a.doy - b.doy)


def group_by_tile(scenes) -> dict:
    """Scenes bucketed by MGRS tile.

    A point near a tile edge is covered by several tiles, and each one is a
    separate candidate grid with its own raster origin.
    """
    out: dict = {}
    for scene in scenes:
        out.setdefault(scene.mgrs_tile, []).append(scene)
    return out


#: Human-readable statement of the rule, recorded in every manifest so a
#: dataset can always be traced to the logic that produced it.
SELECTION_RULE = ("minimise |DOY(before) - DOY(after)| over all acceptable "
                  "Y-1 x Y+1 pairs inside the seasonal window; ties broken by "
                  "lower summed cloud cover, then by (before, after) STAC id")


def select_pair(scenes, year: int):
    """Choose the phenologically closest BEFORE/AFTER pair for one MGRS tile.

    BEFORE must come from Y-1 and AFTER from Y+1. Year Y itself is never used:
    TMF records only the YEAR of first deforestation, not the date, so a scene
    from within Y could fall on either side of the event.

    Among all acceptable pairs the one with the smallest day-of-year separation
    wins. The previous rule - last scene in Y-1, first in Y+1 - produced a
    September BEFORE against an August AFTER in 42 of 43 samples, a systematic
    seasonal bias pointing the same way every time. Minimising the separation
    removes that by construction: nothing in the rule prefers early or late
    scenes, so any residual offset is scatter rather than direction.

    Ties are broken deterministically, in order:
      1. smaller |DOY difference|
      2. lower summed cloud cover of the two scenes
      3. lexicographically smaller (before STAC id, after STAC id)

    Returns (t1, t2, None, info) on success, or (None, None, reason, info).
    `info` carries the selection provenance either way, so a rejected candidate
    records what was considered rather than vanishing.
    """
    usable = [s for s in scenes if acceptable(s)]
    before = sorted((s for s in usable if s.dt.year == year - 1), key=lambda s: s.dt)
    after = sorted((s for s in usable if s.dt.year == year + 1), key=lambda s: s.dt)

    info = {
        "selection_rule": SELECTION_RULE,
        "seasonal_window": {"start": list(SEASON_START), "end": list(SEASON_END)},
        "n_before_candidates": len(before),
        "n_after_candidates": len(after),
        "before_candidate_dates": [s.date for s in before],
        "after_candidate_dates": [s.date for s in after],
    }
    if not before or not after:
        return None, None, "no_s2_pair", info

    # Filter to grid-compatible pairs first, so one mismatched scene cannot
    # veto a perfectly good pair that exists alongside it.
    pairs = [(b, a) for b in before for a in after
             if b.epsg == a.epsg and b.utm_zone == a.utm_zone]
    info["n_pairs_considered"] = len(pairs)
    if not pairs:
        return None, None, "crs_grid", info

    t1, t2 = min(pairs, key=lambda p: (doy_gap(p[0], p[1]),
                                       p[0].cloud_pct + p[1].cloud_pct,
                                       p[0].stac_id, p[1].stac_id))
    info.update({"before_doy": t1.doy, "after_doy": t2.doy,
                 "doy_difference": doy_gap(t1, t2),
                 "selected_before": t1.date, "selected_after": t2.date,
                 "selected_reason": "smallest day-of-year separation available"})
    if doy_gap(t1, t2) > SEASONAL_MAX_DOY_GAP:
        return None, None, "seasonal_window", info
    return t1, t2, None, info


def search_scenes(lon: float, lat: float, years, timeout: int = 90):
    """NETWORK. Every scene near (lon, lat) in the seasonal window of `years`.

    Queried per year so the seasonal window is applied server-side; the cloud
    gate is re-checked locally by `acceptable` so that rule lives in one place.
    """
    import requests

    found: dict = {}
    for year in years:
        start = f"{year:04d}-{SEASON_START[0]:02d}-{SEASON_START[1]:02d}T00:00:00Z"
        end = f"{year:04d}-{SEASON_END[0]:02d}-{SEASON_END[1]:02d}T23:59:59Z"
        body = {"collections": [COLLECTION],
                "intersects": {"type": "Point", "coordinates": [lon, lat]},
                "datetime": f"{start}/{end}",
                "query": {"eo:cloud_cover": {"lt": CLOUD_MAX_PCT}},
                "limit": 100}
        response = requests.post(STAC_URL, json=body, timeout=timeout)
        response.raise_for_status()
        for item in response.json().get("features", []):
            try:
                scene = scene_from_item(item)
            except (KeyError, ValueError):
                continue          # missing protocol metadata: not usable
            found[scene.stac_id] = scene
    return sorted(found.values(), key=lambda s: s.dt)
