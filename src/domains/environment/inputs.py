"""The environment detector's input contract: six Sentinel-2 bands, or nothing.

The model reads B02, B03, B04, B08, B11 and B12. Two of the six - the SWIR
bands B11 and B12 - carry most of what separates bare soil and slash from
canopy, and they have no RGB equivalent. There is no defensible way to supply
them from a three-band image, so this module refuses rather than inventing:
no channel duplication, no zero padding, no index synthesised from visible
bands. An RGB image reaching this engine is a caller error, and it is reported
as one.

Accepted inputs
---------------
    numpy array      H x W x 6, or 6 x H x W, surface reflectance
    numpy .npy path  the same, as written by the v24 dataset builder
    mapping          {"B02": 2-D array, ...} with all six bands present

Reflectance, not DN
-------------------
Sentinel-2 L2A products store scaled integers, and since processing baseline
04.00 they also carry an additive offset. Getting that conversion wrong silently
shifts every band, so integer input is rejected with instructions rather than
divided by a guessed constant. Use `reflectance_from_dn`, which delegates to the
frozen acquisition code - there is exactly one implementation of that rule.

Geometry
--------
Nothing here reprojects, resamples or co-registers. The two dates must already
share a grid; a size mismatch is an error, not something to fix by resizing.
"""
from __future__ import annotations

import os

import numpy as np

from src.domains.environment.data import sentinel2 as s2

#: The six bands the model was trained on, in channel order.
REQUIRED_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")

#: The single user-facing sentence for a wrong-band input. One string, so the
#: CLI, the API and any future UI cannot drift from each other.
REQUIRED_BANDS_MESSAGE = (
    "Environmental forest-loss analysis requires six-band Sentinel-2 imagery: "
    "B02, B03, B04, B08, B11 and B12.")

#: Sentinel-2 L2A is distributed on a 10 m grid (B11/B12 resampled from 20 m).
#: Exposed as a named constant for a caller that knows its imagery is on that
#: grid. The engine never assumes it - see the note in engine.py on why a
#: ground sample distance is taken from the caller and never invented.
SENTINEL2_GSD_M = 10.0

#: Above this, values cannot be surface reflectance: the v24 TRAIN maximum over
#: 27.1 million pixels per band is 1.55, while L2A digital numbers are in the
#: thousands. Used only to tell the two apart, never to rescale anything.
REFLECTANCE_CEILING = 10.0


class BandContractError(ValueError):
    """Input does not satisfy the six-band contract.

    `user_message` is a complete sentence safe to show a user; str(exc) adds the
    technical detail for a log.
    """

    def __init__(self, user_message, detail=None):
        self.user_message = user_message
        super().__init__(f"{user_message} ({detail})" if detail else user_message)


def reflectance_from_dn(dn, offset_applied, baseline=None) -> np.ndarray:
    """Sentinel-2 L2A digital numbers -> surface reflectance.

    Thin re-export of the frozen acquisition rule, including the BOA_ADD_OFFSET
    handling that applies only from processing baseline 04.00. Not reimplemented
    here: src/domains/environment/data/sentinel2.py owns it.
    """
    return s2.to_reflectance(dn, offset_applied, baseline)


def _describe(array) -> str:
    if array.ndim == 2:
        return "a single-band image"
    if array.ndim == 3:
        return f"a {array.shape[-1]}-band image"
    # 0-d and 1-d inputs have no band axis at all. Described rather than
    # indexed: building the refusal message must never itself raise.
    return f"a {array.ndim}-dimensional array"


def _from_mapping(mapping, name):
    missing = [b for b in REQUIRED_BANDS if b not in mapping]
    if missing:
        raise BandContractError(
            REQUIRED_BANDS_MESSAGE,
            f"{name} is missing {', '.join(missing)}")
    planes = []
    for band in REQUIRED_BANDS:
        plane = np.asarray(mapping[band])
        if plane.ndim != 2:
            raise BandContractError(
                REQUIRED_BANDS_MESSAGE,
                f"{name}[{band!r}] has shape {plane.shape}, expected a 2-D array")
        planes.append(plane)
    shapes = {p.shape for p in planes}
    if len(shapes) != 1:
        raise BandContractError(
            "The six bands must all cover the same extent at the same size.",
            f"{name} band shapes {sorted(shapes)}")
    return np.stack(planes, axis=-1)


def _from_path(path, name):
    text = os.fspath(path)
    if text.lower().endswith(".npy"):
        return np.load(text)

    # Anything else is an ordinary image file. Open it only to establish how
    # many bands it has, so the caller gets the band message rather than a
    # decoding error. Reading is delegated to Pillow, already a dependency.
    try:
        from PIL import Image
        with Image.open(text) as img:
            count = len(img.getbands())
            array = np.array(img)
    except BandContractError:
        raise
    except Exception as exc:                                     # noqa: BLE001
        raise BandContractError(
            f"{name} could not be read as imagery.", f"{text}: {exc}") from exc
    if count != len(REQUIRED_BANDS):
        raise BandContractError(
            REQUIRED_BANDS_MESSAGE,
            f"{text} is a {count}-band file")
    return array


def _from_stream(stream, name):
    """An open file object - what a browser upload arrives as.

    Reading only: a .npy payload, or an ordinary image whose band count is
    checked so an RGB upload is refused with the published sentence rather than
    with a decoding error.
    """
    def rewind():
        try:
            stream.seek(0)
        except Exception:                                        # noqa: BLE001
            pass

    rewind()
    try:
        return np.load(stream, allow_pickle=False)
    except Exception:                                            # noqa: BLE001
        pass

    rewind()
    try:
        from PIL import Image
        with Image.open(stream) as img:
            count = len(img.getbands())
            array = np.array(img)
    except Exception as exc:                                     # noqa: BLE001
        raise BandContractError(
            f"{name} could not be read. Supply a six-band Sentinel-2 array "
            f"(.npy) for this analysis.", f"{exc}") from exc
    if count != len(REQUIRED_BANDS):
        raise BandContractError(
            REQUIRED_BANDS_MESSAGE,
            f"{name} is an uploaded {count}-band image file")
    return array


def as_six_band(image, name="image") -> np.ndarray:
    """Coerce an accepted input to an H x W x 6 float32 reflectance array.

    Raises BandContractError - never guesses, pads, duplicates or synthesises a
    missing band.
    """
    if hasattr(image, "keys"):
        array = _from_mapping(image, name)
    elif isinstance(image, (str, os.PathLike)):
        array = _from_path(image, name)
    elif hasattr(image, "read"):
        array = _from_stream(image, name)
    else:
        array = np.asarray(image)

    n = len(REQUIRED_BANDS)
    if array.ndim == 3 and array.shape[0] == n and array.shape[-1] != n:
        array = np.moveaxis(array, 0, -1)       # unambiguous CHW
    if array.ndim != 3 or array.shape[-1] != n:
        raise BandContractError(
            REQUIRED_BANDS_MESSAGE,
            f"{name} is {_describe(array)} with shape {array.shape}")

    if np.issubdtype(array.dtype, np.integer):
        raise BandContractError(
            "Supply surface reflectance, not raw Sentinel-2 digital numbers.",
            f"{name} has integer dtype {array.dtype}; convert with "
            "src.domains.environment.inputs.reflectance_from_dn, which applies "
            "the BOA offset correctly for the product's processing baseline")

    array = array.astype(np.float32, copy=False)
    finite = array[np.isfinite(array)]
    if finite.size and float(finite.max()) > REFLECTANCE_CEILING:
        raise BandContractError(
            "Supply surface reflectance, not raw Sentinel-2 digital numbers.",
            f"{name} reaches {float(finite.max()):.1f}; surface reflectance is "
            f"below {REFLECTANCE_CEILING:g}. Convert with "
            "src.domains.environment.inputs.reflectance_from_dn")
    return array


def check_pair(before, after):
    """Both dates as H x W x 6 reflectance, verified to share one grid."""
    a = as_six_band(before, "before")
    b = as_six_band(after, "after")
    if a.shape[:2] != b.shape[:2]:
        raise BandContractError(
            "The two dates must cover the same extent at the same size.",
            f"before {a.shape[:2]} vs after {b.shape[:2]}; Earth Guardian does "
            "not resample or co-register")
    return a, b


# --------------------------------------------------------------- presentation
#: Fixed display scale for a true-colour composite. NOT a per-image stretch: a
#: 2-98 percentile stretch amplified faint brightness into apparent cloud three
#: separate times during Phase 4B analysis, so the contact sheets were switched
#: to this fixed scale and the same rule is reused here. Two images rendered
#: this way are directly comparable to each other and across dates.
PREVIEW_MAX_REFLECTANCE = 0.30
PREVIEW_GAMMA = 0.6


def preview_rgb(reflectance) -> np.ndarray:
    """H x W x 6 reflectance -> H x W x 3 uint8 true-colour composite.

    Display only. Nothing here ever reaches the model: inference reads the six
    bands through normalization.py. Lives in the domain rather than in an
    application because which bands are red, green and blue is a property of
    Sentinel-2, not of the CLI or the UI.
    """
    array = np.asarray(reflectance, dtype=np.float32)
    index = {b: i for i, b in enumerate(REQUIRED_BANDS)}
    visible = np.stack([array[:, :, index["B04"]],      # red
                        array[:, :, index["B03"]],      # green
                        array[:, :, index["B02"]]], -1)  # blue
    scaled = np.clip(visible / PREVIEW_MAX_REFLECTANCE, 0, 1) ** PREVIEW_GAMMA
    return (scaled * 255).astype(np.uint8)


def prepare_pair(before, after):
    """Application entry point: sources -> (before, after, preview_a, preview_b).

    The application layer calls this and passes the result straight to
    engine.analyze(). It does not know what a band is, and no environmental
    normalisation happens here - that belongs to the engine.
    """
    a, b = check_pair(before, after)
    return a, b, preview_rgb(a), preview_rgb(b)


def _georef_of(before, after):
    """Georeferencing, when both sources are georeferenced raster FILES.

    Uses the shared GeoTIFF reader - there is no second parser here - and
    returns None the moment anything is uncertain. The v24 arrays are .npy and
    carry no georeferencing at all, which is reported honestly rather than
    filled in: a caller who knows the grid passes gsd_m instead.
    """
    paths = []
    for source in (before, after):
        if not isinstance(source, (str, os.PathLike)):
            return None
        text = os.fspath(source)
        if text.lower().endswith(".npy"):
            return None
        paths.append(text)
    try:
        from PIL import Image

        from src.common.georef import pair_georef, read_raster_info
        infos = []
        for text in paths:
            with Image.open(text) as img:
                infos.append(read_raster_info(img))
        return pair_georef(*infos)
    except Exception:                                            # noqa: BLE001
        return None


def prepare_application_pair(before, after, input_spec=None):
    """Input adapter for this domain, used via src/common/pair_input.py.

    A BandContractError raised here propagates to pair_input.prepare(), which
    turns it into a rejected PreparedPair carrying this module's exact sentence.
    """
    from src.common.pair_input import PreparedPair

    a, b, preview_a, preview_b = prepare_pair(before, after)
    georef = _georef_of(before, after)
    notes = (f"Six-band Sentinel-2 pair, {a.shape[1]}x{a.shape[0]} px. "
             + ("Georeferenced: ground area in m2 is available."
                if (georef is not None and georef.has_scale)
                else "No georeferencing: results are in pixels. Pass a ground "
                     "sample distance to report m2."))
    return PreparedPair(domain="environment", before=a, after=b,
                        preview_before=preview_a, preview_after=preview_b,
                        georef=georef, notes=notes)
