"""The environment detector's input normalisation contract.

Why the numbers live here and not in a JSON file
------------------------------------------------
Training read them from data/environment/characterization_v24_train_bands.json.
That directory is git-ignored, so a checkout of this repository does not contain
it. A production engine that could only normalise when an untracked data file
happened to be present would be loading a model it cannot correctly feed. The
twelve constants are therefore part of the code, exactly as the shared
ImageNet constants are for the built-environment engine.

They are NOT new values. They are the Phase 4B-1A statistics verbatim, and
`verify_against_file()` re-reads the characterization file when it is available
and reports any disagreement rather than trusting this copy.

What they are
-------------
Per-band mean and standard deviation of surface REFLECTANCE, measured over the
v24 TRAIN split alone - 27,131,904 pixels per band. No validation or test pixel
contributed. They are applied globally, to Amazon and SE Asia alike, because a
per-region normalisation would hide the geographic domain shift the evaluation
exists to expose.

ImageNet statistics are NOT used and must never be substituted: they describe
8-bit sRGB photographs and have no meaning for surface reflectance, least of all
for SWIR.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

#: Canonical band order. Matches environment.data.sentinel2.BANDS and the order
#: the six input channels are stacked in.
BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")

#: v24 TRAIN-split per-band reflectance mean, in BANDS order.
MEAN = (0.04413012791877784, 0.06321721185877703, 0.0564930349856759,
        0.27439223812305985, 0.20611148870348356, 0.11236270837461315)

#: v24 TRAIN-split per-band reflectance standard deviation, in BANDS order.
STD = (0.025441759989742286, 0.02748462185418511, 0.044444973492246354,
       0.06249170404340661, 0.08492849143361803, 0.07320270276557037)

SOURCE = "v24 TRAIN split only (Phase 4B-1A characterization)"
SOURCE_FILE = os.path.join("data", "environment",
                           "characterization_v24_train_bands.json")
#: sha256 of the characterization file these constants were copied from.
SOURCE_FILE_SHA256 = "95e58cc37815f1759e2661ec3fd010bffd0954dd80d0951a32be3ac326d3c9c1"
#: Pixels per band the statistics were measured over.
SOURCE_PIXELS_PER_BAND = 27131904


def subset_digest(bands, mean, std) -> str:
    """Stable hash of the exact (band, mean, std) triples a model uses.

    Defined once here and reused by the Phase 4B data loader, so a checkpoint's
    recorded normalisation digest and the engine's are comparable by
    construction rather than by coincidence.
    """
    payload = json.dumps([[b, m, s] for b, m, s in zip(bands, mean, std)],
                         sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def statistics(bands=BANDS) -> dict:
    """Mean/std for a band subset, in the same shape the experiments recorded.

    Selecting a subset of train-only statistics is still train-only.
    """
    bands = tuple(bands)
    unknown = [b for b in bands if b not in BANDS]
    if unknown:
        raise KeyError(f"no train statistics for {unknown}; known: {list(BANDS)}")
    index = {b: i for i, b in enumerate(BANDS)}
    mean = [MEAN[index[b]] for b in bands]
    std = [STD[index[b]] for b in bands]
    return {"path": SOURCE_FILE, "sha256": SOURCE_FILE_SHA256,
            "subset_sha256": subset_digest(bands, mean, std),
            "mean": mean, "std": std, "bands": list(bands), "source": SOURCE}


def normalize(reflectance: np.ndarray, bands=BANDS) -> np.ndarray:
    """(reflectance - mean) / std, per band. H x W x len(bands) -> same shape.

    Identical arithmetic to the training data loader, so the engine feeds the
    model exactly what it was trained on.
    """
    stats = statistics(bands)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return (np.asarray(reflectance, dtype=np.float32) - mean) / std


def verify_against_file(path=None, bands=BANDS, tolerance: float = 0.0):
    """Compare these constants with the characterization file, when present.

    Returns None when the file is absent - that is expected in a fresh checkout
    and is not an error. Otherwise returns a report dict whose "agrees" field
    says whether the file and this module state the same thing.
    """
    from src import config
    path = path or config.resolve(SOURCE_FILE)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        raw = fh.read()
    table = json.loads(raw.decode("utf-8"))
    stats = statistics(bands)
    mismatches = []
    for band, mean, std in zip(bands, stats["mean"], stats["std"]):
        if band not in table:
            mismatches.append(f"{band}: absent from {path}")
            continue
        for field, here in (("mean", mean), ("std", std)):
            there = float(table[band][field])
            if abs(there - here) > tolerance:
                mismatches.append(f"{band}.{field}: file {there!r} != module {here!r}")
    return {"path": path, "file_sha256": hashlib.sha256(raw).hexdigest(),
            "expected_file_sha256": SOURCE_FILE_SHA256,
            "subset_sha256": stats["subset_sha256"],
            "agrees": not mismatches, "mismatches": mismatches}
