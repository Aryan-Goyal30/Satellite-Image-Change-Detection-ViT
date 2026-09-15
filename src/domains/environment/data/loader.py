"""Dataset and preprocessing for the frozen environmental dataset (Phase 4B-1B).

Reads dataset_v24 read-only. Nothing here writes to the dataset, and the split,
eligibility, label and provenance fields are consumed exactly as frozen.

Normalisation
-------------
Per-band mean and standard deviation come from the TRAIN split only, measured in
Phase 4B-1A over 27.1 million pixels per band and stored in
characterization_v24_train_bands.json. They are applied globally, to Amazon and
SE Asia alike - deliberately, because a per-region normalisation would hide the
geographic domain shift this experiment exists to expose.

ImageNet statistics are NOT used. They describe 8-bit sRGB photographs and have
no meaning for surface reflectance, least of all for SWIR.

Augmentation
------------
Only the dihedral group: four rotations by 90 degrees, optionally flipped. The
SAME transform is applied to BEFORE, AFTER, label and the invalid mask, so the
temporal relationship and the pixel correspondence to the label both survive.

Nothing that could manufacture spectral change is allowed: no date swapping, no
independent jitter of one date, no per-date geometry. Those would create pairs
whose apparent change is an artefact of augmentation rather than of the ground.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from src.domains.environment.data import sentinel2 as s2

#: The eight dihedral transforms, as (number of 90 degree rotations, flip).
DIHEDRAL = tuple((k, f) for k in range(4) for f in (False, True))

AUGMENTATION_POLICY = {
    "transforms": "dihedral group: rot90 x {0,1,2,3} optionally followed by horizontal flip",
    "applied_identically_to": ["before", "after", "label", "invalid"],
    "date_swap": False,
    "independent_per_date_geometry": False,
    "colour_jitter": False,
    "rationale": ("only transforms that cannot manufacture spectral change; "
                  "anything applied to one date alone would create apparent "
                  "change that is an artefact of augmentation"),
}


def load_band_statistics(path: str, bands=None) -> dict:
    """TRAIN-only mean/std for a band subset, plus provenance hashes.

    The per-band statistics were measured over the TRAIN split alone in Phase
    4B-1A. Selecting a subset of them is therefore still train-only - no
    validation or test pixel contributes to any experiment's normalisation.

    `subset_sha256` hashes the exact (band, mean, std) triples an experiment
    uses, so two experiments on different band sets are distinguishable in their
    manifests rather than sharing the file-level hash.
    """
    import hashlib

    from src.domains.environment import normalization

    bands = tuple(bands) if bands is not None else tuple(s2.BANDS)
    with open(path, "rb") as fh:
        raw = fh.read()
    stats = json.loads(raw.decode("utf-8"))
    mean = [float(stats[b]["mean"]) for b in bands]
    std = [float(stats[b]["std"]) for b in bands]
    return {
        "path": os.path.relpath(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        # The digest rule lives in src/domains/environment/normalization.py so
        # that a checkpoint's recorded digest and the engine's are comparable by
        # construction. The bytes hashed are unchanged.
        "subset_sha256": normalization.subset_digest(bands, mean, std),
        "mean": mean, "std": std, "bands": list(bands),
        "source": normalization.SOURCE,
    }


class EnvironmentChangeDataset(Dataset):
    """One frozen v24 split, as (before, after, label, invalid) tensors.

    `eligible_only` applies the dataset's own frozen `training_eligibility`
    field: `small_event` positives are retained in the archive but excluded from
    the optimisation pool. The rule is read, never redefined here.
    """

    def __init__(self, root: str, split: str, stats: dict,
                 augment: bool = False, eligible_only: bool = False,
                 seed: int = 0, bands=None):
        self.root = root
        self.split = split
        # Stored arrays always carry all six bands; an experiment selects the
        # channels it uses. Selection happens AFTER reflectance conversion, so
        # the conversion is identical in every experiment.
        self.bands = tuple(bands) if bands is not None else tuple(s2.BANDS)
        self.band_index = [s2.BAND_INDEX[b] for b in self.bands]
        with open(os.path.join(root, "manifest.json"), encoding="utf-8") as fh:
            manifest = json.load(fh)
        records = [r for r in manifest["samples"] if r["split"] == split]
        if eligible_only:
            records = [r for r in records
                       if r.get("training_eligibility") == "training_eligible"]
        self.records = sorted(records, key=lambda r: r["sample_id"])
        self.augment = augment
        self.mean = np.asarray(stats["mean"], dtype=np.float32)
        self.std = np.asarray(stats["std"], dtype=np.float32)
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.records)

    def _reflectance(self, record, side):
        dn = np.load(os.path.join(self.root, f"{record['sample_id']}_{side}.npy"))
        full = s2.to_reflectance(dn, record[f"{side}_offset_applied"],
                                 record[f"{side}_baseline"])
        return full[:, :, self.band_index]

    def __getitem__(self, index):
        record = self.records[index]
        before = self._reflectance(record, "before")
        after = self._reflectance(record, "after")
        label = np.load(os.path.join(self.root,
                                     f"{record['sample_id']}_label.npy")).astype(np.float32)
        invalid = np.load(os.path.join(self.root,
                                       f"{record['sample_id']}_invalid.npy")).astype(np.float32)

        before = (before - self.mean) / self.std
        after = (after - self.mean) / self.std

        if self.augment:
            k, flip = DIHEDRAL[int(self._rng.integers(len(DIHEDRAL)))]
            before, after = np.rot90(before, k, (0, 1)), np.rot90(after, k, (0, 1))
            label, invalid = np.rot90(label, k), np.rot90(invalid, k)
            if flip:
                before, after = before[:, ::-1], after[:, ::-1]
                label, invalid = label[:, ::-1], invalid[:, ::-1]

        to_chw = lambda a: torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))
        return {
            "before": to_chw(before).float(),
            "after": to_chw(after).float(),
            "label": torch.from_numpy(np.ascontiguousarray(label))[None].float(),
            "invalid": torch.from_numpy(np.ascontiguousarray(invalid))[None].float(),
            "sample_id": record["sample_id"],
            "region": "SE Asia" if record["tmf_tile"] in ("N0_E110", "N10_E100") else "Amazon",
            "sample_type": record["sample_type"],
        }

    def summary(self) -> dict:
        positives = sum(1 for r in self.records if r["sample_type"] == "positive")
        asia = sum(1 for r in self.records
                   if r["tmf_tile"] in ("N0_E110", "N10_E100"))
        return {"split": self.split, "n": len(self.records), "positive": positives,
                "negative": len(self.records) - positives,
                "se_asia": asia, "amazon": len(self.records) - asia,
                "augmented": self.augment, "bands": list(self.bands)}
