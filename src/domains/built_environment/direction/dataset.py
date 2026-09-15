"""Region-crop dataset for the Stage 3B-2 direction classifier.

Reads the memory-mapped crops produced by scripts/build_s2looking_crops.py.
Region geometry originates from the production region extractor via the Stage
3B-1 targets; no second region implementation exists.

Augmentation
------------
Dihedral only: random horizontal flip, vertical flip and k*90 degree rotation,
applied **identically to the BEFORE and AFTER halves**. The two halves are never
swapped and never augmented independently - swapping them would invert the
ground-truth direction, and independent augmentation would destroy the pixel
correspondence the task depends on.

No photometric augmentation: brightness/contrast jitter applied to one date
would simulate exactly the illumination difference the model must not rely on,
and applying it jointly adds nothing.

Normalisation uses the same ImageNet statistics as the rest of the project.

Memmap handling (why the array is opened lazily)
------------------------------------------------
The crops are a memory-mapped .npy array. numpy **materialises a memmap when it
is pickled**, and on Windows the DataLoader uses ``spawn``, which pickles the
dataset to every worker - so holding the array as an attribute pushed 2.6 GB per
worker down a pipe on every epoch and eventually exhausted the commit charge
(``OSError: [Errno 22]`` on ``reduction.dump``, children dying with truncated
pickles). The dataset therefore stores only the **path**, opens the memmap
lazily per process, and drops the handle in ``__getstate__``.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from src.domains.built_environment.data.levir import IMAGENET_MEAN, IMAGENET_STD

from .model import CLASS_NAMES, select_channels


class RegionCrops(Dataset):
    """Stored crops for one split.

    Parameters
    ----------
    crops_dir : directory written by build_s2looking_crops.py
    split     : "train" | "val" | "test"
    mode      : "both" | "before" | "after"
    augment   : dihedral augmentation (training only)
    seed      : augmentation RNG seed; per-sample draws are reproducible
    """

    def __init__(self, crops_dir, split, mode="both", augment=False, seed=0):
        # Only the PATH is stored. See the note on memmap pickling above.
        self._x_path = os.path.join(crops_dir, f"{split}_x.npy")
        self._x = None
        self.y = np.load(os.path.join(crops_dir, f"{split}_y.npy"))
        self.split = split
        self.mode = mode
        self.augment = augment
        self.seed = seed
        self.epoch = 0
        mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
        std = np.asarray(IMAGENET_STD, dtype=np.float32)
        # One 6-channel normaliser: the same statistics for each date.
        self._mean = np.concatenate([mean, mean]).reshape(6, 1, 1)
        self._std = np.concatenate([std, std]).reshape(6, 1, 1)

    @property
    def x(self):
        """Memory-mapped crops, opened lazily in whichever process asks."""
        if self._x is None:
            self._x = np.load(self._x_path, mmap_mode="r")
        return self._x

    def __getstate__(self):
        """Never ship the memmap through a pickle.

        numpy materialises a memmap when pickling it, so a spawned worker would
        receive the whole array (2.6 GB for train) down a pipe. Dropping the
        handle keeps the pickled dataset a few hundred bytes; the child reopens
        the file itself on first access.
        """
        state = self.__dict__.copy()
        state["_x"] = None
        return state

    def __len__(self):
        return len(self.y)

    def set_epoch(self, epoch: int):
        """Vary augmentation across epochs while staying reproducible."""
        self.epoch = int(epoch)

    def _rng(self, index):
        return np.random.default_rng((self.seed, self.epoch, index))

    def __getitem__(self, index):
        crop = np.asarray(self.x[index])            # (H, W, 6) uint8
        if self.augment:
            rng = self._rng(index)
            if rng.random() < 0.5:
                crop = crop[:, ::-1]
            if rng.random() < 0.5:
                crop = crop[::-1, :]
            k = int(rng.integers(0, 4))
            if k:
                crop = np.rot90(crop, k, axes=(0, 1))
        arr = np.ascontiguousarray(crop.transpose(2, 0, 1), dtype=np.float32) / 255.0
        arr = (arr - self._mean) / self._std
        x = torch.from_numpy(arr)
        x = select_channels(x.unsqueeze(0), self.mode).squeeze(0)
        return x, int(self.y[index])

    # -------------------------------------------------------------- helpers
    def class_counts(self):
        return {name: int((self.y == i).sum()) for i, name in enumerate(CLASS_NAMES)}

    def class_weights(self):
        """Inverse-frequency weights normalised to mean 1.

        The construction:demolition ratio is about 1.75:1 in train, so the
        minority class is up-weighted rather than the imbalance being ignored.
        """
        counts = np.array([(self.y == i).sum() for i in range(len(CLASS_NAMES))],
                          dtype=np.float64)
        w = counts.sum() / (len(counts) * np.maximum(counts, 1))
        return torch.tensor(w / w.mean(), dtype=torch.float32)
