"""LEVIR-CD tile dataset - Built Environment domain.

Expects the output of scripts/prepare_tiles.py:

    data/levir_cd_tiles/{train,val,test}/{A,B,label}/*.png

A = "before" image, B = "after" image, label = binary change mask (0 / 255).

Normalisation comes from src/common/preprocessing.py (ImageNet statistics,
because the encoder is an ImageNet pretrained ResNet). It used to be defined
here, which meant the inference engine imported this dataset module just to get
two constants. The arithmetic is unchanged.

Augmentation is applied to the TRAIN split only, and identically to A, B and the
label (geometric ops must stay pixel-aligned or the supervision is corrupted).

Task symmetry
-------------
The current task is BINARY change: the label marks "these pixels differ", not
which direction the change went. That makes the task symmetric under swapping
the two dates, which is why date-swap is a valid augmentation below.
Construction-vs-demolition directionality is a FUTURE capability; if it is ever
added, the date-swap augmentation must be removed first, because direction would
then be part of the label.
"""
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.common.preprocessing import (  # re-exported: callers still import these here
    IMAGENET_MEAN,
    IMAGENET_STD,
    denormalize,
)
from src.common.preprocessing import to_tensor as _to_tensor

__all__ = ["LevirCDTiles", "denormalize", "IMAGENET_MEAN", "IMAGENET_STD"]


class LevirCDTiles(Dataset):
    """Paired before/after tiles with a binary change mask.

    Args:
        root:  data/levir_cd_tiles
        split: 'train' | 'val' | 'test'
        augment: geometric + photometric augmentation (train only)
    """

    def __init__(self, root, split, augment=False):
        self.dir = os.path.join(root, split)
        self.split = split
        self.augment = augment
        a_dir = os.path.join(self.dir, "A")
        if not os.path.isdir(a_dir):
            raise FileNotFoundError(
                f"{a_dir} not found. Run: python scripts/prepare_tiles.py")
        self.names = sorted(f for f in os.listdir(a_dir) if f.endswith(".png"))
        if not self.names:
            raise RuntimeError(f"No tiles in {a_dir}")

    def __len__(self):
        return len(self.names)

    def _load(self, name):
        a = np.array(Image.open(os.path.join(self.dir, "A", name)).convert("RGB"))
        b = np.array(Image.open(os.path.join(self.dir, "B", name)).convert("RGB"))
        m = np.array(Image.open(os.path.join(self.dir, "label", name)).convert("L"))
        return a, b, (m > 127).astype(np.float32)

    def _augment(self, a, b, m):
        # --- geometric: applied identically to all three ---
        if random.random() < 0.5:                      # horizontal flip
            a, b, m = a[:, ::-1], b[:, ::-1], m[:, ::-1]
        if random.random() < 0.5:                      # vertical flip
            a, b, m = a[::-1], b[::-1], m[::-1]
        k = random.randint(0, 3)                       # 90 degree rotations
        if k:
            a, b, m = np.rot90(a, k), np.rot90(b, k), np.rot90(m, k)

        # --- temporal symmetry -------------------------------------------------
        # LEVIR change is symmetric under swapping the two dates (the mask marks
        # "these pixels differ", not a direction), so swapping is a valid and
        # useful augmentation. NOTE: if we later predict build-vs-demolish this
        # must be removed, because direction would then be part of the label.
        if random.random() < 0.5:
            a, b = b, a

        # --- photometric: applied INDEPENDENTLY per date ------------------------
        # This is deliberate. Illumination differs between acquisition dates, and
        # forcing the model to stay invariant to that is exactly the failure mode
        # the Stage 0 heuristic baseline had.
        a = self._jitter(a)
        b = self._jitter(b)
        return a, b, m

    @staticmethod
    def _jitter(img):
        if random.random() < 0.5:
            f = np.float32(img)
            f *= random.uniform(0.85, 1.15)                 # brightness
            f = (f - f.mean()) * random.uniform(0.9, 1.1) + f.mean()   # contrast
            img = np.clip(f, 0, 255).astype(np.uint8)
        return img

    def __getitem__(self, i):
        name = self.names[i]
        a, b, m = self._load(name)
        if self.augment:
            a, b, m = self._augment(a, b, m)
        return {
            "a": _to_tensor(np.ascontiguousarray(a)),
            "b": _to_tensor(np.ascontiguousarray(b)),
            "mask": torch.from_numpy(np.ascontiguousarray(m))[None],  # 1xHxW
            "name": name,
        }
