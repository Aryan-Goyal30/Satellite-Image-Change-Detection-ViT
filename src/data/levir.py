"""LEVIR-CD tile dataset.

Expects the output of scripts/prepare_tiles.py:

    data/levir_cd_tiles/{train,val,test}/{A,B,label}/*.png

A = "before" image, B = "after" image, label = binary change mask (0 / 255).

Normalisation uses ImageNet statistics because the encoder is an ImageNet
pretrained ResNet; both dates go through the *same* transform so the siamese
branches see comparable inputs.

Augmentation is applied to the TRAIN split only, and identically to A, B and the
label (geometric ops must stay pixel-aligned or the supervision is corrupted).
"""
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _to_tensor(img_np):
    """HWC uint8 -> CHW float tensor, ImageNet-normalised."""
    x = img_np.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def denormalize(t):
    """CHW normalised tensor -> HWC uint8, for visualisation."""
    x = t.detach().cpu().numpy().transpose(1, 2, 0)
    x = x * IMAGENET_STD + IMAGENET_MEAN
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


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
