"""Shared image preprocessing contract.

These constants and helpers used to live in src/data/levir.py, which meant the
inference engine imported a dataset module just to normalise an array. Normalisation
belongs to the model's input contract, not to one domain's dataset.

The numerical behaviour is UNCHANGED: same constants, same order of operations,
same dtypes. Both call sites (the LEVIR dataset and the inference engine) keep
their exact original arithmetic; only the source of the constants moved.
"""
import numpy as np
import torch

# ImageNet statistics, because the Built Environment encoder is an ImageNet
# pretrained ResNet. A future engine with different pretraining declares its own.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_tensor(img_np, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """HWC uint8 -> CHW float tensor, normalised. (Dataset path.)"""
    x = img_np.astype(np.float32) / 255.0
    x = (x - mean) / std
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def to_batched_tensor(img_np, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """HWC uint8 -> 1CHW float tensor, normalised. (Inference path.)"""
    x = img_np.astype(np.float32) / 255.0
    x = (x - mean) / std
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)


def denormalize(t, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """CHW normalised tensor -> HWC uint8, for visualisation."""
    x = t.detach().cpu().numpy().transpose(1, 2, 0)
    x = x * std + mean
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)
