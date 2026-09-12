"""Shared checkpoint loading.

Previously `load_model` lived in src/eval/evaluate.py, so the visualisation code
imported an evaluation CLI module just to build a model. Loading is a service,
not an evaluation concern.

Behaviour is identical to the previous src.eval.evaluate.load_model: same
torch.load call, same encoder resolution, same .eval() state.
"""
import hashlib
import os

import torch

from src.models.siamese_unet import build_model

# architecture name -> builder. A future domain registers its own here.
MODEL_BUILDERS = {
    "siamese-unet": lambda encoder: build_model(encoder, pretrained=False),
}

DEFAULT_ARCH = "siamese-unet"
DEFAULT_ENCODER = "resnet34"


def checkpoint_sha256(path):
    """SHA-256 of the checkpoint file, for provenance. None if unreadable."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def read_checkpoint(path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=False)


def build_from_checkpoint(ck, device="cpu"):
    """Instantiate the architecture a checkpoint was saved from (weights not loaded)."""
    arch = ck.get("arch", DEFAULT_ARCH)
    if arch not in MODEL_BUILDERS:
        raise KeyError(f"unknown architecture '{arch}'. Known: {sorted(MODEL_BUILDERS)}")
    return MODEL_BUILDERS[arch](ck.get("encoder", DEFAULT_ENCODER)).to(device)


def load_model(ckpt_path, device="cpu"):
    """Load a trained model in eval mode. Returns (model, checkpoint_dict)."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\nTrain one first: python -m src.train.train")
    ck = read_checkpoint(ckpt_path, device)
    model = build_from_checkpoint(ck, device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, ck
