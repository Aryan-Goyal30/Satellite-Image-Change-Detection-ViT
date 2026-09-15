"""Shared checkpoint loading.

Previously `load_model` lived in src/eval/evaluate.py, so the visualisation code
imported an evaluation CLI module just to build a model. Loading is a service,
not an evaluation concern.

Behaviour for existing callers is unchanged: same torch.load call, same encoder
resolution, same .eval() state, same results.

Registering a domain's architecture
-----------------------------------
A builder takes the loaded checkpoint dict and returns an un-weighted module, so
it can read whatever that domain recorded - band list, encoder, channel count -
instead of being limited to an encoder name. Domains register their own rather
than this module importing them, which would invert the dependency:

    from src.common import model_loader
    model_loader.register_builder("environment-siamese-unet", build)

State-dict key
--------------
Checkpoints in this project store the weights under "model_state" (built
environment) or "model" (the Phase 4B environment experiments). Both are read;
anything else raises rather than being guessed at.
"""
import hashlib
import os

import torch

from src.models.siamese_unet import build_model

DEFAULT_ARCH = "siamese-unet"
DEFAULT_ENCODER = "resnet34"

#: Keys under which a checkpoint may store its state dict, in priority order.
STATE_DICT_KEYS = ("model_state", "model")

# architecture name -> builder(checkpoint) -> nn.Module. Domains register their
# own via register_builder(); see src/domains/environment/loading.py.
MODEL_BUILDERS = {
    "siamese-unet": lambda ck: build_model(ck.get("encoder", DEFAULT_ENCODER),
                                           pretrained=False),
}


def register_builder(arch, builder, *, overwrite=False):
    """Register `arch` -> builder(checkpoint_dict) -> un-weighted module."""
    if arch in MODEL_BUILDERS and not overwrite and MODEL_BUILDERS[arch] is not builder:
        raise ValueError(f"architecture '{arch}' is already registered")
    MODEL_BUILDERS[arch] = builder


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


def state_dict_of(ck):
    """The weights inside a checkpoint, whichever supported key holds them."""
    for key in STATE_DICT_KEYS:
        value = ck.get(key)
        if isinstance(value, dict):
            return value
    raise KeyError(
        f"checkpoint has no state dict: expected one of {list(STATE_DICT_KEYS)}, "
        f"found keys {sorted(ck)}")


def build_from_checkpoint(ck, device="cpu", arch=None):
    """Instantiate the architecture a checkpoint was saved from (weights not loaded).

    `arch` overrides the checkpoint's own "arch" field, for checkpoints written
    before the field existed. Nothing is inferred from tensor shapes.
    """
    arch = arch or ck.get("arch", DEFAULT_ARCH)
    if arch not in MODEL_BUILDERS:
        raise KeyError(f"unknown architecture '{arch}'. Known: {sorted(MODEL_BUILDERS)}")
    return MODEL_BUILDERS[arch](ck).to(device)


def load_model(ckpt_path, device="cpu", arch=None):
    """Load a trained model in eval mode. Returns (model, checkpoint_dict)."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\nTrain one first: python -m src.train.train")
    ck = read_checkpoint(ckpt_path, device)
    model = build_from_checkpoint(ck, device, arch=arch)
    model.load_state_dict(state_dict_of(ck))
    model.eval()
    return model, ck
