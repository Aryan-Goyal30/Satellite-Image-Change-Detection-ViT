"""Loading the environment checkpoint through the shared model loader.

The Phase 4B checkpoints predate the loader's `arch` field and store their
weights under "model" rather than "model_state", so they cannot be loaded by
inspection alone. Rather than add a second loader, this module registers the
environment architecture with the shared one and passes the architecture name
explicitly. src/common/model_loader.py stays generic and keeps knowing nothing
about any domain.

Band set comes from the checkpoint
----------------------------------
E0 (3 bands), E1 (4) and E2 (6) share this architecture and differ only in their
input stem, so the builder reads the band list the checkpoint recorded instead
of assuming six. A checkpoint that recorded none is rejected rather than
defaulted - silently building a six-band stem for a three-band checkpoint would
fail later with an opaque shape error.

pretrained=False is deliberate: the encoder's ImageNet weights are already
inside the checkpoint. Passing True would reach out to download them, and this
engine must never touch the network to answer a request.
"""
from __future__ import annotations

from src.common import model_loader
from src.domains.environment.model import build_model

#: Architecture name for MODEL_BUILDERS. Not stored in the Phase 4B
#: checkpoints, so callers here pass it explicitly.
ARCH = "environment-siamese-unet"

DEFAULT_ENCODER = "resnet34"


def bands_of(ck):
    """The band list a checkpoint was trained on.

    Read from the top-level "bands" field (E3 and later) or from the recorded
    architecture description (E0-E2). Never defaulted.
    """
    bands = ck.get("bands")
    if not bands:
        bands = (ck.get("description") or {}).get("bands")
    if not bands:
        raise KeyError(
            "checkpoint records no band list, so its input stem cannot be "
            "rebuilt; expected a 'bands' field or description['bands']")
    return tuple(bands)


def build(ck):
    """Builder registered with the shared loader: checkpoint -> un-weighted model."""
    return build_model(bands=bands_of(ck),
                       encoder=ck.get("encoder", DEFAULT_ENCODER),
                       pretrained=False)


model_loader.register_builder(ARCH, build, overwrite=True)


def load_environment_model(path, device="cpu"):
    """Load an environment checkpoint in eval mode. Returns (model, checkpoint)."""
    return model_loader.load_model(path, device, arch=ARCH)
