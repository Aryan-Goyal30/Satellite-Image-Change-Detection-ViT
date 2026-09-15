"""ResNet-18 region direction classifier: construction vs demolition.

Research component for Stage 3B-2. The production Siamese U-Net change detector
is NOT touched, imported or modified by anything here.

Architecture
------------
torchvision ResNet-18 (ImageNet weights), first convolution widened to accept a
6-channel early-fusion input, classifier head replaced with 2 logits.

    input  : 6 x 128 x 128   (channels 0:3 BEFORE RGB, 3:6 AFTER RGB)
    output : 2 logits        (0 = construction, 1 = demolition)

First-convolution initialisation - the decision and why
-------------------------------------------------------
ResNet-18's pretrained ``conv1`` expects 3 channels. Widening it to 6 needs an
explicit choice; duplicating weights "because it works" is not good enough, so
the strategy used here is:

    W_new[:, 0:3] = W_pretrained * 0.5
    W_new[:, 3:6] = W_pretrained * 0.5

**Property that makes this the right default:** for an unchanged pair, where
BEFORE == AFTER == I, the widened convolution reproduces the original exactly

    conv6([I, I]) = 0.5 * conv3(I) + 0.5 * conv3(I) = conv3(I)

so every downstream ImageNet feature starts in exactly the distribution the
pretrained weights expect - no activation-scale drift, no re-tuning of the
batch-norm statistics that follow. The network begins effectively *blind to
change* (it sees the average of the two dates) and must learn the difference,
which is the honest starting point for a directionality task.

It is also **deterministic**: a pure function of the pretrained tensor, drawing
no random numbers, so two builds on any machine produce bit-identical weights.

Alternatives considered and rejected:

* ``W[:, 0:3] = W_pre, W[:, 3:6] = 0`` - asymmetric; the model ignores the AFTER
  image at initialisation and the two dates enter training on unequal footing.
* ``W[:, 0:3] = +W_pre/2, W[:, 3:6] = -W_pre/2`` - computes a difference image
  at initialisation, which is appealing, but it yields exactly zero response for
  unchanged input and feeds negative-image responses into ImageNet filters,
  discarding the pretrained feature semantics.
* Random initialisation of the extra channels - introduces an RNG dependence in
  what should be a reproducible transform, and breaks the equality property
  above.

``tests/test_stage3b2.py`` asserts the equality property numerically, asserts
determinism across two builds, and pins the parameter count.
"""
from __future__ import annotations

import torch
import torch.nn as nn

#: Input configurations. "both" is the primary model; the other two are the
#: required ablations and use the unmodified 3-channel pretrained conv1.
INPUT_MODES = ("both", "before", "after")

CLASS_NAMES = ("construction", "demolition")
NUM_CLASSES = 2

#: Channel slices of the stored 6-channel crop.
BEFORE_SLICE = slice(0, 3)
AFTER_SLICE = slice(3, 6)

FIRST_CONV_INIT = (
    "duplicate-and-halve: W_new[:, 0:3] = W_new[:, 3:6] = W_pretrained * 0.5, so "
    "conv6([I, I]) == conv3(I) exactly for an unchanged pair. Deterministic, "
    "draws no random numbers."
)


def in_channels_for(mode: str) -> int:
    if mode not in INPUT_MODES:
        raise ValueError(f"mode must be one of {INPUT_MODES}, got {mode!r}")
    return 6 if mode == "both" else 3


def widen_first_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """Return a copy of `conv` accepting `in_channels`, per FIRST_CONV_INIT."""
    if in_channels == conv.in_channels:
        return conv
    if in_channels % conv.in_channels != 0:
        raise ValueError(
            f"cannot widen {conv.in_channels}-channel conv to {in_channels}")
    repeats = in_channels // conv.in_channels
    wide = nn.Conv2d(in_channels, conv.out_channels,
                     kernel_size=conv.kernel_size, stride=conv.stride,
                     padding=conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        wide.weight.copy_(conv.weight.repeat(1, repeats, 1, 1) / repeats)
        if conv.bias is not None:
            wide.bias.copy_(conv.bias)
    return wide


def build_model(mode: str = "both", pretrained: bool = True,
                num_classes: int = NUM_CLASSES) -> nn.Module:
    """ResNet-18 adapted to `mode`, with a 2-logit head."""
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.conv1 = widen_first_conv(model.conv1, in_channels_for(mode))
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.input_mode = mode
    return model


def select_channels(x: torch.Tensor, mode: str) -> torch.Tensor:
    """Slice a stored 6-channel batch down to what `mode` consumes."""
    if mode == "both":
        return x
    if mode == "before":
        return x[:, BEFORE_SLICE]
    if mode == "after":
        return x[:, AFTER_SLICE]
    raise ValueError(f"unknown mode {mode!r}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def verify_first_conv_init(model: nn.Module) -> dict:
    """Check the documented initialisation actually holds on a built model.

    Returns the evidence rather than just a boolean, so it can be recorded in
    the model card.
    """
    from torchvision.models import ResNet18_Weights, resnet18

    reference = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).conv1
    # The model may already live on GPU; the reference is always on CPU, so the
    # comparison is done on CPU rather than assuming a device.
    w = model.conv1.weight.detach().cpu()
    bias = (model.conv1.bias.detach().cpu()
            if model.conv1.bias is not None else None)
    out = {"in_channels": int(model.conv1.in_channels),
           "strategy": FIRST_CONV_INIT}
    if model.conv1.in_channels == 6:
        half_a, half_b = w[:, 0:3], w[:, 3:6]
        out["halves_identical"] = bool(torch.equal(half_a, half_b))
        out["equals_pretrained_over_two"] = bool(
            torch.allclose(half_a, reference.weight.detach() / 2, atol=0, rtol=0))
        # The property that matters: an unchanged pair reproduces conv3. The
        # identity is exact in exact arithmetic, so in float32 the two differ
        # only by accumulation error - hence a RELATIVE tolerance, not a bare
        # absolute one. The probe input is seeded because this evidence is
        # recorded in the model card and must be reproducible.
        gen = torch.Generator().manual_seed(0)
        img = torch.randn(2, 3, 64, 64, generator=gen)
        with torch.no_grad():
            ref_out = torch.nn.functional.conv2d(
                img, reference.weight, reference.bias,
                stride=reference.stride, padding=reference.padding)
            wide_out = torch.nn.functional.conv2d(
                torch.cat([img, img], dim=1), w, bias,
                stride=model.conv1.stride, padding=model.conv1.padding)
        max_abs = float((ref_out - wide_out).abs().max())
        scale = float(ref_out.abs().max())
        out["probe_seed"] = 0
        out["max_abs_diff_unchanged_pair"] = max_abs
        out["max_relative_diff_unchanged_pair"] = (max_abs / scale) if scale else 0.0
        out["reproduces_pretrained_on_unchanged_pair"] = bool(
            torch.allclose(ref_out, wide_out, atol=1e-4, rtol=1e-4))
    else:
        out["equals_pretrained"] = bool(
            torch.equal(w, reference.weight.detach()))
    return out
