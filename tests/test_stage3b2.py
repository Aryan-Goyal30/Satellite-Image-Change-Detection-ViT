"""Focused tests for Stage 3B-2: the region direction classifier.

pytest is not a project dependency, so these are plain functions that assert.
Run either way:

    python tests/test_stage3b2.py
    pytest tests/test_stage3b2.py

These cover the model's first-convolution initialisation contract and the
dataset's augmentation invariants. They need no trained checkpoint; the dataset
tests build tiny synthetic crop files in a temp directory. The Stage 1, Stage 3A
and Stage 3B suites are untouched.
"""
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.domains.built_environment.direction.dataset import RegionCrops
from src.domains.built_environment.direction.model import (
    CLASS_NAMES, INPUT_MODES, build_model, count_parameters, in_channels_for,
    select_channels, verify_first_conv_init, widen_first_conv)


# ------------------------------------------------------- first convolution
def test_six_channel_conv_reproduces_pretrained_on_unchanged_pair():
    """The documented property: conv6([I, I]) == conv3(I) exactly."""
    m = build_model("both")
    ev = verify_first_conv_init(m)
    assert ev["in_channels"] == 6
    assert ev["halves_identical"]
    assert ev["equals_pretrained_over_two"]
    assert ev["reproduces_pretrained_on_unchanged_pair"]
    # Exact in exact arithmetic; in float32 only accumulation error remains, so
    # the bound is on RELATIVE error against a seeded probe.
    assert ev["probe_seed"] == 0
    assert ev["max_relative_diff_unchanged_pair"] < 1e-5, ev["max_relative_diff_unchanged_pair"]


def test_first_conv_init_is_deterministic():
    a = build_model("both").conv1.weight.detach()
    b = build_model("both").conv1.weight.detach()
    assert torch.equal(a, b), "first-conv init must draw no random numbers"


def test_widen_is_pure_scaling_of_pretrained_weights():
    import torch.nn as nn
    conv = nn.Conv2d(3, 8, 3, bias=False)
    with torch.no_grad():
        conv.weight.fill_(1.0)
    wide = widen_first_conv(conv, 6)
    assert wide.in_channels == 6
    assert torch.allclose(wide.weight, torch.full_like(wide.weight, 0.5))


def test_widen_rejects_incompatible_channel_count():
    import torch.nn as nn
    try:
        widen_first_conv(nn.Conv2d(3, 8, 3), 7)
    except ValueError:
        return
    raise AssertionError("widening to a non-multiple channel count must raise")


def test_parameter_count_is_about_11m():
    n = count_parameters(build_model("both"))
    assert 10_500_000 < n < 12_000_000, n


def test_ablation_models_keep_pretrained_first_conv():
    for mode in ("before", "after"):
        ev = verify_first_conv_init(build_model(mode))
        assert ev["in_channels"] == 3
        assert ev["equals_pretrained"]


def test_input_modes_and_channel_selection():
    assert INPUT_MODES == ("both", "before", "after")
    assert in_channels_for("both") == 6
    assert in_channels_for("before") == 3
    x = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1, 1)
    assert select_channels(x, "both").shape[1] == 6
    assert torch.equal(select_channels(x, "before").flatten(), torch.tensor([0., 1., 2.]))
    assert torch.equal(select_channels(x, "after").flatten(), torch.tensor([3., 4., 5.]))


def test_model_forward_shapes():
    m = build_model("both", pretrained=False).eval()
    with torch.no_grad():
        out = m(torch.zeros(2, 6, 128, 128))
    assert out.shape == (2, 2)
    assert len(CLASS_NAMES) == 2


# ---------------------------------------------------------------- dataset
def _make_crops(tmp, n=8, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 255, (n, 128, 128, 6), dtype=np.uint8)
    y = np.array([i % 2 for i in range(n)], dtype=np.int64)
    np.save(os.path.join(tmp, "train_x.npy"), x)
    np.save(os.path.join(tmp, "train_y.npy"), y)
    return x, y


def test_augmentation_is_applied_identically_to_both_dates():
    """Swapping or independently augmenting the halves would destroy the label."""
    tmp = tempfile.mkdtemp()
    try:
        x, _ = _make_crops(tmp)
        # Make BEFORE and AFTER identical, so any divergence after augmentation
        # proves the halves were transformed differently.
        x[:, :, :, 3:6] = x[:, :, :, 0:3]
        np.save(os.path.join(tmp, "train_x.npy"), x)
        ds = RegionCrops(tmp, "train", mode="both", augment=True, seed=7)
        for epoch in (0, 1, 2):
            ds.set_epoch(epoch)
            for i in range(len(ds)):
                got, _ = ds[i]
                assert torch.allclose(got[0:3], got[3:6], atol=1e-6), (
                    "BEFORE and AFTER halves diverged under augmentation")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_augmentation_is_reproducible_for_a_given_epoch():
    tmp = tempfile.mkdtemp()
    try:
        _make_crops(tmp)
        a = RegionCrops(tmp, "train", mode="both", augment=True, seed=11)
        b = RegionCrops(tmp, "train", mode="both", augment=True, seed=11)
        a.set_epoch(3)
        b.set_epoch(3)
        for i in range(len(a)):
            assert torch.equal(a[i][0], b[i][0])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_augmentation_varies_across_epochs():
    tmp = tempfile.mkdtemp()
    try:
        _make_crops(tmp)
        ds = RegionCrops(tmp, "train", mode="both", augment=True, seed=5)
        ds.set_epoch(0)
        first = [ds[i][0].clone() for i in range(len(ds))]
        ds.set_epoch(1)
        second = [ds[i][0] for i in range(len(ds))]
        assert any(not torch.equal(a, b) for a, b in zip(first, second))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_eval_split_is_not_augmented():
    tmp = tempfile.mkdtemp()
    try:
        _make_crops(tmp)
        a = RegionCrops(tmp, "train", mode="both", augment=False, seed=1)
        b = RegionCrops(tmp, "train", mode="both", augment=False, seed=99)
        for i in range(len(a)):
            assert torch.equal(a[i][0], b[i][0]), "eval crops must be deterministic"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_mode_selects_correct_channel_count():
    tmp = tempfile.mkdtemp()
    try:
        _make_crops(tmp)
        assert RegionCrops(tmp, "train", mode="both")[0][0].shape[0] == 6
        assert RegionCrops(tmp, "train", mode="before")[0][0].shape[0] == 3
        assert RegionCrops(tmp, "train", mode="after")[0][0].shape[0] == 3
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_class_weights_upweight_the_minority_class():
    tmp = tempfile.mkdtemp()
    try:
        x = np.zeros((10, 128, 128, 6), np.uint8)
        y = np.array([0] * 8 + [1] * 2, dtype=np.int64)   # 4:1 imbalance
        np.save(os.path.join(tmp, "train_x.npy"), x)
        np.save(os.path.join(tmp, "train_y.npy"), y)
        ds = RegionCrops(tmp, "train")
        w = ds.class_weights()
        assert w[1] > w[0], "minority class must be up-weighted"
        assert abs(float(w.mean()) - 1.0) < 1e-5, "weights are mean-normalised"
        assert ds.class_counts() == {"construction": 8, "demolition": 2}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ runner
def _run():
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS] {name}")
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run())
