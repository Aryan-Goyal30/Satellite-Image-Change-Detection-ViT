"""Stage 0 baseline: score the frozen-ViT proof of concept against LEVIR-CD.

Stage 0 is NOT a trained change detector. It is the original proof of concept
(baseline/main.py): a frozen ImageNet-pretrained ViT-B/16 whose patch
embeddings are compared between the two dates. No weight is updated and nothing
here learns from change labels.

Pipeline per 256x256 tile (the exact tiles Stage 1 is evaluated on):

    tile -> resize to 224 -> frozen ViT-B/16 -> 196 patch tokens per date
         -> mean |f_before - f_after| over the 768 channels -> 14x14 map
         -> normalise to [0, 1] -> bilinear upsample to 256x256 -> change score

Two normalisations are scored, because the choice decides whether the baseline
is a straw man:

    poc_minmax      per-tile min-max, exactly as baseline/main.py does. Uses
                    only the tile's own map (no labels, no test statistics),
                    but it throws away how large the difference was.
    val_calibrated  raw distance mapped to [0, 1] with two fixed constants: the
                    0.5th and 99.5th percentiles of VALIDATION patch distances.
                    Keeps magnitude. The strongest fair use of the same features.

Protocol (the same as src/eval/evaluate.py, reusing its metric classes):

    1. Every tunable quantity is fixed on the VALIDATION split: the calibration
       constants and each variant's operating threshold (argmax F1 of a
       ThresholdSweep on validation).
    2. Only after that is the test split opened. The frozen constants and
       thresholds are applied unchanged. Test AP needs no threshold. The
       test-optimal threshold is reported for transparency and never used.
    3. TP/FP/FN/TN are accumulated globally over every test pixel.

Diagnostic only: the POC's own decision rule (per-tile min-max, patches above
mean + 1.5 * std) is also scored. It has no tunable part.

The Stage 1 checkpoint and evaluation.json are only read. Their SHA-256 hashes
and the ViT weight hash are checked before and after; the script aborts if any
changed.

Usage:
    python -m src.eval.evaluate_baseline
"""
import argparse
import hashlib
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.levir import IMAGENET_MEAN, IMAGENET_STD, LevirCDTiles
from src.eval.metrics import ConfusionAccumulator, ThresholdSweep, format_metrics

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TILES = os.path.join(ROOT, "data", "levir_cd_tiles")
RESULTS = os.path.join(ROOT, "outputs", "results")
STAGE1_EVAL = os.path.join(RESULTS, "evaluation.json")
STAGE1_CKPT = os.path.join(ROOT, "checkpoints", "siamese_unet_r34_best.pt")

VIT_NAME = "vit_base_patch16_224"          # same model string as baseline/main.py
POC_SIGMA = 1.5                            # baseline/main.py default sensitivity
CAL_QUANTILES = (0.005, 0.995)
SCORED = ("poc_minmax", "val_calibrated")

VARIANT_LABELS = {
    "poc_minmax": "Stage 0 - frozen ImageNet ViT-B/16 feature distance, "
                  "per-tile min-max (as in the original POC)",
    "val_calibrated": "Stage 0 - frozen ImageNet ViT-B/16 feature distance, "
                      "calibrated on validation",
    "poc_native_rule": "Stage 0 - original POC decision rule "
                       "(per-tile mean + 1.5 std), no tuning",
}


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_vit(device):
    import timm
    vit = timm.create_model(VIT_NAME, pretrained=True).to(device).eval()
    for p in vit.parameters():
        p.requires_grad_(False)
    return vit


def weights_sha256(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def to_vit_input(x, mean, std):
    """ImageNet-normalised 256px tile -> the ViT checkpoint's own 224px input.

    LevirCDTiles normalises with ImageNet statistics for the Stage 1 encoder.
    That is inverted exactly in float (no uint8 rounding), the tile is resized
    to 224px, and the checkpoint's own mean/std is applied - the 0.5 / 0.5 that
    baseline/main.py hard-codes for this model.
    """
    im_mean = torch.as_tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    im_std = torch.as_tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    x01 = x * im_std + im_mean
    x01 = F.interpolate(x01, size=(224, 224), mode="bilinear",
                        align_corners=False, antialias=True)
    m = torch.as_tensor(mean, device=x.device).view(1, 3, 1, 1)
    s = torch.as_tensor(std, device=x.device).view(1, 3, 1, 1)
    return (x01 - m) / s


@torch.no_grad()
def patch_distance(vit, a, b):
    """B x 1 x 14 x 14 map of mean |f_before - f_after| over the patch tokens."""
    cfg = vit.pretrained_cfg
    fa = vit.forward_features(to_vit_input(a, cfg["mean"], cfg["std"]))
    fb = vit.forward_features(to_vit_input(b, cfg["mean"], cfg["std"]))
    k = getattr(vit, "num_prefix_tokens", 1)      # drop [CLS], as the POC does
    fa, fb = fa[:, k:, :], fb[:, k:, :]
    n = int(round(fa.shape[1] ** 0.5))
    return (fa - fb).abs().mean(dim=2).reshape(-1, 1, n, n).float()


def norm_poc_minmax(d):
    """Per-tile min-max to [0, 1], as compute_change_map in baseline/main.py."""
    flat = d.flatten(1)
    lo = flat.min(1).values.view(-1, 1, 1, 1)
    hi = flat.max(1).values.view(-1, 1, 1, 1)
    rng = hi - lo
    return torch.where(rng > 0, (d - lo) / rng.clamp_min(1e-12), torch.zeros_like(d))


def norm_calibrated(d, lo, hi):
    return ((d - lo) / (hi - lo)).clamp(0.0, 1.0)


def upsample(score, size):
    return F.interpolate(score, size=size, mode="bilinear",
                         align_corners=False).clamp(0.0, 1.0)


def poc_rule_mask(d, size):
    """The POC's own decision: patches above mean + 1.5 * std of the min-max map.

    Population std (numpy's default, as in baseline/main.py) and a strict '>'.
    Nearest upsampling keeps the result binary.
    """
    s = norm_poc_minmax(d)
    flat = s.flatten(1)
    mu = flat.mean(1)
    sd = ((flat - mu[:, None]) ** 2).mean(1).sqrt()
    th = (mu + POC_SIGMA * sd).view(-1, 1, 1, 1)
    return F.interpolate((s > th).float(), size=size, mode="nearest")


def scores(d, size, cal_lo, cal_hi):
    return {"poc_minmax": upsample(norm_poc_minmax(d), size),
            "val_calibrated": upsample(norm_calibrated(d, cal_lo, cal_hi), size)}


def make_loader(split, batch_size, workers, device):
    ds = LevirCDTiles(TILES, split, augment=False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                    pin_memory=(device == "cuda"))
    return ds, dl


def curve_dict(sweep):
    th, p, r, f = sweep.curve()
    return {"threshold": th.tolist(), "precision": p.tolist(),
            "recall": r.tolist(), "f1": f.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    t_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bs = args.batch_size

    # Stage 1 artefacts are read, never written. Hash them to prove it.
    protected = [p for p in (STAGE1_CKPT, STAGE1_EVAL) if os.path.exists(p)]
    hashes_before = {os.path.relpath(p, ROOT): file_sha256(p) for p in protected}

    vit = load_vit(device)
    cfg = vit.pretrained_cfg
    w_before = weights_sha256(vit)
    n_trainable = sum(p.numel() for p in vit.parameters() if p.requires_grad)
    n_params = sum(p.numel() for p in vit.parameters())
    print(f"Stage 0 model : {VIT_NAME} [{cfg.get('hf_hub_id') or cfg.get('tag')}]")
    print(f"                frozen: {n_trainable} trainable of {n_params:,} parameters")
    print(f"ViT input norm: mean {tuple(cfg['mean'])}  std {tuple(cfg['std'])}")
    print(f"device        : {device}\n")

    # ------------------------------------------------------------------ 1. VAL
    # Every tunable quantity is fixed here, before the test split is opened.
    val_ds, val_dl = make_loader("val", bs, args.workers, device)
    val_maps, val_masks = [], []
    for batch in val_dl:
        a = batch["a"].to(device, non_blocking=True)
        b = batch["b"].to(device, non_blocking=True)
        val_maps.append(patch_distance(vit, a, b).cpu())
        val_masks.append(batch["mask"] > 0.5)
    val_maps = torch.cat(val_maps)
    val_masks = torch.cat(val_masks)
    size = tuple(val_masks.shape[-2:])

    q = torch.quantile(val_maps.flatten().double(),
                       torch.tensor(CAL_QUANTILES, dtype=torch.float64))
    cal_lo, cal_hi = float(q[0]), float(q[1])

    val_sweep = {k: ThresholdSweep() for k in SCORED}
    for i in range(0, len(val_maps), bs):
        d = val_maps[i:i + bs].to(device)
        m = val_masks[i:i + bs].to(device).float()
        for k, s in scores(d, size, cal_lo, cal_hi).items():
            val_sweep[k].update(s, m)
    tau = {k: val_sweep[k].best_f1()["threshold"] for k in SCORED}

    val_conf = {k: ConfusionAccumulator(threshold=tau[k]) for k in SCORED}
    val_rule = ConfusionAccumulator(threshold=0.5)          # rule output is 0/1
    for i in range(0, len(val_maps), bs):
        d = val_maps[i:i + bs].to(device)
        m = val_masks[i:i + bs].to(device).float()
        for k, s in scores(d, size, cal_lo, cal_hi).items():
            val_conf[k].update(s, m)
        val_rule.update(poc_rule_mask(d, size), m)

    frozen = {
        "calibration": {"lo": cal_lo, "hi": cal_hi, "quantiles": list(CAL_QUANTILES),
                        "fit_on": "validation patch distances"},
        "thresholds": dict(tau),
        "threshold_rule": "argmax F1 of ThresholdSweep on validation",
    }
    print("FROZEN on validation - the test pass below cannot change these:")
    print(f"  calibration   lo = {cal_lo:.4f}   hi = {cal_hi:.4f}")
    for k in SCORED:
        print(f"  tau[{k:<14}] = {tau[k]:.3f}   (val F1 {val_conf[k].compute()['f1']:.4f})")
    print()
    n_val = len(val_ds)
    del val_maps, val_masks, val_dl, val_ds

    # ----------------------------------------------------------------- 2. TEST
    test_ds, test_dl = make_loader("test", bs, args.workers, device)
    test_conf = {k: ConfusionAccumulator(threshold=frozen["thresholds"][k]) for k in SCORED}
    test_sweep = {k: ThresholdSweep() for k in SCORED}
    test_rule = ConfusionAccumulator(threshold=0.5)
    t_test = time.time()
    lo, hi = frozen["calibration"]["lo"], frozen["calibration"]["hi"]
    for batch in test_dl:
        a = batch["a"].to(device, non_blocking=True)
        b = batch["b"].to(device, non_blocking=True)
        m = batch["mask"].to(device, non_blocking=True)
        d = patch_distance(vit, a, b)
        for k, s in scores(d, size, lo, hi).items():
            test_conf[k].update(s, m)
            test_sweep[k].update(s, m)
        test_rule.update(poc_rule_mask(d, size), m)
    t_test = time.time() - t_test

    # ----------------------------------------------------------- 3. INTEGRITY
    w_after = weights_sha256(vit)
    hashes_after = {k: file_sha256(os.path.join(ROOT, k)) for k in hashes_before}
    if w_after != w_before:
        raise SystemExit("ABORT: ViT weights changed during scoring - Stage 0 must be frozen.")
    if hashes_after != hashes_before:
        raise SystemExit("ABORT: a Stage 1 artefact changed during Stage 0 scoring.")
    if frozen["thresholds"] != tau:
        raise SystemExit("ABORT: thresholds changed after being frozen on validation.")

    s1 = json.load(open(STAGE1_EVAL)) if os.path.exists(STAGE1_EVAL) else None
    test_m = {k: test_conf[k].compute() for k in SCORED}
    rule_m = test_rule.compute()
    n_px = lambda c: c["tp"] + c["fp"] + c["fn"] + c["tn"]
    same_tiles = bool(s1) and s1["n_test_tiles"] == len(test_ds) and s1["n_val_tiles"] == n_val
    same_pixels = bool(s1) and n_px(s1["test"]) == n_px(test_m["poc_minmax"])

    # -------------------------------------------------------------- 4. REPORT
    for k in SCORED:
        print(format_metrics(test_m[k], f"TEST - {VARIANT_LABELS[k]}"))
        oracle = test_sweep[k].best_f1()
        print(f"  Test average precision: {test_sweep[k].average_precision():.4f}")
        print(f"  [transparency] test-optimal threshold would be {oracle['threshold']:.3f} "
              f"-> F1 {oracle['f1']:.4f} (NOT used; gap {oracle['f1'] - test_m[k]['f1']:+.4f})\n")
    print(format_metrics(rule_m, f"TEST - {VARIANT_LABELS['poc_native_rule']}"))
    print()

    print("=" * 78)
    print(f"{'LEVIR-CD test, change class':<40}{'P':>7}{'R':>8}{'F1':>8}{'IoU':>8}{'AP':>8}")
    print("-" * 78)
    for k in SCORED:
        t = test_m[k]
        print(f"{'Stage 0 ' + k:<40}{t['precision']:>7.4f}{t['recall']:>8.4f}"
              f"{t['f1']:>8.4f}{t['iou']:>8.4f}{test_sweep[k].average_precision():>8.4f}")
    print(f"{'Stage 0 poc_native_rule':<40}{rule_m['precision']:>7.4f}{rule_m['recall']:>8.4f}"
          f"{rule_m['f1']:>8.4f}{rule_m['iou']:>8.4f}{'n/a':>8}")
    if s1:
        t = s1["test"]
        print(f"{'Stage 1 trained Siamese U-Net':<40}{t['precision']:>7.4f}{t['recall']:>8.4f}"
              f"{t['f1']:>8.4f}{t['iou']:>8.4f}{s1['test_average_precision']:>8.4f}")
    print("=" * 78)
    print(f"same test tiles as Stage 1: {same_tiles} | same pixel count: {same_pixels} | "
          f"ViT weights unchanged: {w_after == w_before} | Stage 1 files unchanged: "
          f"{hashes_after == hashes_before}")

    out = {
        "stage": 0,
        "description": ("Frozen ImageNet ViT-B/16 feature-distance baseline (the original "
                        "proof of concept). Not a trained change detector."),
        "model": {
            "timm_name": VIT_NAME,
            "pretrained": cfg.get("hf_hub_id") or cfg.get("tag"),
            "input_mean": list(cfg["mean"]), "input_std": list(cfg["std"]),
            "parameters": n_params, "trainable_parameters": n_trainable,
            "weights_sha256_before": w_before, "weights_sha256_after": w_after,
            "weights_unchanged": w_after == w_before,
        },
        "pipeline": {
            "tile": list(size), "vit_input": [224, 224], "patch_grid": [14, 14],
            "distance": "mean |f_before - f_after| over 768 channels, [CLS] dropped",
            "upsampling": "bilinear 14x14 -> 256x256",
            "resize_to_vit": "bilinear with antialias, 256 -> 224",
        },
        "protocol": {
            "tuned_on": "validation",
            "tuned_quantities": ["calibration lo/hi (val_calibrated only)",
                                 "operating threshold per variant"],
            "frozen_before_test_opened": frozen,
            "test_used_for": "final scoring only; test-optimal threshold reported, not used",
            "aggregation": "global TP/FP/FN/TN over all pixels of the split",
            "n_val_tiles": n_val, "n_test_tiles": len(test_ds),
            "same_test_tiles_as_stage1": same_tiles,
            "same_test_pixel_count_as_stage1": same_pixels,
        },
        "variants": {},
        "stage1_reference": ({"source": os.path.relpath(STAGE1_EVAL, ROOT),
                              "model_name": s1["model_name"],
                              "selected_threshold": s1["selected_threshold"],
                              "test": s1["test"],
                              "test_average_precision": s1["test_average_precision"]}
                             if s1 else None),
        "protected_artifacts_sha256": {"before": hashes_before, "after": hashes_after,
                                       "unchanged": hashes_after == hashes_before},
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "test_seconds": t_test,
        "total_seconds": time.time() - t_start,
    }
    for k in SCORED:
        v = {
            "label": VARIANT_LABELS[k],
            "threshold": frozen["thresholds"][k],
            "val": val_conf[k].compute(),
            "val_average_precision": val_sweep[k].average_precision(),
            "test": test_m[k],
            "test_average_precision": test_sweep[k].average_precision(),
            "test_oracle_threshold": test_sweep[k].best_f1(),
            "pr_curve_test": curve_dict(test_sweep[k]),
        }
        if k == "val_calibrated":
            v["calibration"] = frozen["calibration"]
        out["variants"][k] = v
    out["variants"]["poc_native_rule"] = {
        "label": VARIANT_LABELS["poc_native_rule"],
        "rule": f"per-tile min-max, patches > mean + {POC_SIGMA} * std (population)",
        "tuned": False,
        "val": val_rule.compute(),
        "test": rule_m,
    }

    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "baseline_evaluation.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {path}  ({out['total_seconds']:.0f}s total)")


if __name__ == "__main__":
    main()
