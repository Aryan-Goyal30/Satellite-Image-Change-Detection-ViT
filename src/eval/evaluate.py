"""Evaluate a trained checkpoint against LEVIR-CD ground truth.

Protocol (this matters for defensibility):
  1. The operating threshold is selected on the VALIDATION split.
  2. That threshold is then applied, unchanged, to the TEST split.
     The test set is never used to tune anything.
  3. Metrics are globally aggregated over all pixels of the split
     (the LEVIR-CD convention) - see src/eval/metrics.py.

Usage:
    python -m src.eval.evaluate --checkpoint checkpoints/siamese_unet_r34_best.pt
"""
import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.levir import LevirCDTiles
from src.eval.metrics import ConfusionAccumulator, ThresholdSweep, format_metrics
from src.models.siamese_unet import build_model

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TILES = os.path.join(ROOT, "data", "levir_cd_tiles")
RESULTS = os.path.join(ROOT, "outputs", "results")


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ck.get("encoder", "resnet34"), pretrained=False).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, ck


@torch.no_grad()
def run_split(model, split, device, batch_size, workers, threshold, amp, tiles_root=None):
    ds = LevirCDTiles(tiles_root or TILES, split, augment=False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=workers, pin_memory=(device == "cuda"))
    conf = ConfusionAccumulator(threshold=threshold)
    sweep = ThresholdSweep()
    t0 = time.time()
    for batch in dl:
        a = batch["a"].to(device, non_blocking=True)
        b = batch["b"].to(device, non_blocking=True)
        m = batch["mask"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits = model(a, b)
        probs = torch.sigmoid(logits.float())
        conf.update(probs, m)
        sweep.update(probs, m)
    return conf.compute(), sweep, len(ds), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default=None, help="override tile root (default data/levir_cd_tiles)")
    ap.add_argument("--checkpoint", default="checkpoints/siamese_unet_r34_best.pt")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = args.amp and device == "cuda"
    ckpt_path = args.checkpoint if os.path.isabs(args.checkpoint) \
        else os.path.join(ROOT, args.checkpoint)
    model, ck = load_model(ckpt_path, device)

    print(f"checkpoint : {ckpt_path}")
    print(f"model      : {ck.get('model_name')} v{ck.get('version')} "
          f"(epoch {ck.get('epoch')})")
    print(f"device     : {device}\n")

    # --- 1. select threshold on VALIDATION ---------------------------------
    val_m, val_sweep, n_val, t_val = run_split(
        model, "val", device, args.batch_size, args.workers, 0.5, amp, args.tiles)
    best = val_sweep.best_f1()
    tau = best["threshold"]
    print(format_metrics(val_m, f"VALIDATION @ 0.5  ({n_val:,} tiles)"))
    print(f"\n  Best threshold on val: tau = {tau:.3f}  (val F1 {best['f1']:.4f})")
    print(f"  Val average precision : {val_sweep.average_precision():.4f}\n")

    # --- 2. apply that threshold to TEST, untouched -------------------------
    test_m, test_sweep, n_test, t_test = run_split(
        model, "test", device, args.batch_size, args.workers, tau, amp, args.tiles)
    print(format_metrics(test_m, f"TEST @ tau={tau:.3f} (selected on val)  ({n_test:,} tiles)"))
    ap_test = test_sweep.average_precision()
    print(f"\n  Test average precision: {ap_test:.4f}")
    print(f"  Test inference        : {t_test:.1f}s for {n_test:,} tiles "
          f"({n_test/t_test:.0f} tiles/s)")

    # For transparency we also report what test would have scored at its own
    # optimum. The GAP between the two is the honest cost of threshold transfer.
    test_oracle = test_sweep.best_f1()
    print(f"\n  [transparency] test-optimal threshold would be "
          f"{test_oracle['threshold']:.3f} -> F1 {test_oracle['f1']:.4f} "
          f"(we do NOT use this; gap = {test_oracle['f1'] - test_m['f1']:+.4f})")

    os.makedirs(RESULTS, exist_ok=True)
    th, p, r, f = test_sweep.curve()
    out = {
        "checkpoint": os.path.relpath(ckpt_path, ROOT),
        "model_name": ck.get("model_name"),
        "version": ck.get("version"),
        "epoch": ck.get("epoch"),
        "encoder": ck.get("encoder"),
        "selected_threshold": tau,
        "threshold_selected_on": "validation",
        "val": val_m,
        "val_average_precision": val_sweep.average_precision(),
        "test": test_m,
        "test_average_precision": ap_test,
        "test_oracle_threshold": test_oracle,
        "n_val_tiles": n_val,
        "n_test_tiles": n_test,
        "test_seconds": t_test,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "pr_curve_test": {"threshold": th.tolist(),
                          "precision": p.tolist(),
                          "recall": r.tolist(),
                          "f1": f.tolist()},
    }
    path = os.path.join(RESULTS, "evaluation.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
