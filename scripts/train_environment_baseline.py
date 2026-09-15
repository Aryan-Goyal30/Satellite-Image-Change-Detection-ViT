"""Train the Phase 4B-1B six-band environmental baseline on frozen v24.

The first environmental model experiment for Earth Guardian. It establishes a
baseline and nothing more: no ablation, no comparison, no integration.

Everything that could leak evaluation information is confined to TRAIN and VAL:

  * normalisation statistics come from the TRAIN split only (Phase 4B-1A);
  * `pos_weight` comes from TRAIN pixels only;
  * the checkpoint is selected on VALIDATION F1;
  * the decision threshold is selected on VALIDATION only, after training.

The TEST split is not read by this script at all.

Usage:  python scripts/train_environment_baseline.py [--epochs 80]
"""
import argparse
import datetime
import hashlib
import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config                                              # noqa: E402
from src.domains.environment.data.loader import (                   # noqa: E402
    AUGMENTATION_POLICY, EnvironmentChangeDataset, load_band_statistics)
from src.domains.environment.model import BAND_SETS, build_model    # noqa: E402
from src.eval.metrics import ConfusionAccumulator, ThresholdSweep   # noqa: E402
from src.train.losses import BCEDiceLoss                            # noqa: E402

EXPERIMENT = "phase4b-1b-sixband-baseline"
DATASET_DIR = os.path.join(config.DATA_DIR, "environment", "dataset_v24")
STATS_PATH = os.path.join(config.DATA_DIR, "environment",
                          "characterization_v24_train_bands.json")
#: Deliberately OUTSIDE the frozen dataset directory.
OUT_DIR = os.path.join(config.OUTPUTS, "environment_baseline")

SEED = 20260913

#: TRAIN-pixel positive weighting measured in Phase 4B-1A:
#: 12,621,541 negative / 944,411 positive.
POS_WEIGHT = 13.364


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def validate(model, loader, device, amp_dtype):
    """Validation pass. Returns fixed-threshold metrics and a threshold sweep."""
    model.eval()
    fixed = ConfusionAccumulator(threshold=0.5)
    sweep = ThresholdSweep()
    total = 0.0
    for batch in loader:
        before = batch["before"].to(device, non_blocking=True)
        after = batch["after"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            logits = model(before, after)
        probs = torch.sigmoid(logits.float())
        fixed.update(probs, label)
        sweep.update(probs, label)
        total += float(label.numel())
    return fixed.compute(), sweep


def main():
    global OUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--bands", default="six_band", choices=sorted(BAND_SETS),
                        help="band subset for the Phase 4B spectral ablation; "
                             "the ONLY variable that may differ between runs")
    parser.add_argument("--out-dir", default=None,
                        help="experiment directory; defaults to the baseline dir")
    args = parser.parse_args()

    bands = BAND_SETS[args.bands]
    OUT_DIR = args.out_dir or OUT_DIR
    os.makedirs(OUT_DIR, exist_ok=True)
    set_determinism(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (device.type == "cuda"
                                   and torch.cuda.is_bf16_supported()) else torch.float16

    stats = load_band_statistics(STATS_PATH, bands)
    train_set = EnvironmentChangeDataset(DATASET_DIR, "train", stats, augment=True,
                                         eligible_only=True, seed=SEED, bands=bands)
    val_set = EnvironmentChangeDataset(DATASET_DIR, "val", stats, augment=False,
                                       eligible_only=False, seed=SEED, bands=bands)
    print(f"[data] train {train_set.summary()}")
    print(f"[data] val   {val_set.summary()}")
    print(f"[data] normalisation from {stats['source']} sha256 {stats['sha256'][:16]}")

    generator = torch.Generator()
    generator.manual_seed(SEED)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, generator=generator, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=0)

    model = build_model(bands=bands, encoder=args.encoder, pretrained=True).to(device)
    description = model.describe()
    print(f"[model] {description['name']} {args.encoder} "
          f"{description['parameters']/1e6:.2f}M parameters, "
          f"{description['in_channels']} input channels")

    criterion = BCEDiceLoss(w_bce=0.5, w_dice=0.5, pos_weight=POS_WEIGHT).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    history, best = [], {"val_f1": -1.0, "epoch": -1}
    checkpoint_path = os.path.join(OUT_DIR, "environment_best.pt"
                                   if args.bands != "six_band"
                                   else "environment_sixband_best.pt")
    started = time.time()
    stale = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, steps = 0.0, 0
        for batch in train_loader:
            before = batch["before"].to(device, non_blocking=True)
            after = batch["after"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                logits = model(before, after)
            loss = criterion(logits.float(), label)
            loss.backward()
            optimiser.step()
            running += float(loss.detach())
            steps += 1
        scheduler.step()

        fixed, sweep = validate(model, val_loader, device, amp_dtype)
        peak = sweep.best_f1()
        row = {"epoch": epoch, "train_loss": running / max(steps, 1),
               "lr": scheduler.get_last_lr()[0],
               "val_f1_at_0.5": fixed["f1"], "val_iou_at_0.5": fixed["iou"],
               "val_precision_at_0.5": fixed["precision"],
               "val_recall_at_0.5": fixed["recall"],
               "val_best_f1": peak["f1"], "val_best_threshold": peak["threshold"],
               "val_ap": sweep.average_precision()}
        history.append(row)
        marker = ""
        if peak["f1"] > best["val_f1"]:
            best = {"val_f1": peak["f1"], "epoch": epoch,
                    "threshold": peak["threshold"], "metrics": row}
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "encoder": args.encoder, "description": description,
                        "bands": list(bands), "val_f1": peak["f1"],
                        "threshold_hint": peak["threshold"]},
                       checkpoint_path)
            stale = 0
            marker = "  <- best"
        else:
            stale += 1
        print(f"  epoch {epoch:3d}  loss {row['train_loss']:.4f}  "
              f"val F1@0.5 {fixed['f1']:.4f}  val best-F1 {peak['f1']:.4f} "
              f"@{peak['threshold']:.3f}  AP {row['val_ap']:.4f}{marker}", flush=True)
        if stale >= args.patience:
            print(f"  early stop: {args.patience} epochs without improvement")
            break

    duration = time.time() - started
    digest = hashlib.sha256(open(checkpoint_path, "rb").read()).hexdigest()

    manifest = {
        "experiment": EXPERIMENT,
        "band_set": args.bands,
        "bands": list(bands),
        "dataset_version": "v24",
        "dataset_dir": os.path.relpath(DATASET_DIR),
        "architecture": description,
        "normalisation": stats,
        "loss": {"formulation": "0.5 * BCEWithLogits(pos_weight) + 0.5 * (1 - soft Dice)",
                 "w_bce": 0.5, "w_dice": 0.5, "pos_weight": POS_WEIGHT,
                 "pos_weight_source": "TRAIN pixels only: 12,621,541 neg / 944,411 pos",
                 "effective_number_scheme": "not used (degenerate at pixel scale)"},
        "optimiser": {"name": "AdamW", "lr": args.lr,
                      "weight_decay": args.weight_decay,
                      "schedule": f"CosineAnnealingLR(T_max={args.epochs})"},
        "batch_size": args.batch_size, "epochs_requested": args.epochs,
        "epochs_run": len(history), "seed": SEED,
        "precision": str(amp_dtype).replace("torch.", ""),
        "augmentation": AUGMENTATION_POLICY,
        "splits": {"train": train_set.summary(), "val": val_set.summary(),
                   "test": "not read by this script"},
        "checkpoint_selection": "highest validation best-F1 over the threshold sweep",
        "threshold_selection": "validation only, after training (see evaluate script)",
        "best_epoch": best["epoch"], "best_val_f1": best["val_f1"],
        "best_val_threshold": best["threshold"],
        "training_seconds": round(duration, 1),
        "hardware": (torch.cuda.get_device_name(0) if device.type == "cuda"
                     else "cpu"),
        "checkpoint": os.path.relpath(checkpoint_path),
        "checkpoint_sha256": digest,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc)
                         .strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(os.path.join(OUT_DIR, "experiment_manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    with open(os.path.join(OUT_DIR, "training_history.json"), "w",
              encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)

    print(f"\n[done] best epoch {best['epoch']} val best-F1 {best['val_f1']:.4f} "
          f"in {duration/60:.1f} min")
    print(f"       checkpoint {checkpoint_path}")
    print(f"       sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
