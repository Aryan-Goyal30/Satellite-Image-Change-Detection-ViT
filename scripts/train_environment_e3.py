"""E3: train the six-band model with the loss at TMF's native ~30 m support.

The ONLY change from E2 is the spatial support at which the loss is evaluated.
Architecture, bands, normalisation, optimiser, schedule, batch size, seed,
augmentation, early stopping and checkpoint rule are all E2's, unchanged. The
model still emits a 256 x 256 10 m logit map; pooling happens inside the loss.

Usage:  python scripts/train_environment_e3.py
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
from src.domains.environment import support                         # noqa: E402
from src.domains.environment.data.loader import (                   # noqa: E402
    AUGMENTATION_POLICY, EnvironmentChangeDataset, load_band_statistics)
from src.domains.environment.model import BAND_SETS, build_model    # noqa: E402
from src.eval.metrics import ConfusionAccumulator, ThresholdSweep   # noqa: E402
from src.train.losses import BCEDiceLoss                            # noqa: E402

EXPERIMENT = "phase4b-4-e3-native-30m-supervision"
DATASET_DIR = os.path.join(config.DATA_DIR, "environment", "dataset_v24")
STATS_PATH = os.path.join(config.DATA_DIR, "environment",
                          "characterization_v24_train_bands.json")
OUT_DIR = os.path.join(config.OUTPUTS, "environment_e3")

SEED = 20260913
#: E2's weighting, reused unchanged so spatial support is the only variable.
#: Measured alternatives are recorded in the manifest but NOT used.
POS_WEIGHT = 13.364


def set_determinism(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def measured_pos_weights(dataset):
    """Verify the positive weighting from TRAIN pixels at both supports."""
    pos10 = neg10 = pos30 = neg30 = 0
    for record in dataset.records:
        label = np.load(os.path.join(DATASET_DIR,
                                     f"{record['sample_id']}_label.npy")) > 0.5
        pos10 += int(label.sum())
        neg10 += int((~label).sum())
        coarse = support.label_30m(label) > 0.5
        pos30 += int(coarse.sum())
        neg30 += int((~coarse).sum())
    return {"train_10m": round(neg10 / max(pos10, 1), 4),
            "train_30m": round(neg30 / max(pos30, 1), 4),
            "used": POS_WEIGHT,
            "note": ("E2's value is reused so that spatial support is the only "
                     "experimental variable; the measured values differ because "
                     "majority pooling removes isolated speckle")}


@torch.no_grad()
def validate(model, loader, device, amp_dtype):
    """Validation at BOTH supports. 30 m is the primary objective."""
    model.eval()
    fixed30 = ConfusionAccumulator(threshold=0.5)
    sweep30, sweep10 = ThresholdSweep(), ThresholdSweep()
    for batch in loader:
        before = batch["before"].to(device, non_blocking=True)
        after = batch["after"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            logits = model(before, after)
        logits = logits.float()
        probs10 = torch.sigmoid(logits)
        probs30 = torch.sigmoid(support.pool_mean_torch(logits))
        coarse = (support.pool_mean_torch(label) >= support.LABEL_MAJORITY).float()
        fixed30.update(probs30, coarse)
        sweep30.update(probs30, coarse)
        sweep10.update(probs10, label)
    return fixed30.compute(), sweep30, sweep10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--encoder", default="resnet34")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    set_determinism(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (device.type == "cuda"
                                   and torch.cuda.is_bf16_supported()) else torch.float16

    bands = BAND_SETS["six_band"]
    stats = load_band_statistics(STATS_PATH, bands)
    train_set = EnvironmentChangeDataset(DATASET_DIR, "train", stats, augment=True,
                                         eligible_only=True, seed=SEED, bands=bands)
    val_set = EnvironmentChangeDataset(DATASET_DIR, "val", stats, augment=False,
                                       eligible_only=False, seed=SEED, bands=bands)
    weights = measured_pos_weights(train_set)
    print(f"[data] train {train_set.summary()}")
    print(f"[data] val   {val_set.summary()}")
    print(f"[data] normalisation sha256 {stats['subset_sha256'][:16]} "
          f"(file {stats['sha256'][:16]})")
    print(f"[loss] pos_weight used {POS_WEIGHT}; measured 10m "
          f"{weights['train_10m']}, 30m {weights['train_30m']}")
    print(f"[support] {support.DESCRIPTION['pooling']} -> {support.GRID}x{support.GRID}")

    generator = torch.Generator()
    generator.manual_seed(SEED)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, generator=generator)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=0)

    model = build_model(bands=bands, encoder=args.encoder, pretrained=True).to(device)
    description = model.describe()
    print(f"[model] {description['name']} {description['parameters']:,} parameters "
          f"(E2 topology, unchanged)")

    criterion = BCEDiceLoss(w_bce=0.5, w_dice=0.5, pos_weight=POS_WEIGHT).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    history, best = [], {"val_f1": -1.0, "epoch": -1}
    checkpoint_path = os.path.join(OUT_DIR, "environment_e3_best.pt")
    started, stale = time.time(), 0

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
            # THE experimental change: pool the 10 m logits to ~30 m support and
            # supervise there. The model's own output stays 10 m.
            pooled = support.pool_mean_torch(logits.float())
            target = (support.pool_mean_torch(label) >= support.LABEL_MAJORITY).float()
            loss = criterion(pooled, target)
            loss.backward()
            optimiser.step()
            running += float(loss.detach())
            steps += 1
        scheduler.step()

        fixed30, sweep30, sweep10 = validate(model, val_loader, device, amp_dtype)
        peak30, peak10 = sweep30.best_f1(), sweep10.best_f1()
        row = {"epoch": epoch, "train_loss": running / max(steps, 1),
               "lr": scheduler.get_last_lr()[0],
               "val_f1_30m_at_0.5": fixed30["f1"], "val_iou_30m_at_0.5": fixed30["iou"],
               "val_best_f1_30m": peak30["f1"], "val_best_threshold_30m": peak30["threshold"],
               "val_ap_30m": sweep30.average_precision(),
               "val_best_f1_10m": peak10["f1"], "val_best_threshold_10m": peak10["threshold"],
               "val_ap_10m": sweep10.average_precision()}
        history.append(row)
        marker = ""
        if peak30["f1"] > best["val_f1"]:
            best = {"val_f1": peak30["f1"], "epoch": epoch,
                    "threshold_30m": peak30["threshold"],
                    "threshold_10m": peak10["threshold"]}
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "encoder": args.encoder, "bands": list(bands),
                        "description": description, "val_f1_30m": peak30["f1"],
                        "support": support.DESCRIPTION}, checkpoint_path)
            stale, marker = 0, "  <- best"
        else:
            stale += 1
        print(f"  epoch {epoch:3d}  loss {row['train_loss']:.4f}  "
              f"val30 best-F1 {peak30['f1']:.4f}@{peak30['threshold']:.3f}  "
              f"AP30 {row['val_ap_30m']:.4f}  |  val10 best-F1 {peak10['f1']:.4f}"
              f"{marker}", flush=True)
        if stale >= args.patience:
            print(f"  early stop: {args.patience} epochs without improvement")
            break

    duration = time.time() - started
    digest = hashlib.sha256(open(checkpoint_path, "rb").read()).hexdigest()
    manifest = {
        "experiment": EXPERIMENT, "dataset_version": "v24",
        "hypothesis": ("does supervising at TMF's native ~30 m support resolve the "
                       "apparent region/topology failure?"),
        "only_change_from_e2": "spatial support at which the loss is evaluated",
        "spatial_support": support.DESCRIPTION,
        "architecture": description, "bands": list(bands),
        "normalisation": stats,
        "loss": {"formulation": "0.5 * BCEWithLogits(pos_weight) + 0.5 * (1 - soft Dice), "
                                "computed on 3x3 mean-pooled logits vs the majority-pooled label",
                 "w_bce": 0.5, "w_dice": 0.5, "pos_weight": POS_WEIGHT,
                 "pos_weight_measured": weights},
        "optimiser": {"name": "AdamW", "lr": args.lr, "weight_decay": args.weight_decay,
                      "schedule": f"CosineAnnealingLR(T_max={args.epochs})"},
        "batch_size": args.batch_size, "seed": SEED,
        "precision": str(amp_dtype).replace("torch.", ""),
        "augmentation": AUGMENTATION_POLICY,
        "splits": {"train": train_set.summary(), "val": val_set.summary(),
                   "test": "not read by this script"},
        "checkpoint_selection": "highest validation best-F1 at 30 m support",
        "best_epoch": best["epoch"], "best_val_f1_30m": best["val_f1"],
        "epochs_run": len(history), "epochs_requested": args.epochs,
        "training_seconds": round(duration, 1),
        "hardware": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "checkpoint": os.path.relpath(checkpoint_path),
        "checkpoint_sha256": digest,
        "dataset_manifest_sha256": "23721c09c864297a7857a3576e4398f700e642c3ab1e4531462eb603a2f827d8",
        "dataset_arrays_sha256": "b53d4b3d89d111de653313bcaad9984c6bc5e240905771a5d09c5fa6cc32d248",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc)
                         .strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(os.path.join(OUT_DIR, "experiment_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    with open(os.path.join(OUT_DIR, "training_history.json"), "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)

    print(f"\n[done] best epoch {best['epoch']} val30 best-F1 {best['val_f1']:.4f} "
          f"in {duration/60:.1f} min")
    print(f"       sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
