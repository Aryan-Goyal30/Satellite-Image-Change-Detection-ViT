"""Train the Stage 3B-2 region direction classifier (construction vs demolition).

Research training only. The frozen LEVIR Siamese U-Net detector, its checkpoint,
the production engine, the UI and the ChangeResult schema are all untouched.

Training configuration and why
------------------------------
loss        : cross-entropy with inverse-frequency class weights (mean-normalised).
              The train split is ~1.749:1 construction:demolition, so the
              minority class is explicitly up-weighted rather than the imbalance
              being absorbed into a misleadingly high accuracy.
optimizer   : AdamW, weight_decay 1e-4 - same family already used for the detector.
lr          : 3e-4 with a 1-epoch linear warmup, then cosine annealing to ~0.
              Fine-tuning pretrained ImageNet features, so a modest lr.
scheduler   : cosine annealing over the full epoch budget.
batch size  : 128 (128x128x6 crops fit comfortably in 8.5 GB VRAM).
augmentation: dihedral only (flips + k*90 rotations), applied identically to the
              BEFORE and AFTER halves. See dataset.py for why nothing else.
epochs      : 30, early stopping patience 7 on validation macro-F1.
seed        : fixed; torch/numpy/python seeded, cuDNN deterministic.
selection   : best validation MACRO-F1, not accuracy - the splits have different
              class ratios (train 1.749, val 2.148, test 1.635), so accuracy on
              val would reward predicting the majority class.

Usage:
    python scripts/train_direction_classifier.py --mode both
    python scripts/train_direction_classifier.py --mode before   # ablation
    python scripts/train_direction_classifier.py --mode after    # ablation
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config  # noqa: E402
from src.domains.built_environment.data.s2looking import TEMPORAL_ORDERING  # noqa: E402
from src.domains.built_environment.direction.dataset import RegionCrops  # noqa: E402
from src.domains.built_environment.direction.model import (  # noqa: E402
    CLASS_NAMES, FIRST_CONV_INIT, build_model, count_parameters,
    verify_first_conv_init)

CROPS_DIR = os.path.join(config.S2LOOKING_RAW, "crops")
SEED = 1337


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def macro_f1(cm):
    """Macro-F1 from a 2x2 confusion matrix (rows = true, cols = predicted)."""
    f1s = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s)), f1s


@torch.no_grad()
def evaluate(model, loader, device, n_classes=2):
    model.eval()
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    loss_sum, n = 0.0, 0
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x)
            loss = criterion(logits, y)
        loss_sum += float(loss) * y.numel()
        n += y.numel()
        pred = logits.argmax(1)
        for t, p in zip(y.cpu().numpy(), pred.cpu().numpy()):
            cm[t, p] += 1
    mf1, per_class = macro_f1(cm)
    acc = float(np.trace(cm) / max(1, cm.sum()))
    return {"loss": loss_sum / max(1, n), "macro_f1": mf1, "accuracy": acc,
            "per_class_f1": per_class, "confusion": cm.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("both", "before", "after"), default="both")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=7)
    # 0 by default: the per-sample work is a memmap slice plus a normalise, so
    # worker processes buy little here and cost a spawn-pickle per epoch on
    # Windows. Not raised to chase throughput.
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    if not os.path.exists(os.path.join(CROPS_DIR, "train_x.npy")):
        print(f"NOT FOUND: crops in {CROPS_DIR}\nRun scripts/build_s2looking_crops.py first.")
        return 1

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = RegionCrops(CROPS_DIR, "train", mode=args.mode, augment=True, seed=args.seed)
    val_ds = RegionCrops(CROPS_DIR, "val", mode=args.mode, augment=False, seed=args.seed)
    print(f"train {len(train_ds)} {train_ds.class_counts()}")
    print(f"val   {len(val_ds)} {val_ds.class_counts()}")

    weights = train_ds.class_weights().to(device)
    print(f"class weights ({', '.join(CLASS_NAMES)}): {weights.tolist()}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = build_model(args.mode).to(device)
    init_evidence = verify_first_conv_init(model)
    print(f"parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    warmup_steps = steps_per_epoch
    total_steps = args.epochs * steps_per_epoch

    def lr_at(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * min(1.0, p)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    ckpt_path = os.path.join(config.CHECKPOINTS, f"direction_resnet18_{args.mode}_best.pt")
    os.makedirs(config.CHECKPOINTS, exist_ok=True)

    history, best, best_epoch, bad = [], -1.0, -1, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        train_ds.set_epoch(epoch)
        model.train()
        run_loss, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            run_loss += loss.detach().item() * y.numel()
            seen += y.numel()
        val = evaluate(model, val_loader, device)
        entry = {"epoch": epoch, "train_loss": run_loss / max(1, seen),
                 "lr": optimizer.param_groups[0]["lr"], **val}
        history.append(entry)
        star = ""
        if val["macro_f1"] > best:
            best, best_epoch, bad = val["macro_f1"], epoch, 0
            torch.save({
                "model_state": model.state_dict(),
                "mode": args.mode,
                "class_names": list(CLASS_NAMES),
                "epoch": epoch,
                "val_macro_f1": best,
                "seed": args.seed,
                "first_conv_init": FIRST_CONV_INIT,
                "temporal_ordering": TEMPORAL_ORDERING,
                "config": vars(args),
            }, ckpt_path)
            star = "  *"
        else:
            bad += 1
        print(f"epoch {epoch:3d}  train_loss {entry['train_loss']:.4f}  "
              f"val_loss {val['loss']:.4f}  val_macroF1 {val['macro_f1']:.4f}  "
              f"val_acc {val['accuracy']:.4f}{star}", flush=True)
        if bad >= args.patience:
            print(f"early stopping: no val macro-F1 improvement for {bad} epochs")
            break

    digest = hashlib.sha256(open(ckpt_path, "rb").read()).hexdigest()
    out = {
        "mode": args.mode,
        "seed": args.seed,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "parameters": count_parameters(model),
        "first_conv_init": FIRST_CONV_INIT,
        "first_conv_verification": init_evidence,
        "class_weights": weights.tolist(),
        "train_counts": train_ds.class_counts(),
        "val_counts": val_ds.class_counts(),
        "config": vars(args),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best,
        "checkpoint": os.path.relpath(ckpt_path, config.ROOT),
        "checkpoint_sha256": digest,
        "selection_criterion": "best validation macro-F1",
        "temporal_ordering": TEMPORAL_ORDERING,
        "minutes": round((time.time() - t0) / 60, 2),
        "history": history,
    }
    os.makedirs(config.S2LOOKING_META, exist_ok=True)
    report = os.path.join(config.S2LOOKING_META, f"direction_train_{args.mode}.json")
    json.dump(out, open(report, "w", encoding="utf-8"), indent=2)
    print(f"\nbest val macro-F1 {best:.4f} at epoch {best_epoch}")
    print(f"[write] {ckpt_path}\n  sha256 {digest}")
    print(f"[write] {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
