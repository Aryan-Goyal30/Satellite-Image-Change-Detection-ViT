"""Train the Siamese U-Net change detector on LEVIR-CD tiles.

Usage:
    python -m src.train.train --epochs 50 --batch-size 32 --workers 4

Checkpoints the best model by VALIDATION F1 (not by loss - loss is dominated by
the background class and is a poor model-selection signal here).
"""
import argparse
import json
import os
import sys
import time
from datetime import timedelta

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src import config
from src.domains.built_environment.data.levir import LevirCDTiles
from src.eval.metrics import ConfusionAccumulator, ThresholdSweep
from src.models.siamese_unet import build_model
from src.train.losses import BCEDiceLoss

ROOT = config.ROOT
TILES = config.LEVIR_TILES
CKPT_DIR = config.CHECKPOINTS


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", default="resnet34", choices=["resnet18", "resnet34", "resnet50"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--pos-weight", type=float, default=2.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=10, help="early stop on val F1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--limit-train", type=int, default=0, help="debug: cap train tiles")
    p.add_argument("--tiles", default=None, help="override tile root (default data/levir_cd_tiles)")
    p.add_argument("--run-name", default="siamese_unet_r34")
    return p.parse_args()


def set_seed(s):
    import random
    import numpy as np
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def validate(model, loader, device, amp):
    model.eval()
    conf = ConfusionAccumulator(threshold=0.5)
    sweep = ThresholdSweep()
    for batch in loader:
        a = batch["a"].to(device, non_blocking=True)
        b = batch["b"].to(device, non_blocking=True)
        m = batch["mask"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits = model(a, b)
        probs = torch.sigmoid(logits.float())
        conf.update(probs, m)
        sweep.update(probs, m)
    return conf.compute(), sweep


def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(CKPT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("!! CUDA not available - training on CPU will be impractically slow.")
    else:
        print(f"Device: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB)")
    amp = args.amp and device == "cuda"

    tiles_root = args.tiles or TILES
    train_ds = LevirCDTiles(tiles_root, "train", augment=True)
    val_ds = LevirCDTiles(tiles_root, "val", augment=False)
    if args.limit_train:
        train_ds.names = train_ds.names[:args.limit_train]
    print(f"train tiles: {len(train_ds):,} | val tiles: {len(val_ds):,}")

    common = dict(num_workers=args.workers, pin_memory=(device == "cuda"),
                  persistent_workers=args.workers > 0)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          drop_last=True, **common)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **common)

    model = build_model(args.encoder, pretrained=True).to(device)
    print(f"model: {model.NAME}-{args.encoder} | params: {model.n_params/1e6:.2f}M")

    criterion = BCEDiceLoss(pos_weight=args.pos_weight).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best_f1, best_epoch, bad = -1.0, -1, 0
    history = []
    ckpt_path = os.path.join(CKPT_DIR, f"{args.run_name}_best.pt")
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n = 0.0, 0
        t0 = time.time()
        for i, batch in enumerate(train_dl, 1):
            a = batch["a"].to(device, non_blocking=True)
            b = batch["b"].to(device, non_blocking=True)
            m = batch["mask"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = criterion(model(a, b), m)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += loss.item() * a.size(0); n += a.size(0)
            if i % 50 == 0:
                print(f"  ep{epoch} [{i}/{len(train_dl)}] loss {running/n:.4f}", flush=True)

        sched.step()
        train_loss = running / max(n, 1)
        val_m, sweep = validate(model, val_dl, device, amp)
        best_t = sweep.best_f1()
        dt = time.time() - t0

        print(f"epoch {epoch:>3}/{args.epochs} | loss {train_loss:.4f} | "
              f"val F1 {val_m['f1']:.4f} IoU {val_m['iou']:.4f} "
              f"P {val_m['precision']:.4f} R {val_m['recall']:.4f} | "
              f"best-thr F1 {best_t['f1']:.4f}@{best_t['threshold']:.2f} | "
              f"{dt:.0f}s", flush=True)

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val": {k: v for k, v in val_m.items() if isinstance(v, float)},
                        "val_best_threshold": best_t, "seconds": dt})

        if val_m["f1"] > best_f1:
            best_f1, best_epoch, bad = val_m["f1"], epoch, 0
            torch.save({
                "model_state": model.state_dict(),
                "encoder": args.encoder,
                "epoch": epoch,
                "val_metrics": val_m,
                "val_best_threshold": best_t,
                "args": vars(args),
                "model_name": f"{model.NAME}-{args.encoder}",
                "version": "1.0.0",
            }, ckpt_path)
            print(f"  -> saved best checkpoint (val F1 {best_f1:.4f})", flush=True)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop: no val F1 improvement for {args.patience} epochs")
                break

    total = time.time() - t_start
    summary = {
        "run_name": args.run_name,
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "total_seconds": total,
        "total_time": str(timedelta(seconds=int(total))),
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "args": vars(args),
        "history": history,
    }
    with open(os.path.join(CKPT_DIR, f"{args.run_name}_history.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone in {summary['total_time']}. Best val F1 {best_f1:.4f} "
          f"(epoch {best_epoch}) -> {ckpt_path}")


if __name__ == "__main__":
    main()
