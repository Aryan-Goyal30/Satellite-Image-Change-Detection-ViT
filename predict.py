"""Earth Guardian CLI - analyze one image pair.

    python predict.py --before images/before1.png --after images/after1.png
    python predict.py --before A.png --after B.png --out outputs/demo --json

The CLI is a thin wrapper: all logic lives in src/inference/engine.py, which the
Streamlit app also calls. One code path, two entry points.
"""
import argparse, json, os, sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.inference.engine import ChangeEngine, overlay


def main():
    ap = argparse.ArgumentParser(description="Earth Guardian - Built-Environment Change Monitor")
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--checkpoint", default="checkpoints/siamese_unet_r34_best.pt")
    ap.add_argument("--out", default="outputs/predictions")
    ap.add_argument("--threshold", type=float, default=None,
                    help="default: value selected on the validation split")
    ap.add_argument("--min-area-px", type=int, default=32)
    ap.add_argument("--gsd-m", type=float, default=None,
                    help="metres per pixel; only then are m2 areas reported")
    ap.add_argument("--json", action="store_true", help="print JSON only")
    args = ap.parse_args()

    engine = ChangeEngine(args.checkpoint)
    out = engine.analyze(args.before, args.after, threshold=args.threshold,
                         min_area_px=args.min_area_px, gsd_m=args.gsd_m)
    res, mask, prob = out["result"], out["mask"], out["probability"]

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.before))[0]
    Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(args.out, f"{stem}_mask.png"))
    Image.fromarray((prob * 255).astype(np.uint8)).save(os.path.join(args.out, f"{stem}_prob.png"))
    Image.fromarray(overlay(out["after"], mask)).save(os.path.join(args.out, f"{stem}_overlay.png"))
    with open(os.path.join(args.out, f"{stem}_result.json"), "w") as f:
        json.dump(res, f, indent=2)

    if args.json:
        print(json.dumps(res, indent=2)); return

    s = res["summary"]
    print(f"\n  Earth Guardian - {res['model']['name']} v{res['model']['version']}")
    print(f"  capability : {res['model']['capability']}")
    print(f"  trained on : {res['model']['trained_on']}  (val F1 {res['model']['val_f1']})")
    print("  " + "-" * 52)
    print(f"  changed pixels   : {s['changed_pixels']:,} / {s['total_pixels']:,}")
    print(f"  changed area     : {s['changed_area_pct']:.3f} %")
    print(f"  regions detected : {s['n_regions']}")
    print(f"  mean confidence  : {s['mean_confidence']:.3f}")
    if s["changed_area_m2"] is None:
        print("  area in m2       : n/a (no GSD supplied - not fabricated)")
    else:
        print(f"  area in m2       : {s['changed_area_m2']:,}")
    print(f"  threshold        : {res['params']['threshold']}")
    print(f"  runtime          : {res['runtime_seconds']}s")
    print(f"\n  wrote -> {args.out}/")


if __name__ == "__main__":
    main()
