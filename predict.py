"""Earth Guardian CLI - analyze one image pair.

    python predict.py --before images/before1.png --after images/after1.png
    python predict.py --before A.png --after B.png --out outputs/demo --json

The CLI is a thin wrapper: it asks the engine registry for a domain engine and
works with the ChangeResult contract. The Streamlit app uses the same engine and
the same contract, so there is one code path and one result representation.

    engine.analyze(...) -> ChangeResult -> presentation / export

The legacy result dictionary is still available behind --legacy-json for
consumers that have not migrated. It is deprecated.
"""
import argparse, json, os, sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.core import registry
from src.domains.built_environment.engine import overlay


def main():
    ap = argparse.ArgumentParser(description="Earth Guardian - Built-Environment Change Monitor")
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--engine", default="built_environment",
                    help=f"engine to use. Available: {registry.available()}")
    ap.add_argument("--checkpoint", default=None,
                    help="override; by default the engine resolves its own "
                         "configured checkpoint")
    ap.add_argument("--out", default="outputs/predictions")
    ap.add_argument("--threshold", type=float, default=None,
                    help="default: value selected on the validation split")
    ap.add_argument("--min-area-px", type=int, default=32)
    ap.add_argument("--gsd-m", type=float, default=None,
                    help="metres per pixel; only then are m2 areas reported")
    ap.add_argument("--json", action="store_true", help="print the ChangeResult JSON only")
    ap.add_argument("--legacy-json", action="store_true",
                    help="DEPRECATED: also write the pre-v1 result dictionary "
                         "as <stem>_result_legacy.json")
    args = ap.parse_args()

    engine = (registry.get(args.engine, checkpoint=args.checkpoint)
              if args.checkpoint else registry.get(args.engine))
    out = engine.analyze(args.before, args.after, threshold=args.threshold,
                         min_area_px=args.min_area_px, gsd_m=args.gsd_m)

    # ChangeResult is the primary representation.
    cr = out["change_result"]
    mask, prob = out["mask"], out["probability"]
    md = engine.metadata

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.before))[0]
    Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(args.out, f"{stem}_mask.png"))
    Image.fromarray((prob * 255).astype(np.uint8)).save(os.path.join(args.out, f"{stem}_prob.png"))
    Image.fromarray(overlay(out["after"], mask)).save(os.path.join(args.out, f"{stem}_overlay.png"))
    with open(os.path.join(args.out, f"{stem}_result.json"), "w") as f:
        json.dump(cr.to_dict(), f, indent=2)

    if args.legacy_json:   # deprecated compatibility path
        with open(os.path.join(args.out, f"{stem}_result_legacy.json"), "w") as f:
            json.dump(out["result"], f, indent=2)

    if args.json:
        print(json.dumps(cr.to_dict(), indent=2)); return

    q = cr.quantities
    layer = cr.primary_layer
    prov = cr.provenance
    print(f"\n  Earth Guardian - {md.display_name} ({md.name} v{md.version})")
    print(f"  task       : {md.task}")
    print(f"  trained on : {prov.dataset}  (val F1 {round(engine.val_f1, 4)})")
    print("  " + "-" * 52)
    print(f"  layer            : {layer.name}")
    print(f"  changed pixels   : {q.changed_pixels:,} / {q.total_pixels:,}")
    print(f"  changed area     : {q.changed_percentage:.3f} %")
    print(f"  regions detected : {len(cr.regions)}")
    print(f"  mean confidence  : {layer.mean_confidence:.3f}")
    if q.area_m2 is None:
        print("  area in m2       : n/a (no GSD supplied - not fabricated)")
    else:
        print(f"  area in m2       : {q.area_m2:,}")
    print(f"  threshold        : {cr.params['threshold']}")
    print(f"  runtime          : {cr.runtime_seconds}s")
    for w in cr.warnings:
        print(f"  warning          : {w}")
    print(f"\n  wrote -> {args.out}/  (schema v{cr.schema_version})")


if __name__ == "__main__":
    main()
