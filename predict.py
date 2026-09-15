"""Earth Guardian CLI - analyze one image pair.

    python predict.py --before images/before1.png --after images/after1.png
    python predict.py --before A.tif --after B.tif        # GeoTIFF: adds m2
    python predict.py --engine environment \
        --before s_before.npy --after s_after.npy        # six-band Sentinel-2

The CLI is a thin wrapper: it asks the engine registry for a domain engine and
works with the ChangeResult contract. The Streamlit app uses the same engine and
the same contract, so there is one code path and one result representation.

    engine.analyze(...) -> ChangeResult -> presentation / export

Domains do not share an input contract, so the CLI does not assume one. Reading
the pair is delegated to src/common/pair_input.py, which dispatches on the
engine's own domain: RGB images for built_environment, six-band Sentinel-2
surface reflectance for environment. The CLI itself contains no band handling
and no normalisation.

The legacy result dictionary is still available behind --legacy-json for
consumers that have not migrated. It is deprecated, and only the
built-environment engine produces one.
"""
import argparse, json, os, sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import config
from src.common import pair_input
from src.common.visualization import overlay
from src.core import registry


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
    ap.add_argument("--min-area-px", type=int, default=config.DEFAULT_MIN_AREA_PX,
                    help="connected components smaller than this are discarded")
    ap.add_argument("--gsd-m", type=float, default=None,
                    help="metres per pixel; only then are m2 areas reported for "
                         "imagery that is not a georeferenced GeoTIFF")
    ap.add_argument("--json", action="store_true", help="print the ChangeResult JSON only")
    ap.add_argument("--legacy-json", action="store_true",
                    help="DEPRECATED: also write the pre-v1 result dictionary "
                         "as <stem>_result_legacy.json")
    args = ap.parse_args()

    engine = (registry.get(args.engine, checkpoint=args.checkpoint)
              if args.checkpoint else registry.get(args.engine))

    # Read the pair the way THIS domain requires. For built_environment that is
    # the unchanged open_image / validate_pair / to_rgb sequence, including
    # GeoTIFF metadata; for environment it is the six-band contract, which
    # refuses RGB rather than fabricating the missing bands.
    pair = pair_input.prepare(engine.domain, args.before, args.after,
                              engine.metadata.input_spec)
    for message in pair.warnings:
        print(f"  warning: {message}")
    if not pair.ok:
        for message in pair.errors:
            print(f"  error: {message}")
        raise SystemExit(1)

    out = engine.analyze(pair.before, pair.after,
                         threshold=args.threshold, min_area_px=args.min_area_px,
                         gsd_m=args.gsd_m, georef=pair.georef)

    # ChangeResult is the primary representation.
    cr = out["change_result"]
    mask, prob = out["mask"], out["probability"]
    md = engine.metadata

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.before))[0]
    Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(args.out, f"{stem}_mask.png"))
    Image.fromarray((prob * 255).astype(np.uint8)).save(os.path.join(args.out, f"{stem}_prob.png"))
    # The overlay is drawn on the domain's DISPLAY image. For an RGB domain that
    # is the same array the model read; for the environment domain it is the
    # true-colour composite the domain itself renders, because a six-band
    # reflectance array is not displayable.
    if pair.preview_after is not None:
        Image.fromarray(overlay(pair.preview_after, mask)).save(
            os.path.join(args.out, f"{stem}_overlay.png"))
    with open(os.path.join(args.out, f"{stem}_result.json"), "w") as f:
        json.dump(cr.to_dict(), f, indent=2)

    if args.legacy_json:   # deprecated compatibility path
        if "result" in out:
            with open(os.path.join(args.out, f"{stem}_result_legacy.json"), "w") as f:
                json.dump(out["result"], f, indent=2)
        else:
            print(f"  warning: --legacy-json is not available for the "
                  f"{engine.domain} engine; it is a deprecated built-environment "
                  f"output shape. The ChangeResult JSON was written as usual.")

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
        print("  area in m2       : n/a (no metric georeferencing - not fabricated)")
    else:
        print(f"  area in m2       : {q.area_m2:,}")
    if cr.georef is not None:
        print(f"  crs              : {cr.georef.crs or 'not declared'}")
    print(f"  threshold        : {cr.params['threshold']}")
    print(f"  min region size  : {cr.params['min_area_px']} px")
    print(f"  runtime          : {cr.runtime_seconds}s")
    for w in cr.warnings:
        print(f"  warning          : {w}")
    print(f"\n  wrote -> {args.out}/  (schema v{cr.schema_version})")


if __name__ == "__main__":
    main()
