"""Generate examples/catalogue.json - the bundled demo example list.

The catalogue lets applications select inputs by logical ID instead of knowing
a dataset's directory layout.

Two kinds of entry are produced:

  1. LEVIR-CD held-out TEST tiles whose ground truth contains at least
     --min-change of changed pixels, in sorted filename order. This is exactly
     the rule the Streamlit demo used to apply inline, so the default example is
     unchanged. These entries only resolve on a machine where the dataset has
     been prepared.
  2. The bundled sample pairs in images/, which are tracked in the repository
     and therefore always resolve. They have no ground-truth mask.

Only measured facts are recorded. These images carry no location or acquisition
date, so no such metadata is written.

Usage:  python scripts/build_example_catalogue.py [--limit 24] [--min-change 0.05]
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config  # noqa: E402

SCHEMA_VERSION = "1.0"
DOMAIN = "built_environment"

# Bundled pairs tracked in the repository. Their LEVIR-CD split membership was
# never recorded, so we do not claim one.
SAMPLE_PAIRS = [
    ("sample_pair_1", "Sample pair 1", "before1.png", "after1.png"),
    ("sample_pair_2", "Sample pair 2", "before.png", "after.png"),
    ("sample_pair_3", "Sample pair 3", "before2.png", "after2.png"),
]


def rel(path):
    return os.path.relpath(path, config.ROOT).replace("\\", "/")


def levir_test_examples(limit, min_change):
    label_dir = os.path.join(config.LEVIR_TILES, "test", "label")
    if not os.path.isdir(label_dir):
        print(f"  !! {label_dir} not found - skipping dataset examples")
        return []
    out = []
    for name in sorted(os.listdir(label_dir)):
        if not name.endswith(".png"):
            continue
        frac = float((np.array(Image.open(os.path.join(label_dir, name))
                               .convert("L")) > 127).mean())
        if frac < min_change:
            continue
        stem = os.path.splitext(name)[0]
        out.append({
            "id": f"levir_{stem}",
            "name": f"LEVIR-CD test tile {stem}",
            "domain": DOMAIN,
            "before": rel(os.path.join(config.LEVIR_TILES, "test", "A", name)),
            "after": rel(os.path.join(config.LEVIR_TILES, "test", "B", name)),
            "ground_truth": rel(os.path.join(config.LEVIR_TILES, "test", "label", name)),
            "description": (f"Held-out LEVIR-CD test tile, 256x256. "
                            f"Ground truth marks {100 * frac:.1f}% of pixels as changed."),
            "source": "LEVIR-CD official test split",
        })
        if len(out) >= limit:
            break
    return out


def sample_examples():
    out = []
    for eid, name, before, after in SAMPLE_PAIRS:
        b = os.path.join(config.DEMO_IMAGES, before)
        a = os.path.join(config.DEMO_IMAGES, after)
        if not (os.path.exists(b) and os.path.exists(a)):
            continue
        with Image.open(b) as im:
            w, h = im.size
        out.append({
            "id": eid,
            "name": name,
            "domain": DOMAIN,
            "before": rel(b),
            "after": rel(a),
            "ground_truth": None,
            "description": (f"Bundled sample pair, {w}x{h}. No ground-truth mask "
                            f"is available for this pair."),
            "source": "LEVIR-CD sample (split membership not recorded)",
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=24,
                    help="max dataset-derived examples")
    ap.add_argument("--min-change", type=float, default=0.05,
                    help="minimum ground-truth changed fraction")
    args = ap.parse_args()

    examples = levir_test_examples(args.limit, args.min_change) + sample_examples()

    # Entries from other domains are preserved: this script owns the
    # built-environment entries only, and re-running it must not delete the
    # environmental bundle that scripts/build_environment_examples.py wrote.
    other, previous = [], {}
    if os.path.exists(config.EXAMPLES_CATALOGUE):
        with open(config.EXAMPLES_CATALOGUE, encoding="utf-8") as f:
            previous = json.load(f)
        other = [e for e in previous.get("examples", []) if e.get("domain") != DOMAIN]
    examples = examples + other

    doc = {
        "schema_version": previous.get("schema_version", SCHEMA_VERSION),
        "generated_by": "scripts/build_example_catalogue.py",
        "note": ("Bundled demo inputs. These images carry no location or "
                 "acquisition-date metadata, so none is recorded. Dataset-derived "
                 "entries resolve only where the LEVIR-CD tiles have been prepared."),
        "selection_rule": (f"LEVIR-CD test tiles in sorted filename order whose "
                           f"ground truth marks at least {args.min_change:.0%} of "
                           f"pixels as changed, limited to {args.limit}."),
        "examples": examples,
    }
    if previous.get("domain_notes"):
        doc["domain_notes"] = previous["domain_notes"]

    os.makedirs(config.EXAMPLES_DIR, exist_ok=True)
    with open(config.EXAMPLES_CATALOGUE, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")

    n_gt = sum(1 for e in examples if e["ground_truth"])
    print(f"wrote {config.EXAMPLES_CATALOGUE}")
    print(f"  {len(examples)} examples ({n_gt} with ground truth)")
    if examples:
        print(f"  first (UI default): {examples[0]['id']}")


if __name__ == "__main__":
    main()
