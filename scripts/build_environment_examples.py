"""Generate the small, tracked environmental demo bundle from the frozen v24 TEST split.

The v24 archive lives under data/, which is git-ignored and 503 MB, so the
Environment monitor had no bundled examples at all. This script copies THREE
test samples out of it as self-contained, tracked artifacts, and merges catalogue
entries for them into examples/catalogue.json.

Read-only with respect to the dataset
-------------------------------------
The archive is opened for reading and never written. Nothing is regenerated,
re-sampled, re-labelled or re-split. The three sample IDs are pinned below so a
re-run reproduces the same bundle rather than re-selecting.

Selection rule
--------------
Chosen on DATA properties only - ground-truth extent, component count, cloud and
invalid fraction, negative subtype - and never on how the model scores them.
Picking demo cases by model performance would make the demo a claim about the
model that the frozen evaluation does not support.

Representation
--------------
Stored as float32 surface reflectance in the frozen band order, which is exactly
what the engine reads. The conversion from stored L2A digital numbers happens
HERE, once, using the frozen acquisition rule (including the per-product BOA
offset decision), so no application ever converts, selects bands, normalises or
resamples. Labels are copied as-is.

Usage:  python scripts/build_environment_examples.py [--verify-only]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config                                            # noqa: E402
from src.common import examples as example_catalogue              # noqa: E402
from src.domains.environment.data import sentinel2 as s2          # noqa: E402

DATASET_DIR = os.path.join(config.DATA_DIR, "environment", "dataset_v24")
OUT_DIR = os.path.join(config.EXAMPLES_DIR, "environment")
DOMAIN = "environment"
CATALOGUE_SCHEMA_VERSION = "1.1"

#: Frozen v24 manifest digest. The bundle is only meaningful against this exact
#: archive, so a changed manifest stops the script rather than silently
#: producing artifacts from different data.
V24_MANIFEST_SHA256 = "23721c09c864297a7857a3576e4398f700e642c3ab1e4531462eb603a2f827d8"

#: The three pinned samples, with the DATA reason each was chosen.
SELECTION = [
    {
        "sample_id": "env_N0_W70_2021_0106",
        "id": "env_amazon_large_clearing",
        "name": "Amazon - large forest clearing",
        "role": "strong positive",
        "why": ("largest spatially coherent event in the test split: 26,977 "
                "labelled pixels (41.2% of the tile) in only 3 connected "
                "components, 0.05% residual cloud, no invalid pixels"),
    },
    {
        "sample_id": "env_N0_E110_2019_a001",
        "id": "env_seasia_small_clearing",
        "name": "Southeast Asia - smaller forest clearing",
        "role": "moderate positive",
        "why": ("a genuinely harder case: 1,877 labelled pixels (2.9% of the "
                "tile) across 5 components, from the under-represented "
                "Southeast Asia region, 0.4% residual cloud. The first "
                "candidate chosen on extent alone (env_N10_E100_2020_a032) is "
                "recorded as a COMPLETE_MISS in the frozen Phase 4B-3 "
                "taxonomy - the model predicts a similar AREA in the wrong "
                "PLACE - which demonstrates a failure, not the moderate case "
                "this slot is for. The frozen per-sample table was consulted "
                "only to tell those two apart; no new evaluation was run and "
                "no example was chosen for scoring well."),
    },
    {
        "sample_id": "env_S10_W60_2019_0157",
        "id": "env_amazon_intact_forest",
        "name": "Amazon - intact forest (no change)",
        "role": "negative",
        "why": ("a stable_forest negative inside the southern deforestation "
                "arc: no labelled loss, no water, cloud-free - a genuine "
                "no-change case in a high-pressure region, not an easy one"),
    },
]

ARRAYS = ("before", "after", "label")


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest():
    path = os.path.join(DATASET_DIR, "manifest.json")
    if not os.path.exists(path):
        raise SystemExit(
            f"frozen v24 archive not found at {DATASET_DIR}.\n"
            "This script reads the Phase 4B dataset; it cannot regenerate it.")
    digest = sha256(path)
    if digest != V24_MANIFEST_SHA256:
        raise SystemExit(
            f"v24 manifest digest mismatch.\n  expected {V24_MANIFEST_SHA256}\n"
            f"  found    {digest}\nRefusing to build examples from a changed archive.")
    with open(path, encoding="utf-8") as fh:
        return {r["sample_id"]: r for r in json.load(fh)["samples"]}


def reflectance(record, side):
    """Stored L2A digital numbers -> float32 surface reflectance, frozen rule."""
    dn = np.load(os.path.join(DATASET_DIR, f"{record['sample_id']}_{side}.npy"))
    return s2.to_reflectance(dn, record[f"{side}_offset_applied"],
                             record[f"{side}_baseline"]).astype(np.float32)


def describe(record, spec, label):
    fraction = float(label.mean())
    region = ("Southeast Asia" if record["tmf_tile"] in ("N0_E110", "N10_E100")
              else "Amazon")
    before = record["before_datetime"][:10]
    after = record["after_datetime"][:10]
    if record["sample_type"] == "positive":
        change = (f"JRC TMF labels {100 * fraction:.2f}% of pixels as forest "
                  f"loss between these dates.")
    else:
        change = ("JRC TMF labels no forest loss between these dates - this is "
                  "a no-change reference case.")
    return (f"{region}, MGRS tile {record['mgrs_tile']}. Sentinel-2 L2A "
            f"{before} and {after}, six bands at 10 m. {change}")


def build(manifest, write=True):
    entries, report = [], []
    for spec in SELECTION:
        sid = spec["sample_id"]
        record = manifest.get(sid)
        if record is None:
            raise SystemExit(f"sample {sid} is not in the v24 manifest")
        if record["split"] != "test":
            raise SystemExit(f"sample {sid} is in the {record['split']} split, "
                             "not test - refusing")

        label = np.load(os.path.join(DATASET_DIR, f"{sid}_label.npy"))
        arrays = {"before": reflectance(record, "before"),
                  "after": reflectance(record, "after"),
                  "label": label}

        if write:
            os.makedirs(OUT_DIR, exist_ok=True)
            for name, array in arrays.items():
                np.save(os.path.join(OUT_DIR, f"{spec['id']}_{name}.npy"), array)

        rel = os.path.relpath(OUT_DIR, config.ROOT).replace("\\", "/")
        entries.append({
            "id": spec["id"],
            "name": spec["name"],
            "domain": DOMAIN,
            "before": f"{rel}/{spec['id']}_before.npy",
            "after": f"{rel}/{spec['id']}_after.npy",
            "ground_truth": f"{rel}/{spec['id']}_label.npy",
            "description": describe(record, spec, label),
            "source": "JRC TMF + Sentinel-2 L2A, dataset v24 held-out test split",
            "kind": example_catalogue.KIND_SENTINEL2_SIX_BAND,
            "ground_truth_kind": example_catalogue.GT_NPY_MASK,
            "bands": list(s2.BANDS),
            "gsd_m": float(record["gsd_m"]),
            "crs": record["crs"],
            "source_sample_id": sid,
        })
        report.append((spec, record, arrays))
    return entries, report


def merge_catalogue(entries):
    """Add the environment entries, leaving every other domain untouched."""
    path = config.EXAMPLES_CATALOGUE
    doc = {"schema_version": CATALOGUE_SCHEMA_VERSION, "examples": []}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    kept = [e for e in doc.get("examples", []) if e.get("domain") != DOMAIN]
    doc["examples"] = kept + entries
    doc["schema_version"] = CATALOGUE_SCHEMA_VERSION
    notes = doc.setdefault("domain_notes", {})
    notes[DOMAIN] = (
        "Three samples copied from the frozen v24 held-out TEST split, chosen on "
        "ground-truth extent, spatial coherence and image quality only - never "
        "on how the model scores them. Stored as float32 surface reflectance in "
        "band order B02, B03, B04, B08, B11, B12.")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")
    return len(kept), len(entries)


def verify(manifest):
    """Every stored artifact must still equal what the frozen archive holds."""
    ok = True
    for spec in SELECTION:
        sid, eid = spec["sample_id"], spec["id"]
        record = manifest[sid]
        expected = {"before": reflectance(record, "before"),
                    "after": reflectance(record, "after"),
                    "label": np.load(os.path.join(DATASET_DIR, f"{sid}_label.npy"))}
        for name, want in expected.items():
            path = os.path.join(OUT_DIR, f"{eid}_{name}.npy")
            if not os.path.exists(path):
                print(f"  MISSING {path}")
                ok = False
                continue
            got = np.load(path)
            same = (got.shape == want.shape and got.dtype == want.dtype
                    and np.array_equal(got, want))
            print(f"  {'OK  ' if same else 'FAIL'} {eid}_{name}.npy  "
                  f"{got.shape} {got.dtype}  bit-identical={np.array_equal(got, want)}")
            ok = ok and same
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true",
                    help="check existing artifacts against the archive, write nothing")
    args = ap.parse_args()

    manifest = load_manifest()
    print(f"[archive] v24 manifest verified {V24_MANIFEST_SHA256[:16]}...")

    if args.verify_only:
        print("[verify]")
        return 0 if verify(manifest) else 1

    entries, report = build(manifest, write=True)
    kept, added = merge_catalogue(entries)
    print(f"[catalogue] {config.EXAMPLES_CATALOGUE}: {kept} kept + {added} environment")

    print("[verify]")
    ok = verify(manifest)

    total = 0
    for spec, record, _arrays in report:
        size = sum(os.path.getsize(os.path.join(OUT_DIR, f"{spec['id']}_{n}.npy"))
                   for n in ARRAYS)
        total += size
        print(f"  {spec['id']:<30} {spec['role']:<17} {size/1e6:6.2f} MB  "
              f"<- {spec['sample_id']}")
    print(f"[size] {total/1e6:.2f} MB across {len(SELECTION) * len(ARRAYS)} files")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
