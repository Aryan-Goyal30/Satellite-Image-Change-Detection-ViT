"""Inspect the extracted S2Looking dataset and report what is ACTUALLY there.

Assumes nothing about directory names, split names, label encoding or channel
count. Everything printed is read off disk. Stage 3B-0 uses this to confirm (or
contradict) the dataset audit before any label is generated.

Usage:  python scripts/inspect_s2looking.py
"""
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config  # noqa: E402

ROOT = config.S2LOOKING_RAW
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def tree(root, max_depth=3):
    """Directory tree with file counts - no assumption about names."""
    print(f"=== directory tree under {root} (depth {max_depth}) ===")
    root = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath[len(root):].count(os.sep)
        if depth > max_depth:
            dirnames[:] = []
            continue
        dirnames.sort()
        rel = os.path.relpath(dirpath, root)
        imgs = [f for f in filenames if f.lower().endswith(IMAGE_EXT)]
        others = [f for f in filenames if not f.lower().endswith(IMAGE_EXT)]
        indent = "  " * depth
        label = "." if rel == "." else os.path.basename(dirpath)
        bits = []
        if imgs:
            bits.append(f"{len(imgs)} images")
        if others:
            bits.append(f"{len(others)} other: {sorted(others)[:4]}")
        print(f"{indent}{label}/  {'  '.join(bits)}")
        if imgs:
            print(f"{indent}   e.g. {sorted(imgs)[:3]}")


def leaf_dirs_with_images(root):
    """Every directory that directly contains images -> sorted file list."""
    out = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        imgs = sorted(f for f in filenames if f.lower().endswith(IMAGE_EXT))
        if imgs:
            out[os.path.relpath(dirpath, root)] = imgs
    return out


def probe(path, n_values=12):
    """Read one raster and report its real properties."""
    with Image.open(path) as im:
        mode, size = im.mode, im.size
        a = np.array(im)
    info = {
        "mode": mode, "size_wh": size, "shape": a.shape, "dtype": str(a.dtype),
        "nan": bool(np.isnan(a).any()) if a.dtype.kind == "f" else False,
    }
    u = np.unique(a)
    info["n_unique"] = int(u.size)
    info["unique_sample"] = u[:n_values].tolist()
    if a.ndim == 3:
        info["channels_identical"] = bool(
            all(np.array_equal(a[..., 0], a[..., c]) for c in range(1, a.shape[2])))
    return info


def main():
    if not os.path.isdir(ROOT):
        print(f"NOT FOUND: {ROOT}\nRun scripts/download_s2looking.py first.")
        return 1

    tree(ROOT)

    print("\n=== directories that directly contain images ===")
    leaves = leaf_dirs_with_images(ROOT)
    for rel, files in sorted(leaves.items()):
        print(f"  {rel:40s} {len(files):6d} files   first={files[0]}  last={files[-1]}")

    # Group leaves by their parent, so <split>/<kind>/ structures surface
    # without us having named either the split or the kind in advance.
    groups = defaultdict(dict)
    for rel, files in leaves.items():
        parent, kind = os.path.split(rel)
        groups[parent][kind] = files

    print("\n=== grouped by parent directory ===")
    for parent, kinds in sorted(groups.items()):
        counts = {k: len(v) for k, v in sorted(kinds.items())}
        print(f"  {parent or '.'}: {counts}")
        names = {k: set(v) for k, v in kinds.items()}
        if len(names) > 1:
            common = set.intersection(*names.values())
            allnames = set.union(*names.values())
            print(f"     filenames identical across sub-dirs: {common == allnames} "
                  f"(shared {len(common)} of {len(allnames)})")

    print("\n=== raster properties (first file of each leaf dir) ===")
    for rel, files in sorted(leaves.items()):
        p = os.path.join(ROOT, rel, files[0])
        try:
            print(f"  {rel:40s} {probe(p)}")
        except Exception as e:
            print(f"  {rel:40s} ERROR {type(e).__name__}: {e}")

    print("\n=== value distribution over a sample of label-like rasters ===")
    for rel, files in sorted(leaves.items()):
        if "label" not in rel.lower():
            continue
        counter = Counter()
        for f in files[:25]:
            a = np.array(Image.open(os.path.join(ROOT, rel, f)))
            if a.ndim == 3:
                a = a[..., 0]
            for v, c in zip(*np.unique(a, return_counts=True)):
                counter[int(v)] += int(c)
        total = sum(counter.values()) or 1
        print(f"  {rel}  (first {min(25, len(files))} files)")
        for v, c in sorted(counter.items()):
            print(f"      value {v:3d}: {c:12,d}  ({100 * c / total:7.4f}%)")

    print("\n=== non-image files (splits, readme, licence?) ===")
    for dirpath, _d, filenames in os.walk(ROOT):
        for f in sorted(filenames):
            if not f.lower().endswith(IMAGE_EXT) and not f.startswith("."):
                full = os.path.join(dirpath, f)
                print(f"  {os.path.relpath(full, ROOT)}  ({os.path.getsize(full):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
