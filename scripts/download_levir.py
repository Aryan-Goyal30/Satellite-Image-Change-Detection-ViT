"""Download the official LEVIR-CD train/val/test archives and extract them.

Source: https://huggingface.co/datasets/satellite-image-deep-learning/LEVIR-CD
This mirror preserves the OFFICIAL pair-level split (445 / 64 / 128 pairs),
which is what makes our numbers comparable to published results.

Usage:  python scripts/download_levir.py
Resumable: re-running skips files that are already complete.
"""
import os, sys, zipfile, urllib.request, hashlib

BASE = "https://huggingface.co/datasets/satellite-image-deep-learning/LEVIR-CD/resolve/main/"
FILES = {"train.zip": 1721956862, "val.zip": 246152048, "test.zip": 496305323}
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "levir_cd")
RAW = os.path.join(ROOT, "raw")


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def download(name, expected):
    dest = os.path.join(RAW, name)
    if os.path.exists(dest) and os.path.getsize(dest) == expected:
        print(f"[skip] {name} already complete ({human(expected)})")
        return dest
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    req = urllib.request.Request(BASE + name, headers={"User-Agent": "earth-guardian/0.1"})
    if have:
        req.add_header("Range", f"bytes={have}-")
        print(f"[resume] {name} from {human(have)}")
    mode = "ab" if have else "wb"
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, mode) as f:
        done = have
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            pct = 100 * done / expected
            print(f"\r  {name}: {human(done)}/{human(expected)} ({pct:.1f}%)", end="", flush=True)
    print()
    got = os.path.getsize(dest)
    if got != expected:
        raise RuntimeError(f"{name}: expected {expected} bytes, got {got}")
    return dest


def extract(path):
    marker = os.path.join(ROOT, os.path.basename(path).replace(".zip", ""), ".extracted")
    if os.path.exists(marker):
        print(f"[skip] {os.path.basename(path)} already extracted")
        return
    print(f"[extract] {os.path.basename(path)} -> {ROOT}")
    with zipfile.ZipFile(path) as z:
        z.extractall(ROOT)
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    open(marker, "w").write("ok")


if __name__ == "__main__":
    os.makedirs(RAW, exist_ok=True)
    for name, size in FILES.items():
        p = download(name, size)
        extract(p)
    print("\nDone. Contents of", ROOT)
    for d in sorted(os.listdir(ROOT)):
        full = os.path.join(ROOT, d)
        if os.path.isdir(full):
            print(" ", d, "->", sorted(os.listdir(full))[:5])
