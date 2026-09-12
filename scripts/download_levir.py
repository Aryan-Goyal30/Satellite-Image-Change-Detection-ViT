"""Download the official LEVIR-CD train/val/test archives and extract them.

Source: https://huggingface.co/datasets/satellite-image-deep-learning/LEVIR-CD
This mirror preserves the OFFICIAL pair-level split (445 / 64 / 128 pairs),
which is what makes our numbers comparable to published results.

Usage:  python scripts/download_levir.py
Resumable: re-running skips files that are already complete.
"""
import os, sys, time, zipfile, urllib.request, urllib.error

BASE = "https://huggingface.co/datasets/satellite-image-deep-learning/LEVIR-CD/resolve/main/"
FILES = {"train.zip": 1721956862, "val.zip": 246152048, "test.zip": 496305323}
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config  # noqa: E402

ROOT = config.LEVIR_RAW
RAW = os.path.join(ROOT, "raw")


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def _attempt(name, expected, dest):
    """One download attempt, resuming from whatever is already on disk."""
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    if have >= expected:
        return True
    req = urllib.request.Request(BASE + name, headers={"User-Agent": "earth-guardian/0.1"})
    if have:
        req.add_header("Range", f"bytes={have}-")
        print(f"  [resume] from {human(have)} ({100*have/expected:.1f}%)", flush=True)
    mode = "ab" if have else "wb"
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, mode) as f:
        done = have
        last = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if done - last > (32 << 20):          # log every 32 MB
                last = done
                print(f"  {name}: {human(done)}/{human(expected)} "
                      f"({100*done/expected:.1f}%)", flush=True)
    return os.path.getsize(dest) >= expected


def download(name, expected, max_attempts=40):
    """Download with resume + retry.

    Large files over a slow link get reset by the CDN fairly often; a reset must
    not lose the bytes already on disk, so every attempt resumes via Range.
    """
    dest = os.path.join(RAW, name)
    if os.path.exists(dest) and os.path.getsize(dest) == expected:
        print(f"[skip] {name} already complete ({human(expected)})")
        return dest

    for attempt in range(1, max_attempts + 1):
        try:
            if _attempt(name, expected, dest):
                break
            print(f"  [retry {attempt}] short read, resuming...", flush=True)
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            got = os.path.getsize(dest) if os.path.exists(dest) else 0
            print(f"  [retry {attempt}/{max_attempts}] {type(e).__name__}: {e} "
                  f"- have {human(got)}, resuming in 5s", flush=True)
            time.sleep(5)
    else:
        raise RuntimeError(f"{name}: gave up after {max_attempts} attempts")

    got = os.path.getsize(dest)
    if got != expected:
        raise RuntimeError(f"{name}: expected {expected} bytes, got {got}")
    print(f"[done] {name} ({human(expected)})", flush=True)
    return dest


def extract(path):
    # Marker lives in raw/ next to the archive. It must NOT be placed at
    # <root>/<split>/.extracted: this mirror extracts FLAT (A/B/label shared by
    # all splits, with the split in the filename prefix), so creating a
    # <root>/train/ directory would fabricate a layout that does not exist and
    # confuse the tiler's layout detection.
    base = os.path.basename(path).replace(".zip", "")
    marker = os.path.join(RAW, f".{base}.extracted")
    if os.path.exists(marker):
        print(f"[skip] {os.path.basename(path)} already extracted")
        return
    print(f"[extract] {os.path.basename(path)} -> {ROOT}", flush=True)
    with zipfile.ZipFile(path) as z:
        z.extractall(ROOT)
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
