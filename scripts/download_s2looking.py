"""Download the official S2Looking dataset archive and extract it.

Official source (the only one used here):
    https://github.com/S2Looking/Dataset   ->   Google Drive folder
    https://drive.google.com/drive/folders/1zzb6hif2hwWx4z8UIMLpMAInkhbmmrFY
    folder contains exactly one file: S2Looking.zip (file id below)

No unofficial mirror is used. The Google Drive "large file" flow needs a
confirm token, so a plain GET is not enough: we fetch the interstitial form,
resubmit it, and stream the result.

Usage:  python scripts/download_s2looking.py
Resumable: re-running skips or resumes whatever is already on disk.

This script only acquires data. It trains nothing and changes no behaviour.
"""
import hashlib
import json
import os
import re
import sys
import time
import zipfile
from datetime import datetime, timezone

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config  # noqa: E402

FOLDER_URL = "https://drive.google.com/drive/folders/1zzb6hif2hwWx4z8UIMLpMAInkhbmmrFY"
FILE_ID = "1YAUrUp3QtZ6VMM1nSuqOjveIVO9ADOY1"
FILE_NAME = "S2Looking.zip"
EXPECTED_BYTES = 10_959_772_884          # from Content-Range on the official host

ROOT = config.S2LOOKING_RAW
RAW = os.path.join(ROOT, "raw")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def _resolve_download(session):
    """Return (url, params) for the real byte stream behind Drive's confirm page."""
    r = session.get("https://drive.google.com/uc",
                    params={"export": "download", "id": FILE_ID}, timeout=60)
    r.raise_for_status()
    if "text/html" not in r.headers.get("Content-Type", ""):
        return r.url, {}
    form = re.search(r'<form[^>]+action="([^"]+)"[^>]*>(.*?)</form>', r.text, re.S)
    if not form:
        raise RuntimeError("Drive confirm form not found - the sharing link may have changed")
    action = form.group(1).replace("&amp;", "&")
    fields = dict(re.findall(r'name="([^"]+)"\s+value="([^"]*)"', form.group(2)))
    return action, fields


def _attempt(session, dest):
    """One streaming attempt, resuming from whatever is already on disk."""
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    if have >= EXPECTED_BYTES:
        return True
    url, params = _resolve_download(session)
    headers = {"Range": f"bytes={have}-"} if have else {}
    if have:
        print(f"  [resume] from {human(have)} ({100 * have / EXPECTED_BYTES:.1f}%)", flush=True)
    with session.get(url, params=params, headers=headers, stream=True, timeout=120) as r:
        if have and r.status_code != 206:
            raise RuntimeError(f"resume refused (HTTP {r.status_code}); will retry")
        r.raise_for_status()
        with open(dest, "ab" if have else "wb") as f:
            done, last = have, have
            for chunk in r.iter_content(1 << 20):
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if done - last > (128 << 20):          # log every 128 MB
                    last = done
                    print(f"  {FILE_NAME}: {human(done)}/{human(EXPECTED_BYTES)} "
                          f"({100 * done / EXPECTED_BYTES:.1f}%)", flush=True)
    return os.path.getsize(dest) >= EXPECTED_BYTES


def download(max_attempts=40):
    dest = os.path.join(RAW, FILE_NAME)
    if os.path.exists(dest) and os.path.getsize(dest) == EXPECTED_BYTES:
        print(f"[skip] {FILE_NAME} already complete ({human(EXPECTED_BYTES)})")
        return dest

    session = requests.Session()
    session.headers["User-Agent"] = UA
    for attempt in range(1, max_attempts + 1):
        try:
            if _attempt(session, dest):
                break
            print(f"  [retry {attempt}] short read, resuming...", flush=True)
        except Exception as e:                      # network/CDN resets are routine
            got = os.path.getsize(dest) if os.path.exists(dest) else 0
            print(f"  [retry {attempt}/{max_attempts}] {type(e).__name__}: {e} "
                  f"- have {human(got)}, resuming in 5s", flush=True)
            time.sleep(5)
    else:
        raise RuntimeError(f"{FILE_NAME}: gave up after {max_attempts} attempts")

    got = os.path.getsize(dest)
    if got != EXPECTED_BYTES:
        raise RuntimeError(f"{FILE_NAME}: expected {EXPECTED_BYTES} bytes, got {got}")
    print(f"[done] {FILE_NAME} ({human(EXPECTED_BYTES)})", flush=True)
    return dest


def sha256(path, blocksize=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(blocksize), b""):
            h.update(block)
    return h.hexdigest()


def extract(path):
    marker = os.path.join(RAW, ".extracted")
    if os.path.exists(marker):
        print("[skip] archive already extracted")
        return
    print(f"[extract] {os.path.basename(path)} -> {ROOT}", flush=True)
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        print(f"  {len(names)} entries; first: {names[:3]}", flush=True)
        z.extractall(ROOT)
    open(marker, "w").write("ok")


def write_provenance(path, digest):
    os.makedirs(config.S2LOOKING_META, exist_ok=True)
    out = os.path.join(config.S2LOOKING_META, "provenance.json")
    json.dump({
        "dataset": "S2Looking",
        "official_repository": "https://github.com/S2Looking/Dataset",
        "official_folder_url": FOLDER_URL,
        "drive_file_id": FILE_ID,
        "file_name": FILE_NAME,
        "downloaded_bytes": os.path.getsize(path),
        "sha256": digest,
        "access_date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "paper": "Shen et al., S2Looking: A Satellite Side-Looking Dataset for "
                 "Building Change Detection, Remote Sensing 13(24):5094, 2021. "
                 "https://doi.org/10.3390/rs13245094",
        "license_status": "No dataset license is stated by the authors. The GitHub "
                          "repository declares no license and the paper's Data "
                          "Availability Statement only says the data are publicly "
                          "available. Treat as research use; do not redistribute "
                          "imagery or label rasters.",
        "mirror_used": False,
    }, open(out, "w", encoding="utf-8"), indent=2)
    print("[provenance]", out)


if __name__ == "__main__":
    os.makedirs(RAW, exist_ok=True)
    archive = download()
    print("[sha256] hashing archive (a few minutes)...", flush=True)
    digest = sha256(archive)
    print("[sha256]", digest, flush=True)
    write_provenance(archive, digest)
    extract(archive)
    print("\nDone. Contents of", ROOT)
    for d in sorted(os.listdir(ROOT)):
        full = os.path.join(ROOT, d)
        print(" ", d, "->", sorted(os.listdir(full))[:6] if os.path.isdir(full) else "(file)")
