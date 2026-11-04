#!/usr/bin/env python3
"""Fetch only the selected CREMA-D videos, verified against the upstream LFS pointers.

Each clip's sha256 and size come from its pointer on GitHub. The total is checked
against --max-bytes before any video is requested, so an oversized subset stops without
writing anything. Bytes come from the GitLab mirror through the LFS batch API, so no
git-lfs and no checkout are needed.

gitlab.com limits unauthenticated traffic per IP and login nodes share one, so downloads
are sequential, paced, and wait out any Retry-After. The pointer manifest is written
before the first download, so a rerun resumes without asking GitHub again.

Runs on stock Python 3.6, standard library only:

    python3 -I crema_fetch_videos.py --selection crema_d_selection.csv --out-dir DIR
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

POINTER_URL = "https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/master/VideoFlash/{}.flv"
BATCH_URL = "https://gitlab.com/cs-cooper-lab/crema-d-mirror.git/info/lfs/objects/batch"
LFS_JSON = "application/vnd.git-lfs+json"
MANIFEST = "crema_d_lfs_manifest.csv"
USER_AGENT = "crema-subset-fetch/1.0"
CHUNK = 1 << 16
BATCH_SIZE = 100
RETRIES = 4
MAX_RATE_LIMIT_WAITS = 12
DEFAULT_RETRY_AFTER = 60


def with_retries(fn, *args):
    """Back off on transient errors; on 429 wait as long as the server asks."""
    failures = rate_waits = 0
    while True:
        try:
            return fn(*args)
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and rate_waits < MAX_RATE_LIMIT_WAITS:
                rate_waits += 1
                wait = int(exc.headers.get("Retry-After") or DEFAULT_RETRY_AFTER) + 1
                print("429, waiting {} s as asked".format(wait), flush=True)
                time.sleep(wait)
                continue
            failures = RETRIES
            error = exc
        except (urllib.error.URLError, OSError, ValueError) as exc:
            failures += 1
            error = exc
        if failures >= RETRIES:
            raise RuntimeError("{}: {}".format(args[:1], error))
        time.sleep(2 ** failures)


def parse_pointer(text):
    oid = re.search(r"^oid sha256:([0-9a-f]{64})$", text, flags=re.M)
    size = re.search(r"^size (\d+)$", text, flags=re.M)
    if not text.startswith("version https://git-lfs") or not oid or not size:
        raise ValueError("not an LFS pointer: {!r}".format(text[:80]))
    return oid.group(1), int(size.group(1))


def fetch_pointer(clip_id):
    request = urllib.request.Request(POINTER_URL.format(clip_id), headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return (clip_id,) + parse_pointer(response.read(1024).decode("utf-8"))


def batch_downloads(pointers):
    """Ask the LFS server for one download action per object."""
    body = json.dumps(
        {
            "operation": "download",
            "transfers": ["basic"],
            "objects": [{"oid": oid, "size": size} for _, oid, size in pointers],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        BATCH_URL,
        data=body,
        headers={"Accept": LFS_JSON, "Content-Type": LFS_JSON, "User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        objects = json.loads(response.read().decode("utf-8"))["objects"]
    actions = {}
    for obj in objects:
        if "actions" not in obj or "download" not in obj["actions"]:
            raise ValueError("no download action for {}: {}".format(obj["oid"], obj.get("error")))
        actions[obj["oid"]] = obj["actions"]["download"]
    return actions


def verified(path, oid, size):
    if not os.path.isfile(path) or os.path.getsize(path) != size:
        return False
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest() == oid


def fetch_video(clip_id, oid, size, action, out_dir):
    final = os.path.join(out_dir, clip_id + ".flv")
    part = final + ".part"
    headers = dict(action.get("header") or {})
    headers["User-Agent"] = USER_AGENT
    request = urllib.request.Request(action["href"], headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=120) as response, open(part, "wb") as fh:
            for chunk in iter(lambda: response.read(CHUNK), b""):
                fh.write(chunk)
        if not verified(part, oid, size):
            raise ValueError("{}: bytes do not match the LFS pointer".format(clip_id))
        with open(part, "rb") as fh:
            if fh.read(3) != b"FLV":
                raise ValueError("{}: not an FLV file".format(clip_id))
        os.replace(part, final)
    finally:
        if os.path.exists(part):
            os.remove(part)


def load_pointers(clips, manifest, workers):
    if os.path.exists(manifest):
        with open(manifest, newline="", encoding="utf-8") as fh:
            known = {r["clip_id"]: (r["clip_id"], r["sha256"], int(r["size"])) for r in csv.DictReader(fh)}
        if set(known) == set(clips):
            return [known[c] for c in clips]
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(lambda c: with_retries(fetch_pointer, c), clips))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-bytes", type=int, default=1500000000)
    parser.add_argument("--max-files", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=4, help="pointer requests only")
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between video requests")
    parser.add_argument("--plan-only", action="store_true", help="stop after the size check")
    args = parser.parse_args()

    with open(args.selection, newline="", encoding="utf-8") as fh:
        clips = [row["clip_id"] for row in csv.DictReader(fh)]
    manifest = os.path.join(args.out_dir, MANIFEST)
    pointers = load_pointers(clips, manifest, args.workers)
    total = sum(p[2] for p in pointers)
    print("{} clips, {} bytes planned, largest {}".format(len(pointers), total, max(p[2] for p in pointers)))
    if total > args.max_bytes or len(pointers) + 1 > args.max_files:
        print("STOP: over budget, nothing downloaded", file=sys.stderr)
        return 2
    if args.plan_only:
        return 0

    os.makedirs(args.out_dir, exist_ok=True)
    with open(manifest, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["clip_id", "sha256", "size"])
        writer.writerows(pointers)

    todo = [p for p in pointers if not verified(os.path.join(args.out_dir, p[0] + ".flv"), p[1], p[2])]
    print("{} already verified, {} to fetch".format(len(pointers) - len(todo), len(todo)), flush=True)
    done = 0
    for start in range(0, len(todo), BATCH_SIZE):
        chunk = todo[start : start + BATCH_SIZE]
        actions = with_retries(batch_downloads, chunk)
        for clip_id, oid, size in chunk:
            with_retries(fetch_video, clip_id, oid, size, actions[oid], args.out_dir)
            done += 1
            time.sleep(args.delay)
        print("fetched {}/{}".format(done, len(todo)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
