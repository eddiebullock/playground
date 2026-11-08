#!/usr/bin/env python3
"""Helpers for research/scripts/phase2_sync_check.sh (ROADMAP step 2.1).

Stdlib only. Nothing here touches the network; the shell script does the SSH.

Subcommands
  pins [--json OUT]                 parse the freeze sections, print a summary
  local-check [--strict-a6]         check every pinned file against the working tree
  remote-paths                      print the unique pinned paths, one per line
  rsync-excludes                    print the rsync exclude patterns, one per line
  jobs-gate [--other-jobs-ok R]     read `squeue -h -o '%i|%j|%T'` on stdin, decide
  annotate                          read rsync --itemize-changes on stdin, flag pins
  compare REMOTE_OUT --out JSON     compare cluster hashes with the pins, write JSON

What is parsed, per amendment (A2, A4, A5, A7 frozen; A6 prepared, not frozen):
  - the preamble before the first "## " heading (A2 and A4 state their freeze there),
  - any "## Freeze ..." section, any "## Code hashes" section, and "## Deviations"
    with its "###" subsections,
  - the JSON fence inside "## Design block" (path-keyed hashes only).
Pairs are recognised in three written forms:
  `path` `hash`   or   `path`: `hash`    (prose and bullet lists)
  path hash                              (lines inside a ```text fence)
  "path": "hash"                         (the JSON design block)
Any other 64-hex value in those sections, and any PENDING field, is reported as
not parsed as a file pin, so the coordinator sees exactly what was skipped.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import fnmatch
import hashlib
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

AMENDMENTS = [
    ("A2", "research/PREREG_A2_DRAFT.md", True),
    ("A4", "research/PREREG_A4_DRAFT.md", True),
    ("A5", "research/PREREG_A5_DRAFT.md", True),
    ("A6", "research/PREREG_A6_DRAFT.md", False),  # freeze prepared, not tagged
    ("A7", "research/PREREG_A7_DRAFT.md", True),
]

ROTATION_JOB = "s3_rotation"
OTHER_PROJECT_PREFIX = "cas_"  # CASCADE, a separate project directory
ACTIVE_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED",
                 "REQUEUED", "REQUEUE_HOLD", "REQUEUE_FED", "RESIZING", "SIGNALING",
                 "STAGE_OUT", "STOPPED"}

# Pins whose hash is written in a freeze section but whose path is not. Each is checked
# on the cluster only (the file is not committed). The hash must appear verbatim in the
# named amendment's text, or collect() reports an error.
CLUSTER_ONLY_PINS = [
    {"amendment": "A4", "path": "results/a4/manifest.csv",
     "sha256": "2ef1161e895224d499b360144690a07c5440bb018105e9c2ffc762f68badd2d1",
     "why": "A4 preamble 'Pinned manifest sha256' names no path; path from "
            "scripts/reproduce.py step a4_manifest, value equals a4_design.MANIFEST_SHA256"},
]

# rsync excludes (rsync filter syntax). A leading "/" anchors at the repo root; a
# trailing "/" matches directories only. Data, results, stimuli, media, caches and .git
# never go over. Never used with --delete.
RSYNC_EXCLUDES = [
    "/.git/", "/data/", "/results/", "/logs/", "/research/logs/", "/study4_rmet/",
    "/.tmp_rmet/", "stimuli/", "media/", "renders/", "frames/", "checkpoints/", "models/",
    "hf_cache/", "conda_envs/", ".venv/", "venv/", "venvs/", "__pycache__/",
    ".pytest_cache/", ".mypy_cache/", ".ipynb_checkpoints/", ".DS_Store", "*.pyc",
    "*.log", ".env", ".env.*", "*.npy", "*.npz", "*.pt", "*.bin", "*.safetensors",
    "*.mp4", "*.flv", "*.avi", "*.mov", "*.mkv", "*.webm", "*.wav", "*.mp3", "*.png",
    "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.tif", "*.tiff", "*.webp", "*.tar", "*.zip",
    "*.gz",
]


def excluded(path: str, patterns=RSYNC_EXCLUDES) -> bool:
    """Approximate rsync exclude matching for a file path relative to the repo root."""
    parts = path.split("/")
    dirs, name = parts[:-1], parts[-1]
    for pat in patterns:
        if pat.startswith("/") and pat.endswith("/"):
            if path.startswith(pat[1:]):
                return True
        elif pat.endswith("/"):
            if pat[:-1] in dirs:
                return True
        elif fnmatch.fnmatchcase(name, pat):
            return True
    return False

HEX64 = r"[0-9a-f]{64}"
PATH_CHARS = r"[A-Za-z0-9_./-]+"
RE_TICK_PAIR = re.compile(r"`(?P<path>" + PATH_CHARS + r")`\s*:?\s*`(?P<h>" + HEX64 + r")`")
RE_TEXT_LINE = re.compile(r"^\s*(?P<path>" + PATH_CHARS + r")\s+(?P<h>" + HEX64 + r")\s*$")
RE_JSON_PAIR = re.compile(r'"(?P<path>' + PATH_CHARS + r')"\s*:\s*"(?P<h>' + HEX64 + r')"')
RE_HEX = re.compile(r"(?<![0-9a-f])" + HEX64 + r"(?![0-9a-f])")
RE_PENDING = re.compile(r"PENDING")

SECTION_HEAD = re.compile(r"^(Freeze\b|Code hashes\b|Deviations\b)")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def looks_like_file(path: str) -> bool:
    if path.startswith("/") or ".." in path.split("/"):
        return False
    name = path.rsplit("/", 1)[-1]
    return "." in name.strip(".") and not re.fullmatch(r"[0-9a-f.]+", name)


def split_sections(text: str):
    """Yield (label, start_line, lines) for the sections parsed for pins.

    Lines are 1-based. The preamble is everything before the first '## ' heading."""
    lines = text.splitlines()
    heads = [i for i, ln in enumerate(lines) if ln.startswith("## ")]
    first = heads[0] if heads else len(lines)
    out = [("preamble", 1, lines[:first])]
    for k, i in enumerate(heads):
        end = heads[k + 1] if k + 1 < len(heads) else len(lines)
        title = lines[i][3:].strip()
        body = lines[i:end]
        if SECTION_HEAD.match(title):
            out.append((title, i + 1, body))
        elif title.startswith("Design block"):
            # only the JSON fence
            in_fence, fence = False, []
            for j, ln in enumerate(body):
                if ln.startswith("```json"):
                    in_fence, start = True, i + j + 2
                    continue
                if in_fence and ln.startswith("```"):
                    break
                if in_fence:
                    fence.append(ln)
            if fence:
                out.append(("Design block (JSON)", start, fence))
    return out


def parse_amendment(text: str, amendment: str, frozen: bool):
    """Return (pins, unparsed). pins: list of dicts; unparsed: list of dicts."""
    pins, unparsed = [], []
    for label, start, lines in split_sections(text):
        in_fence, in_text_fence = False, False
        for off, ln in enumerate(lines):
            lineno = start + off
            if ln.startswith("```"):
                if in_fence:
                    in_fence, in_text_fence = False, False
                else:
                    in_fence = True
                    in_text_fence = ln.strip() in ("```text", "```")
                continue
            consumed = set()
            matches = list(RE_TICK_PAIR.finditer(ln)) + list(RE_JSON_PAIR.finditer(ln))
            if in_text_fence:
                m = RE_TEXT_LINE.match(ln)
                if m:
                    matches.append(m)
            for m in matches:
                path, h = m.group("path"), m.group("h")
                consumed.add(h)
                if not looks_like_file(path):
                    unparsed.append({"amendment": amendment, "line": lineno, "section": label,
                                     "value": h, "context": f"key '{path}' is not a file path"})
                    continue
                pins.append({"amendment": amendment, "path": path, "sha256": h,
                             "section": label, "line": lineno, "frozen": frozen})
            for m in RE_HEX.finditer(ln):
                if m.group(0) not in consumed:
                    ctx = ln[max(0, m.start() - 70):m.start()].strip()
                    unparsed.append({"amendment": amendment, "line": lineno, "section": label,
                                     "value": m.group(0), "context": ctx[-70:]})
            if RE_PENDING.search(ln) and label != "preamble":
                unparsed.append({"amendment": amendment, "line": lineno, "section": label,
                                 "value": "PENDING", "context": ln.strip()[:160]})
    return pins, unparsed


def collect(repo: Path = REPO, amendments=AMENDMENTS, cluster_only_pins=CLUSTER_ONLY_PINS):
    """Parse every amendment. Returns a dict with per-amendment pins, unique files,
    conflicts and unparsed items."""
    per, unparsed, errors, texts = {}, [], [], {}
    for name, rel, frozen in amendments:
        p = repo / rel
        if not p.is_file():
            errors.append(f"{rel} not found")
            continue
        texts[name] = p.read_text(encoding="utf-8")
        pins, unp = parse_amendment(texts[name], name, frozen)
        per[name] = {"file": rel, "frozen": frozen, "pins": pins}
        unparsed.extend(unp)
        if not pins:
            errors.append(f"{name}: no file pins parsed from {rel}")

    files, conflicts = {}, []
    for name, info in per.items():
        for pin in info["pins"]:
            f = files.setdefault(pin["path"], {"path": pin["path"], "sha256": pin["sha256"],
                                               "amendments": [], "frozen": False,
                                               "sources": []})
            if pin["sha256"] != f["sha256"]:
                conflicts.append({"path": pin["path"], "a": f["sha256"], "b": pin["sha256"],
                                  "where": f"{name} line {pin['line']}",
                                  "frozen": pin["frozen"] or f["frozen"]})
                continue
            if name not in f["amendments"]:
                f["amendments"].append(name)
            f["frozen"] = f["frozen"] or pin["frozen"]
            f["sources"].append(f"{name}:{pin['line']}")
    cluster_only = []
    for c in cluster_only_pins:
        if c["amendment"] not in texts:
            continue
        if c["sha256"] not in texts[c["amendment"]]:
            errors.append(f"cluster-only pin {c['path']}: hash not found in {c['amendment']}")
            continue
        cluster_only.append({**c, "frozen": True})
    return {"amendments": per, "files": [files[k] for k in sorted(files)],
            "cluster_only": cluster_only,
            "conflicts": conflicts, "unparsed": unparsed, "errors": errors}


def local_check(info, repo: Path = REPO):
    rows = []
    for f in info["files"]:
        p = repo / f["path"]
        local = sha256_file(p) if p.is_file() else None
        status = "match" if local == f["sha256"] else ("missing" if local is None else "mismatch")
        rows.append({**f, "local_sha256": local, "local_status": status})
    return rows


def jobs_gate(squeue_text: str, other_jobs_ok: str = ""):
    """Decide from `squeue -h -o '%i|%j|%T'` output. Returns (ok, messages)."""
    rotation, other, msgs = [], [], []
    for ln in squeue_text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split("|")
        if len(parts) != 3:
            return False, [f"cannot parse squeue line: {ln!r}"]
        jid, name, state = (x.strip() for x in parts)
        if state.upper() not in ACTIVE_STATES:
            continue
        if name == ROTATION_JOB:
            rotation.append(f"{jid} {name} {state}")
        elif not name.startswith(OTHER_PROJECT_PREFIX):
            other.append(f"{jid} {name} {state}")
    ok = True
    if rotation:
        ok = False
        msgs.append(f"{len(rotation)} {ROTATION_JOB} job(s) pending or running; no file read "
                    "by a running job may change: " + "; ".join(rotation))
    if other:
        if other_jobs_ok:
            msgs.append("other active jobs accepted by --other-jobs-ok "
                        f"({other_jobs_ok}): " + "; ".join(other))
        else:
            ok = False
            msgs.append("other active non-cas jobs may read study3 files; pass "
                        "--other-jobs-ok REASON after checking them: " + "; ".join(other))
    return ok, msgs


def annotate(itemized: str, info):
    pinned = {f["path"]: f for f in info["files"]}
    out, touched = [], []
    for ln in itemized.splitlines():
        parts = ln.split(" ", 1)
        if len(parts) == 2 and parts[1].strip() in pinned:
            path = parts[1].strip()
            out.append(f"{ln}    <-- PINNED ({','.join(pinned[path]['amendments'])})")
            touched.append(path)
        else:
            out.append(ln)
    return out, touched


def parse_remote(text: str):
    """Lines '<hash>  <path>' from sha256sum, or 'MISSING  <path>'."""
    res = {}
    for ln in text.splitlines():
        ln = ln.rstrip("\n")
        if not ln.strip():
            continue
        m = re.match(r"^(" + HEX64 + r"|MISSING)\s+\*?(.+)$", ln)
        if not m:
            raise ValueError(f"cannot parse remote hash line: {ln!r}")
        res[m.group(2).strip()] = None if m.group(1) == "MISSING" else m.group(1)
    return res


def compare(info, remote: dict, mode: str, remote_root: str, rsync_applied: bool,
            extra: dict | None = None, repo: Path = REPO):
    rows = []
    for r in local_check(info, repo):
        cl = remote.get(r["path"])
        rows.append({"path": r["path"], "amendments": r["amendments"],
                     "frozen": r["frozen"], "pinned_sha256": r["sha256"],
                     "local_sha256": r["local_sha256"], "cluster_sha256": cl,
                     "match": cl == r["sha256"],
                     "synced_by_rsync": not excluded(r["path"])})
    for c in info.get("cluster_only", []):
        cl = remote.get(c["path"])
        rows.append({"path": c["path"], "amendments": [c["amendment"]], "frozen": True,
                     "pinned_sha256": c["sha256"], "local_sha256": None,
                     "cluster_sha256": cl, "match": cl == c["sha256"],
                     "synced_by_rsync": False, "cluster_only": True, "why": c["why"]})
    frozen_bad = [r["path"] for r in rows if r["frozen"] and not r["match"]]
    a6_bad = [r["path"] for r in rows if not r["frozen"] and not r["match"]]
    doc = {
        "step": "ROADMAP 2.1 cluster hash check",
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": mode,
        "rsync_applied": rsync_applied,
        "remote_root": remote_root,
        "amendments": {k: {"file": v["file"], "frozen": v["frozen"],
                           "n_pins": len(v["pins"])}
                       for k, v in info["amendments"].items()},
        "note_a6": "A6 freeze is prepared, not tagged; its pins are checked but A6 is not frozen.",
        "cluster_only_pins_skipped": bool(info.get("cluster_only_pins_skipped")),
        "n_files": len(rows),
        "n_mismatch_frozen": len(frozen_bad),
        "n_mismatch_a6_only": len(a6_bad),
        "all_frozen_match": not frozen_bad,
        "files": rows,
    }
    if extra:
        doc.update(extra)
    code = 1 if frozen_bad else (3 if a6_bad else 0)
    return doc, code


def _print_summary(info):
    for name, a in info["amendments"].items():
        tag = "frozen" if a["frozen"] else "PREPARED, NOT FROZEN"
        print(f"{name} ({tag}): {len(a['pins'])} pins, "
              f"{len({p['path'] for p in a['pins']})} unique files")
    print(f"unique pinned files: {len(info['files'])}")
    for c in info.get("cluster_only", []):
        print(f"cluster-only pin ({c['amendment']}): {c['path']} -- {c['why']}")
    for c in info["conflicts"]:
        print(f"CONFLICT: {c['path']} pinned as {c['a']} and {c['b']} ({c['where']})")
    for e in info["errors"]:
        print(f"ERROR: {e}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", default=str(REPO))
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pins")
    p.add_argument("--json")
    p.add_argument("--show-unparsed", action="store_true")
    p = sub.add_parser("local-check")
    p.add_argument("--strict-a6", action="store_true")
    sub.add_parser("remote-paths")
    sub.add_parser("rsync-excludes")
    p = sub.add_parser("jobs-gate")
    p.add_argument("--other-jobs-ok", default="")
    sub.add_parser("annotate")
    p = sub.add_parser("compare")
    p.add_argument("remote_out")
    p.add_argument("--out", required=True)
    p.add_argument("--mode", required=True)
    p.add_argument("--remote-root", required=True)
    p.add_argument("--rsync-applied", action="store_true")
    p.add_argument("--extra-json", default="")
    a = ap.parse_args(argv)
    repo = Path(a.repo)
    skip_co = os.environ.get("PHASE2_SKIP_CLUSTER_ONLY_PINS") == "1"  # tests only
    info = collect(repo, cluster_only_pins=[] if skip_co else CLUSTER_ONLY_PINS)
    info["cluster_only_pins_skipped"] = skip_co

    if a.cmd == "pins":
        _print_summary(info)
        if a.show_unparsed:
            for u in info["unparsed"]:
                print(f"UNPARSED {u['amendment']} line {u['line']} [{u['section']}]: "
                      f"{u['value'][:16]} ... {u['context']}")
        if a.json:
            Path(a.json).write_text(json.dumps(info, indent=2) + "\n")
        return 1 if (info["errors"] or info["conflicts"]) else 0

    if info["errors"] or any(c["frozen"] for c in info["conflicts"]):
        _print_summary(info)
        print("STOP: freeze sections did not parse cleanly", file=sys.stderr)
        return 2

    if a.cmd == "local-check":
        rows = local_check(info, repo)
        bad_frozen = [r for r in rows if r["frozen"] and r["local_status"] != "match"]
        bad_a6 = [r for r in rows if not r["frozen"] and r["local_status"] != "match"]
        for r in rows:
            flag = "ok " if r["local_status"] == "match" else r["local_status"].upper()
            print(f"{flag:8s} {','.join(r['amendments']):12s} {r['path']}")
        for r in bad_a6:
            print(f"WARNING: A6-only pin (not frozen) differs locally: {r['path']}",
                  file=sys.stderr)
        under_excl = [r["path"] for r in rows if excluded(r["path"])]
        if under_excl:
            print("NOTE: pinned but excluded from rsync (hash-checked only): "
                  + ", ".join(under_excl), file=sys.stderr)
        if bad_frozen:
            print(f"STOP: {len(bad_frozen)} frozen file(s) differ from their pins in the "
                  "working tree", file=sys.stderr)
            return 1
        if bad_a6 and a.strict_a6:
            print("STOP: --strict-a6 and an A6 pin differs", file=sys.stderr)
            return 1
        print(f"local check: {len(rows)} pinned files, frozen all match"
              + (f", {len(bad_a6)} A6-only differ" if bad_a6 else ""))
        return 0

    if a.cmd == "remote-paths":
        for f in info["files"] + info.get("cluster_only", []):
            print(f["path"])
        return 0

    if a.cmd == "rsync-excludes":
        for pat in RSYNC_EXCLUDES:
            print(pat)
        return 0

    if a.cmd == "jobs-gate":
        ok, msgs = jobs_gate(sys.stdin.read(), a.other_jobs_ok)
        for m in msgs:
            print(m, file=sys.stderr)
        return 0 if ok else 1

    if a.cmd == "annotate":
        out, touched = annotate(sys.stdin.read(), info)
        print("\n".join(out))
        print(f"rsync would update {len(touched)} pinned file(s)"
              + (": " + ", ".join(touched) if touched else ""))
        return 0

    if a.cmd == "compare":
        remote = parse_remote(Path(a.remote_out).read_text())
        extra = json.loads(a.extra_json) if a.extra_json else None
        doc, code = compare(info, remote, a.mode, a.remote_root, a.rsync_applied, extra, repo)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(doc, indent=2) + "\n")
        for r in doc["files"]:
            if not r["match"]:
                tag = "FROZEN" if r["frozen"] else "A6-only"
                print(f"MISMATCH [{tag}] {r['path']}: pinned {r['pinned_sha256'][:12]} "
                      f"cluster {(r['cluster_sha256'] or 'MISSING')[:12]}", file=sys.stderr)
        print(f"cluster check: {doc['n_files']} files, {doc['n_mismatch_frozen']} frozen "
              f"mismatches, {doc['n_mismatch_a6_only']} A6-only mismatches; wrote {a.out}")
        return code
    return 2


if __name__ == "__main__":
    sys.exit(main())
