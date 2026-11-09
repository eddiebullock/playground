#!/usr/bin/env python
"""A8 preflight: the right weights, safe output paths, room on RDS. Run first in every A8 job.

Outputs of the frozen scripts carry the small sibling's key (`qwen3vl`, `gemma4`), so a job
that loaded the wrong weights, or wrote to a default path, would mix models silently. This
check stops that before any forward.

    python research/scripts/a8_preflight.py --a8_id qwen3vl_32b --models_dir DIR \
        --output PATH [--output PATH ...] [--full_hash] [--quota_file FILE] \
        [--provenance PATH]

Checks, each a hard stop:
- the key resolves under --models_dir exactly as scripts/evaluate.resolve_model_path does
  (models_dir / key), and its real path is the design's weights directory;
- config.json matches the design's expected architecture, layer count and hidden size;
- every weight file in the design is present at its pinned size, and no other
  .safetensors file is present; with --full_hash, every sha256 equals the pin;
- every output path sits under an A8 output root and does not exist yet;
- RDS usage, read from `quota` output, is under the storage stop rule.

Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a8_design  # noqa: E402

ROOT = a8_design.ROOT


def resolve_key_dir(models_dir: Path, key: str) -> Path:
    """What resolve_model_path sees with MR_EU_HPC_MODELS_DIR=models_dir (config.MODELS hpc_path)."""
    return Path(os.path.expanduser(str(models_dir))) / key


def check_weights_dir(models_dir: Path, entry: Mapping[str, Any]) -> Path:
    path = resolve_key_dir(models_dir, entry["code_key"])
    if not path.exists():
        raise SystemExit(f"{path} does not exist; the frozen loader would fall back to another path.")
    real, expected = path.resolve(), Path(entry["weights_dir"]).resolve()
    if real != expected:
        raise SystemExit(f"{path} resolves to {real}, not the design's {expected}.")
    return real


def check_config(weights: Path, expected: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = json.loads((weights / "config.json").read_text(encoding="utf-8"))
    text = cfg.get("text_config") or {}
    seen = {
        "architectures": cfg.get("architectures"),
        "model_type": cfg.get("model_type"),
        "text_num_hidden_layers": text.get("num_hidden_layers"),
        "text_hidden_size": text.get("hidden_size"),
    }
    wrong = {k: v for k, v in seen.items() if v != expected[k]}
    if wrong:
        raise SystemExit(f"config.json does not match the A8 model: {wrong}")
    return seen


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def check_files(weights: Path, files: Mapping[str, Mapping[str, Any]], full_hash: bool = False) -> Dict[str, Any]:
    present = {p.name for p in weights.glob("*.safetensors")}
    extra = sorted(present - set(files))
    missing = sorted(set(files) - present)
    if missing or extra:
        raise SystemExit(f"Weight files differ from the design: missing {missing}, extra {extra}")
    out = {}
    for name, pin in files.items():
        path = weights / name
        stat = path.stat()
        if stat.st_size != pin["size"]:
            raise SystemExit(f"{name} is {stat.st_size} bytes, not {pin['size']}.")
        row: Dict[str, Any] = {"size": stat.st_size, "mtime": stat.st_mtime}
        if full_hash:
            row["sha256"] = sha256_file(path)
            if row["sha256"] != pin["sha256"]:
                raise SystemExit(f"{name} sha256 {row['sha256']} is not the pinned {pin['sha256']}.")
        out[name] = row
    return out


def check_outputs(outputs: Sequence[Path], roots: Sequence[str], root: Path = ROOT, allow_existing: bool = False) -> List[str]:
    """Every output under an A8 root, given relative to the project root, and not already there."""
    bases = [(root / r).resolve() for r in roots]
    checked = []
    for out in outputs:
        full = (root / out).resolve() if not Path(out).is_absolute() else Path(out).resolve()
        if not any(full == b or b in full.parents for b in bases):
            raise SystemExit(f"{out} is outside the A8 output roots {list(roots)}.")
        if full.exists() and not allow_existing:
            raise SystemExit(f"{out} exists; A8 never overwrites an output.")
        checked.append(str(full))
    return checked


_QUOTA_ROW = re.compile(r"^\s*/rds-d\d+/user/\S+\s+([\d.]+)\s+[\d.]+\s+[\d.]+\s+\S+\s+(\d+)", re.M)


def parse_quota(text: str) -> Dict[str, float]:
    """GB and files of the RDS user row in CSD3 `quota` output."""
    found = _QUOTA_ROW.search(text)
    if found is None:
        raise SystemExit("No RDS user row in the quota output; storage cannot be checked.")
    return {"gb": float(found.group(1)), "files": int(found.group(2))}


def check_storage(usage: Mapping[str, float], stop: Mapping[str, float]) -> None:
    if usage["gb"] > stop["rds_gb"] or usage["files"] > stop["rds_files"]:
        raise SystemExit(f"RDS at {usage['gb']} GB and {usage['files']} files is over the stop rule {dict(stop)}.")


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a8_id", required=True)
    ap.add_argument("--models_dir", type=Path, help="Defaults to the design's models_dir.")
    ap.add_argument("--output", type=Path, action="append", default=[])
    ap.add_argument("--allow_existing", action="store_true", help="Merge and analysis steps that read earlier A8 files.")
    ap.add_argument("--full_hash", action="store_true")
    ap.add_argument("--quota_file", type=Path, help="Saved `quota` output; default runs `quota`.")
    ap.add_argument("--skip_storage", action="store_true", help="Local tests only.")
    ap.add_argument("--provenance", type=Path)
    args = ap.parse_args(argv)

    design = a8_design.load_design()
    a8_design.check_frozen_code(design)
    entry = a8_design.model(design, args.a8_id)
    weights = check_weights_dir(args.models_dir or Path(entry["models_dir"]), entry)
    config = check_config(weights, entry["expected_config"])
    files = check_files(weights, entry["weight_files"], full_hash=args.full_hash)
    outputs = check_outputs(args.output, design["output_roots"], allow_existing=args.allow_existing)
    usage = None
    if not args.skip_storage:
        text = args.quota_file.read_text() if args.quota_file else subprocess.run(
            ["quota"], capture_output=True, text=True, check=False).stdout
        usage = parse_quota(text)
        check_storage(usage, design["storage_stop"])
    report = {
        "a8_id": args.a8_id,
        "code_key": entry["code_key"],
        "repo": entry["repo"],
        "revision": entry["revision"],
        "weights_dir": str(weights),
        "config": config,
        "weight_files": files,
        "full_hash": bool(args.full_hash),
        "outputs": outputs,
        "rds_usage": usage,
        "design_sha256": a8_design.block_hash(),
        "preflight_sha256": a8_design.file_sha256(Path(__file__).resolve()),
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if args.provenance:
        check_outputs([args.provenance], design["output_roots"], allow_existing=False)
        args.provenance.parent.mkdir(parents=True, exist_ok=True)
        args.provenance.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("a8_id", "weights_dir", "full_hash", "design_sha256")}))
    return report


if __name__ == "__main__":
    main()
