#!/usr/bin/env python
"""A8, dated deviation (DECISIONS 2026-11-17, row 102): the check of the device-aware gate hooks on the frozen 8B.

Three gate files of Qwen3-VL-8B, the frozen sibling, from the frozen gate command at bfloat16: the frozen
script twice on one GPU each (frozen_a, frozen_b), and the same command through
research/scripts/a8_gates_device.py on two GPUs (patched). The rule, set before any of the three exists, is
that of A8's a1_gpu_rule, leaf for leaf:

1. Each file is a complete run: not a smoke, with a verdict.
2. The three files have the same leaves: the same keys, and lists of the same lengths.
3. Apart from the timings (leaves named `seconds` or `model_load_seconds`), the two frozen files are equal,
   so the frozen script is shown to repeat itself on one GPU.
4. Apart from the timings, the patched file equals them.

The check passes only if all four hold. Timings are listed in the report and never compared. This file reads
no A8 output.

    python research/scripts/a8_gates_device_check.py --frozen A.json B.json --patched P.json --report R.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple

TIMINGS = ("seconds", "model_load_seconds")


def leaves(obj: Any, path: str = "") -> Iterator[Tuple[str, Any]]:
    """Every leaf of a JSON value as (path, value); an empty dict or list is a leaf."""
    if isinstance(obj, dict) and obj:
        for key in sorted(obj):
            yield from leaves(obj[key], f"{path}/{key}")
    elif isinstance(obj, list) and obj:
        for i, item in enumerate(obj):
            yield from leaves(item, f"{path}[{i}]")
    else:
        yield path, obj


def is_timing(path: str) -> bool:
    return path.rsplit("/", 1)[-1] in TIMINGS


def same(a: Any, b: Any) -> bool:
    """Equal as JSON text: NaN equals NaN, and 1, 1.0 and true differ."""
    return json.dumps(a) == json.dumps(b)


def compare(frozen_a: Mapping[str, Any], frozen_b: Mapping[str, Any], patched: Mapping[str, Any]) -> Dict[str, Any]:
    files = {"frozen_a": frozen_a, "frozen_b": frozen_b, "patched": patched}
    flat = {name: dict(leaves(obj)) for name, obj in files.items()}
    a, b, p = flat["frozen_a"], flat["frozen_b"], flat["patched"]
    shared = set(a) & set(b) & set(p)
    compared = sorted(k for k in shared if not is_timing(k))
    report: Dict[str, Any] = {
        "complete": {name: (obj.get("smoke") is False and "verdict" in obj) for name, obj in files.items()},
        "n_leaves": {name: len(x) for name, x in flat.items()},
        "leaves_not_in_all_three": {name: sorted(set(x) - shared) for name, x in flat.items()},
        "n_compared": len(compared),
        "timings_not_compared": sorted(k for k in shared if is_timing(k)),
        "frozen_runs_differ_at": [k for k in compared if not same(a[k], b[k])],
        "patched_differs_at": [k for k in compared if same(a[k], b[k]) and not same(p[k], a[k])],
        "verdicts": {name: obj.get("verdict") for name, obj in files.items()},
    }
    report["passed"] = bool(
        all(report["complete"].values())
        and not any(report["leaves_not_in_all_three"].values())
        and not report["frozen_runs_differ_at"]
        and not report["patched_differs_at"]
        and compared
    )
    return report


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--frozen", type=Path, nargs=2, required=True)
    ap.add_argument("--patched", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.report.exists():
        raise SystemExit(f"{args.report} exists; the check runs once.")
    load = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))  # noqa: E731
    report = compare(load(args.frozen[0]), load(args.frozen[1]), load(args.patched))
    report["files"] = {"frozen_a": str(args.frozen[0]), "frozen_b": str(args.frozen[1]), "patched": str(args.patched)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("passed", "n_compared", "frozen_runs_differ_at", "patched_differs_at")}, indent=1))
    if not report["passed"]:
        raise SystemExit("The check of the device-aware gate hooks did not pass.")
    return report


if __name__ == "__main__":
    main()
