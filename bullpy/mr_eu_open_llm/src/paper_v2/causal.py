"""Phase 3 ablation summaries. Runs on the trials CSV; does not call the model."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[2]


def _bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"cannot parse bool {v!r}")


def load_trials(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarise_trials(rows: Iterable[Dict[str, str]]) -> Dict[str, Any]:
    """Compare each ablation axis to the baseline row of the same trial.

    A flip is a prediction change vs baseline. Δacc is mean(correct_ablate −
    correct_base). The n=36 pilot is 5 flips / 180 axis-trials.
    """
    by_trial: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_trial[row["trial_id"]][row["axis"]] = row

    axes_seen = sorted({ax for blocks in by_trial.values() for ax in blocks if ax != "baseline"})
    per_axis: Dict[str, Dict[str, Any]] = {}
    flip_labels: Counter[str] = Counter()
    n_axis_trials = 0
    n_flips = 0

    for axis in axes_seen:
        n = 0
        flips = 0
        d_acc = 0.0
        d_conf = 0.0
        flip_examples: List[Dict[str, str]] = []
        for tid, blocks in by_trial.items():
            base = blocks.get("baseline")
            abl = blocks.get(axis)
            if base is None or abl is None:
                continue
            n += 1
            n_axis_trials += 1
            d_acc += float(_bool(abl["correct"])) - float(_bool(base["correct"]))
            d_conf += float(_bool(abl["confused_with_top_foil"])) - float(
                _bool(base["confused_with_top_foil"])
            )
            if abl["prediction"] != base["prediction"]:
                flips += 1
                n_flips += 1
                flip_labels[axis] += 1
                if len(flip_examples) < 8:
                    flip_examples.append(
                        {
                            "trial_id": tid,
                            "from": base["prediction"],
                            "to": abl["prediction"],
                            "label": base.get("correct_label", ""),
                        }
                    )
        per_axis[axis] = {
            "n_trials": n,
            "n_flips": flips,
            "mean_delta_accuracy": d_acc / n if n else None,
            "mean_delta_confusion_rate": d_conf / n if n else None,
            "flip_examples": flip_examples,
        }
        flip_labels.setdefault(axis, 0)

    n_trials = len(by_trial)
    return {
        "n_trials": n_trials,
        "n_axes": len(axes_seen),
        "n_axis_trials": n_axis_trials,
        "n_flips": n_flips,
        "flip_rate": n_flips / n_axis_trials if n_axis_trials else None,
        "flips_by_axis": dict(flip_labels),
        "axes": per_axis,
        "nonselective": all(v["mean_delta_accuracy"] == 0 for v in per_axis.values())
        if per_axis
        else None,
    }


def summarise_path(trials_csv: Path, out: Optional[Path] = None) -> Dict[str, Any]:
    payload = {
        "source": str(trials_csv),
        **summarise_trials(load_trials(trials_csv)),
    }
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    payload = summarise_path(args.trials, args.out)
    print(json.dumps({k: v for k, v in payload.items() if k != "axes"}, indent=2))
    if args.out:
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
