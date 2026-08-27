#!/usr/bin/env python3
"""Forced-choice response entropy (RQ1.1b), the primary calibration metric.

Distinct from RQ1.1a (scripts/semantic_entropy.py), which is model-internal generative
uncertainty over embedded free text. This module samples the model's N-AFC answer N times
per item at temperature > 0 and takes Shannon entropy of the resulting response
distribution, in the same categorical space as the human forced-choice data, so the two
entropies are directly comparable.

Two roles:

  sample_forced_choice   called from evaluate.py during a run (model already loaded and
                         frames already decoded, so the N draws are cheap relative to a
                         separate pass)
  CLI                    joins per-item model entropy from finished eval JSONs against
                         data/eu_emotions_human_entropy.json and reports the RQ1.1b
                         correlation

Each draw is seeded from sha256(trial_id|seed|draw), matching the foil-generation scheme
in trial_foils.py, so a rerun reproduces the same distribution rather than merely the
same mean.

Usage:

  python -m scripts.forced_choice_entropy \\
    --eval results/baseline/eu_emotions/qwen3vl/eval_v2_*.json \\
    --human data/eu_emotions_human_entropy.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from scripts.human_entropy import shannon_entropy

UNPARSED = "__unparsed__"


def forced_choice_distribution(
    predictions: Sequence[Optional[str]],
    options: Sequence[str],
) -> Dict[str, float]:
    """
    Empirical response distribution over the option set.

    Unparseable draws are kept under UNPARSED rather than discarded: dropping them would
    understate entropy for exactly the ambiguous items where the model equivocates.
    """
    counts: Dict[str, int] = {option: 0 for option in options}
    for prediction in predictions:
        key = prediction if prediction in counts else UNPARSED
        counts[key] = counts.get(key, 0) + 1
    total = sum(counts.values())
    if total == 0:
        return {option: 0.0 for option in options}
    return {label: count / total for label, count in counts.items()}


def entropy_of_distribution(distribution: Dict[str, float], *, log_base: str = "e") -> float:
    return shannon_entropy(list(distribution.values()), log_base=log_base)


def sample_forced_choice(
    generate: Callable[[float], str],
    parse: Callable[[str], Optional[str]],
    options: Sequence[str],
    *,
    n_samples: int,
    temperature: float,
    seed: int,
    trial_id: str,
    seed_fn: Optional[Callable[..., None]] = None,
    generate_batch: Optional[Callable[[float, int], List[str]]] = None,
) -> Dict[str, Any]:
    """
    Draw the model's forced choice n_samples times and summarise the distribution.

    `generate` takes a temperature and returns raw model text; `parse` maps that text to
    an option or None. Both are supplied by the caller so this stays independent of any
    particular model-loading path.

    When `generate_batch` is given, all draws come from one call (num_return_sequences),
    which is far cheaper than a loop. Seeding then happens once per trial rather than once
    per draw; both are reproducible, so the mode is recorded in the result.
    """
    raw_outputs: List[str] = []
    if generate_batch is not None:
        if seed_fn is not None:
            seed_fn(seed, trial_id, 0)
        raw_outputs = list(generate_batch(float(temperature), int(n_samples)))
        # A backend that ignores num_return_sequences would silently give one draw and an
        # entropy of 0, which looks like a confident model rather than a bug.
        if len(raw_outputs) != int(n_samples):
            raise RuntimeError(
                f"batched sampling returned {len(raw_outputs)} of {n_samples} draws "
                f"for {trial_id}; refusing to report entropy over an incomplete sample"
            )
    else:
        for draw in range(int(n_samples)):
            if seed_fn is not None:
                seed_fn(seed, trial_id, draw)
            raw_outputs.append(generate(float(temperature)))

    predictions: List[Optional[str]] = [parse(raw) for raw in raw_outputs]

    distribution = forced_choice_distribution(predictions, options)
    n_unparsed = sum(1 for p in predictions if p not in set(options))
    modal = max(distribution.items(), key=lambda kv: kv[1])[0] if distribution else None
    return {
        "n_samples": int(n_samples),
        "temperature": float(temperature),
        "sampling_mode": "batched" if generate_batch is not None else "sequential",
        "predictions": predictions,
        "raw_outputs": raw_outputs,
        "response_distribution": distribution,
        "forced_choice_entropy": entropy_of_distribution(distribution),
        "modal_prediction": None if modal == UNPARSED else modal,
        "n_unparsed": n_unparsed,
    }


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Spearman rho with average ranks for ties; None when undefined."""
    n = len(xs)
    if n < 3:
        return None

    def ranks(values: Sequence[float]) -> List[float]:
        order = sorted(range(n), key=lambda i: values[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = average
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def correlate_with_human(eval_path: Path, human_path: Path) -> Dict[str, Any]:
    """RQ1.1b (and RQ1.1a alongside it) against per-item human response entropy."""
    metrics = json.loads(eval_path.read_text(encoding="utf-8"))
    human = json.loads(human_path.read_text(encoding="utf-8"))["trials"]

    paired: List[Dict[str, Any]] = []
    for trial in metrics.get("trials", []):
        entry = human.get(trial.get("trial_id"))
        if entry is None:
            continue
        stage2 = trial.get("stage2") or {}
        stage1 = trial.get("stage1") or {}
        paired.append(
            {
                "trial_id": trial["trial_id"],
                "human_entropy": entry["human_entropy"],
                "forced_choice_entropy": stage2.get("forced_choice_entropy"),
                "semantic_entropy": stage1.get("semantic_entropy"),
                "correct": stage2.get("correct"),
            }
        )

    def paired_rho(field: str) -> Dict[str, Any]:
        rows = [r for r in paired if r[field] is not None]
        return {
            "n": len(rows),
            "spearman_rho": _spearman(
                [r["human_entropy"] for r in rows], [r[field] for r in rows]
            ),
        }

    return {
        "eval": eval_path.name,
        "model": metrics.get("model"),
        "condition": metrics.get("condition"),
        "protocol_version": metrics.get("protocol_version"),
        "n_trials_matched_to_human": len(paired),
        "rq1_1b_forced_choice_vs_human": paired_rho("forced_choice_entropy"),
        "rq1_1a_semantic_vs_human": paired_rho("semantic_entropy"),
        "trials": paired,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval", required=True, nargs="+", help="eval JSON path(s) or glob(s)")
    parser.add_argument("--human", default=Path("data/eu_emotions_human_entropy.json"), type=Path)
    parser.add_argument("--out", type=Path, default=None, help="optional JSON output path")
    args = parser.parse_args()

    paths: List[Path] = []
    for pattern in args.eval:
        matched = [Path(p) for p in glob.glob(pattern)]
        paths.extend(matched or [Path(pattern)])

    reports = []
    for path in paths:
        if not path.exists():
            print(f"missing: {path}")
            continue
        report = correlate_with_human(path, args.human)
        reports.append(report)
        rq1b = report["rq1_1b_forced_choice_vs_human"]
        rq1a = report["rq1_1a_semantic_vs_human"]
        print(f"{report['model']} | {report['condition']} | matched {report['n_trials_matched_to_human']}")
        print(f"   RQ1.1b forced-choice vs human : rho={rq1b['spearman_rho']} (n={rq1b['n']})")
        print(f"   RQ1.1a semantic      vs human : rho={rq1a['spearman_rho']} (n={rq1a['n']})")
        if rq1b["n"] == 0:
            print("   note: no forced_choice_entropy in this eval; rerun with --fc_samples > 1")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(reports, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
