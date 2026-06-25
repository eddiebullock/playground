#!/usr/bin/env python3
"""
B0 gating: verify fine-tuned EU eval is genuine degradation vs parse/format artifact.

Re-scores an existing eval JSON with tolerant parsing and free-response judge
without re-running the model. Prints raw output samples for manual inspection.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from scripts.eu_emotion_synonyms import synonym_config_source
from scripts.tolerant_parse import parse_emotion_tolerant, rescore_eval_json
from scripts.trial_foils import resolve_eu_emotion_pool


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sample_raw_outputs(trials: List[dict], n: int = 8) -> List[Dict[str, Any]]:
    out = []
    for t in trials[:n]:
        s2 = t.get("stage2") or {}
        out.append(
            {
                "trial_id": t.get("trial_id"),
                "label": t.get("label"),
                "strict_prediction": s2.get("prediction"),
                "raw_model_output": (s2.get("raw_model_output") or "")[:500],
            }
        )
    return out


def _prediction_histogram(trials: List[dict]) -> Counter:
    c: Counter = Counter()
    for t in trials:
        s2 = t.get("stage2") or {}
        pred = s2.get("prediction")
        c[str(pred) if pred is not None else "<null>"] += 1
    return c


def _tolerant_histogram(trials: List[dict], pool: List[str]) -> Counter:
    c: Counter = Counter()
    for t in trials:
        s2 = t.get("stage2") or {}
        opts = s2.get("options") or []
        raw = s2.get("raw_model_output") or ""
        pred, _, method = parse_emotion_tolerant(raw, opts, full_label_pool=pool)
        c[f"{pred}|{method}"] += 1
    return c


def main() -> None:
    ap = argparse.ArgumentParser(description="B0: verify fine-tune EU eval rescoring")
    ap.add_argument(
        "--eval_json",
        type=Path,
        required=True,
        help="Post-FT or baseline eval_v2 JSON",
    )
    ap.add_argument(
        "--baseline_json",
        type=Path,
        default=None,
        help="Optional baseline JSON for side-by-side comparison",
    )
    ap.add_argument("--sample_n", type=int, default=8)
    ap.add_argument("--output", type=Path, default=None, help="Write full report JSON")
    args = ap.parse_args()

    data = _load(args.eval_json)
    trials = data.get("trials") or []
    pool = resolve_eu_emotion_pool()
    rescore = rescore_eval_json(trials)

    report: Dict[str, Any] = {
        "eval_json": str(args.eval_json),
        "synonym_config_source": synonym_config_source(),
        "model": data.get("model"),
        "lora_adapter": data.get("lora_adapter"),
        "original_accuracy": data.get("accuracy"),
        "original_n_correct": data.get("n_correct"),
        "original_n_scored": data.get("n_scored"),
        "rescore": rescore,
        "strict_prediction_histogram": dict(_prediction_histogram(trials)),
        "raw_output_samples": _sample_raw_outputs(trials, n=args.sample_n),
    }

    tol_hist = _tolerant_histogram(trials, pool)
    report["tolerant_parse_histogram"] = dict(tol_hist.most_common(20))

    # Verdict helper
    strict = rescore.get("strict_4afc_accuracy")
    tolerant = rescore.get("tolerant_4afc_accuracy")
    if strict is not None and tolerant is not None:
        delta = tolerant - strict
        if tolerant >= 0.20 and strict < 0.10:
            verdict = "LIKELY_PARSE_ARTIFACT: tolerant score near chance+ while strict very low"
        elif tolerant < 0.15:
            verdict = "GENUINE_DEGRADATION: tolerant rescore still far below chance (~25%)"
        elif delta > 0.10:
            verdict = "PARTIAL_FORMAT_ISSUE: tolerant rescore materially higher than strict"
        else:
            verdict = "MIXED: review raw samples"
        report["verdict"] = verdict
        report["tolerant_minus_strict_pp"] = delta * 100 if delta is not None else None

    if args.baseline_json and args.baseline_json.is_file():
        base = _load(args.baseline_json)
        base_trials = base.get("trials") or []
        base_rescore = rescore_eval_json(base_trials)
        report["baseline"] = {
            "path": str(args.baseline_json),
            "original_accuracy": base.get("accuracy"),
            "rescore": base_rescore,
        }

    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
