#!/usr/bin/env python3
"""Augment an existing eval_v2 JSON with judge + selective-prediction metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from scripts.free_response_judge import judge_free_response
from scripts.selective_prediction import selective_prediction_report
from scripts.tolerant_parse import rescore_eval_json


def augment_eval_json(data: Dict[str, Any]) -> Dict[str, Any]:
    trials = data.get("trials") or []
    stage1_judged = 0
    stage1_correct = 0
    for t in trials:
        s1 = t.get("stage1") or {}
        text = s1.get("free_response_text")
        label = t.get("label")
        if text and label:
            jr = judge_free_response(text, label, use_llm=False)
            s1["judge"] = jr
            t["stage1"] = s1
            stage1_judged += 1
            stage1_correct += int(jr.get("correct", False))

    out = dict(data)
    out["artifact_metrics"] = {
        "primary": "free_response_judge",
        "free_response_judge_accuracy": stage1_correct / stage1_judged if stage1_judged else None,
        "free_response_judge_n": stage1_judged,
        "selective_prediction": selective_prediction_report(trials),
        "tolerant_rescore": rescore_eval_json(trials),
        "secondary_4afc_accuracy": data.get("accuracy"),
    }
    out["trials"] = trials
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Add artifact metrics to eval JSON")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    augmented = augment_eval_json(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(augmented, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(augmented.get("artifact_metrics"), indent=2))


if __name__ == "__main__":
    main()
