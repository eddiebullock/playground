"""
Select best baseline model by EU-Emotions Stage 2 accuracy (protocol v2).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import LOCAL_RESULTS_DIR, PROTOCOL_VERSION, SEED, STUDY_MODELS


def load_v2_result(path: Path) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if obj.get("protocol_version") != PROTOCOL_VERSION:
        return None
    if obj.get("accuracy") is None:
        return None
    return obj


def select_best(
    result_paths: List[Path],
    tie_break: str = "alphabetical",
    *,
    one_per_model: bool = True,
) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    for p in result_paths:
        obj = load_v2_result(p)
        if obj is None:
            continue
        model_key = obj.get("model")
        n_scored = obj.get("n_scored") or 0
        if model_key not in STUDY_MODELS or n_scored < 100:
            continue
        candidates.append(
            {
                "model_key": model_key,
                "accuracy": float(obj["accuracy"]),
                "path": str(p),
                "n_scored": n_scored,
                "condition": obj.get("condition"),
                "mean_semantic_entropy": obj.get("mean_semantic_entropy"),
            }
        )
    if not candidates:
        raise RuntimeError("No protocol v2 baseline JSONs found.")

    if one_per_model:
        by_model: Dict[str, Dict[str, Any]] = {}
        for c in candidates:
            mk = c["model_key"]
            if mk not in by_model or c["accuracy"] > by_model[mk]["accuracy"]:
                by_model[mk] = c
        candidates = list(by_model.values())

    candidates.sort(key=lambda x: (-x["accuracy"], x["model_key"] if tie_break == "alphabetical" else x["model_key"]))
    best = candidates[0]
    return {
        "model_key": best["model_key"],
        "accuracy": best["accuracy"],
        "condition": best.get("condition"),
        "mean_semantic_entropy": best.get("mean_semantic_entropy"),
        "result_path": best["path"],
        "protocol_version": PROTOCOL_VERSION,
        "candidates": candidates,
        "tie_break": tie_break,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Select best model from Study 1 v2 baseline JSONs.")
    ap.add_argument(
        "--input_glob",
        type=str,
        default=str(LOCAL_RESULTS_DIR / "baseline" / "eu_emotions" / "*" / "eval_v2_*.json"),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=LOCAL_RESULTS_DIR / "stats" / "best_model.json",
    )
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    paths = sorted(Path().glob(args.input_glob))
    summary = select_best(paths)
    summary["seed"] = args.seed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
