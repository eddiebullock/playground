#!/usr/bin/env python3
from __future__ import annotations

"""
Study 1 post-baseline analysis: error structure, best-model selection, summary tables.

Run on HPC after EU-Emotions 118-trial baselines complete:
  python -m scripts.run_study1_postbaseline
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config import LOCAL_RESULTS_DIR, PROTOCOL_VERSION, SEED, STUDY_MODELS
from scripts.error_analysis import run_error_analysis
from scripts.per_emotion_analysis import compute_per_emotion_accuracy, intensity_summary, remove_neutral
from scripts.select_best_model import select_best
from scripts.study1_baselines import CANONICAL_CONDITION, discover_all_canonical

try:
    from scripts.entropy_sensitivity import recompute_trials
    from config import ENTROPY_SENSITIVITY_EMBEDDING_MODELS, ENTROPY_SENSITIVITY_TEMPERATURES
except ImportError:
    recompute_trials = None  # type: ignore


def _load_eval(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"Not protocol v2: {path}")
    return obj


def trials_to_dataframe(obj: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for t in obj.get("trials", []):
        s1 = t.get("stage1") or {}
        s2 = t.get("stage2") or {}
        rows.append(
            {
                "trial_id": t.get("trial_id"),
                "model": obj.get("model"),
                "condition": obj.get("condition"),
                "mental_state": t.get("label"),
                "predicted_label": s2.get("prediction"),
                "is_correct": s2.get("correct"),
                "semantic_entropy": s1.get("semantic_entropy"),
            }
        )
    return pd.DataFrame(rows)


def entropy_accuracy_correlation(df: pd.DataFrame) -> Optional[float]:
    sub = df[df["semantic_entropy"].notna() & df["is_correct"].notna()].copy()
    if len(sub) < 5:
        return None
    # Point-biserial: entropy vs binary correct (exploratory, not confirmatory).
    correct = sub["is_correct"].astype(float)
    ent = sub["semantic_entropy"].astype(float)
    if ent.std() == 0 or correct.std() == 0:
        return None
    return float(np.corrcoef(ent, correct)[0, 1])


def baseline_metrics_row(path: Path) -> Dict[str, Any]:
    obj = _load_eval(path)
    model = obj.get("model")
    return {
        "model": model,
        "condition": obj.get("condition"),
        "canonical_condition": CANONICAL_CONDITION.get(str(model)),
        "path": str(path),
        "n_scored": obj.get("n_scored"),
        "accuracy": obj.get("accuracy"),
        "mean_semantic_entropy": obj.get("mean_semantic_entropy"),
        "median_semantic_entropy": obj.get("median_semantic_entropy"),
        "p_binom_gt_chance": obj.get("p_binom_gt_chance"),
        "stage1_policy": obj.get("stage1_policy"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Study 1 EU baseline post-processing.")
    ap.add_argument(
        "--results-root",
        type=Path,
        default=LOCAL_RESULTS_DIR / "baseline" / "eu_emotions",
    )
    ap.add_argument("--stats-dir", type=Path, default=LOCAL_RESULTS_DIR / "stats")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    stats_dir = args.stats_dir
    stats_dir.mkdir(parents=True, exist_ok=True)

    canonical = discover_all_canonical(results_root=args.results_root)
    missing = [m for m in STUDY_MODELS if m not in canonical]
    if missing:
        raise RuntimeError(f"Missing canonical 118-trial baselines for: {missing}")

    summary_rows = [baseline_metrics_row(p) for p in canonical.values()]
    all_frames: List[pd.DataFrame] = []
    error_reports: Dict[str, Any] = {}

    for model, path in sorted(canonical.items()):
        err = run_error_analysis(path, top_k=args.top_k)
        error_reports[model] = err
        pairs_path = stats_dir / f"confused_pairs_{model}.json"
        err_path = stats_dir / f"error_analysis_{model}.json"
        pairs_path.write_text(
            json.dumps({"model": model, "confused_pairs": err["confused_pairs"]}, indent=2) + "\n",
            encoding="utf-8",
        )
        err_path.write_text(json.dumps(err, indent=2) + "\n", encoding="utf-8")

        df = trials_to_dataframe(_load_eval(path))
        corr = entropy_accuracy_correlation(df)
        for row in summary_rows:
            if row["model"] == model:
                row["entropy_accuracy_corr"] = corr
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)
    per_emotion = compute_per_emotion_accuracy(remove_neutral(combined))
    intens = intensity_summary(per_emotion)
    per_emotion_path = stats_dir / "per_emotion_eu_baselines.csv"
    intensity_path = stats_dir / "intensity_eu_baselines.csv"
    per_emotion.to_csv(per_emotion_path, index=False)
    intens.to_csv(intensity_path, index=False)

    paths_all = list(canonical.values())
    best_overall = select_best(paths_all)
    best_overall["selection_policy"] = "best_eu_accuracy_any_condition"
    best_overall["seed"] = SEED

    video_paths = [p for p in paths_all if _load_eval(p).get("condition") == "video_only"]
    best_video = select_best(video_paths) if video_paths else None
    if best_video is not None:
        best_video["selection_policy"] = "video_only_comparable"
        best_video["seed"] = SEED
        best_video["note"] = (
            "Fair comparison for LoRA if fine-tuning uses video_only (config.FINETUNE_MODALITY). "
            "Excludes Gemma multimodal."
        )

    out_summary = {
        "protocol_version": PROTOCOL_VERSION,
        "canonical_baselines": summary_rows,
        "entropy_scale": {
            "log_base": "natural",
            "fine_labels": 26,
            "base_labels": 20,
            "neutral_excluded": True,
            "intensity_collapsed": True,
            "max_entropy_ln20": float(np.log(20)),
            "interpretation": (
                "Primary semantic_entropy = H over ~20 base emotions after softmax on "
                "26 fine labels (neutral excluded), rich label prompts, intensity collapsed."
            ),
        },
        "per_emotion_csv": str(per_emotion_path),
        "intensity_csv": str(intensity_path),
    }
    (stats_dir / "study1_eu_baseline_summary.json").write_text(
        json.dumps(out_summary, indent=2) + "\n", encoding="utf-8"
    )
    (stats_dir / "best_model_overall.json").write_text(
        json.dumps(best_overall, indent=2) + "\n", encoding="utf-8"
    )
    if best_video is not None:
        (stats_dir / "best_model_video_only.json").write_text(
            json.dumps(best_video, indent=2) + "\n", encoding="utf-8"
        )

    if recompute_trials is not None:
        import json as _json

        sens_reports = []
        for path in canonical.values():
            obj = _json.loads(path.read_text(encoding="utf-8"))
            obj["_path"] = str(path)
            sens_reports.append(
                recompute_trials(
                    obj,
                    temperatures=ENTROPY_SENSITIVITY_TEMPERATURES,
                    embedding_models=ENTROPY_SENSITIVITY_EMBEDDING_MODELS,
                )
            )
        sens_path = stats_dir / "entropy_sensitivity.json"
        sens_path.write_text(
            _json.dumps(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "reports": sens_reports,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        out_summary["entropy_sensitivity_json"] = str(sens_path)

    print(json.dumps(out_summary, indent=2))
    print("\nBest overall (any condition):")
    print(json.dumps(best_overall, indent=2))
    if best_video is not None:
        print("\nBest video_only (LoRA-comparable):")
        print(json.dumps(best_video, indent=2))


if __name__ == "__main__":
    main()
