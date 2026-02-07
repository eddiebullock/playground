#!/usr/bin/env python3
"""
Analyze selection bias in MindReading video decode failures.

Groups failures by emotion category, folder/directory, and actor ID;
performs chi-square tests for non-random patterns; outputs CSV/JSON
and optional visualization. Prepares two result sets: (a) full test set
with missing treated as wrong/excluded, (b) valid trials only.

Usage:
  python analyze_mindreading_selection_bias.py \\
    --trial-definitions data/trial_definitions/mindreading_emotions_test.json \\
    --summary results/mindreading_multimodal/summary.json \\
    --output-dir results/mindreading_selection_bias
"""

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def extract_actor_id(stimulus_path: str) -> str:
    """Extract actor ID from path, e.g. 05/0507601/0507601M4Vcommitted.mov -> M4V."""
    filename = Path(stimulus_path).name
    m = re.match(r"\d{7}([A-Z]\d+[A-Z])[a-z-]+\.(mov|mp4)", filename, re.I)
    return m.group(1) if m else "unknown"


def load_trials_and_failures(
    trial_definitions_path: str,
    summary_path: str,
) -> Tuple[List[Dict], Set[str]]:
    """Load trial list and set of failed trial_id."""
    with open(trial_definitions_path, "r") as f:
        data = json.load(f)
    trials = data.get("trials", data) if isinstance(data, dict) else data

    with open(summary_path, "r") as f:
        summary = json.load(f)
    failed_ids = set(summary.get("failed_trials", []))

    return trials, failed_ids


def build_trial_id_to_trial(trials: List[Dict]) -> Dict[str, Dict]:
    """Map trial_id -> trial dict."""
    return {t.get("trial_id", ""): t for t in trials if t.get("trial_id")}


def group_by_dimensions(
    trials: List[Dict],
    failed_ids: Set[str],
    trial_by_id: Dict[str, Dict],
) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """
    Group success/fail counts by emotion, folder, actor.
    Returns (by_emotion, by_folder, by_actor) as dicts of {dim_value: {"success": n, "fail": n}}.
    """
    by_emotion = defaultdict(lambda: {"success": 0, "fail": 0})
    by_folder = defaultdict(lambda: {"success": 0, "fail": 0})
    by_actor = defaultdict(lambda: {"success": 0, "fail": 0})

    for t in trials:
        trial_id = t.get("trial_id", "")
        emotion = t.get("correct_label") or t.get("emotion") or "unknown"
        folder = t.get("folder") or (Path(t.get("stimulus_path", "")).parts[0] if t.get("stimulus_path") else "unknown")
        actor = extract_actor_id(t.get("stimulus_path", ""))

        failed = trial_id in failed_ids
        if failed:
            by_emotion[emotion]["fail"] += 1
            by_folder[folder]["fail"] += 1
            by_actor[actor]["fail"] += 1
        else:
            by_emotion[emotion]["success"] += 1
            by_folder[folder]["success"] += 1
            by_actor[actor]["success"] += 1

    return dict(by_emotion), dict(by_folder), dict(by_actor)


def chi2_test(by_dim: Dict[str, Dict]) -> Tuple[float, float]:
    """
    Chi-square test: are failures independent of dimension (e.g. emotion)?
    Returns (chi2_stat, p_value). Uses scipy if available else None for p.
    """
    # Contingency: rows = dimension value, cols = success / fail
    rows = []
    for dim_val, counts in by_dim.items():
        rows.append((counts.get("success", 0), counts.get("fail", 0)))
    if not rows:
        return 0.0, 1.0

    try:
        from scipy.stats import chi2_contingency
        table = [list(r) for r in rows]
        chi2, p, dof, expected = chi2_contingency(table)
        return float(chi2), float(p)
    except ImportError:
        logger.warning("scipy not installed; skipping chi-square p-value. Install with: pip install scipy")
        # Compute chi2 manually (no p-value)
        total_success = sum(r[0] for r in rows)
        total_fail = sum(r[1] for r in rows)
        n = total_success + total_fail
        if n == 0:
            return 0.0, float("nan")
        exp_success = total_success / n
        exp_fail = total_fail / n
        chi2 = 0.0
        for s, f in rows:
            exp_s = (s + f) * exp_success
            exp_f = (s + f) * exp_fail
            if exp_s > 0:
                chi2 += (s - exp_s) ** 2 / exp_s
            if exp_f > 0:
                chi2 += (f - exp_f) ** 2 / exp_f
        return chi2, float("nan")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze selection bias in MindReading decode failures"
    )
    parser.add_argument(
        "--trial-definitions",
        type=str,
        required=True,
        help="Path to trial definitions JSON",
    )
    parser.add_argument(
        "--summary",
        type=str,
        required=True,
        help="Path to experiment summary.json (contains failed_trials)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/mindreading_selection_bias",
        help="Output directory for report and artifacts",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate bar chart visualizations (requires matplotlib)",
    )
    args = parser.parse_args()

    trials, failed_ids = load_trials_and_failures(args.trial_definitions, args.summary)
    trial_by_id = build_trial_id_to_trial(trials)
    n_total = len(trials)
    n_fail = len(failed_ids)
    n_success = n_total - n_fail

    logger.info("Total trials: %d, failed: %d, success: %d", n_total, n_fail, n_success)

    by_emotion, by_folder, by_actor = group_by_dimensions(trials, failed_ids, trial_by_id)

    # Chi-square tests
    chi2_emotion, p_emotion = chi2_test(by_emotion)
    chi2_folder, p_folder = chi2_test(by_folder)
    chi2_actor, p_actor = chi2_test(by_actor)

    logger.info("Chi-square (fail vs dimension): emotion chi2=%.2f p=%.4f; folder chi2=%.2f p=%.4f; actor chi2=%.2f p=%.4f",
                chi2_emotion, p_emotion, chi2_folder, p_folder, chi2_actor, p_actor)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV-style report: by emotion
    rows_emotion = []
    for emotion, counts in sorted(by_emotion.items()):
        s, f = counts.get("success", 0), counts.get("fail", 0)
        tot = s + f
        fail_pct = 100.0 * f / tot if tot else 0
        rows_emotion.append({"emotion": emotion, "success": s, "fail": f, "total": tot, "fail_pct": round(fail_pct, 2)})
    csv_path = out_dir / "failure_by_emotion.csv"
    with open(csv_path, "w") as f:
        f.write("emotion,success,fail,total,fail_pct\n")
        for r in rows_emotion:
            f.write("%s,%d,%d,%d,%.2f\n" % (r["emotion"], r["success"], r["fail"], r["total"], r["fail_pct"]))
    logger.info("Wrote %s", csv_path)

    # By folder
    rows_folder = []
    for folder, counts in sorted(by_folder.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
        s, f = counts.get("success", 0), counts.get("fail", 0)
        tot = s + f
        fail_pct = 100.0 * f / tot if tot else 0
        rows_folder.append({"folder": folder, "success": s, "fail": f, "total": tot, "fail_pct": round(fail_pct, 2)})
    csv_folder = out_dir / "failure_by_folder.csv"
    with open(csv_folder, "w") as f:
        f.write("folder,success,fail,total,fail_pct\n")
        for r in rows_folder:
            f.write("%s,%d,%d,%d,%.2f\n" % (r["folder"], r["success"], r["fail"], r["total"], r["fail_pct"]))
    logger.info("Wrote %s", csv_folder)

    # By actor (may be many; summarize)
    rows_actor = []
    for actor, counts in sorted(by_actor.items()):
        s, f = counts.get("success", 0), counts.get("fail", 0)
        tot = s + f
        fail_pct = 100.0 * f / tot if tot else 0
        rows_actor.append({"actor": actor, "success": s, "fail": f, "total": tot, "fail_pct": round(fail_pct, 2)})
    csv_actor = out_dir / "failure_by_actor.csv"
    with open(csv_actor, "w") as f:
        f.write("actor,success,fail,total,fail_pct\n")
        for r in rows_actor:
            f.write("%s,%d,%d,%d,%.2f\n" % (r["actor"], r["success"], r["fail"], r["total"], r["fail_pct"]))
    logger.info("Wrote %s", csv_actor)

    # JSON report
    report = {
        "n_total": n_total,
        "n_failed": n_fail,
        "n_success": n_success,
        "fail_rate_pct": round(100.0 * n_fail / n_total, 2) if n_total else 0,
        "chi2_tests": {
            "emotion": {"chi2": chi2_emotion, "p": p_emotion},
            "folder": {"chi2": chi2_folder, "p": p_folder},
            "actor": {"chi2": chi2_actor, "p": p_actor},
        },
        "by_emotion": by_emotion,
        "by_folder": by_folder,
        "by_actor": by_actor,
        "result_sets": {
            "full_test_set": {
                "description": "All trials; missing (failed decode) treated as excluded from accuracy denominator, or as incorrect if included.",
                "n_trials": n_total,
            },
            "valid_only": {
                "description": "Only trials that decoded successfully; accuracy reported on this set (e.g. 77.29%%)",
                "n_trials": n_success,
            },
        },
    }
    report_path = out_dir / "selection_bias_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote %s", report_path)

    # Discussion template
    discussion = """
# Selection Bias Discussion (Template)

## Decode failure rate
- Total test trials: %d
- Failed to decode (OpenCV): %d (%.1f%%)
- Successfully decoded: %d

## Statistical tests (failure vs dimension)
- Emotion: chi2 = %.2f, p = %s. %s
- Folder (video directory): chi2 = %.2f, p = %s. %s
- Actor: chi2 = %.2f, p = %s. %s

## Interpretation
- If p < 0.05 for a dimension, failures are NOT random with respect to that dimension; the evaluated subset may be biased.
- If p >= 0.05 (or scipy not used), we cannot reject that failures are random; reported accuracy on the valid subset may still be generalizable, but the high attrition (%.1f%%) should be reported as a limitation.

## Result sets for reporting
1. **Full intended test set**: N = %d. Either (a) exclude failed trials from denominator and report accuracy on successful decodes only, or (b) treat failed trials as incorrect and report accuracy = correct / %d.
2. **Valid trials only**: N = %d. Accuracy 77.29%% (or recomputed) applies to this subset; state clearly that results are for the subset that decoded successfully.
""" % (
        n_total,
        n_fail,
        100.0 * n_fail / n_total if n_total else 0,
        n_success,
        chi2_emotion,
        "%.4f" % p_emotion if not (p_emotion != p_emotion) else "N/A (scipy not installed)",
        "Failures may be biased by emotion." if (p_emotion < 0.05 if hasattr(p_emotion, "__lt__") else False) else "No significant association with emotion.",
        chi2_folder,
        "%.4f" % p_folder if not (p_folder != p_folder) else "N/A",
        "Failures may be biased by folder." if (p_folder < 0.05 if hasattr(p_folder, "__lt__") else False) else "No significant association with folder.",
        chi2_actor,
        "%.4f" % p_actor if not (p_actor != p_actor) else "N/A",
        "Failures may be biased by actor." if (p_actor < 0.05 if hasattr(p_actor, "__lt__") else False) else "No significant association with actor.",
        100.0 * n_fail / n_total if n_total else 0,
        n_total,
        n_total,
        n_success,
    )
    discussion_path = out_dir / "SELECTION_BIAS_DISCUSSION.md"
    with open(discussion_path, "w") as f:
        f.write(discussion.strip())
    logger.info("Wrote %s", discussion_path)

    # Optional plot
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3, figsize=(14, 5))
            # By folder (usually few categories)
            folders = sorted(by_folder.keys(), key=lambda x: int(x) if x.isdigit() else 0)
            f_success = [by_folder[k].get("success", 0) for k in folders]
            f_fail = [by_folder[k].get("fail", 0) for k in folders]
            axes[0].bar([f + " (s)" for f in folders], f_success, color="green", alpha=0.7)
            axes[0].bar([f + " (f)" for f in folders], f_fail, color="red", alpha=0.7)
            axes[0].set_title("Decode success (s) vs fail (f) by folder")
            axes[0].tick_params(axis="x", rotation=45)
            # By emotion (top 20 by total)
            emo_sorted = sorted(by_emotion.items(), key=lambda x: x[1]["success"] + x[1]["fail"], reverse=True)[:20]
            emo_names = [e[0] for e in emo_sorted]
            e_success = [e[1].get("success", 0) for e in emo_sorted]
            e_fail = [e[1].get("fail", 0) for e in emo_sorted]
            axes[1].barh(range(len(emo_names)), e_success, color="green", alpha=0.7, label="success")
            axes[1].barh(range(len(emo_names)), [-e for e in e_fail], color="red", alpha=0.7, label="fail")
            axes[1].set_yticks(range(len(emo_names)))
            axes[1].set_yticklabels(emo_names, fontsize=8)
            axes[1].set_title("Top 20 emotions: success vs fail")
            axes[1].legend()
            # Fail rate by folder
            axes[2].bar(folders, [100.0 * by_folder[k].get("fail", 0) / (by_folder[k].get("success", 0) + by_folder[k].get("fail", 0)) if (by_folder[k].get("success", 0) + by_folder[k].get("fail", 0)) else 0 for k in folders], color="orange", alpha=0.7)
            axes[2].set_title("Fail rate (%) by folder")
            axes[2].tick_params(axis="x", rotation=45)
            plt.tight_layout()
            plot_path = out_dir / "selection_bias_plot.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()
            logger.info("Wrote %s", plot_path)
        except ImportError:
            logger.warning("matplotlib not installed; skip --plot. Install with: pip install matplotlib")


if __name__ == "__main__":
    main()
