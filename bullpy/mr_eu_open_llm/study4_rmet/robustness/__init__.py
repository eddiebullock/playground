"""
Robustness / rigor layer for study4_rmet A1 (item-level H1).

Does not modify the existing Spearman+permutation pipeline in
scripts/alignment_analyses.py — these modules run alongside it.

Run report:
  python study4_rmet/robustness/run_robustness_report.py
Tests:
  pytest study4_rmet/robustness/test_robustness.py
"""

__all__ = [
    "power_analysis",
    "equivalence_bayes",
    "disattenuation",
    "trial_level_model",
    "meta_analysis",
]
