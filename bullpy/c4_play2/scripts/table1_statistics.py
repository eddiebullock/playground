"""
Table 1 statistical comparisons: autism vs non-autism groups.
For each cohort: Mann-Whitney U tests for continuous variables,
chi-square tests for categorical variables.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

# Ensure src/ is importable when running as a script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if os.path.join(REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from study_utils import load_cohort_c4, load_cohort_card, load_cohort_ybt  # type: ignore


def compare_groups(df, groupcol="diagnosis", continuous_cols=None, categorical_cols=None):
    """
    For each continuous column: Mann-Whitney U test (non-parametric).
    For each categorical column: chi-square test.
    Returns a DataFrame of results.
    """
    results = []
    autism = df[df[groupcol] == 1]
    non_autism = df[df[groupcol] == 0]

    if continuous_cols:
        for col in continuous_cols:
            if col not in df.columns:
                continue
            a = autism[col].dropna()
            b = non_autism[col].dropna()
            if len(a) == 0 or len(b) == 0:
                continue
            stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            results.append(
                {
                    "Variable": col,
                    "Test": "Mann-Whitney U",
                    "Autism_mean_SD": f"{a.mean():.2f} ({a.std():.2f})",
                    "NonAutism_mean_SD": f"{b.mean():.2f} ({b.std():.2f})",
                    "Statistic": round(stat, 2),
                    "p_value": round(p, 4),
                    "Significant": p < 0.05,
                }
            )

    if categorical_cols:
        for col in categorical_cols:
            if col not in df.columns:
                continue
            ct = pd.crosstab(df[groupcol], df[col])
            if ct.shape[0] < 2 or ct.shape[1] < 2:
                continue
            stat, p, dof, expected = stats.chi2_contingency(ct)
            results.append(
                {
                    "Variable": col,
                    "Test": "Chi-square",
                    "Autism_mean_SD": str(autism[col].value_counts().to_dict()),
                    "NonAutism_mean_SD": str(non_autism[col].value_counts().to_dict()),
                    "Statistic": round(stat, 2),
                    "p_value": round(p, 4),
                    "Significant": p < 0.05,
                }
            )

    return pd.DataFrame(results)


def main():
    continuous = ["age", "aq_total", "eq_total", "sqr_total"]
    continuous_with_spq = continuous + ["spq_total"]
    categorical = ["sex_num"]

    results_dir = os.path.join(REPO_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "table1_statistics.csv")

    all_results = []

    # C4
    c4_path = os.path.join(REPO_ROOT, "data", "processed", "data_c4_final_recreated_cleaned.csv")
    df_c4, feat_c4, target_c4 = load_cohort_c4(
        c4_path, age_min=18, age_max=55, balance_50_50=False, apply_aq_filter=True, keep_all_columns=False
    )
    res_c4 = compare_groups(df_c4, groupcol=target_c4, continuous_cols=continuous_with_spq, categorical_cols=categorical)
    if not res_c4.empty:
        res_c4.insert(0, "Cohort", "C4")
        all_results.append(res_c4)

    # CARD
    card_path = os.path.join(REPO_ROOT, "data", "processed", "card_aligned.csv")
    df_card, feat_card, target_card = load_cohort_card(
        card_path, age_min=18, age_max=55, balance_50_50=False, apply_aq_filter=True
    )
    res_card = compare_groups(df_card, groupcol=target_card, continuous_cols=continuous_with_spq, categorical_cols=categorical)
    if not res_card.empty:
        res_card.insert(0, "Cohort", "CARD")
        all_results.append(res_card)

    # YBT
    ybt_aligned_path = os.path.join(REPO_ROOT, "data", "processed", "ybt_aligned.csv")
    _default_ybt = os.path.expanduser("~/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/YBT.csv")
    _repo_ybt = os.path.join(REPO_ROOT, "data", "raw", "YBT.csv")
    ybt_path = ybt_aligned_path if os.path.isfile(ybt_aligned_path) else _default_ybt if os.path.isfile(_default_ybt) else _repo_ybt
    df_ybt, feat_ybt, target_ybt = load_cohort_ybt(
        ybt_path, age_min=18, age_max=55, balance_50_50=False, apply_aq_filter=True
    )
    res_ybt = compare_groups(df_ybt, groupcol=target_ybt, continuous_cols=continuous, categorical_cols=categorical)
    if not res_ybt.empty:
        res_ybt.insert(0, "Cohort", "YBT")
        all_results.append(res_ybt)

    if all_results:
        final = pd.concat(all_results, ignore_index=True)
        final.to_csv(out_path, index=False)
        print(f"Saved to {out_path}")
    else:
        print("No results to save; check data loading and column names.")


if __name__ == "__main__":
    main()

