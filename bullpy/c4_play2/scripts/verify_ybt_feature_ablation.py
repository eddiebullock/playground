#!/usr/bin/env python3
"""Verify YBT feature ablation: AUROC values in feature_comparison_table.csv and CV vs test."""
import pandas as pd
import os

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
results_path = os.path.join(REPO_ROOT, "results", "study3_features", "feature_comparison_table.csv")

if os.path.exists(results_path):
    feat_comp = pd.read_csv(results_path)
    # CSV uses 'YBT' not 'Dataset3'
    ybt = feat_comp[feat_comp["Cohort"].isin(["Dataset3", "YBT"])].copy()

    print("=" * 60)
    print("YBT (Dataset3) FEATURE SET RESULTS")
    print("=" * 60)
    print(ybt.to_string(index=False))
    print("\n")

    print("=" * 60)
    print("TABLE 4 VALUES CHECK")
    print("=" * 60)

    for feat_set in ["demographics", "eq_only", "sqr_only", "aq_only", "eq_sq_only", "all_features"]:
        row = ybt[ybt["Feature_Set"] == feat_set]
        if len(row) > 0:
            auroc = row["AUROC"].values[0]
            n_feat = row["N_Features"].values[0]
            print(f"{feat_set:20s} (n={n_feat:2d}): AUROC = {auroc:.3f}")
        else:
            print(f"{feat_set:20s}: NOT FOUND")

    print("\n")

    all_feat_auroc = ybt[ybt["Feature_Set"] == "all_features"]["AUROC"].values
    table2_test_auroc = 0.82
    if len(all_feat_auroc) > 0:
        print("=" * 60)
        print("CRITICAL VALIDATION")
        print("=" * 60)
        print(f"YBT 'all_features' AUROC from Table 4:  {all_feat_auroc[0]:.3f}")
        print(f"YBT best test AUROC from Table 2:      {table2_test_auroc:.2f}")
        print(f"Difference:                              {all_feat_auroc[0] - table2_test_auroc:.3f}")
        print("")

        if abs(all_feat_auroc[0] - table2_test_auroc) < 0.01:
            print("MATCH - Table 4 values are TEST SET metrics")
        elif 0.03 <= (all_feat_auroc[0] - table2_test_auroc) <= 0.05:
            print("LIKELY OK - Table 4 values are probably CV metrics")
            print("   (CV typically 0.03-0.05 higher than test)")
        elif table2_test_auroc - all_feat_auroc[0] > 0.05:
            print("Table 4 values are CROSS-VALIDATION (lower than test).")
            print("   CV on full cohort; Table 2 = held-out test set.")
        else:
            print("WARNING - Significant discrepancy detected!")
            print("   Need to investigate data processing differences")

    print("\n" + "=" * 60)
    print("METHODOLOGY")
    print("=" * 60)
    print("Study 3 uses cv_auroc_f1 with StratifiedKFold(n_splits=5).")
    print("=> Table 4 values are CROSS-VALIDATION metrics (5-fold CV on full cohort),")
    print("   NOT held-out test set metrics. Table 2 = test set AUROC.")

    print("\n" + "=" * 60)
    print("COPY-PASTE READY OUTPUT FOR TABLE 4 UPDATE")
    print("=" * 60)
    mapping = {
        "demographics": "Demographics",
        "eq_only": "EQ-10 only",
        "sqr_only": "SQ-R-10 only",
        "aq_only": "AQ-10 only",
        "eq_sq_only": "EQ + SQ-R",
        "all_features": "Full model (all_features)",
    }
    print("\nYBT Column for Table 4:")
    print("Feature Set          | AUROC")
    print("---------------------|-------")
    for code, label in mapping.items():
        row = ybt[ybt["Feature_Set"] == code]
        if len(row) > 0:
            auroc = row["AUROC"].values[0]
            print(f"{label:20s} | {auroc:.2f}")
        else:
            print(f"{label:20s} | NOT FOUND")

else:
    print(f"ERROR: File not found at {results_path}")
