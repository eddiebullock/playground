#!/usr/bin/env python3
"""
Run Study 3 feature set comparison and write Table 4: Feature Set Performance.
Outputs AUROC for each (Feature Set x Cohort) and Marginal Gain over demographics.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from study_utils import (
    load_cohort_c4,
    load_cohort_card,
    load_cohort_ybt,
    get_models,
    RANDOM_STATE,
    DEMOGRAPHICS_FEATURES,
    AQ_ITEM_FEATURES,
    EQ_SQ_ONLY_FEATURES,
    SPQ_ITEM_FEATURES,
    FEATURE_NAMES_45,
    FEATURE_NAMES_35,
    REPO_ROOT,
    RESULTS_DIR,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score

EQ_ONLY_FEATURES = [f"eq_{i}" for i in range(1, 11)]
SQR_ONLY_FEATURES = [f"sqr_{i}" for i in range(1, 11)]
SPQ_COLS = [f"spq_{i}" for i in range(1, 11)] + ["spq_total"]


def get_feature_set_columns(df, set_name, has_spq, has_aq=False):
    available = set(df.columns)
    if set_name == "demographics":
        return [f for f in DEMOGRAPHICS_FEATURES if f in available]
    if set_name == "eq_only":
        return [f for f in EQ_ONLY_FEATURES if f in available]
    if set_name == "sqr_only":
        return [f for f in SQR_ONLY_FEATURES if f in available]
    if set_name == "aq_only" and has_aq:
        return [f for f in AQ_ITEM_FEATURES if f in available]
    if set_name == "eq_sq_only":
        return [f for f in EQ_SQ_ONLY_FEATURES if f in available]
    if set_name == "spq_only" and has_spq:
        return [f for f in SPQ_ITEM_FEATURES if f in available]
    if set_name == "all_no_aq":
        base = FEATURE_NAMES_45 if has_spq else FEATURE_NAMES_35
        return [f for f in base if f in available and f not in AQ_ITEM_FEATURES and f != "aq_total"]
    if set_name == "all_features":
        return [f for f in (FEATURE_NAMES_45 if has_spq else FEATURE_NAMES_35) if f in available]
    return []


def cv_auroc_f1(X, y, n_splits=5):
    if X.shape[1] == 0 or len(np.unique(y)) < 2:
        return np.nan, np.nan
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    probas = np.zeros_like(y, dtype=float)
    for train_idx, val_idx in skf.split(Xs, y):
        m = get_models()["xgboost"]
        m.fit(Xs[train_idx], y[train_idx])
        probas[val_idx] = m.predict_proba(Xs[val_idx])[:, 1]
    auroc = roc_auc_score(y, probas)
    pred = (probas >= 0.5).astype(int)
    f1 = f1_score(y, pred, zero_division=0)
    return auroc, f1


def main():
    c4_path = os.path.join(REPO_ROOT, "data", "processed", "data_c4_final_recreated_cleaned.csv")
    card_path = os.path.join(REPO_ROOT, "data", "processed", "card_aligned.csv")
    ybt_aligned = os.path.join(REPO_ROOT, "data", "processed", "ybt_aligned.csv")
    ybt_raw = os.path.expanduser("~/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/YBT.csv")
    if not os.path.isfile(ybt_raw):
        ybt_raw = os.path.join(REPO_ROOT, "data", "raw", "YBT.csv")
    ybt_path = ybt_aligned if os.path.isfile(ybt_aligned) else ybt_raw

    df_c4, _, target_c4 = load_cohort_c4(
        c4_path, age_min=18, age_max=120, balance_50_50=True, apply_aq_filter=True, keep_all_columns=True
    )
    df_card = None
    if os.path.isfile(card_path):
        df_card, _, target_card = load_cohort_card(
            card_path, age_min=18, age_max=120, balance_50_50=True, apply_aq_filter=True
        )
    df_ybt, _, target_ybt = load_cohort_ybt(
        ybt_path, age_min=18, age_max=120, balance_50_50=True, apply_aq_filter=True
    )

    feature_set_names = [
        "demographics",
        "eq_only",
        "sqr_only",
        "spq_only",
        "aq_only",
        "eq_sq_only",
        "all_no_aq",
        "all_features",
    ]
    cohorts = [
        ("C4", df_c4, True, True, target_c4),
        ("CARD", df_card, True, False, target_card if df_card is not None else None),
        ("YBT", df_ybt, False, False, target_ybt),
    ]

    rows = []
    for cohort_name, df, has_spq, has_aq, target_col in cohorts:
        if df is None or target_col is None:
            continue
        for set_name in feature_set_names:
            cols = get_feature_set_columns(df, set_name, has_spq, has_aq)
            if not cols:
                continue
            X = df[cols].fillna(0).values
            y = df[target_col].values
            auroc, f1 = cv_auroc_f1(X, y)
            rows.append({"Cohort": cohort_name, "Feature_Set": set_name, "N_Features": len(cols), "AUROC": auroc, "F1": f1})

    feat_comp = pd.DataFrame(rows)

    study3_dir = os.path.join(RESULTS_DIR, "study3_features")
    os.makedirs(study3_dir, exist_ok=True)
    feat_comp.to_csv(os.path.join(study3_dir, "feature_comparison_table.csv"), index=False)

    # Build Table 4: rows = feature set, cols = C4 AUROC, CARD AUROC, YBT AUROC, Marginal Gain
    demo_auroc = {}
    for c in ["C4", "CARD", "YBT"]:
        d = feat_comp[(feat_comp["Cohort"] == c) & (feat_comp["Feature_Set"] == "demographics")]
        demo_auroc[c] = d["AUROC"].iloc[0] if len(d) else np.nan

    display_names = {
        "demographics": "Demographics",
        "eq_only": "EQ-10",
        "sqr_only": "SQ-R-10",
        "spq_only": "SPQ-10",
        "aq_only": "AQ-10",
        "eq_sq_only": "EQ + SQ-R",
        "all_no_aq": "EQ + SQ-R + SPQ (or Full no AQ)",
        "all_features": "Full Model (all items + demo)",
    }
    table4_rows = [
        ("Demographics", "Baseline"),
        ("EQ-10", "Single questionnaire"),
        ("SQ-R-10", "Single questionnaire"),
        ("SPQ-10", "Single questionnaire"),
        ("AQ-10", "Single questionnaire"),
        ("EQ + SQ-R", "Core combination"),
        ("EQ + SQ-R + SPQ", "Extended combination"),
        ("EQ + SQ-R + AQ", "Extended (not in pipeline)"),
        ("Full Model", "All items + demo"),
    ]

    def get_auroc(cohort, set_name):
        if set_name == "EQ + SQ-R":
            set_name = "eq_sq_only"
        elif set_name == "EQ + SQ-R + SPQ":
            set_name = "all_no_aq"
        elif set_name == "Full Model":
            set_name = "all_features"
        elif set_name == "EQ-10":
            set_name = "eq_only"
        elif set_name == "SQ-R-10":
            set_name = "sqr_only"
        elif set_name == "SPQ-10":
            set_name = "spq_only"
        elif set_name == "AQ-10":
            set_name = "aq_only"
        elif set_name == "Demographics":
            set_name = "demographics"
        d = feat_comp[(feat_comp["Cohort"] == cohort) & (feat_comp["Feature_Set"] == set_name)]
        return d["AUROC"].iloc[0] if len(d) else np.nan

    out_lines = [
        "Table 4: Feature Set Performance (5-fold CV AUROC)",
        "Marginal Gain = AUROC - Demographics AUROC for that cohort.",
        "",
    ]
    header = "Feature Set          | C4 AUROC | CARD AUROC | YBT AUROC | Marginal Gain"
    out_lines.append(header)
    out_lines.append("-" * len(header))

    for row_name, _ in table4_rows:
        c4_a = get_auroc("C4", row_name)
        card_a = get_auroc("CARD", row_name)
        ybt_a = get_auroc("YBT", row_name) if row_name != "AQ-10" and row_name != "SPQ-10" else np.nan
        if row_name == "EQ + SQ-R + AQ":
            c4_a = card_a = ybt_a = np.nan
        marg = np.nan
        if row_name != "Demographics" and not np.isnan(c4_a) and not np.isnan(demo_auroc.get("C4")):
            marg = c4_a - demo_auroc["C4"]
        c4_s = f"{c4_a:.2f}" if not np.isnan(c4_a) else "—"
        card_s = f"{card_a:.2f}" if not np.isnan(card_a) else "—"
        ybt_s = f"{ybt_a:.2f}" if not np.isnan(ybt_a) else "—"
        marg_s = f"+{marg:.2f}" if not np.isnan(marg) and row_name != "Demographics" else ("—" if row_name == "Demographics" else "")
        out_lines.append(f"{row_name:20} | {c4_s:^8} | {card_s:^10} | {ybt_s:^9} | {marg_s}")

    out_path = os.path.join(RESULTS_DIR, "table4_feature_set_performance.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(out_lines))
    print("\n".join(out_lines))
    print(f"\nSaved to {out_path}")
    print("\nFull feature_comparison_table.csv saved to results/study3_features/")


if __name__ == "__main__":
    main()
