#!/usr/bin/env python3
"""
Extract all information needed to fill appendix placeholders.
Run from repo root: python scripts/extract_appendix_data.py
Uses only stdlib + pandas to avoid slow ML/font cache imports.
"""
import os
import sys
import json
import platform

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
out_lines = []

def log(s):
    out_lines.append(s)
    print(s)

# ---------- A. QUESTIONNAIRE ITEM MAPPINGS (from study_utils constants) ----------
log("=" * 70)
log("APPENDIX A: QUESTIONNAIRE ITEM FEATURE NAMES")
log("=" * 70)
AQ_ITEM_FEATURES = [f"aq_{i}" for i in range(1, 11)]
EQ_ITEMS = [f"eq_{i}" for i in range(1, 11)]
SQR_ITEMS = [f"sqr_{i}" for i in range(1, 11)]
SPQ_ITEM_FEATURES = [f"spq_{i}" for i in range(1, 11)]
log("\nAQ-10 items (feature names):")
log(str(AQ_ITEM_FEATURES))
log("\nEQ-10 items (feature names):")
log(str(EQ_ITEMS))
log("\nSQ-R-10 items (feature names):")
log(str(SQR_ITEMS))
log("\nSPQ-10 items (feature names):")
log(str(SPQ_ITEM_FEATURES))
log("\nNote: Actual questionnaire item text and full-to-short mappings (e.g. AQ-50 to AQ-10)")
log("are in CARD_ITEM_MAPPING_REQUIREMENTS.md and card_c4_validation.ipynb (ITEM_MAPPINGS).")

# ---------- B. MODEL HYPERPARAMETERS (from study_utils get_models) ----------
log("\n" + "=" * 70)
log("APPENDIX B.1: MODEL HYPERPARAMETERS")
log("=" * 70)
log("""
XGBOOST:
  max_depth=5, learning_rate=0.05, n_estimators=200, scale_pos_weight=1.0,
  random_state=42, eval_metric='logloss'

LIGHTGBM:
  max_depth=5, learning_rate=0.05, n_estimators=200, random_state=42, verbose=-1

RANDOM_FOREST:
  n_estimators=200, max_depth=10, random_state=42

LOGISTIC:
  max_iter=1000, random_state=42
""".strip())

# ---------- B.2 PREPROCESSING ----------
log("\n" + "=" * 70)
log("APPENDIX B.2: DATA PREPROCESSING")
log("=" * 70)
log("StandardScaler: used for all models (fit on train, transform train and test).")
log("Missing values: filled with 0 in cohort loading and feature matrices (fillna(0)).")
log("Stratification: StratifiedKFold(n_splits=5, shuffle=True, random_state=42) for CV;")
log("  train_test_split(..., stratify=y) for 80/20 train/test.")
log("Train/test split: test_size=0.2 (80% train, 20% test).")
log("CV folds: 5. Random state: 42.")

# ---------- B.3 SPLIT SIZES ----------
log("\n" + "=" * 70)
log("APPENDIX B.3: TRAIN-TEST SPLIT SIZES (80/20)")
log("=" * 70)
datasets = {"C4 (Channel 4)": 26152, "CARD": 3893, "YBT (Dataset3)": 684}
for name, total_n in datasets.items():
    train_n = int(total_n * 0.8)
    test_n = total_n - train_n
    log(f"\n{name}:")
    log(f"  Total (balanced): {total_n}")
    log(f"  Train (80%):     {train_n}")
    log(f"  Test (20%):      {test_n}")

# ---------- C. CONFUSION MATRICES ----------
log("\n" + "=" * 70)
log("APPENDIX C: CONFUSION MATRICES (TEST SET)")
log("=" * 70)
for cohort, subdir in [("C4", "c4"), ("CARD", "card"), ("YBT (Dataset3)", "dataset3")]:
    pm_path = os.path.join(RESULTS_DIR, "study1_within_cohort", subdir, "performance_metrics.json")
    if not os.path.exists(pm_path):
        continue
    with open(pm_path) as f:
        pm = json.load(f)
    log(f"\n--- {cohort} ---")
    for model_name, m in pm.items():
        tp, tn, fp, fn = m.get("TP", 0), m.get("TN", 0), m.get("FP", 0), m.get("FN", 0)
        if tp + tn + fp + fn > 0:
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            log(f"  {model_name}: TP={tp}, TN={tn}, FP={fp}, FN={fn} | Sens={sens:.3f}, Spec={spec:.3f}")

# ---------- D. SUBGROUP SAMPLE SIZES ----------
log("\n" + "=" * 70)
log("APPENDIX D: SUBGROUP SAMPLE SIZES (Study 2)")
log("=" * 70)
import pandas as pd
for subdir, label in [
    ("age_stratified", "Age strata"),
    ("sex_stratified", "Sex"),
    ("comorbidity_stratified", "Comorbidity"),
]:
    path = os.path.join(RESULTS_DIR, "study2_subgroups", subdir, "subgroup_comparison_table.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        log(f"\n{label}:")
        log(df[["Cohort", "Subgroup", "Category", "n"]].to_string(index=False))
    else:
        log(f"\n{label}: file not found")

# ---------- E. SHAP RESULTS ----------
log("\n" + "=" * 70)
log("APPENDIX E: TOP 20 SHAP FEATURES (C4)")
log("=" * 70)
shap_path = os.path.join(RESULTS_DIR, "study3_features", "shap_values", "c4_shap_importance.csv")
if os.path.exists(shap_path):
    shap_df = pd.read_csv(shap_path)
    log(shap_df.head(20).to_string(index=False))
else:
    log("SHAP file not found.")

# ---------- F. SOFTWARE VERSIONS ----------
log("\n" + "=" * 70)
log("APPENDIX F: SOFTWARE ENVIRONMENT")
log("=" * 70)
log(f"\nPython: {sys.version.split()[0]}")
log(f"Platform: {platform.platform()}")
for pkg in ["numpy", "pandas", "sklearn", "xgboost", "lightgbm", "shap"]:
    try:
        mod = __import__(pkg)
        v = getattr(mod, "__version__", "?")
        log(f"  {pkg}: {v}")
    except ImportError:
        log(f"  {pkg}: not installed")
log("\nFull pip freeze (first 2500 chars):")
try:
    r = __import__("subprocess").run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=15
    )
    if r.returncode == 0:
        log(r.stdout[:2500] + ("..." if len(r.stdout) > 2500 else ""))
except Exception as e:
    log(f"(pip freeze failed: {e})")

# ---------- K. DATASET SIZES ----------
log("\n" + "=" * 70)
log("APPENDIX K: DATASET SIZES")
log("=" * 70)
log("\nBalanced (50/50, age 18-55, AQ>=6 for autism):")
bal_path = os.path.join(RESULTS_DIR, "balanced_demographics.csv")
if os.path.exists(bal_path):
    b = pd.read_csv(bal_path)
    for _, row in b.iterrows():
        log(f"  {row['Cohort']}: N_total={row['N_total']}, N_autism={row['N_autism']}, N_non_autism={row['N_non_autism']}")
log("\nPre-balancing (after exclusions, before 50/50):")
log("  C4: ~28,003 total (13,076 autism, 14,927 non-autism) - from pipeline.")
log("  CARD: ~3,893 (2,033 autism, 1,860 non-autism) - card_aligned pre-balance.")
log("  YBT: balanced N in table above; raw varies by data source.")

# ---------- E.3 FEATURE ABLATION ----------
log("\n" + "=" * 70)
log("APPENDIX E.3: FEATURE SET ABLATION (Study 3)")
log("=" * 70)
feat_path = os.path.join(RESULTS_DIR, "study3_features", "feature_comparison_table.csv")
if os.path.exists(feat_path):
    fc = pd.read_csv(feat_path)
    log(fc.to_string(index=False))
    log("\nNote: Values are 5-fold cross-validation AUROC and F1. Per-fold results not saved.")
else:
    log("feature_comparison_table.csv not found.")

log("\n" + "=" * 70)
log("CV: Fold-level details")
log("=" * 70)
log("Only mean and std saved (e.g. cv_auroc_mean, cv_auroc_std in performance_metrics.json).")

# Write output file
out_path = os.path.join(REPO_ROOT, "results", "APPENDIX_DATA_EXTRACTION.txt")
with open(out_path, "w") as f:
    f.write("\n".join(out_lines))
log(f"\n\nOutput saved to: {out_path}")
