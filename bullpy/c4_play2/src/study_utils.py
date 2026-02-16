"""
Shared utilities for ML autism prediction study (Study 1, 2, 3).
Handles cohort loading, stratification, CV training, evaluation, and feature schemas.
"""

from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List, Any
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, roc_curve, brier_score_loss,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Default paths (relative to repo root)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(REPO_ROOT, "models", "cross_validation")
FEATURE_INFO_PATH = os.path.join(ARTIFACT_DIR, "feature_info_original.json")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

# 45-feature schema (C4/CARD): AQ excluded from model input
FEATURE_NAMES_45 = [
    "age", "sex",
    "spq_1", "spq_2", "spq_3", "spq_4", "spq_5", "spq_6", "spq_7", "spq_8", "spq_9", "spq_10",
    "eq_1", "eq_2", "eq_3", "eq_4", "eq_5", "eq_6", "eq_7", "eq_8", "eq_9", "eq_10",
    "sqr_1", "sqr_2", "sqr_3", "sqr_4", "sqr_5", "sqr_6", "sqr_7", "sqr_8", "sqr_9", "sqr_10",
    "spq_total", "eq_total", "sqr_total", "d_score", "sqrt_age", "age_x_eq", "eq_sqr_ratio",
    "is_stem_occupation", "sex_num",
    "age_group_19-30", "age_group_31-45", "age_group_46-60", "age_group_61+",
]

# 35-feature schema (no SPQ, for Dataset3/YBT)
SPQ_COLS = [f"spq_{i}" for i in range(1, 11)] + ["spq_total"]
FEATURE_NAMES_35 = [f for f in FEATURE_NAMES_45 if f not in SPQ_COLS]

# Feature sets for Study 3 (names -> list of feature names or pattern)
DEMOGRAPHICS_FEATURES = ["age", "sex_num", "sqrt_age", "is_stem_occupation", "age_group_19-30", "age_group_31-45", "age_group_46-60", "age_group_61+"]
AQ_ITEM_FEATURES = [f"aq_{i}" for i in range(1, 11)]
EQ_SQ_ONLY_FEATURES = [f"eq_{i}" for i in range(1, 11)] + [f"sqr_{i}" for i in range(1, 11)]
SPQ_ITEM_FEATURES = [f"spq_{i}" for i in range(1, 11)]

# Subgroup age bins (for stratification and subgroup analysis)
AGE_GROUPS_STUDY = {"18-30": (18, 30), "31-40": (31, 40), "41-50": (41, 50), "51-55": (51, 56)}

# Comorbidity columns for Study 2 subgroup analysis (if present in data)
COMOBIDITY_COLUMNS = ["has_adhd", "has_anxiety", "has_depression"]

CV_FOLDS = 5
RANDOM_STATE = 42
TEST_SIZE = 0.2


def get_feature_schema(include_spq: bool = True) -> List[str]:
    """Return feature list: 45 with SPQ, 35 without."""
    if include_spq:
        return list(FEATURE_NAMES_45)
    return list(FEATURE_NAMES_35)


def load_feature_info(path: Optional[str] = None) -> Dict[str, Any]:
    """Load feature names from feature_info_original.json if present."""
    path = path or FEATURE_INFO_PATH
    if os.path.isfile(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"feature_names": FEATURE_NAMES_45, "excluded_features": []}


def _ensure_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            continue
        if df[col].dtype == object or df[col].dtype.name == "category":
            df = df.copy()
            df[col] = pd.to_numeric(df[col].replace("unknown", np.nan), errors="coerce").fillna(0)
    return df


def create_stratified_split(
    df: pd.DataFrame,
    target_col: str = "diagnosis",
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    strata_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/test split. Uses composite strata (age_group + sex + target)
    when possible to keep proportions.
    """
    if strata_cols is None:
        strata_cols = []
    if "strata" not in df.columns:
        if "age_group" in df.columns and "sex" in df.columns:
            df = df.copy()
            df["strata"] = (
                df["age_group"].astype(str) + "_" + df["sex"].astype(str) + "_" + df[target_col].astype(str)
            )
        else:
            df = df.copy()
            df["strata"] = df[target_col].astype(str)
    strata = df["strata"]
    if strata.nunique() > len(df) * 0.9:
        stratify = df[target_col]
    else:
        stratify = strata
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=stratify, random_state=random_state
    )
    return train_df, test_df


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """TN / (TN + FP)."""
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    denom = tn + fp
    return tn / denom if denom > 0 else 0.0


def npv_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """TN / (TN + FN)."""
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    denom = tn + fn
    return tn / denom if denom > 0 else 0.0


def ppv_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Same as precision: TP / (TP + FP)."""
    return precision_score(y_true, y_pred, zero_division=0)


def evaluate_model(
    model: Any,
    X: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 0.5,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Calculate AUROC, sensitivity, specificity, F1, PPV, NPV, accuracy.
    Returns metrics dict.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:
        proba = model.predict(X).astype(float)
    y_pred = (proba >= threshold).astype(int)
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    try:
        auroc = roc_auc_score(y_true, proba)
    except Exception:
        auroc = 0.0
    return {
        "auroc": auroc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "ppv": ppv,
        "npv": npv,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": sensitivity,
    }


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray, metric: str = "f1") -> float:
    """Find threshold that maximizes metric (e.g. f1)."""
    best_thresh, best_val = 0.5, 0.0
    for t in np.linspace(0.2, 0.8, 61):
        pred = (y_proba >= t).astype(int)
        if metric == "f1":
            v = f1_score(y_true, pred, zero_division=0)
        elif metric == "sensitivity":
            tp = ((y_true == 1) & (pred == 1)).sum()
            fn = ((y_true == 1) & (pred == 0)).sum()
            v = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        else:
            v = accuracy_score(y_true, pred)
        if v > best_val:
            best_val, best_thresh = v, t
    return best_thresh


def get_models() -> Dict[str, Any]:
    """Default models for Study 1."""
    return {
        "xgboost": XGBClassifier(
            max_depth=5,
            learning_rate=0.05,
            n_estimators=200,
            scale_pos_weight=1.0,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
        ),
        "lightgbm": LGBMClassifier(
            max_depth=5,
            learning_rate=0.05,
            n_estimators=200,
            random_state=RANDOM_STATE,
            verbose=-1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=RANDOM_STATE,
        ),
        "logistic": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    }


def train_with_cv(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = CV_FOLDS,
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = True,
) -> Tuple[Any, StandardScaler, Dict[str, float], float]:
    """
    Fit scaler on X_train, run stratified K-fold CV, fit model on full train.
    Returns: fitted_model, scaler, cv_metrics (auroc_mean, auroc_std, etc.), optimal_threshold.
    """
    if scaler is None:
        scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train) if fit_scaler else scaler.transform(X_train)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    cv_aurocs = []
    for train_idx, val_idx in skf.split(X_tr, y_train):
        X_f, X_v = X_tr[train_idx], X_tr[val_idx]
        y_f, y_v = y_train[train_idx], y_train[val_idx]
        m = __clone_model(model)
        m.fit(X_f, y_f)
        proba = m.predict_proba(X_v)[:, 1]
        cv_aurocs.append(roc_auc_score(y_v, proba))
    full_proba = np.zeros_like(y_train, dtype=float)
    for train_idx, val_idx in skf.split(X_tr, y_train):
        m = __clone_model(model)
        m.fit(X_tr[train_idx], y_train[train_idx])
        full_proba[val_idx] = m.predict_proba(X_tr[val_idx])[:, 1]
    opt_threshold = find_optimal_threshold(y_train, full_proba, metric="f1")
    model.fit(X_tr, y_train)
    cv_metrics = {
        "cv_auroc_mean": float(np.mean(cv_aurocs)),
        "cv_auroc_std": float(np.std(cv_aurocs)),
    }
    return model, scaler, cv_metrics, opt_threshold


def __clone_model(model: Any) -> Any:
    from sklearn.base import clone
    try:
        return clone(model)
    except Exception:
        return model


def load_cohort_c4(
    data_path: str,
    age_min: int = 18,
    age_max: int = 55,
    balance_50_50: bool = True,
    apply_aq_filter: bool = True,
    keep_all_columns: bool = False,
) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Load C4 cohort from data/processed/data_c4_final_recreated_cleaned.csv (or provided path).
    Applies age 18-55, AQ>=6 for autism cases, optional 50/50 balance.
    If keep_all_columns=True, returned df keeps all CSV columns (e.g. aq_1..aq_10 for Study 3).
    Returns: (df with diagnosis column, feature_names_45, target_column_name).
    """
    df = pd.read_csv(data_path)
    if "autism_target" in df.columns:
        df = df.rename(columns={"autism_target": "diagnosis"})
    target_col = "diagnosis"

    if "age" in df.columns:
        df = df[(df["age"] >= age_min) & (df["age"] <= age_max)].copy()
    if apply_aq_filter and "aq_total" in df.columns:
        autism = df[df[target_col] == 1]
        non_autism = df[df[target_col] == 0]
        autism = autism[autism["aq_total"] >= 6]
        df = pd.concat([autism, non_autism], ignore_index=True)

    feat_info = load_feature_info()
    feature_names = feat_info.get("feature_names", FEATURE_NAMES_45)
    comorbidity_cols = [c for c in COMOBIDITY_COLUMNS if c in df.columns]
    if not keep_all_columns:
        for m in [f for f in feature_names if f not in df.columns]:
            df[m] = 0
        df = df[feature_names + comorbidity_cols + [target_col]].copy()
    else:
        for m in [f for f in feature_names if f not in df.columns]:
            df[m] = 0
    df = _ensure_numeric(df, feature_names)

    if balance_50_50:
        pos = df[df[target_col] == 1]
        neg = df[df[target_col] == 0]
        n = min(len(pos), len(neg))
        pos = pos.sample(n=n, random_state=RANDOM_STATE)
        neg = neg.sample(n=n, random_state=RANDOM_STATE)
        df = pd.concat([pos, neg], ignore_index=True).sample(frac=1, random_state=RANDOM_STATE)

    return df, feature_names, target_col


def load_cohort_card(
    card_path_or_df: str | pd.DataFrame,
    age_min: int = 18,
    age_max: int = 55,
    balance_50_50: bool = True,
    apply_aq_filter: bool = True,
) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Load CARD cohort from preprocessed card_aligned.csv (from card_c4_validation.ipynb)
    or a DataFrame that already has 45 C4-aligned columns + autism_target, age, aq_total.
    Applies age filter, AQ filter, optional balance. Returns (df, feature_names, target_col).
    """
    if isinstance(card_path_or_df, str) and os.path.isfile(card_path_or_df):
        card_df = pd.read_csv(card_path_or_df)
    else:
        card_df = card_path_or_df if isinstance(card_path_or_df, pd.DataFrame) else pd.DataFrame()
    if card_df.empty:
        return card_df, [], "diagnosis"
    if "autism_target" in card_df.columns:
        card_df = card_df.rename(columns={"autism_target": "diagnosis"})
    target_col = "diagnosis"
    df = card_df.copy()
    if "age" in df.columns:
        df = df[(df["age"] >= age_min) & (df["age"] <= age_max)]
    if apply_aq_filter and "aq_total" in df.columns:
        autism = df[df[target_col] == 1]
        non_autism = df[df[target_col] == 0]
        autism = autism[autism["aq_total"] >= 6]
        df = pd.concat([autism, non_autism], ignore_index=True)
    feat_info = load_feature_info()
    feature_names = feat_info.get("feature_names", FEATURE_NAMES_45)
    for f in feature_names:
        if f not in df.columns:
            df[f] = 0
    comorbidity_cols = [c for c in COMOBIDITY_COLUMNS if c in df.columns]
    df = df[feature_names + comorbidity_cols + [target_col]].copy()
    df = _ensure_numeric(df, feature_names)
    if balance_50_50:
        pos = df[df[target_col] == 1]
        neg = df[df[target_col] == 0]
        n = min(len(pos), len(neg))
        pos = pos.sample(n=n, random_state=RANDOM_STATE)
        neg = neg.sample(n=n, random_state=RANDOM_STATE)
        df = pd.concat([pos, neg], ignore_index=True).sample(frac=1, random_state=RANDOM_STATE)
    return df, feature_names, target_col


def load_cohort_ybt(
    data_path: str,
    age_min: int = 18,
    age_max: int = 55,
    balance_50_50: bool = True,
    apply_aq_filter: bool = True,
) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Load YBT (Dataset3). If path is to preprocessed ybt_aligned.csv (from external_validation_ybt.ipynb),
    loads it and applies age/AQ/balance only (45 features, SPQ=0). Otherwise loads raw YBT and
    harmonizes to 35-feature schema (no SPQ) with scoring matching external_validation_ybt.
    """
    if not os.path.isfile(data_path):
        return pd.DataFrame(), list(FEATURE_NAMES_35), "diagnosis"
    df = pd.read_csv(data_path, low_memory=False)
    # Preprocessed aligned file: 45 features + autism_target, age, aq_total
    is_aligned = (
        "ybt_aligned" in data_path
        or (len(df.columns) >= 45 and "autism_target" in df.columns and "age" in df.columns and all(f in df.columns for f in FEATURE_NAMES_45[:5]))
    )
    if is_aligned:
        if "autism_target" in df.columns:
            df = df.rename(columns={"autism_target": "diagnosis"})
        target_col = "diagnosis"
        df = df[(df["age"] >= age_min) & (df["age"] <= age_max)].copy()
        if apply_aq_filter and "aq_total" in df.columns:
            autism = df[df[target_col] == 1]
            non_autism = df[df[target_col] == 0]
            autism = autism[autism["aq_total"] >= 6]
            df = pd.concat([autism, non_autism], ignore_index=True)
        for f in FEATURE_NAMES_45:
            if f not in df.columns:
                df[f] = 0
        comorbidity_cols = [c for c in COMOBIDITY_COLUMNS if c in df.columns]
        df = df[FEATURE_NAMES_45 + comorbidity_cols + [target_col]].copy()
        df = _ensure_numeric(df, FEATURE_NAMES_45)
        if balance_50_50:
            pos = df[df[target_col] == 1]
            neg = df[df[target_col] == 0]
            n = min(len(pos), len(neg))
            if n > 0:
                pos = pos.sample(n=n, random_state=RANDOM_STATE)
                neg = neg.sample(n=n, random_state=RANDOM_STATE)
                df = pd.concat([pos, neg], ignore_index=True).sample(frac=1, random_state=RANDOM_STATE)
        return df, list(FEATURE_NAMES_45), target_col

    # Raw YBT: parse comorbidity from diagnosis text before overwriting diagnosis
    if "diagnosis" in df.columns and df["diagnosis"].dtype == object:
        diag_str = df["diagnosis"].astype(str)
        df["has_adhd"] = diag_str.str.contains("ADHD", case=False, na=False).astype(int)
        df["has_anxiety"] = diag_str.str.contains("Anxiety", case=False, na=False).astype(int)
        df["has_depression"] = diag_str.str.contains("Depression", case=False, na=False).astype(int)

    if "autism_target" in df.columns:
        df = df.rename(columns={"autism_target": "diagnosis"})
    elif "diagnosis_yes_no" in df.columns and "diagnosis" in df.columns:
        autism_keywords = ["autism", "asd", "asperger", "autistic"]
        df["diagnosis"] = df["diagnosis"].astype(str).str.contains(
            "|".join(autism_keywords), case=False, na=False
        ).astype(int)
    target_col = "diagnosis"

    eq_map = {f"eq10_{i}": f"eq_{i}" for i in range(1, 11)}
    sq_map = {f"sq10_{i}": f"sqr_{i}" for i in range(1, 11)}
    df = df.rename(columns={**eq_map, **sq_map})
    for i in range(1, 11):
        if f"eq_{i}" not in df.columns and f"eq10_{i}" in df.columns:
            df[f"eq_{i}"] = df[f"eq10_{i}"]
        if f"sqr_{i}" not in df.columns and f"sq10_{i}" in df.columns:
            df[f"sqr_{i}"] = df[f"sq10_{i}"]

    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df = df[df["age"].notna() & (df["age"] >= age_min) & (df["age"] <= age_max)]

    # YBT raw stores EQ/SQ-R/AQ as text ("strongly agree", etc.). Map to 1-4 then apply C4 binary scoring.
    _ybt_map = {"strongly agree": 4, "slightly agree": 3, "slightly disagree": 2, "strongly disagree": 1}

    def _ybt_to_14(ser: pd.Series) -> pd.Series:
        out = ser.astype(str).str.strip().str.lower().replace(_ybt_map)
        return pd.to_numeric(out, errors="coerce")

    for i in range(1, 11):
        for col, reverse in [
            (f"eq_{i}", i == 3),
            (f"sqr_{i}", i in (2, 4, 6, 8, 10)),
            (f"aq_{i}", i in (2, 3, 4, 5, 6, 9)),
        ]:
            if col not in df.columns:
                continue
            if df[col].dtype == object:
                raw = _ybt_to_14(df[col])
                if reverse:
                    df[col] = raw.apply(lambda x: 1 if pd.notna(x) and 1 <= x <= 2 else (0 if pd.notna(x) and 3 <= x <= 4 else np.nan))
                else:
                    df[col] = raw.apply(lambda x: 1 if pd.notna(x) and 3 <= x <= 4 else (0 if pd.notna(x) and 1 <= x <= 2 else np.nan))
                df[col] = df[col].fillna(0)
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["eq_total"] = df[[f"eq_{i}" for i in range(1, 11) if f"eq_{i}" in df.columns]].sum(axis=1)
    df["sqr_total"] = df[[f"sqr_{i}" for i in range(1, 11) if f"sqr_{i}" in df.columns]].sum(axis=1)
    df["aq_total"] = df[[f"aq_{i}" for i in range(1, 11) if f"aq_{i}" in df.columns]].sum(axis=1)
    df["d_score"] = df["sqr_total"] - df["eq_total"]
    df["eq_sqr_ratio"] = df["eq_total"] / (df["sqr_total"].replace(0, np.nan).fillna(1) + 1)
    df["sqrt_age"] = np.sqrt(df["age"])
    df["age_x_eq"] = df["age"] * df["eq_total"]
    df["is_stem_occupation"] = 0
    sex_map = {"male": 0, "female": 1, "m": 0, "f": 1, "Male": 0, "Female": 1}
    df["sex_num"] = df["sex"].astype(str).str.strip().str.lower().map(sex_map).fillna(0)
    if "sex" not in df.columns:
        df["sex"] = df["sex_num"]

    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 18, 30, 45, 60, 100],
        labels=["0-18", "19-30", "31-45", "46-60", "61+"],
    )
    dummies = pd.get_dummies(df["age_group"], prefix="age_group").astype(int)
    for c in ["age_group_19-30", "age_group_31-45", "age_group_46-60", "age_group_61+"]:
        df[c] = dummies[c] if c in dummies.columns else 0

    feature_names = list(FEATURE_NAMES_35)
    for f in feature_names:
        if f not in df.columns:
            df[f] = 0
    comorbidity_cols = [c for c in COMOBIDITY_COLUMNS if c in df.columns]
    df = df[feature_names + comorbidity_cols + [target_col]].copy()
    df = _ensure_numeric(df, feature_names)

    if apply_aq_filter and "aq_total" in df.columns:
        autism = df[df[target_col] == 1]
        non_autism = df[df[target_col] == 0]
        autism = autism[autism["aq_total"] >= 6]
        df = pd.concat([autism, non_autism], ignore_index=True)

    if balance_50_50:
        pos = df[df[target_col] == 1]
        neg = df[df[target_col] == 0]
        n = min(len(pos), len(neg))
        if n == 0:
            return df, FEATURE_NAMES_35, target_col
        pos = pos.sample(n=n, random_state=RANDOM_STATE)
        neg = neg.sample(n=n, random_state=RANDOM_STATE)
        df = pd.concat([pos, neg], ignore_index=True).sample(frac=1, random_state=RANDOM_STATE)

    return df, FEATURE_NAMES_35, target_col


def bootstrap_ci_auroc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> Tuple[float, float]:
    """Bootstrap 95% CI for AUROC."""
    rng = np.random.RandomState(RANDOM_STATE)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        try:
            scores.append(roc_auc_score(y_true[idx], y_proba[idx]))
        except Exception:
            pass
    scores = np.array(scores)
    alpha = 1 - ci
    return float(np.percentile(scores, 100 * alpha / 2)), float(np.percentile(scores, 100 * (1 - alpha / 2)))
