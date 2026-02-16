# Study 1: Data Leakage and Score Robustness Verification

This document confirms that the Study 1 within-cohort CV pipeline has **no data leakage** and that the reported test scores are **accurate and robust** for C4 and CARD. Dataset3 (YBT) test metrics are unbiased but **not robust** due to small sample and model behaviour.

---

## 1. No data leakage

### 1.1 Target not used as a feature

- The target is `diagnosis` (binary). It is **not** in the feature list.
- `load_cohort_c4`, `load_cohort_card`, and `load_cohort_ybt` all return dataframes with exactly `feature_names + [target_col]`. The model inputs are `df[feature_names]`; the target is only `df[target_col]`.
- **AQ is not a predictor:** AQ items and `aq_total` are **not** in `FEATURE_NAMES_45` or `FEATURE_NAMES_35`. AQ is used only as an inclusion filter (AQ >= 6 for autism cases), not as a model input.

So there is no target leakage via AQ or diagnosis in the feature set.

### 1.2 Strict train/test split

- `create_stratified_split(df, target_col=target_col)` is called **once** per cohort. It uses `train_test_split(..., test_size=0.2, stratify=stratify, random_state=42)`.
- `train_df` and `test_df` are disjoint. All downstream steps use:
  - `X_train`, `y_train` from `train_df`
  - `X_test`, `y_test` from `test_df`
- The test set is never used for fitting. Reported **Test_AUROC**, **Test_F1**, **Test_Sens**, **Test_Spec** are computed only on this held-out test set.

So test data are never used for training or for choosing the model.

### 1.3 Scaler fit only on training data

- In `run_cohort`, a single `StandardScaler` is created. The **first** model calls `train_with_cv(..., scaler=None, fit_scaler=True)`.
- In `train_with_cv`, when `fit_scaler` is True, the scaler is fit on **X_train only**: `X_tr = scaler.fit_transform(X_train)`.
- Test data are only **transformed**: `X_test_s = scaler.transform(X_test)`. Test is never used in `fit` or `fit_transform`.
- The same scaler (fit on train) is reused for all four models in that cohort.

So there is no leakage from test into the scaling step.

### 1.4 Optimal threshold from CV on training data only

- The decision threshold is chosen by `find_optimal_threshold(y_train, full_proba, metric="f1")`.
- `full_proba` is the 5-fold CV predicted probabilities on the **training set** (each fold’s validation predictions). So the threshold is chosen using only training/CV data.
- The same threshold is then applied to the test set for computing sensitivity, specificity, F1, etc. Test labels are not used to pick the threshold.

So there is no threshold leakage from the test set.

### 1.5 Cross-validation uses only training data

- Stratified 5-fold CV is run on `X_train`, `y_train` only (`skf.split(X_tr, y_train)`).
- CV is used for: (a) reporting `cv_auroc_mean` / `cv_auroc_std`, and (b) building `full_proba` for the optimal threshold. No test data are involved.

**Conclusion:** There is no data leakage in the Study 1 evaluation pipeline. Test metrics are unbiased estimates of performance on held-out data.

---

## 2. Accuracy and robustness of the reported scores

### 2.1 What “accurate” and “robust” mean here

- **Accurate:** The reported numbers (e.g. Test_AUROC 0.916 for C4 XGBoost) are correct given the current split and pipeline; they are not inflated by leakage.
- **Robust:** The evaluation is stable and generalisable (same cohort, different splits or external data would give similar performance).

### 2.2 C4 and CARD

- **Accurate:** Yes. Test set is held out, scaler and threshold are train-only, no target or AQ in features.
- **Robust:** Reasonably so.
  - Large samples (C4 ~26.7k, CARD ~5.3k after your age change), 80/20 split, stratified by target (and strata when available).
  - Fixed `random_state=42` so the split is reproducible.
  - CV AUROC (on train) and Test AUROC are in a similar range (e.g. C4 ~0.91), which suggests the test estimate is not a fluke of one split.
  - Your reported scores (e.g. C4 XGBoost Test_AUROC 0.916, CARD 0.901) are consistent with within-cohort generalisation for these cohorts.

So for **C4 and CARD**, the scores are **accurate and reasonably robust**.

### 2.3 Dataset3 (YBT)

- **Accurate:** Yes. Same pipeline: no leakage; test metrics are unbiased for that single split.
- **Robust:** No.
  - Small sample (~1.2k after balance), so a 20% test set is small and metrics are high-variance.
  - Test AUROC ~0.47–0.55 (near chance) with sensitivity ~1.0 and specificity 0.0: the model is effectively predicting “positive” for almost everyone. So the **numbers** are correct for this run, but they do **not** represent a useful or stable predictor. Interpretation: the model has not learned a generalisable rule on YBT (and/or there is distribution/label mismatch when using raw YBT).

So for **Dataset3**, the scores are **accurate but not robust**; they should not be interpreted as evidence of good performance.

---

## 3. Summary table

| Check                          | Result |
|--------------------------------|--------|
| Target / AQ in features?       | No     |
| Train/test split strict?      | Yes    |
| Scaler fit on train only?     | Yes    |
| Threshold from test?           | No (from CV on train) |
| Test metrics on held-out only? | Yes    |
| C4/CARD scores accurate?      | Yes    |
| C4/CARD scores robust?        | Reasonably yes |
| Dataset3 scores accurate?     | Yes (for this split) |
| Dataset3 scores robust?       | No (small n, poor calibration) |

**Bottom line:** There is no data leakage. C4 and CARD test scores (e.g. AUROC ~0.90–0.92) are accurate and reasonably robust. Dataset3 test scores are accurate for the current split but not robust or generalisable; treat them as “no useful signal” rather than as good performance.
