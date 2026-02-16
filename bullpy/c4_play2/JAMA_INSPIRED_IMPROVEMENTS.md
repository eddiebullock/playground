# Improving the Experiment: Plan Inspired by the JAMA Paper

This document outlines concrete changes to improve model performance and impact, taking inspiration from "Machine Learning Prediction of Autism Spectrum Disorder From a Minimal Set of Medical and Background Information" (JAMA Network Open 2024).

---

## 1. Study design and reporting (TRIPOD-style)

**What JAMA did:** Followed TRIPOD guidelines; clear cohort definitions; 60% train / 20% validation / 20% test; validation set for tuning then merged for final model; 10-fold CV for reported metrics with mean and SD.

**What to do:**

- **Define cohorts explicitly:** "C4 = [source, N, age range, inclusion]. CARD = [source, N, age range]. External validation = CARD only (no refit)."
- **Split C4 into train/validation/test (e.g. 60/20/20):** Use validation for threshold and hyperparameter tuning; report test-set metrics only for final model. Optionally report mean (SD) over 10-fold CV on the test set for AUROC/F1.
- **Pre-register or document:** "Primary outcome: AUROC on held-out CARD. Secondary: sensitivity, specificity, PPV, F1, calibration."

---

## 2. Minimal, reproducible feature set

**What JAMA did:** 28 variables from "basic medical screening and background history" – easily obtainable, non-invasive. Clear eTable of all predictors.

**What to do:**

- **Define a minimal set:** e.g. "SPQ-10 total, EQ-10 total, SQR-10 total (or d_score), age, sex, plus optionally AQ-10 total" – all obtainable from short self-report. List exact items and scoring (or cite CARD_VALIDATION_SPEC.md).
- **Two models in the paper:**
  - **Model A (conservative):** No AQ in model input (current 45-feature setup). Emphasises non-circularity.
  - **Model B (best prediction):** Include AQ (and age_x_aq if used in C4). Emphasises "best performance from screening questionnaires."
- **Document feature mapping** across cohorts (C4 vs CARD) in a supplement table so replication is clear.

---

## 3. Address age (and sex) skew for generalisability

**What JAMA did:** Reported demographics by group; stratified analysis by age and sex.

**What to do:**

- **Restrict age range for training and validation:** e.g. 18–55 or 18–60 so both C4 and CARD are in a "supported" range (C4 has almost no one >60). Apply same range to CARD when validating.
- **Stratified splits:** When creating train/validation/test (and when doing 50/50 balance), stratify by age band (e.g. 18–25, 26–35, 36–50, 51+) and optionally sex so each set has a similar mix. This reduces overfitting to young adults.
- **Oversample or weight older participants** in training (e.g. sample 36–50 and 51+ with replacement so they are better represented), or use sample weights inverse to age-group frequency.
- **Report performance by age band and sex** on CARD (e.g. AUROC for 18–30, 31–45, 46+; and by sex). State clearly where the model is not supported by data (e.g. 60+).

---

## 4. Model development (tuning and selection)

**What JAMA did:** 4 algorithms (logistic regression, decision tree, random forest, XGBoost); 10-fold CV; Bayesian hyperparameter optimisation; early stopping for XGBoost; chose best model (XGBoost) for further testing.

**What to do:**

- **Keep 4–5 algorithms** (e.g. logistic regression, random forest, XGBoost, LightGBM, gradient boosting). Use a **held-out validation set** (20% of C4) for:
  - Hyperparameter tuning (e.g. GridSearchCV or Bayesian optimisation on validation).
  - Threshold optimisation (F1 or balanced accuracy) on validation.
- **Report mean (SD) AUROC** over 10-fold CV on the **test** set (final 20% of C4) so readers see stability.
- **Pick one best model** (e.g. XGBoost or LightGBM) for external validation on CARD; report that model's metrics and calibration. Optionally report the others in a supplement.

---

## 5. External validation that can "get better scores" and stay impactful

**What JAMA did:** Tested on SPARK v10 (same ecosystem) and SSC (different cohort, ASD-only). Main "external" success was same-ecosystem (SPARK v8 → v10).

**What to do:**

- **Primary analysis (rigorous):** Train on C4 only → validate on CARD (no refit). Report AUROC, sensitivity, specificity, PPV, F1, 95% CIs (e.g. DeLong for AUROC), and calibration plot. This is your main scientific result; if it's modest, that's an honest, impactful finding.
- **Secondary analysis (better scores, still valid):** **Pooled training:** Train on C4 + CARD (or C4 + CARD + YBT if available), with a **held-out test set from CARD** (e.g. 20% of CARD never used in training). Report metrics on that held-out CARD. This mirrors "training on one wave, testing on another" and often improves CARD performance while still being a fair test.
- **Optional:** If you have a second external cohort (e.g. YBT), report sensitivity/specificity there too (even if only one class is available, report what you can, as JAMA did for SSC).

---

## 6. Metrics and calibration

**What JAMA did:** AUROC with 95% CI (DeLong); accuracy, sensitivity, specificity, PPV, F1; calibrated PPV and F1 for imbalance.

**What to do:**

- **On CARD:** Report AUROC (95% CI), accuracy, sensitivity, specificity, PPV, NPV, F1. Use the same threshold(s) chosen on C4 validation (no tuning on CARD).
- **Calibration:** Plot predicted probability vs observed proportion (calibration curve) on CARD. If the model is poorly calibrated, report it and optionally add a note that calibration could be refit on a small CARD subset for deployment (without retraining the model).
- **Class imbalance:** If you ever report on an imbalanced CARD subset (e.g. natural prevalence), report calibrated PPV/NPV or sensitivity/specificity so the numbers are interpretable.

---

## 7. Interpretability and phenotype associations

**What JAMA did:** SHAP for feature importance; phenotype associations (CBCL, FSIQ, SCQ) for correctly vs incorrectly predicted groups.

**What to do:**

- **SHAP (or equivalent):** Compute SHAP values for the best model on C4 (and optionally on CARD). Report top 10–15 predictors (e.g. SPQ_total, EQ_total, d_score, age, sex) in a figure or table.
- **Phenotype associations on CARD:** For individuals with ASD: compare mean (SD) SPQ_total, EQ_total, SQR_total, AQ_total (and age, sex) in "correctly predicted ASD" vs "incorrectly predicted as non-ASD." For non-ASD: compare "correctly predicted non-ASD" vs "incorrectly predicted as ASD." Use simple tests (e.g. Wilcoxon, t-test) and report P values. This shows "who does the model miss?" and strengthens the paper.

---

## 8. Framing and title for impact

**What JAMA did:** "Machine Learning Prediction of Autism Spectrum Disorder From a **Minimal Set** of Medical and Background Information" – clear, minimal, screening-oriented.

**What to do:**

- **Title idea:** "Prediction of autism diagnosis from a minimal set of self-report questionnaires: development in C4 and external validation in CARD" (or similar). Emphasise "minimal set," "self-report," and "external validation."
- **Abstract:** (1) Objective: Can we predict adult autism from short questionnaires and generalise to an independent cohort? (2) Methods: C4 development with 60/20/20 and stratification; CARD external validation, no refit; two models (with/without AQ). (3) Results: In-domain AUROC X.XX; external (CARD) AUROC X.XX; pooled training AUROC X.XX. (4) Conclusion: ML predicts autism from questionnaires in-domain; generalisation to CARD was [modest/improved with pooled training]; we recommend [validation in target population or pooled training].
- **Key message:** "We provide a rigorous external validation and show that generalisation depends on cohort similarity; we show how pooled training can improve external performance."

---

## 9. Implementation checklist (priority order)

| Priority | Task | Where |
|----------|------|--------|
| 1 | Restrict age to 18–55 (or 18–60) in C4 and CARD for training and validation | data_pipeline_recreation (after filtering); card_c4_validation |
| 2 | Stratify train/validation/test and 50/50 balance by age band (and sex) | data_pipeline_recreation (split and balance steps) |
| 3 | Implement 60/20/20 split; use validation for tuning only; report test metrics with 10-fold CV mean (SD) | data_pipeline_recreation |
| 4 | Add "Model B" with AQ in features; train and save; run CARD validation for both Model A and B | data_pipeline_recreation + card_c4_validation |
| 5 | Report 95% CIs (e.g. DeLong AUROC) and calibration plot on CARD | card_c4_validation (or evaluate_card_results) |
| 6 | Pooled training: train on C4 + CARD (80%), test on held-out CARD (20%); report metrics | New notebook or section in data_pipeline_recreation |
| 7 | SHAP for best model; phenotype table (SPQ/EQ/SQR/AQ by prediction correctness) on CARD | New cells or evaluate_card_results |
| 8 | Write methods/results to match TRIPOD; add supplement with feature list and cohort flow | Paper draft |

---

## 10. Why this can improve scores and impact

- **Better scores:** Age restriction + stratification + pooled training (and optionally Model B with AQ) can improve CARD AUROC and calibration while keeping the design interpretable and publishable.
- **Impact:** Rigorous external validation (C4 → CARD), transparent reporting (TRIPOD-style), two models (with/without AQ), and a clear message about generalisation and how to improve it (pooled training, target-population validation) make the paper a strong, credible contribution that others can build on.
