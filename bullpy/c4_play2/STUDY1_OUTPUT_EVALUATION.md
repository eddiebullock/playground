# Study 1 (Within-Cohort CV) Output Evaluation

Evaluation of `notebooks/study1_within_cohort_cv.ipynb` outputs and answers to your specific questions.

---

## 1. Why is the C4 dataset only ~26k? Wasn't it ~30k?

**Short answer:** The C4 *source* file has **35,662 rows**. You get **26,070** because of Study 1’s filters and 50/50 balancing.

**Step-by-step:**

| Step | What happens | Rows |
|------|----------------|-----|
| Source file | `data_c4_final_recreated_cleaned.csv` | **35,662** |
| Age 18–55 | Keep only this age range | **27,912** |
| AQ >= 6 for autism | Keep only autism cases with AQ_total >= 6 (non-autism unchanged) | Autism: 13,035; Non-autism: 14,877 |
| 50/50 balance | Sample down to min(autism, non-autism) in each class | **13,035 + 13,035 = 26,070** |

So the “30k” you remember is in the right ballpark of the *raw* cleaned C4 (~35.6k). The 26k is by design: after age and AQ filters, the smaller class (autism with AQ>=6) has 13,035, so the balanced cohort is 2 × 13,035 = **26,070**. No rows are dropped by mistake; the reduction is from age filter, AQ filter, and balancing.

---

## 2. Why is YBT only ~1k?

**Short answer:** After the same filters (age 18–55, AQ>=6 for autism, 50/50 balance), the **smaller class has only 549 people**, so the balanced cohort is **549 + 549 = 1,098**.

- Raw YBT: **~24,205** rows.
- After age 18–55 and diagnosis/AQ logic, one of the two classes (autism vs non-autism) has only **549** individuals; the other is larger.
- Study 1 then balances 50/50 by taking 549 from each class → **1,098** total.

So YBT is ~1k because of the **combination** of: (1) age 18–55, (2) AQ>=6 for autism cases only, and (3) 50/50 balance, which caps the cohort at twice the size of the smaller class (549). The raw YBT is ~24k; the strict filters and balance explain the drop to ~1k.

---

## 3. What features are used for each model?

**C4 and CARD (same schema):** **45 features** — `FEATURE_NAMES_45` in `study_utils.py`:

- Demographics: `age`, `sex`, `sex_num`, `sqrt_age`, `is_stem_occupation`, `age_group_19-30`, `age_group_31-45`, `age_group_46-60`, `age_group_61+`
- SPQ: `spq_1` … `spq_10`, `spq_total`
- EQ: `eq_1` … `eq_10`, `eq_total`
- SQ-R: `sqr_1` … `sqr_10`, `sqr_total`
- Derived: `d_score`, `age_x_eq`, `eq_sqr_ratio`

**Dataset3 (YBT):** **34 features** — `FEATURE_NAMES_35` (45 minus 11 SPQ-related columns; YBT has no SPQ, so those are filled with 0 and still included in the list, but the *effective* model inputs are the non-SPQ ones):

- Same as above **except** no SPQ items (no `spq_1`…`spq_10`, `spq_total`). So: demographics, EQ-10, SQ-R-10, and the same derived terms (`eq_total`, `sqr_total`, `d_score`, `sqrt_age`, `age_x_eq`, `eq_sqr_ratio`, etc.).

**AQ is not a predictor in any cohort:** AQ items and `aq_total` are **not** in `FEATURE_NAMES_45` or `FEATURE_NAMES_35`. They are used only for the **AQ>=6 inclusion filter** for autism cases (see below).

---

## 4. Are AQ cutoffs used or is AQ a predictor?

**AQ is used only as a cutoff (inclusion criterion), not as a model feature.**

- In `study_utils.py`, `load_cohort_c4`, `load_cohort_card`, and `load_cohort_ybt` all support `apply_aq_filter=True` (the default in Study 1).
- When `apply_aq_filter` is True and `aq_total` exists:
  - Among **autism** rows only, they keep only those with **aq_total >= 6**.
  - Non-autism rows are kept regardless of AQ.
- The 45- and 35-feature lists do **not** include `aq_1`…`aq_10` or `aq_total`, so **AQ is not a predictor** in any of the models.

So: **AQ cutoff (>=6 for autism cases) is used; AQ is not used as a predictor.**

---

## 5. Summary of run outputs

**Cohort sizes (after filters and balance):**

- **C4:** 26,070 (45 features); diagnosis 1: 13,035, 0: 13,035.
- **CARD:** 4,924 (45 features); 2,462 / 2,462.
- **Dataset3 (YBT):** 1,098 (34 features from raw); 549 / 549.

**Performance (from notebook comparison table):**

- **C4:** Test AUROC ~0.91 (XGB/LightGBM/RF/Logistic all ~0.89–0.91); sensitivity ~0.87–0.89; specificity ~0.77–0.79.
- **CARD:** Test AUROC ~0.89–0.89; sensitivity ~0.85–0.88; specificity ~0.75–0.79.
- **Dataset3 (YBT):** Test AUROC ~0.43–0.49 (near chance); sensitivity ~0.96–1.0; specificity ~0.00–0.01.

So C4 and CARD look good; Dataset3 is effectively predicting almost everyone as positive (high sensitivity, near-zero specificity), which suggests the model is not generalising on YBT—likely related to small sample (1k), different population, or label/feature alignment when using raw YBT instead of `ybt_aligned.csv`.

---

## 6. Feature importance (illustrative)

- **C4:** Feature importance (e.g. XGBoost) is spread across EQ, SQ-R, SPQ, and demographics; e.g. `eq_1`, `eq_4`, `eq_10` among the stronger.
- **Dataset3:** Importance is concentrated on a few columns (e.g. `age`, `sex_num`, `age_group_31-45`); EQ/SQR contributions are zero or negligible, consistent with the poor AUROC and the model defaulting to “positive” for almost everyone.

If you want, we can add a short “Recommendations” section (e.g. re-run YBT with `ybt_aligned.csv` when available, or relax AQ/age for YBT to get a larger cohort for exploration).
