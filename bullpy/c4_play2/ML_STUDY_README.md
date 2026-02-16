# ML Autism Prediction Study (Three Studies)

Rigorous within-cohort and subgroup design for predicting autism from self-report questionnaires across three cohorts: C4, CARD, Dataset3 (YBT).

## Data and paths

Studies **reuse preprocessed outputs** from existing notebooks (no duplicated preprocessing):

- **C4**: `data/processed/data_c4_final_recreated_cleaned.csv` from **data_pipeline_recreation.ipynb**.
- **CARD**: `data/processed/card_aligned.csv` from **card_c4_validation.ipynb** (that notebook uses `CARD_PATH` and now saves the 45-feature aligned dataframe at the end of the “Build 45-feature matrix” step).
- **Dataset3 (YBT)**: `data/processed/ybt_aligned.csv` from **external_validation_ybt.ipynb** if present; otherwise raw YBT is loaded from `YBT_PATH` (or default OneDrive path) and preprocessed in `study_utils.load_cohort_ybt`.

All cohorts: age 18–55, AQ ≥ 6 for autism cases only, optional 50/50 balance.

## Shared code

- **`src/study_utils.py`**: `load_cohort_c4`, `load_cohort_card`, `load_cohort_ybt`, `create_stratified_split`, `train_with_cv`, `evaluate_model`, `get_models`, feature schemas (45 with SPQ, 35 without), bootstrap CI, subgroup helpers.

## Study 1: Within-cohort cross-validation

**Notebook**: `notebooks/study1_within_cohort_cv.ipynb`

- Load C4, CARD (if `card_aligned.csv` exists), YBT; apply age 18–55, AQ filter, 50/50 balance.
- Stratified 80/20 train/test; 5-fold CV on train; evaluate on test.
- Models: XGBoost, LightGBM, Random Forest, Logistic Regression.
- Saves: `results/study1_within_cohort/{c4,card,dataset3}/` (models, scaler, performance_metrics.json, cv_scores.json, feature_importance.csv), plus `comparison_table.csv` and optional ROC figure.

## Study 2: Subgroup analysis

**Notebook**: `notebooks/study2_subgroup_analysis.ipynb`

- Same cohorts (no balance to keep subgroup sizes); add study age strata (18–30, 31–40, 41–50, 51–55).
- Age- and sex-stratified performance (XGBoost, 5-fold CV, AUROC and 95% CI).
- Writes: `results/study2_subgroups/age_stratified/`, `sex_stratified/`, `subgroup_comparison_table.csv`, `subgroup_forest_plot.png`.

## Study 3: Feature set comparison and SPQ contribution

**Notebook**: `notebooks/study3_feature_comparison.ipynb`

- Feature sets: demographics, aq_only, eq_sq_only, spq_only (C4/CARD), all_no_aq, all_features.
- 5-fold CV AUROC/F1 per cohort and set; SPQ contribution (with vs without SPQ) for C4 and CARD.
- Optional SHAP summary (if `shap` is installed).
- Saves: `results/study3_features/feature_comparison_table.csv`, `spq_contribution_analysis.json`, `shap_values/` (optional).

## Why the aligned CSVs exist (and how to get them)

All three notebooks write into **`c4_play2/data/processed/`**. The study notebooks look for files there. If a file is missing, either that notebook was not run to completion or the save step was not executed.

| File | Produced by | When it’s written |
|------|-------------|--------------------|
| `data_c4_final_recreated_cleaned.csv` | **data_pipeline_recreation.ipynb** | After you run the full pipeline (Steps A–F, balance, AQ filter, clean). Uses path `data/processed/...` **relative to the notebook’s current working directory**. So you must run the notebook with **working directory = project root** (`c4_play2`), e.g. start Jupyter from `c4_play2` so `data/processed/` is `c4_play2/data/processed/`. |
| `card_aligned.csv` | **card_c4_validation.ipynb** | In the **same cell** that builds the 45-feature matrix (after “Summary: all 45 C4 features”). That cell ends with a block that saves to `REPO_ROOT/data/processed/card_aligned.csv`. Run the notebook **from start through that cell**; ensure `CARD_PATH` is set and the previous aggregation/AQ/balance cells have run so `X_card` and `df_agg` exist. |
| `ybt_aligned.csv` | **external_validation_ybt.ipynb** | In a **separate cell** right after “STEP 8: Feature alignment to C4 schema” (the one that prints “Feature alignment complete - ready for scaling”). The next cell saves to `.../data/processed/ybt_aligned.csv`. Run the notebook through **that save cell**; the path is based on `ARTIFACT_DIR`, so it does not depend on your working directory. |

**How to verify**

From the project root:

```bash
ls -la data/processed/data_c4_final_recreated_cleaned.csv data/processed/card_aligned.csv data/processed/ybt_aligned.csv
```

- If **C4** is missing: run **data_pipeline_recreation.ipynb** from top to bottom with Jupyter started in `c4_play2` (so `data/processed` is under the project).
- If **card_aligned.csv** is missing: run **card_c4_validation.ipynb** from the top through the cell that builds `X_card` and saves `card_aligned.csv` (the cell that prints “Saved CARD aligned dataset to ...”).
- If **ybt_aligned.csv** is missing: run **external_validation_ybt.ipynb** through the cell that prints “Saved YBT aligned dataset to ...”; it is the cell immediately after “Feature alignment complete - ready for scaling”.

## Running

1. **Produce preprocessed data once**: run **data_pipeline_recreation.ipynb** (C4), **card_c4_validation.ipynb** (CARD), and **external_validation_ybt.ipynb** (YBT) as above so that `c4_play2/data/processed/` contains `data_c4_final_recreated_cleaned.csv`, `card_aligned.csv`, and `ybt_aligned.csv` as needed.
2. Run study notebooks 1 → 2 → 3 from repo root or `notebooks/`. They load from those paths; YBT falls back to raw `YBT_PATH` if `ybt_aligned.csv` is missing.

## Success criteria (from spec)

- **Minimum**: AUROC ≥ 0.85 in at least 2/3 cohorts; cohort AUROC difference < 0.10; interpretable subgroup patterns; feature comparison shows AQ highly predictive (AUROC ≥ 0.90 when used).
- **Target**: AUROC ≥ 0.88 in all three; replication difference < 0.05; SPQ contribution about +0.05–0.10 AUROC (significant).
