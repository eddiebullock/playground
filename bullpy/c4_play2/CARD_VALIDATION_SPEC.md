# CARD External Validation Specification

This document captures everything needed to recreate the CARD external validation pipeline elsewhere. The goal is to **validate the high-accuracy models from the C4 training set on the CARD dataset**, using the same preprocessing, feature set, and evaluation methodology.

---

## Goals

1. **Validate C4-trained models on CARD**: Apply the saved C4 models (and only those) to CARD data; do not retrain or refit on CARD.
2. **Convert full questionnaires to short 10-item versions**: Map AQ-50, EQ-60, SQ-R-75, SPQ-92 to AQ-10, EQ-10, SQ-R-10, SPQ-10 using validated item mappings.
3. **Reconstruct the same 45 features as C4**: Build exactly the feature set used in the C4 experiment (no AQ in model input; AQ used only for filtering).
4. **Create a 50/50 autism vs non-autism dataset**: After exclusions, downsample so that the number of autism cases equals the number of non-autism cases (e.g. 508:508).
5. **Exclude autism cases with AQ < 6**: Match C4 preprocessing by removing from the autism group any individual who scores below 6 on the AQ-10 (C4 excluded such cases for a cleaner, more consistent dataset).

---

## Data Source and Structure

- **Source**: CARD dataset (e.g. `CARD_Nov2025(Sheet1).csv`). Path is configurable.
- **Structure**: One row per questionnaire completion (multiple rows per participant). Key columns:
  - **VolunteerID**: Participant identifier.
  - **TestName**: e.g. "AQ", "Adolescent AQ", "EQ", "SQ-R", "SPQ".
  - **Itemised Score**: CSV string of item responses (e.g. "1,2,3,4,...").
  - **ASC diagnosis**: Diagnosis (used for target: autism vs non-autism).
  - **YearOfBirth**, **AgeWhenTestCompleted**: Used to derive age.
  - **Sex**: 1=male, 2=female, 3=other, 4=prefer not to say (or similar).
  - **Occupation** (optional): Used for `is_stem_occupation`.

Aggregation is **per participant**: one row per `VolunteerID`, with questionnaire items taken from the highest-priority test type (adult > adolescent > child).


---

## Item Mappings (Full → Short 10-Item)

All indices below are **0-based** (position in the full questionnaire’s item list).

| Questionnaire | Full length | Short 10-item indices (0-based) | Source |
|---------------|-------------|----------------------------------|--------|
| **AQ-10** from AQ-50 | 50 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (items 1–10) | Allison et al. (2012) JAACAP |
| **EQ-10** from EQ-60 | 60 | [13, 3, 8, 30, 27, 34, 11, 21, 17, 33] (1-based: 14, 4, 9, 31, 28, 35, 12, 22, 18, 34) | Greenberg et al. (2018) PNAS |
| **SQ-R-10** from SQ-R-75 | 75 | [31, 15, 26, 8, 29, 32, 11, 24, 7, 6] (1-based: 32, 16, 27, 9, 30, 33, 12, 25, 8, 7) | Greenberg et al. (2018) PNAS |
| **SPQ-10** from SPQ-92 | 92 | [1, 20, 31, 34, 37, 57, 61, 72, 73, 87] (1-based: 2, 21, 32, 35, 38, 58, 62, 73, 74, 88) | Greenberg et al. (2018) PNAS |

Parse `Itemised Score` per row; detect questionnaire type by row’s `TestName` and length of item list; then extract positions `short_items` and assign to `aq_1`–`aq_10`, `eq_1`–`eq_10`, `sqr_1`–`sqr_10`, `spq_1`–`spq_10`.

---

## Test Priority (Aggregation)

When multiple versions exist (e.g. AQ, Adolescent AQ, Child AQ), prefer **adult** over adolescent over child:

- `aq`: 1 (adult), 2 (adolescent), 3 (child)  
- Same for `eq`, `sqr`, `spq` (and `sq` maps to sqr).  
For each test type, take the row with **lowest priority number** (e.g. adult AQ over adolescent AQ).

---

## Scoring Rules (Match C4)

- **SPQ-10**: Raw 1–4 → score = `4 - raw` (0–3 per item). Sum → `spq_total` (0–30).
- **EQ-10**: 1–4 scale. Reverse item **3** (1-indexed): disagree (1,2)=1, agree (3,4)=0; others: agree=1, disagree=0. Sum → `eq_total` (0–10).
- **SQR-10**: Reverse items **2, 4, 6, 8, 10** (1-indexed): disagree=1, agree=0; others: agree=1, disagree=0. Sum → `sqr_total` (0–10; notebook also reports 0–9 depending on scoring).
- **AQ-10**: Reverse items **2, 3, 4, 5, 6, 9** (1-indexed): disagree (1,2)=1, agree (3,4)=0; others: agree (3,4)=1, disagree (1,2)=0. Sum → `aq_total` (0–10). **Used only for AQ ≥6 filtering and reference; not used as model input.**

---

## Demographics and Derived Variables

- **Age**: Use first column that matches age (e.g. `AgeWhenTestCompleted` or `YearOfBirth`). Convert to numeric; fill NaN with median; if no age column, use 30.
- **Sex**: Map to 1=male, 2=female, 3=other, 4=prefer not to say. Then **sex_num** = {1→0, 2→1, 3→2, 4→3}; unknown → 0.
- **Age groups** (binary):  
  - `age_group_19-30` = (19 ≤ age ≤ 30)  
  - `age_group_31-45` = (31 ≤ age ≤ 45)  
  - `age_group_46-60` = (46 ≤ age ≤ 60)  
  - `age_group_61+` = (age ≥ 61)
- **sqrt_age**: `sqrt(max(0, age))`.
- **d_score**: `sqr_total - eq_total`.
- **age_x_eq**: `age * eq_total`.
- **eq_sqr_ratio**: `eq_total / (sqr_total replaced 0 with NaN + 1e-8)`; replace inf/-inf with NaN then fillna(0).
- **is_stem_occupation**: 1 if occupation string (case-insensitive) contains any of: science, technology, engineering, math, computer, software, data, research; else 0. If no occupation column, 0.

---

## C4 Feature Set (45 Features)

The model input must be **exactly** these 45 features in this order (from `feature_info_original.json`):

```
age, sex, spq_1, spq_2, spq_3, spq_4, spq_5, spq_6, spq_7, spq_8, spq_9, spq_10,
eq_1, eq_2, eq_3, eq_4, eq_5, eq_6, eq_7, eq_8, eq_9, eq_10,
sqr_1, sqr_2, sqr_3, sqr_4, sqr_5, sqr_6, sqr_7, sqr_8, sqr_9, sqr_10,
spq_total, eq_total, sqr_total, d_score, sqrt_age, age_x_eq, eq_sqr_ratio,
is_stem_occupation, sex_num,
age_group_19-30, age_group_31-45, age_group_46-60, age_group_61+
```


When building the feature matrix, use the exact `feature_names` from the JSON; the notebook provides `sex_num`. Fill any missing columns (e.g. `sex`) with 0 if not in your dataframe.

**Excluded (never in model input)**:  
`autism_target`, `aq_1`–`aq_10`, `aq_total`, `log_aq_total`, `aq_eq_interaction`, `sqp_aq_interaction`, `aq_spq_ratio`, `high_aq`, `age_x_aq`.  
These are listed in `feature_info_original.json` under `excluded_features`. AQ is used only for the AQ ≥6 filter and for reporting.

---

## Feature Alignment to C4 Schema

1. Load `feature_info_original.json`; get `feature_names` (45) and `excluded_features`.
2. Build `X_card`: for each name in `feature_names`, copy from `df_card_aggregated` if present; otherwise set to 0.
3. Order columns exactly as `feature_names`: `X_card = X_card[c4_feature_names]`.
4. Fill NaN with 0; convert all columns to numeric; replace any ±inf with 0.

---

## Scaling and Models

- **Scaler**: Load from `models/cross_validation/scaler_original.joblib`. Apply **only** `scaler.transform(X_card)`; do **not** fit on CARD.
- **Models**: Load from `models/cross_validation/`:
  - `logistic_regression_original.joblib`
  - `random_forest_original.joblib`
  - `xgboost_original.joblib`
  - `lightgbm_original.joblib`
  - `gradient_boosting_original.joblib`
- **Predictions**: Report metrics at **two** thresholds: (1) **0.5** (default, no tuning); (2) **C4 F1-optimized thresholds** loaded from `models/cross_validation/optimal_thresholds.json` (or `data/processed/threshold_optimization_results.csv`). Those thresholds were chosen on the C4 test set in data_pipeline_recreation (Experiment C), so using them on CARD is not leakage and matches the C4 decision rule. If the threshold file is missing, use 0.5 for all models.

---

## Artifact Paths (Relative to Repo)

- `models/cross_validation/feature_info_original.json`
- `models/cross_validation/scaler_original.joblib`
- `models/cross_validation/logistic_regression_original.joblib`
- `models/cross_validation/random_forest_original.joblib`
- `models/cross_validation/xgboost_original.joblib`
- `models/cross_validation/lightgbm_original.joblib`
- `models/cross_validation/gradient_boosting_original.joblib`

---

## Summary Checklist

- [ ] Load CARD; parse itemised scores with full→short mappings; aggregate one row per participant (adult preferred).
- [ ] Score SPQ-10, EQ-10, SQR-10, AQ-10 (AQ for filtering/reference only).
- [ ] Create target and demographics on aggregated data.
- [ ] Apply AQ ≥6 exclusion to autism group; drop adolescent/child-only; balance 50/50 (target_n = min(cases, controls)).
- [ ] Build 45 C4 features; align to C4 schema; fill missing/inf with 0.
- [ ] Scale with C4 scaler only; run C4 models with 0.5 threshold; report metrics.

This spec is the single reference to recreate the CARD external validation pipeline elsewhere.

---

## Prompt Improvements (for implementers)

If you hand this spec to an AI or another developer, these additions reduce ambiguity and rework:

1. **Column name variants**: State that CARD may use "Itemised Score" (with space) or "Itemized Score"; diagnosis column may be "ASC diagnosis". Suggest case-insensitive matching for key columns (VolunteerID, TestName, Itemised Score, ASC diagnosis, Sex, Occupation).
2. **Target encoding**: Specify how to derive `autism_target` from "ASC diagnosis": e.g. 1 or "Yes"/"Autism"/"ASC" -> 1, else 0; and that it must be created before AQ filtering and balancing.
3. **TestName → priority mapping**: Give an explicit mapping from CARD `TestName` values to priority (e.g. "AQ" -> 1, "Adolescent AQ" -> 2, "Child AQ" -> 3; "EQ", "Adolescent EQ", "Child EQ"; "SQ"/"SQ-R", "Adolescent SQ", "Child SQ"; "SPQ"). Note that "SQ" and "SQ-R" both map to the same sqr questionnaire type.
4. **Itemised Score parsing**: Clarify that the CSV may have spaces after commas; trim when splitting. If a row has exactly 10 items and TestName is the short form, use items as-is; only apply full→short indices when item count matches full_length (50, 60, 75, 92).
5. **Order of operations**: State the exact sequence: (1) Load raw CARD, (2) Aggregate one row per participant (by priority per test type), (3) Parse and map items to aq_1–aq_10, eq_1–eq_10, sqr_1–sqr_10, spq_1–spq_10, (4) Score all four questionnaires and build derived vars + target, (5) Apply AQ≥6 exclusion to autism cases, (6) Drop participants with only adolescent/child data, (7) Balance 50/50, (8) Build 45-feature matrix and align to C4 schema, (9) Scale and predict.
6. **Artifact paths**: Clarify "relative to repo" means relative to the project root (e.g. `c4_play2/`); if the notebook lives in `notebooks/`, use `os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'cross_validation', ...)` or a single REPO_ROOT variable.
7. **CARD path**: Ask for one configurable variable (e.g. `CARD_PATH`) for the CSV/Excel file; optional fallback to CSV if Excel is missing or encrypted.
8. **Metrics to report**: Request accuracy, precision, recall, F1, and ROC-AUC per model and a small summary table.
9. **Random seed**: Specify seed (e.g. 42) for downsampling controls so the 50/50 split is reproducible.
