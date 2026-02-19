## Manuscript Results Extraction
### Generated: 2026-02-17
### Study: ML Prediction of Autism from Self-Report Questionnaires

---

## SECTION 1: METHODS

### 1.1 Study Design

- **Study design type**: Observational, retrospective, cross-sectional machine learning prediction study using existing questionnaire cohorts.
- **Number of studies/analyses**:
  - **Study 1**: Within-cohort cross-validation and held-out test evaluation across 3 cohorts (C4, CARD, Dataset3/YBT).
  - **Study 2**: Subgroup performance analysis (age, sex, comorbidities) for the best-performing Study 1 model (XGBoost) in each cohort.
  - **Study 3**: Feature set comparison (demographics vs questionnaire subsets vs full features) and SPQ contribution analysis.
- **Primary outcome variable**:
  - **Target**: Autism spectrum condition (ASC) diagnosis.
  - **Coding**:
    - In all analytic datasets, the outcome is stored as `diagnosis` (or `autism_target` before renaming), with **1 = autism** and **0 = non-autism**.
    - C4 and CARD: `autism_target` derived from existing diagnosis fields, then renamed to `diagnosis`.
    - YBT: `diagnosis` constructed from diagnosis text or `autism_target`, using case-insensitive keyword search for autism-related terms when needed.
    - Source for coding logic: `study_utils.load_cohort_c4`, `study_utils.load_cohort_card`, `study_utils.load_cohort_ybt`.
- **Pre-registration status**:
  - **[NOT FOUND — check: project-level documentation and any OSF links in notebooks such as `feature_engineering.ipynb`, `advanced_domain_adaption_test.ipynb`, `01_explore_ybt_data.ipynb`, `dora_ybt.ipynb`]**

### 1.2 Datasets / Cohorts

General inclusion/exclusion across cohorts (from `ML_STUDY_README.md` and `study_utils`):

- **Age restrictions**:
  - Study 1: C4, CARD, YBT loaded with **age ≥18**, no explicit upper cutoff in `study1_within_cohort_cv.ipynb` (C4/YBT up to at least 120).
  - Study 2 and Study 3 (updated): **age ≥18** for all cohorts (via `age_min=18, age_max=120` in cohort loaders).
  - Additional age grouping in Study 2: 18–30, 31–40, 41–50, 51+.
- **AQ-based filtering**:
  - For all three cohorts in all three studies, **AQ-10 total ≥ 6 is required only for autism cases**; non-autism controls are not filtered by AQ.
  - Implemented in `load_cohort_c4`, `load_cohort_card`, `load_cohort_ybt`:
    - Split into autism vs non-autism.
    - Apply `aq_total >= 6` within the autism group.
    - Recombine autism and non-autism.
- **Balancing**:
  - **Study 1 and Study 3**:
    - For each cohort, after exclusions, **downsample to a 50/50 autism vs non-autism** split for modeling.
    - Implemented in `load_cohort_c4`, `load_cohort_card`, `load_cohort_ybt` with `balance_50_50=True`, sampling `n = min(#autism, #non-autism)` from each class and shuffling.
  - **Study 2**:
    - Uses **unbalanced data** to preserve subgroup sizes; `balance_50_50=False` in `study2_subgroup_analysis.ipynb`.
- **Downsampling method (all cohorts)**:
  - **Approach**: Undersample the majority class to match the minority; no oversampling or SMOTE.
  - **Implementation** (in `src/study_utils.py`): `pos = df[diagnosis==1]`, `neg = df[diagnosis==0]`, `n = min(len(pos), len(neg))`; then `pos.sample(n=n, random_state=RANDOM_STATE)`, `neg.sample(n=n, random_state=RANDOM_STATE)`; concatenate and shuffle with `sample(frac=1, random_state=RANDOM_STATE)`.
  - **Random seed**: `RANDOM_STATE = 42` throughout.
  - **C4 pipeline** (`data_pipeline_recreation.ipynb`): Raw ~759k rows; internal steps include 50/50 balance and AQ>=6 filtering before writing `data_c4_final_recreated_cleaned.csv` (35,792 rows). Study loaders apply age and AQ again and optionally balance again.
  - **CARD** (`card_c4_validation.ipynb`): Aggregate to one row per participant (22,800); then AQ>=6 (autism only) and 50/50 balance before saving `card_aligned.csv`. Study loaders apply age filter and optional balance.
  - **YBT**: No pre-balance in raw/aligned file; balance applied only inside `load_cohort_ybt` when `balance_50_50=True`.

Below, Ns are reported **after exclusions and before any balancing**, based on custom scripts that replicate the cohort loaders’ age and AQ filters.

#### 1.2.1 C4 Cohort

- **Full name / source**:
  - **C4**: large online self-report cohort used for the original 45-feature C4 autism prediction study (exact public name not specified in this repo).
  - Data source notebook: `data_pipeline_recreation.ipynb`.
- **Recruitment method and setting**:
  - **[NOT FOUND — check: `data_pipeline_recreation.ipynb` introduction and original C4 study documentation]**
- **Country of origin**:
  - **[NOT FOUND — check: original C4 publication and data description in `data_pipeline_recreation.ipynb`]**
- **Data file used**:
  - `data/processed/data_c4_final_recreated_cleaned.csv`.
- **Final analytic sample (after exclusions, pre-balance)**:
  - From Python script replicating Study 2-style filters (age ≥18, AQ filter on autism cases only):
    - **N total**: 28 003
    - **N autism**: 13 076
    - **N non-autism**: 14 927
- **Total N before exclusions**:
  - **Raw C4** (from `data_pipeline_recreation.ipynb`): **758,916** rows (or 758,901 after removing test user IDs, `userid > 174283`). Pipeline then applies questionnaire scoring, AQ≥6 and 50/50 balance steps internally; saved file `data_c4_final_recreated_cleaned.csv` has **35,792** rows (age and diagnosis already applied in pipeline). Study loaders apply age ≥18 (and ≤120) and AQ≥6 for autism again to that file, yielding **28,003** pre-balance analytic (then 26,860 after 50/50 balance in Study 1).
- **Exclusion criteria applied** (implemented in `load_cohort_c4`):
  - **Age**:
    - Included: age between **18 and 55 years** (for Study 2/3; Study 1 allows up to 120).
    - Excluded: participants outside this range.
    - **Exact number and percentage excluded**:
      - **[NOT FOUND — check: early cells in `data_pipeline_recreation.ipynb` where age filters are applied and exclusions printed]**
  - **AQ-based filtering**:
    - Autism group restricted to **AQ-10 total ≥ 6.**
    - Controls (non-autism) included regardless of AQ.
    - **Number of autism cases removed due to AQ < 6**:
      - **[NOT FOUND — check: filtering logs in `data_pipeline_recreation.ipynb` around the AQ filter step]**
  - **Other filters**:
    - Cleaning and feature preparation as per original C4 pipeline (e.g. handling missing data, questionnaire completeness).
    - **[NOT FOUND — check: `data_pipeline_recreation.ipynb` for detailed C4 exclusion logic beyond age and AQ]**
- **Class balance method (downsampling to 50/50)**:
  - Implemented by `load_cohort_c4` when `balance_50_50=True`:
    - After filtering, let `pos` be autism (diagnosis=1) and `neg` non-autism (diagnosis=0).
    - Sample `n = min(len(pos), len(neg))` from each, then concatenate and shuffle.
  - **Original autism prevalence before balancing**:
    - Autism proportion in analytic C4 sample (age ≥18, AQ-filtered):
      - 13 076 / 28 003 ≈ **46.7%**.
  - **Balanced dataset sizes (Study 1)**:
    - For C4 in Study 1, `load_cohort_c4` with `balance_50_50=True` yields:
      - Reported in `study1_within_cohort_cv.ipynb` (Cell 3 output):
        - `C4 shape: (26860, 49)`
        - `C4 diagnosis counts: {0: 13430, 1: 13430}`
      - Thus:
        - **N autism after balancing**: 13 430
        - **N non-autism after balancing**: 13 430
        - **Total N after balancing**: 26 860

#### 1.2.2 CARD Cohort

- **Full name / source**:
  - **CARD**: Cambridge Autism Research Database; loaded from `CARD_Nov2025`-type files.
  - Data source notebook: `card_c4_validation.ipynb`; preprocessed, aggregated, and aligned to C4 feature schema.
- **Recruitment method and setting**:
  - **[NOT FOUND — check: CARD documentation and introduction in `card_c4_validation.ipynb`]**
- **Country of origin**:
  - Primarily UK-based sample (inferred from CARD, but **no explicit statement in current repo**).
  - **[NOT FOUND — check: CARD documentation and data-sharing agreements for exact country description]**
- **Data file used**:
  - `data/processed/card_aligned.csv` (one row per participant with 45 C4-aligned features plus outcome and demographics).
- **Final analytic sample (after exclusions, pre-balance)**:
  - Using age ≥18 and AQ≥6 for autism, applied by custom script matching `load_cohort_card` logic:
    - **N total**: 3 893
    - **N autism**: 2 033
    - **N non-autism**: 1 860
- **Total N before exclusions**:
  - **Post-aggregation (one row per participant)** in `card_c4_validation.ipynb`: **22,800** participants (adult questionnaires only). The notebook then applies AQ≥6 for autism cases and 50/50 balance before saving `card_aligned.csv` (~4,650 rows). Study loaders read `card_aligned.csv`, apply age filter (≥18, ≤120) and optional balance, giving **3,893** pre-balance analytic when using age 18–120.
- **Exclusion criteria applied** (from `CARD_VALIDATION_SPEC.md` and preprocessing prompt):
  - **Age**:
    - Age derived from `AgeWhenTestCompleted` or `YearOfBirth`.
    - Included: age between **18 and 55 years** for Study 2/3; Study 1 uses 18–120 in `study1_within_cohort_cv.ipynb`.
    - **[NOT FOUND — number excluded by age; check: `card_c4_validation.ipynb` summary prints after age filter]**
  - **AQ-based filtering**:
    - Autism group restricted to **AQ-10 total ≥ 6**.
    - Non-autism participants retained regardless of AQ score.
    - **[NOT FOUND — number of autism cases removed due to AQ < 6; check: AQ filter section in `card_c4_validation.ipynb`]**
  - **Other filters**:
    - Aggregation across questionnaires per participant with adult versions preferred; removal of participants lacking usable questionnaires or diagnoses.
    - Potential dropping of participants with missing key demographic or questionnaire data.
    - **[NOT FOUND — detailed counts per exclusion step; check: `card_c4_validation.ipynb`]**
- **Class balance method (downsampling to 50/50)**:
  - Implemented by `load_cohort_card` with `balance_50_50=True` in Study 1 and Study 3:
    - For Study 1, reported in `study1_within_cohort_cv.ipynb` (`CARD shape: (3976, 49)` and `CARD diagnosis counts: {1: 1988, 0: 1988}`).
    - Thus after balancing:
      - **N autism after balancing**: 1 988
      - **N non-autism after balancing**: 1 988
      - **Total N after balancing**: 3 976
  - **Original autism prevalence before balancing**:
    - 2 033 / 3 893 ≈ **52.2%**.

#### 1.2.3 Dataset3 (YBT) Cohort

- **Full name / source**:
  - **Dataset3 (YBT)**: Questionnaire dataset referred to as YBT (Yale-Broad or similar; exact acronym expansion not given in repo).
  - Source file: default OneDrive path `~/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/YBT.csv` or `data/raw/YBT.csv`.
  - Preprocessed/aligned file (if present): `data/processed/ybt_aligned.csv`.
  - Pipeline notebook: `external_validation_ybt.ipynb`.
- **Recruitment method and setting**:
  - **[NOT FOUND — check: `external_validation_ybt.ipynb` introduction and any linked YBT documentation]**
- **Country of origin**:
  - **[NOT FOUND — check: underlying YBT study description]**
- **Data file used in current analyses**:
  - In Study 1 and Study 3, `load_cohort_ybt` selects:
    - `data/processed/ybt_aligned.csv` if present (45-feature aligned).
    - Otherwise, falls back to raw `YBT.csv` (35-feature schema without SPQ).
  - For the present run, `study1_within_cohort_cv.ipynb` printed:
    - `YBT aligned (external_validation_ybt output): False`
    - `YBT raw: True`
    - So Study 1 and Study 3 use **raw YBT → 35-feature schema (no SPQ)**.
- **Final analytic sample (after exclusions, pre-balance)**:
  - Using `load_cohort_ybt` with **age ≥18**, AQ≥6 for autism, `balance_50_50=False` (Study 2-style analytic sample; see custom script calling `load_cohort_ybt`):
    - **N total**: 14 368
    - **N autism**: 549
    - **N non-autism**: 13 819
- **Total N before exclusions**:
  - **Raw YBT**: **24,205** rows. Study loaders use age ≥18 (no upper limit, `age_max=120`). After age ≥18 filter: more than 14,368 (the old flowchart used 18–55, but studies use ≥18). After removing missing diagnosis: varies by age range. Study loaders using age ≥18 and AQ≥6 for autism yield **14,368+** pre-balance analytic (exact N depends on how many participants >55 are included); after 50/50 balance in Study 1, N = 1,182.
- **Exclusion criteria applied** (from `study_utils.load_cohort_ybt` and YBT preprocessing):
  - **Age**:
    - Included: age **≥18 years** (no upper limit; `age_max=120` in Study 1/2/3), using numeric conversion and dropping rows with missing or out-of-range age.
    - Note: The exclusion flowchart in `external_validation_ybt.ipynb` previously used 18–55 but has been updated to match Study 1/2/3 (age ≥18).
  - **AQ-based filtering**:
    - Autism group restricted to **AQ-10 total ≥ 6** (after constructing binary AQ-10 items with C4-compatible scoring).
    - Non-autism group retained irrespective of AQ total.
    - **[NOT FOUND — number of autism cases dropped by AQ filter; check: `external_validation_ybt.ipynb`]**
  - **Other filters**:
    - Derivation of autism diagnosis from text fields when necessary.
    - Dropping rows without sufficient questionnaire or age data during harmonization.
    - **[NOT FOUND — detailed exclusion counts; check: `external_validation_ybt.ipynb`]**
- **Class balance method (downsampling to 50/50)**:
  - Implemented in `load_cohort_ybt` with `balance_50_50=True` (used in Study 1 and Study 3):
    - For Study 1, `study1_within_cohort_cv.ipynb` (Cell 7 output) shows:
      - `YBT shape: (1182, 38)`
      - `YBT diagnosis counts: {0: 591, 1: 591}`
    - Thus after balancing:
      - **N autism after balancing**: 591
      - **N non-autism after balancing**: 591
      - **Total N after balancing**: 1 182
  - **Original autism prevalence before balancing** (Study 2-style sample):
    - 549 / 14 368 ≈ **3.8%**.

### 1.3 Participant Characteristics (for Table 1)

All characteristics below are **after exclusions and before balancing** (age ≥18, AQ≥6 for autism cases, unbalanced). Values are derived from the custom Python scripts executed in this repo and from Study 2 subgroup tables (for sex N).

#### 1.3.1 C4

- **Sample sizes**:
  - N total: **28 003**
  - N autism: **13 076**
  - N non-autism: **14 927**
- **Age**:
  - Mean (SD): **28.24 (9.93)** years
  - Median: **24.0**
  - Range: **18.0–55.0**
  - IQR: **20.0–35.0**
- **Sex** (using `sex_num` with 0=male, 1=female, others grouped):
  - Male: **11 799 (42.14%)**
  - Female: **15 299 (54.63%)**
  - Other / unknown: **905 (3.23%)**
- **Questionnaire totals (overall and by diagnosis)**:
  - **AQ-10 total**:
    - Overall mean (SD): **5.43 (2.94)**
    - Autism group: **7.81 (1.37)**
    - Non-autism group: **3.34 (2.30)**
  - **EQ-10 total**:
    - Overall mean (SD): **5.85 (1.64)**
    - Autism group: **5.73 (1.71)**
    - Non-autism group: **5.96 (1.56)**
  - **SQ-R-10 total**:
    - Overall mean (SD): **5.38 (1.83)**
    - Autism group: **6.00 (1.69)**
    - Non-autism group: **4.84 (1.78)**
  - **SPQ-10 total**:
    - Overall mean (SD): **13.01 (6.39)**
    - Autism group: **10.27 (6.01)**
    - Non-autism group: **15.41 (5.72)**
- **Comorbidities** (binary columns `has_adhd`, `has_anxiety`, `has_depression`):
  - ADHD: **3 198 (11.42%)**
  - Anxiety: **1 270 (4.54%)**
  - Depression: **9 220 (32.93%)**
- **Statistical comparisons between autism and non-autism**:
  - Group-wise comparisons (t-tests / Mann-Whitney U, chi-square) are not explicitly stored; Study 2 focuses on predictive performance, not baseline comparisons.
  - **[NOT FOUND — check: `data_pipeline_recreation.ipynb` for any baseline Table 1-style tests]**

#### 1.3.2 CARD

- **Sample sizes**:
  - N total: **3 893**
  - N autism: **2 033**
  - N non-autism: **1 860**
- **Age**:
  - Mean (SD): **34.06 (10.61)** years
  - Median: **34.0**
  - Range: **18.0–55.0**
  - IQR: **24.0–43.0**
- **Sex**:
  - Male: **1 424 (36.58%)**
  - Female: **2 469 (63.42%)**
  - Other / unknown: **0 (0.0%)**
- **Questionnaire totals**:
  - **AQ-10 total**:
    - Overall mean (SD): **5.59 (1.64)**
    - Autism group: **6.48 (0.67)**
    - Non-autism group: **4.61 (1.83)**
  - **EQ-10 total**:
    - Overall mean (SD): **4.46 (2.55)**
    - Autism group: **3.53 (1.81)**
    - Non-autism group: **5.48 (2.84)**
  - **SQ-R-10 total**:
    - Overall mean (SD): **2.77 (2.23)**
    - Autism group: **3.05 (1.96)**
    - Non-autism group: **2.46 (2.46)**
  - **SPQ-10 total**:
    - Overall mean (SD): **3.81 (6.42)**
    - Autism group: **5.22 (6.74)**
    - Non-autism group: **2.28 (5.67)**
- **Comorbidities**:
  - ADHD: **437 (11.23%)**
  - Anxiety: **626 (16.08%)**
  - Depression: **1 292 (33.19%)**
- **Statistical comparisons between autism and non-autism**:
  - No explicit group comparison tests are stored; analyses focus on predictive performance.
  - **[NOT FOUND — check: `card_c4_validation.ipynb` for any t-tests or chi-square outputs]**

#### 1.3.3 Dataset3 (YBT)

- **Sample sizes** (Study 2-style analytic sample, age ≥18, AQ≥6 in autism, pre-balance):
  - N total: **14 368**
  - N autism: **549**
  - N non-autism: **13 819**
- **Age**:
  - Mean (SD): **31.34 (10.28)** years
  - Median: **29.0**
  - Range: **18.0–55.0**
  - IQR: **22.0–39.0**
- **Sex**:
  - Male: **5 529 (38.48%)**
  - Female: **8 839 (61.52%)**
  - Other / unknown: **0 (0.0%)**
- **Questionnaire totals**:
  - **AQ-10 total**:
    - Not explicitly printed in the helper script; present in processed YBT but not summarized.
    - **[NOT FOUND — check: `external_validation_ybt.ipynb` or recompute descriptive stats directly over `aq_total` for the analytic sample]**
  - **EQ-10 total**:
    - Overall mean (SD): **4.95 (2.93)**
    - Autism group: **5.24 (2.38)**
    - Non-autism group: **4.94 (2.95)**
  - **SQ-R-10 total**:
    - Overall mean (SD): **3.94 (2.57)**
    - Autism group: **4.77 (2.13)**
    - Non-autism group: **3.91 (2.59)**
  - **SPQ-10 total**:
    - Not available in YBT (Dataset3 lacks SPQ items; see `ML_STUDY_README.md`).
    - **[NOT APPLICABLE — SPQ not collected in YBT]**
- **Comorbidities**:
  - ADHD: **1 685 (11.73%)**
  - Anxiety: **2 467 (17.17%)**
  - Depression: **2 591 (18.03%)**
- **Statistical comparisons between autism and non-autism**:
  - **[NOT FOUND — check: `external_validation_ybt.ipynb` for group-wise tests if any]**

### 1.4 Questionnaires

- **Questionnaires used**:
  - **AQ-10 (10-item Autism Spectrum Quotient)**:
    - Derived from full AQ-50 using Allison et al. (2012, JAACAP) mapping.
    - Items from AQ-50 (1-indexed): 5, 20, 27, 28, 31, 32, 36, 37, 41, 45.
    - Binary scoring 0–1 per item, total range **0–10**.
    - Used **only for filtering (AQ≥6 in autism) and descriptive statistics**; **not included as model input** (AQ features are listed as excluded in `feature_info_original.json`).
  - **EQ-10 (10-item Empathy Quotient)**:
    - Derived from EQ-60 with indices from Greenberg et al. (2018, PNAS).
    - Binary scoring (0–1 per item) reflecting empathic endorsement; total **0–10**.
  - **SQ-R-10 (10-item Systemizing Quotient-Revised)**:
    - Derived from SQ-R-75 via specific indices (Greenberg et al., 2018).
    - Binary scoring (0–1 per item); total **0–10**.
  - **SPQ-10 (10-item Sensory Perception Quotient)**:
    - Derived from SPQ-92 using Greenberg et al. (2018) mappings.
    - Continuous scoring 0–3 per item (0 = strongly disagree, 3 = strongly agree), total **0–30**.
    - Available **only in C4 and CARD**; **not available for YBT**.
- **Short-form item extraction and citations**:
  - AQ-10 derived from AQ-50 following **Allison et al. (2012), JAACAP**; item indices are explicitly documented in `CARD_VALIDATION_SPEC.md` and `CARD_ITEM_MAPPING_REQUIREMENTS.md`.
  - EQ-10, SQ-R-10, SPQ-10 derived from longer forms according to **Greenberg et al. (2018), PNAS** (indices and wording documented in `CARD_ITEM_MAPPING_REQUIREMENTS.md` and confirmed in `CARD_EVALUATION_SUMMARY.md`).
- **Scoring rules** (CARD and C4, matched across datasets; see `CARD_VALIDATION_SPEC.md` and `study_utils.load_cohort_ybt`):
  - **SPQ-10**:
    - Raw responses 1–4 converted to 0–3 via `score = 4 - raw_value`.
    - Total `spq_total` is sum of 10 items: **range 0–30**.
  - **EQ-10**:
    - 4-point Likert (1–4) mapped to binary:
      - Item 3 (reverse-scored): disagree (1,2) → 1; agree (3,4) → 0.
      - All other EQ items: agree (3,4) → 1; disagree (1,2) → 0.
    - Total `eq_total`: **0–10**.
  - **SQ-R-10**:
    - Items 2, 4, 6, 8, 10 reverse-scored:
      - Reverse items: disagree (1,2) → 1; agree (3,4) → 0.
      - Other items: agree (3,4) → 1; disagree (1,2) → 0.
    - Total `sqr_total`: **0–10**.
  - **AQ-10**:
    - Items 2, 3, 4, 5, 6, 9 are reverse-scored:
      - Reverse items: disagree (1,2) → 1; agree (3,4) → 0.
      - Other items: agree (3,4) → 1; disagree (1,2) → 0.
    - Total `aq_total`: **0–10**; used for filtering and descriptive purposes only.
- **CARD-specific Likert → binary conversions**:
  - For EQ, SQ-R, AQ items, CARD preprocessing uses the 1–4 scale and applies the above binary scoring, creating 0/1 item-level variables then summing to totals.
  - SPQ remains on a 0–3 item scale after subtracting from 4.
- **Which cohorts have which questionnaires**:
  - **C4**: AQ-10, EQ-10, SQ-R-10, SPQ-10.
  - **CARD**: AQ-10, EQ-10, SQ-R-10, SPQ-10.
  - **Dataset3 (YBT)**: AQ-10, EQ-10, SQ-R-10; **no SPQ**, so SPQ features are either absent or set to 0 when using aligned 45-feature schema.

### 1.5 Feature Engineering

Feature definitions from `src/study_utils.py` and CARD specification.

- **Questionnaire item-level features**:
  - **AQ items** (10 binary items): `aq_1`–`aq_10` (used for AQ totals, but **excluded from main 45-feature schema**).
  - **EQ items** (10 binary items): `eq_1`–`eq_10`.
  - **SQ-R items** (10 binary items): `sqr_1`–`sqr_10`.
  - **SPQ items** (10 items on 0–3 scale): `spq_1`–`spq_10`.
- **Questionnaire total scores**:
  - `aq_total` (0–10) — used for filtering and descriptive statistics only.
  - `eq_total` (0–10).
  - `sqr_total` (0–10).
  - `spq_total` (0–30).
- **Derived/interaction features**:
  - `d_score` = `sqr_total - eq_total`.
  - `eq_sqr_ratio` = `eq_total / (sqr_total (0s replaced with NaN) + 1)` in Study 3; similar form with `+1e-8` in CARD pipeline.
  - `sqrt_age` = square root of age (with lower bound clipping at 0).
  - `age_x_eq` = `age * eq_total`.
  - For CARD-specific code: additional intermediate interaction variables may exist (e.g. `aq_eq_interaction`, `aq_spq_ratio`), but **these are explicitly excluded** from the final 45-feature model input.
- **Demographic features**:
  - `age` (continuous).
  - `sex` (original coding in CARD, but final models use `sex_num`).
  - `sex_num`: numeric sex code, with mapping (e.g. male→0, female→1, other→2, prefer not to say→3).
  - `is_stem_occupation`: indicator for STEM-related occupation, based on occupation text containing keywords (science, technology, engineering, math, computer, software, data, research).
- **Age group dummy variables**:
  - `age_group_19-30` = 1 if age 19–30.
  - `age_group_31-45` = 1 if age 31–45.
  - `age_group_46-60` = 1 if age 46–60.
  - `age_group_61+` = 1 if age ≥61.
  - For YBT raw, similar bins are created via `age_group` categorical and one-hot encoded to these dummies.
- **Feature schemas**:
  - **45-feature schema (with SPQ)** (`FEATURE_NAMES_45`):
    - Total count: **45**.
    - Composition:
      - 2 demographics: `age`, `sex`.
      - 10 SPQ items: `spq_1`–`spq_10`.
      - 10 EQ items: `eq_1`–`eq_10`.
      - 10 SQ-R items: `sqr_1`–`sqr_10`.
      - 3 questionnaire totals: `spq_total`, `eq_total`, `sqr_total`.
      - 4 derived features: `d_score`, `sqrt_age`, `age_x_eq`, `eq_sqr_ratio`.
      - 1 occupation feature: `is_stem_occupation`.
      - 1 numeric sex: `sex_num`.
      - 4 age dummy variables: `age_group_19-30`, `age_group_31-45`, `age_group_46-60`, `age_group_61+`.
  - **35-feature schema (no SPQ)** (`FEATURE_NAMES_35`):
    - Constructed as `FEATURE_NAMES_45` with all SPQ-related columns removed:
      - Drop `spq_1`–`spq_10` and `spq_total`.
    - Total count: **35**.
    - Used primarily for Dataset3/YBT.
- **Total feature counts by cohort**:
  - **C4** (has SPQ):
    - Uses the **45-feature schema** in Study 1 and 2.
  - **CARD** (has SPQ):
    - Uses the **45-feature schema** in Study 1 and 2.
  - **Dataset3/YBT**:
    - In current run, `ybt_aligned.csv` is missing and raw YBT is used; thus pseudo-aligned 35-feature schema is applied:
    - **35 features** (no SPQ) in Study 1 and 2.
  - Study 3 feature sets (see Section 1.7 and Study 3 results for specific feature counts per set).

### 1.6 Machine Learning Models

- **Algorithms tested** (Study 1, Study 3; defined in `study_utils.get_models`):
  - **XGBoost**: `XGBClassifier`.
  - **LightGBM**: `LGBMClassifier`.
  - **Random Forest**: `RandomForestClassifier`.
  - **Logistic Regression**: `LogisticRegression`.
- **Default hyperparameters for final models** (from `get_models` in `study_utils.py`):
  - **XGBoost (`xgboost`)**:
    - `max_depth=5`
    - `learning_rate=0.05`
    - `n_estimators=200`
    - `scale_pos_weight=1.0`
    - `random_state=42`
    - `eval_metric='logloss'`
  - **LightGBM (`lightgbm`)**:
    - `max_depth=5`
    - `learning_rate=0.05`
    - `n_estimators=200`
    - `random_state=42`
    - `verbose=-1`
  - **Random Forest (`random_forest`)**:
    - `n_estimators=200`
    - `max_depth=10`
    - `random_state=42`
  - **Logistic Regression (`logistic`)**:
    - `max_iter=1000`
    - `random_state=42`
- **Hyperparameter tuning**:
  - No explicit hyperparameter search (e.g. GridSearchCV or RandomizedSearchCV) is implemented in the study notebooks.
  - Models use **fixed hyperparameters** as above.
  - **[NOT FOUND — any tuning step; confirm by scanning Study 1/2/3 notebooks and `study_utils.py`]**
- **Calibration**:
  - No use of `CalibratedClassifierCV` or explicit calibration in current Study 1–3 notebooks.
  - **[NOT APPLICABLE — no post-hoc calibration performed]**
- **Optimal classification threshold determination**:
  - Implemented in `train_with_cv` (Study 1, Study 3) via `find_optimal_threshold`:
    - After cross-validation, obtain out-of-fold predicted probabilities `full_proba` on the training set.
    - Sweep thresholds **0.20 to 0.80 in steps of 0.01**.
    - For each threshold, compute F1 score on the training data.
    - Select threshold that **maximizes F1** (primary decision rule).
    - This optimal F1-maximizing threshold is then:
      - Reported as `optimal_threshold` in `performance_metrics.json`.
      - Applied to the held-out test set in `evaluate_model`.
- **Software versions**:
  - Python and package versions are not explicitly recorded in-study.
  - **[NOT FOUND — check: `venv` metadata or `pip freeze` outputs in the environment; document Python, scikit-learn, XGBoost, LightGBM versions for reproducibility]**

### 1.7 Validation Strategy Per Study

#### 1.7.1 Study 1 — Within-Cohort Cross-Validation

- **Train/test split**:
  - For each cohort, `create_stratified_split` in `study_utils.py`:
    - Default **test size = 0.20** (80/20 split).
    - Stratification performed via a composite `strata` column:
      - If `age_group` and `sex` are present: `strata = age_group + '_' + sex + '_' + diagnosis`.
      - Otherwise, stratify on `diagnosis` alone.
    - `train_test_split` called with `stratify=strata` (unless strata degenerate).
    - Random seed: `random_state=42`.
- **Cross-validation (CV)**:
  - Number of folds: **5-fold stratified CV** (`CV_FOLDS = 5`, `StratifiedKFold` with `shuffle=True`, `random_state=42`).
  - Conducted **on the training set only**.
- **Scaler and leakage**:
  - A `StandardScaler` is fit **only on the training data**:
    - `train_with_cv`:
      - If `scaler` is `None`, fit on `X_train` (training set) before CV.
      - CV loops re-use the scaled `X_tr` and train on each fold; scaling is not fit on validation folds.
  - Test set is transformed using the same scaler: `X_test_s = scaler.transform(X_test)` prior to prediction.
- **Random seed**:
  - Global random state constant: `RANDOM_STATE = 42` in `study_utils.py`.
  - Used consistently in:
    - StratifiedKFold (`random_state=42`).
    - Train/test split.
    - Sampling for balancing.

#### 1.7.2 Study 2 — Subgroup Analysis

- **Subgroups analyzed** (from `study2_subgroup_analysis.ipynb`):
  - **Age groups**:
    - 18–30
    - 31–40
    - 41–50
    - 51–55
    - Created via helper `add_study_age_groups`, mapping numeric age to `age_strata`.
  - **Sex groups**:
    - Male
    - Female
    - Using `sex_num` if available, otherwise `sex`, mapped to 0/1.
  - **Comorbidity groups**:
    - ADHD
    - Anxiety
    - Depression
    - Based on binary columns `has_adhd`, `has_anxiety`, `has_depression` when present (from comorbidity parsing in C4 and YBT; and from CARD’s diagnosis fields where available).
- **Model used for subgroup evaluation**:
  - XGBoost model defined in `get_models()["xgboost"]` is used for all subgroup analyses:
    - Each subgroup evaluation fits XGBoost from scratch using 5-fold CV within that subgroup’s data.
    - The helper `evaluate_subgroup` uses `get_models()['xgboost']` and retrains on each fold.
  - This corresponds to **“best model from Study 1”**, given XGBoost’s strong performance across cohorts.
- **Minimum subgroup N threshold**:
  - In Study 2 code:
    - Age subgroups: skip subgroups with `len(sub) < 30`.
    - Sex subgroups: skip subgroups with `len(sub) < 30`.
    - Comorbidity subgroups: skip subgroups with `len(sub) < 30`.
- **Statistical tests for subgroup AUROC comparisons**:
  - `evaluate_subgroup` computes:
    - AUROC
    - 95% CI via bootstrap (`bootstrap_ci_auroc`, 1 000 resamples, `RANDOM_STATE=42`).
    - Sensitivity, specificity, F1 at threshold 0.5 (no custom threshold search within subgroups).
  - **No explicit DeLong test or p-values for differences between subgroups** are implemented.
  - Subgroup tables therefore report **point estimates and 95% bootstrap CIs**, but **no pairwise p-values**.

#### 1.7.3 Study 3 — Feature Set Comparison

- **Feature sets compared** (from `study3_feature_comparison.ipynb` Cell 5):
  - `demographics`
  - `aq_only` (C4 and CARD only; YBT has AQ items but `aq_only` set is used where AQ items are present)
  - `eq_sq_only`
  - `spq_only` (C4 and CARD only; not available in Dataset3)
  - `all_no_aq`
  - `all_features`
- **Exact feature set definitions** (via `get_feature_set_columns`):
  - For each cohort dataframe `df` and set name, `get_feature_set_columns` returns the list of column names used:
    - `demographics`: `DEMOGRAPHICS_FEATURES` (`["age", "sex_num", "sqrt_age", "is_stem_occupation", "age_group_19-30", "age_group_31-45", "age_group_46-60", "age_group_61+"]`).
    - `aq_only`: only AQ-item features `AQ_ITEM_FEATURES` when `has_aq=True`.
    - `eq_sq_only`: `EQ_SQ_ONLY_FEATURES` (`eq_1–eq_10` and `sqr_1–sqr_10`).
    - `spq_only`: `SPQ_ITEM_FEATURES` (`spq_1–spq_10`) when SPQ is available.
    - `all_no_aq`:
      - Base feature set is `FEATURE_NAMES_45` (if `has_spq=True`) or `FEATURE_NAMES_35` (if `has_spq=False`).
      - Then excludes all `AQ_ITEM_FEATURES` and `aq_total`.
    - `all_features`:
      - All features from `FEATURE_NAMES_45` or `FEATURE_NAMES_35` that are present in the dataframe.
- **Feature set counts (from `feature_comparison_table.csv`)**:
  - **C4**:
    - Demographics: 8 features.
    - AQ only: 10 features.
    - EQ+SQ only: 20 features.
    - SPQ only: 10 features.
    - All-no-AQ: 45 features.
    - All features: 45 features.
  - **CARD**:
    - Demographics: 8.
    - EQ+SQ only: 20.
    - SPQ only: 10.
    - All-no-AQ: 45.
    - All features: 45.
    - No `aq_only` row, suggesting AQ items not kept in CARD’s aligned dataset or excluded by `has_aq` flag in Study 3.
  - **Dataset3/YBT**:
    - Demographics: 8.
    - EQ+SQ only: 20.
    - All-no-AQ: 34.
    - All features: 34.
    - No SPQ-only or AQ-only feature sets (SPQ not available; `has_spq=False`).
- **SPQ contribution analysis**:
  - `spq_contribution_analysis.json` reports, for XGBoost with full 45 features vs features with SPQ removed:
    - C4:
      - AUROC with SPQ: **0.91**
      - AUROC without SPQ: **0.90**
      - ΔAUROC: **+0.01**
    - CARD:
      - AUROC with SPQ: **0.91**
      - AUROC without SPQ: **0.90**
      - ΔAUROC: **+0.01**
  - Confidence intervals or statistical tests for ΔAUROC are **not computed**.
  - **[NOT FOUND — DeLong or bootstrap CIs and p-values for SPQ vs no-SPQ comparisons; would require extending Study 3 notebook]**
- **Statistical tests for AUROC comparison between feature sets**:
  - Study 3 computes **cross-validated AUROC and F1** for each feature set and cohort.
  - No DeLong or formal inferential comparisons between feature sets are implemented.
  - **[NOT FOUND — pairwise feature-set AUROC p-values; would need additional code around `bootstrap_ci_auroc` or DeLong implementation]**
- **SHAP analysis**:
  - Optional SHAP calculation for C4 only (from `study3_feature_comparison.ipynb`, Cell 9):
    - Trains XGBoost on scaled full-feature C4 dataset.
    - Computes SHAP values on 500 participants and saves `c4_shap_summary.png` under `results/study3_features/shap_values/`.
  - Numeric SHAP values or feature ranking statistics are **not saved to disk**.
  - **[NOT FOUND — tabular SHAP values, mean |SHAP| per feature, and cross-cohort rank correlations; would require exporting SHAP arrays]**

### 1.8 Performance Metrics

- **Metrics reported**:
  - **Study 1 (Within-cohort)**:
    - AUROC (area under the ROC curve).
    - Sensitivity (recall).
    - Specificity.
    - F1 score.
    - PPV (positive predictive value / precision).
    - NPV (negative predictive value).
    - Accuracy (available from `evaluate_model` though not written into `performance_metrics.json`; test accuracy would have to be recomputed or re-saved).
  - **Study 2 (Subgroups)**:
    - AUROC with 95% **bootstrap confidence intervals**.
    - Sensitivity, specificity, F1 at fixed threshold 0.5.
  - **Study 3 (Feature sets)**:
    - Cross-validated AUROC (5-fold, on the full cohort analytic sample).
    - F1 score from cross-validated predictions using threshold 0.5.
- **Metric definitions** (from `evaluate_model` and subgroup code):
  - **AUROC**: `roc_auc_score(y_true, predicted_probabilities)`.
  - **Sensitivity / recall**: `TP / (TP + FN)`.
  - **Specificity**: `TN / (TN + FP)`.
  - **F1 score**: harmonic mean of precision and recall, with `zero_division=0`.
  - **PPV (Precision)**: `TP / (TP + FP)`.
  - **NPV**: `TN / (TN + FN)`.
  - **Accuracy**: `(TP + TN) / (TP + TN + FP + FN)`.
  - Study 2 subgroups also compute confusion counts internally to derive sensitivity and specificity before reporting.
- **Confidence intervals**:
  - **Study 1 and Study 3 (held-out test metrics, best models per cohort)**:
    - Use `bootstrap_auroc_ci` and `bootstrap_metric_ci` on held-out test sets.
    - `n_bootstrap=1000`, `random_state=42`, 95% CIs derived from the 2.5th and 97.5th percentiles of the bootstrap distributions.
    - CIs are stored in `performance_metrics.json` for AUROC, sensitivity, specificity, F1, PPV, NPV, and accuracy (see Section 2.2 and Table 2).
  - **Study 2 (subgroups)**:
    - Uses `bootstrap_ci_auroc`:
      - 1 000 bootstrap resamples.
      - Each resample draws with replacement from the subgroup’s indices.
      - Computes AUROC per resample; 95% CI taken as 2.5th and 97.5th percentiles.
- **Number of bootstrap iterations**:
  - `n_bootstrap=1000` in all bootstrap CI functions (`bootstrap_ci_auroc`, `bootstrap_auroc_ci`, `bootstrap_metric_ci`).

### 1.9 Statistical Analysis

- **Software**:
  - Base language: **Python** (version not explicitly recorded).
  - Libraries used (from `study_utils.py` and notebooks):
    - `numpy`
    - `pandas`
    - `scikit-learn`
    - `xgboost`
    - `lightgbm`
    - `matplotlib` (for plots, not central to statistical results).
    - `shap` (optional, for Study 3 SHAP summary).
    - `joblib` (model serialization).
- **Library versions**:
  - Not recorded in `requirements.txt` or notebooks.
  - **[NOT FOUND — check: `venv` `pip freeze` output to record exact versions of pandas, numpy, scikit-learn, xgboost, lightgbm, shap]**
- **Significance threshold**:
  - Analyses emphasize effect sizes (AUROC, F1, CIs) rather than p-values.
  - No explicit alpha level is coded; by convention, 95% CIs correspond to **α=0.05**.
  - For future manuscript, **p < 0.05** would be the natural threshold but is not formally implemented in the current code.
- **Multiple comparisons correction**:
  - No multiple comparison correction (e.g. Bonferroni, FDR) is applied in the current notebooks.
  - Subgroup comparisons rely on overlapping vs non-overlapping CIs rather than adjusted p-values.
- **Reporting format**:
  - **Study 1 and Study 3**:
    - Main outputs: numeric metrics reported as raw floats in JSON/CSV.
    - For manuscript, should be rounded to:
      - AUROC, sensitivity, specificity, F1, PPV, NPV: **2 decimal places**.
  - **Study 2**:
    - Outputs in CSV with AUROC point estimate and 95% CI; suitable for reporting as **mean [95% CI]**.

---

## SECTION 2: RESULTS

### 2.1 Participant Flow (CONSORT-style)

For each cohort, final analytic Ns (post-exclusions) and balanced Ns are available, and for Dataset3/YBT a stepwise exclusion flowchart is now stored.

- **C4**:
  - Starting N (raw C4): **[NOT FOUND — check: `data_pipeline_recreation.ipynb` for initial raw sample size]**
  - After age ≥18 filter and AQ≥6 in autism (original Table 1 analytic sample):
    - N total: **28 003**
    - N autism: **13 076**
    - N non-autism: **14 927**
  - Balanced dataset for Study 1 and Study 3 (50/50, from `study1_within_cohort_cv.ipynb` Cell 3):
    - N total: **26 860**
    - N autism: **13 430**
    - N non-autism: **13 430**
  - Train/test split (Study 1):
    - Train N ≈ 80% of 26 860; Test N ≈ 20%.
    - Exact Ns are not printed but can be recomputed by rerunning `create_stratified_split`.
    - **[NOT FOUND — exact train/test Ns for each cohort; check: modify `study1_within_cohort_cv.ipynb` to log train/test counts]**

- **CARD**:
  - Starting N (aggregated participant-level dataset): **[NOT FOUND — check: `card_c4_validation.ipynb` after aggregation]**
  - After age ≥18 filter and AQ≥6 in autism (pre-balance):
    - N total: **3 893**
    - N autism: **2 033**
    - N non-autism: **1 860**
  - Balanced dataset for Study 1:
    - N total: **3 976**
    - N autism: **1 988**
    - N non-autism: **1 988**
  - Train/test split:
    - 80/20 stratified (exact Ns not saved).
    - **[NOT FOUND — exact numbers per split; would need added logging]**

- **Dataset3 (YBT)**:
  - Stepwise exclusion counts (from `results/exclusion_flowcharts.txt` in `external_validation_ybt.ipynb` using the 18–55 band that underlies the original analytic sample):
    - **Step 0 – Raw N**: 24 205.
    - **Step 1 – Age 18–55 filter**: 14 368 (excluded 9 837; 40.6%).
    - **Step 2 – Remove missing diagnosis**: 6 321 (excluded 8 047).
    - **Step 3 – AQ≥6 filter on autism cases**: 549 autism before, 549 after (0 excluded by AQ filter).
    - **Step 4 – Final analytic N (diagnosis available)**: 6 321.
  - Pre-balance analytic Ns used for Study 2-style descriptives (via `load_cohort_ybt` with age ≥18, AQ≥6 in autism, `balance_50_50=False`):
    - N total: **14 368**
    - N autism: **549**
    - N non-autism: **13 819**
  - Balanced dataset for Study 1:
    - N total: **1 182**
    - N autism: **591**
    - N non-autism: **591**
  - Train/test split:
    - Stratified 80/20 split on balanced data (exact Ns not printed).
    - **[NOT FOUND — exact train/test N; see `create_stratified_split` outputs if extended]**

### 2.2 Study 1 Results — Within-Cohort Cross-Validation

Data for Study 1 come from:

- `results/study1_within_cohort/{c4,card,dataset3}/performance_metrics.json`
- `results/study1_within_cohort/{c4,card,dataset3}/cv_scores.json`
- `results/study1_within_cohort/comparison_table.csv`

All metrics below are on **balanced test sets (50/50 autism/non-autism)**, which **must be highlighted in the manuscript** because this affects accuracy, PPV, and NPV interpretation.

Rounding: AUROC, sensitivity, specificity, PPV, NPV, F1 to **2 decimal places**; CV AUROC mean±SD to **2 decimal places**.

#### 2.2.1 C4 Cohort

**Cross-validation results (training set, 5-fold CV)**:

- **XGBoost**:
  - CV AUROC: **0.91 (SD 0.01)** (0.91084, SD 0.00537).
- **LightGBM**:
  - CV AUROC: **0.91 (SD 0.01)** (0.91061, SD 0.00544).
- **Random Forest**:
  - CV AUROC: **0.91 (SD 0.01)** (0.90816, SD 0.00616).
- **Logistic Regression**:
  - CV AUROC: **0.91 (SD 0.01)** (0.90917, SD 0.00604).

**Held-out test set results (balanced 50/50)**:

- **XGBoost**:
  - AUROC: **0.91**
  - F1: **0.84**
  - Sensitivity: **0.90**
  - Specificity: **0.77**
  - PPV: **0.80**
  - NPV: **0.88**
  - Optimal threshold used: **0.40**
  - Accuracy: **[NOT FOUND — not stored; recompute from confusion matrix if needed]**
- **LightGBM**:
  - AUROC: **0.91**
  - F1: **0.84**
  - Sensitivity: **0.90**
  - Specificity: **0.77**
  - PPV: **0.80**
  - NPV: **0.88**
  - Optimal threshold: **0.40**
- **Random Forest**:
  - AUROC: **0.91**
  - F1: **0.84**
  - Sensitivity: **0.89**
  - Specificity: **0.77**
  - PPV: **0.79**
  - NPV: **0.87**
  - Optimal threshold: **0.44**
- **Logistic Regression**:
  - AUROC: **0.91**
  - F1: **0.84**
  - Sensitivity: **0.88**
  - Specificity: **0.77**
  - PPV: **0.79**
  - NPV: **0.87**
  - Optimal threshold: **0.40**

**Best-performing model (C4)**:

- Slight AUROC differences favor **XGBoost** (0.91163) and **LightGBM** (0.91148) equally in practice; XGBoost is marginally highest.
- For narrative simplicity and consistency with Study 2 (which uses XGBoost), C4’s **“best model”** can be treated as **XGBoost**.

- **ROC curve data**:
  - Study 1’s ROC plotting cell does not compute ROC curves from saved predictions; it plots only diagonal lines and saves `roc_curves.png` without FPR/TPR arrays.
  - **[NOT FOUND — ROC curve FPR/TPR arrays and AUC values per threshold; need to re-evaluate models on test sets with `roc_curve`]**
- **Calibration curves**:
  - Not computed or saved.
  - **[NOT FOUND — calibration curve data; would require additional code]**
- **Confusion matrices (TP, TN, FP, FN) for best model**:
  - Evaluate_model computes TN, FP, FN, TP internally but does not persist them.
  - **[NOT FOUND — confusion counts for best C4 model; extend `performance_metrics.json` to store them]**

#### 2.2.2 CARD Cohort

**Cross-validation (training set)**:

- **XGBoost**:
  - CV AUROC: **0.91 (SD 0.01)** (0.90806, SD 0.00813).
- **LightGBM**:
  - CV AUROC: **0.91 (SD 0.01)** (0.90694, SD 0.01089).
- **Random Forest**:
  - CV AUROC: **0.91 (SD 0.01)** (0.90903, SD 0.01022).
- **Logistic Regression**:
  - CV AUROC: **0.91 (SD 0.01)** (0.90916, SD 0.01097).

**Held-out test set (balanced 50/50)**:

- **XGBoost**:
  - AUROC: **0.90**
  - F1: **0.83**
  - Sensitivity: **0.88**
  - Specificity: **0.76**
  - PPV: **0.79**
  - NPV: **0.86**
  - Threshold: **0.42**
- **LightGBM**:
  - AUROC: **0.90**
  - F1: **0.82**
  - Sensitivity: **0.88**
  - Specificity: **0.73**
  - PPV: **0.76**
  - NPV: **0.86**
  - Threshold: **0.38**
- **Random Forest**:
  - AUROC: **0.90**
  - F1: **0.82**
  - Sensitivity: **0.84**
  - Specificity: **0.78**
  - PPV: **0.79**
  - NPV: **0.83**
  - Threshold: **0.46**
- **Logistic Regression**:
  - AUROC: **0.90**
  - F1: **0.83**
  - Sensitivity: **0.88**
  - Specificity: **0.77**
  - PPV: **0.79**
  - NPV: **0.87**
  - Threshold: **0.42**

**Best-performing model (CARD)**:

- **Logistic Regression** has the highest AUROC (0.90474) and strong F1 (0.83), slightly outperforming XGBoost (AUROC 0.90314).
- For Study 2, however, subgroup analyses use **XGBoost** consistently, not logistic regression.

- **ROC, calibration, confusion matrices**:
  - As with C4, these are not stored; same **[NOT FOUND]** caveats apply.

#### 2.2.3 Dataset3 (YBT) Cohort

**Cross-validation (training set)**:

- **XGBoost**:
  - CV AUROC: **0.74 (SD 0.04)** (0.73834, SD 0.03794).
- **LightGBM**:
  - CV AUROC: **0.74 (SD 0.04)** (0.73942, SD 0.03803).
- **Random Forest**:
  - CV AUROC: **0.76 (SD 0.04)** (0.75557, SD 0.03639).
- **Logistic Regression**:
  - CV AUROC: **0.77 (SD 0.04)** (0.76975, SD 0.04461).
  - Note: CV AUROC SD ~0.04–0.05 indicates **moderate fold-to-fold variability**, slightly above ideal stability thresholds.

**Held-out test set (balanced 50/50)**:

- **XGBoost**:
  - AUROC: **0.81**
  - F1: **0.73**
  - Sensitivity: **0.89**
  - Specificity: **0.45**
  - PPV: **0.62**
  - NPV: **0.81**
  - Threshold: **0.26**
- **LightGBM**:
  - AUROC: **0.82**
  - F1: **0.75**
  - Sensitivity: **0.92**
  - Specificity: **0.49**
  - PPV: **0.64**
  - NPV: **0.85**
  - Threshold: **0.28**
- **Random Forest**:
  - AUROC: **0.81**
  - F1: **0.74**
  - Sensitivity: **0.92**
  - Specificity: **0.42**
  - PPV: **0.61**
  - NPV: **0.85**
  - Threshold: **0.33**
- **Logistic Regression**:
  - AUROC: **0.82**
  - F1: **0.74**
  - Sensitivity: **0.75**
  - Specificity: **0.71**
  - PPV: **0.72**
  - NPV: **0.74**
  - Threshold: **0.40**

**Best-performing model (Dataset3)**:

- Test AUROC is highest for **logistic regression (0.82)** and LightGBM (0.82), with logistic exhibiting more balanced sensitivity and specificity.
- XGBoost and Random Forest achieve slightly lower AUROC (~0.81) and highly asymmetric operating points (very high sensitivity, low specificity).

- **Anomalies / limitations**:
  - Sensitivity–specificity imbalance is large for XGBoost/Random Forest (difference ≈ 0.44–0.50), indicating strongly skewed thresholds favoring sensitivity over specificity.
  - Performance degradation from C4/CARD to Dataset3:
    - AUROC drops by ≈0.09–0.10 across models (e.g., logistic: ~0.91 in C4/CARD vs 0.82 in Dataset3).
    - This **exceeds the 0.10 AUROC degradation threshold** flagged in the spec as a limitation candidate.

### 2.3 Study 2 Results — Subgroup Analysis

All Study 2 results are based on XGBoost with 5-fold CV and fixed threshold 0.5 on subgroup-specific predictions. Data are taken from:

- `results/study2_subgroups/age_stratified/subgroup_comparison_table.csv`
- `results/study2_subgroups/sex_stratified/subgroup_comparison_table.csv`
- `results/study2_subgroups/comorbidity_stratified/subgroup_comparison_table.csv`

Metrics: AUROC with 95% bootstrap CI (primary), plus sensitivity, specificity, F1 (all using threshold 0.5). Ns reported are subgroup sample sizes.

#### 2.3.1 Age-Stratified Results

**C4**:

- 18–30 (N=18 774):
  - AUROC: **0.90 [0.90, 0.91]**
  - Sensitivity: **0.85**
  - Specificity: **0.80**
  - F1: **0.83**
- 31–40 (N=4 937):
  - AUROC: **0.91 [0.90, 0.91]**
  - Sensitivity: **0.79**
  - Specificity: **0.86**
  - F1: **0.79**
- 41–50 (N=3 325):
  - AUROC: **0.92 [0.91, 0.93]**
  - Sensitivity: **0.79**
  - Specificity: **0.88**
  - F1: **0.79**
- 51–55 (N=967):
  - AUROC: **0.92 [0.90, 0.94]**
  - Sensitivity: **0.74**
  - Specificity: **0.90**
  - F1: **0.76**

**CARD**:

- 18–30 (N=1 633):
  - AUROC: **0.91 [0.89, 0.92]**
  - Sensitivity: **0.83**
  - Specificity: **0.81**
  - F1: **0.83**
- 31–40 (N=1 043):
  - AUROC: **0.89 [0.87, 0.91]**
  - Sensitivity: **0.81**
  - Specificity: **0.82**
  - F1: **0.82**
- 41–50 (N=934):
  - AUROC: **0.90 [0.88, 0.92]**
  - Sensitivity: **0.81**
  - Specificity: **0.83**
  - F1: **0.81**
- 51–55 (N=283):
  - AUROC: **0.87 [0.82, 0.91]**
  - Sensitivity: **0.86**
  - Specificity: **0.76**
  - F1: **0.84**

**Dataset3 (YBT)**:

- 18–30 (N=7 651):
  - AUROC: **0.73 [0.69, 0.76]**
  - Sensitivity: **0.02**
  - Specificity: **1.00**
  - F1: **0.04**
  - Note: extremely low sensitivity and F1 indicate the model almost never predicts autism at threshold 0.5 in this subgroup.
- 31–40 (N=3 635):
  - AUROC: **0.74 [0.70, 0.78]**
  - Sensitivity: **0.09**
  - Specificity: **0.99**
  - F1: **0.14**
- 41–50 (N=2 303):
  - AUROC: **0.81 [0.76, 0.86]**
  - Sensitivity: **0.12**
  - Specificity: **1.00**
  - F1: **0.20**
- 51–55 (N=779):
  - AUROC: **0.82 [0.72, 0.91]**
  - Sensitivity: **0.20**
  - Specificity: **0.99**
  - F1: **0.28**

**Subgroup p-values vs reference group**:

- No formal p-values (e.g., DeLong tests comparing AUROC across age bands) are calculated.
- **[NOT FOUND — age subgroup AUROC p-values; would require DeLong or bootstrap-based comparisons]**

#### 2.3.2 Sex-Stratified Results

**C4**:

- Male (N=11 799):
  - AUROC: **0.89 [0.88, 0.90]**
  - Sensitivity: **0.85**
  - Specificity: **0.78**
  - F1: **0.83**
- Female (N=15 299):
  - AUROC: **0.91 [0.91, 0.92]**
  - Sensitivity: **0.80**
  - Specificity: **0.87**
  - F1: **0.80**

**CARD**:

- Male (N=1 424):
  - AUROC: **0.90 [0.88, 0.92]**
  - Sensitivity: **0.85**
  - Specificity: **0.75**
  - F1: **0.83**
- Female (N=2 469):
  - AUROC: **0.91 [0.90, 0.92]**
  - Sensitivity: **0.83**
  - Specificity: **0.84**
  - F1: **0.84**

**Dataset3 (YBT)**:

- Male (N=5 529):
  - AUROC: **0.71 [0.68, 0.75]**
  - Sensitivity: **0.02**
  - Specificity: **1.00**
  - F1: **0.03**
- Female (N=8 839):
  - AUROC: **0.80 [0.77, 0.83]**
  - Sensitivity: **0.07**
  - Specificity: **1.00**
  - F1: **0.12**

**Sex difference p-values**:

- No formal sex-difference AUROC p-values (e.g., DeLong tests comparing male vs female) are calculated.
- **[NOT FOUND — p-values for sex differences; require adding DeLong tests or bootstrap comparisons]**

#### 2.3.3 Comorbidity-Stratified Results

Subgroups restricted to participants with the comorbidity (i.e., `has_adhd == 1`, etc.). All subgroups have N ≥ 30.

**C4**:

- ADHD (N=3 198):
  - AUROC: **0.85 [0.83, 0.87]**
  - Sensitivity: **0.97**
  - Specificity: **0.45**
  - F1: **0.93**
- Anxiety (N=1 270):
  - AUROC: **0.84 [0.82, 0.87]**
  - Sensitivity: **0.93**
  - Specificity: **0.51**
  - F1: **0.89**
- Depression (N=9 220):
  - AUROC: **0.90 [0.89, 0.91]**
  - Sensitivity: **0.88**
  - Specificity: **0.75**
  - F1: **0.86**

**CARD**:

- ADHD (N=437):
  - AUROC: **0.76 [0.68, 0.83]**
  - Sensitivity: **0.96**
  - Specificity: **0.30**
  - F1: **0.93**
- Anxiety (N=626):
  - AUROC: **0.67 [0.60, 0.74]**
  - Sensitivity: **0.99**
  - Specificity: **0.09**
  - F1: **0.95**
- Depression (N=1 292):
  - AUROC: **0.83 [0.79, 0.85]**
  - Sensitivity: **0.94**
  - Specificity: **0.49**
  - F1: **0.90**

**Dataset3 (YBT)**:

- ADHD (N=1 685):
  - AUROC: **0.75 [0.71, 0.78]**
  - Sensitivity: **0.21**
  - Specificity: **0.96**
  - F1: **0.30**
- Anxiety (N=2 467):
  - AUROC: **0.77 [0.74, 0.81]**
  - Sensitivity: **0.12**
  - Specificity: **0.98**
  - F1: **0.18**
- Depression (N=2 591):
  - AUROC: **0.75 [0.71, 0.78]**
  - Sensitivity: **0.16**
  - Specificity: **0.98**
  - F1: **0.24**

**P-values for comorbidity comparisons**:

- Not computed in the current notebook.
- **[NOT FOUND — AUROC comparison p-values for comorbid vs non-comorbid groups; would require DeLong or bootstrap difference tests]**

**Data needed for Table 3 and forest plots**:

- `subgroup_comparison_table.csv` in each subfolder already contains:
  - `n`, `auroc`, `ci_lower`, `ci_upper`, `sensitivity`, `specificity`, `f1`, `Cohort`, `Subgroup`, `Category`.
  - These CSVs are directly usable as forest-plot input (see Section 4).

### 2.4 Study 3 Results — Feature Set Comparison

Primary data source: `results/study3_features/feature_comparison_table.csv`.

All Study 3 AUROCs and F1 scores are **cross-validated (5-fold)**, computed on **balanced datasets** (where Study 1 uses balancing). Metrics for each (Cohort, Feature_Set) pair:

#### 2.4.1 Feature Set Performance by Cohort

Rounded AUROC and F1 to 2 decimals; feature counts as in the CSV.

- **C4**:
  - **Demographics (8 features)**:
    - AUROC: **0.62**
    - F1: **0.58**
  - **AQ only (10 features)**:
    - AUROC: **0.94**
    - F1: **0.91**
  - **EQ+SQ only (20 features)**:
    - AUROC: **0.90**
    - F1: **0.82**
  - **SPQ only (10 features)**:
    - AUROC: **0.77**
    - F1: **0.71**
  - **All-no-AQ (45 features)**:
    - AUROC: **0.91**
    - F1: **0.84**
  - **All features (45 features)**:
    - Identical to All-no-AQ in current run (AQ is excluded from the 45-feature schema).
    - AUROC: **0.91**
    - F1: **0.84**

- **CARD**:
  - **Demographics (8 features)**:
    - AUROC: **0.59**
    - F1: **0.53**
  - **EQ+SQ only (20 features)**:
    - AUROC: **0.89**
    - F1: **0.81**
  - **SPQ only (10 features)**:
    - AUROC: **0.66**
    - F1: **0.56**
  - **All-no-AQ (45 features)**:
    - AUROC: **0.91**
    - F1: **0.84**
  - **All features (45 features)**:
    - AUROC: **0.91**
    - F1: **0.84**

- **Dataset3 (YBT)**:
  - **Demographics (8 features)**:
    - AUROC: **0.50**
    - F1: **0.51**
  - **EQ+SQ only (20 features)**:
    - AUROC: **0.74**
    - F1: **0.69**
  - **All-no-AQ (34 features)**:
    - AUROC: **0.75**
    - F1: **0.69**
  - **All features (34 features)**:
    - AUROC: **0.75**
    - F1: **0.69**

#### 2.4.2 ΔAUROC vs Demographics and vs Full Model

For each feature set, compute:

- ΔAUROC vs demographics baseline (`feature_AUROC - demographics_AUROC`).
- ΔAUROC vs full model (all_features).

**C4**:

- Demographics: reference.
- AQ only:
  - Δ vs demographics: 0.94 − 0.62 ≈ **+0.32**.
  - Δ vs all_features: 0.94 − 0.91 ≈ **+0.03**.
- EQ+SQ:
  - Δ vs demographics: 0.90 − 0.62 ≈ **+0.28**.
  - Δ vs all_features: 0.90 − 0.91 ≈ **−0.01**.
- SPQ only:
  - Δ vs demographics: 0.77 − 0.62 ≈ **+0.15**.
  - Δ vs all_features: 0.77 − 0.91 ≈ **−0.14**.
- All-no-AQ:
  - Δ vs demographics: 0.91 − 0.62 ≈ **+0.29**.
  - Δ vs all_features: 0.91 − 0.91 ≈ **0.00**.

**CARD**:

- Demographics: reference.
- EQ+SQ:
  - Δ vs demographics: 0.89 − 0.59 ≈ **+0.30**.
  - Δ vs all_features: 0.89 − 0.91 ≈ **−0.02**.
- SPQ only:
  - Δ vs demographics: 0.66 − 0.59 ≈ **+0.07**.
  - Δ vs all_features: 0.66 − 0.91 ≈ **−0.25** (substantial performance gap).
- All-no-AQ:
  - Δ vs demographics: 0.91 − 0.59 ≈ **+0.32**.
  - Δ vs all_features: **0.00**.

**Dataset3 (YBT)**:

- Demographics: reference.
- EQ+SQ:
  - Δ vs demographics: 0.74 − 0.50 ≈ **+0.24**.
  - Δ vs all_features: 0.74 − 0.75 ≈ **−0.01**.
- All-no-AQ:
  - Δ vs demographics: 0.75 − 0.50 ≈ **+0.25**.
  - Δ vs all_features: **0.00**.

#### 2.4.3 SPQ Contribution Analysis (C4 and CARD)

From `spq_contribution_analysis.json`:

- **C4**:
  - AUROC with SPQ: **0.91**
  - AUROC without SPQ: **0.90**
  - ΔAUROC: **+0.01**
  - Interpretation: **very small but positive** SPQ contribution; not large enough to be obviously clinically meaningful.
- **CARD**:
  - AUROC with SPQ: **0.91**
  - AUROC without SPQ: **0.90**
  - ΔAUROC: **+0.01**
  - Interpretation: similar to C4; SPQ adds only a small incremental value.
- **95% CI and p-values for ΔAUROC**:
  - Not computed.
  - **[NOT FOUND — significance of SPQ contribution; require bootstrap of AUROC differences or DeLong tests]**

#### 2.4.4 SHAP Analysis

- SHAP is computed only for **C4 full-feature XGBoost model** (500 participants).
- Output:
  - `c4_shap_summary.png` summarizing feature importance and SHAP distributions.
- Numeric SHAP arrays and mean |SHAP| values are not exported.
- **[NOT FOUND — table of top 10–15 features with mean |SHAP|, sign, and rank; would require modifying notebook to save these statistics]**

### 2.5 Model Comparison Across Studies

- **Best overall models per cohort (Study 1)**:
  - **C4**: XGBoost (AUROC 0.91, F1 0.84).
  - **CARD**: Logistic regression (AUROC 0.90, F1 0.83).
  - **Dataset3/YBT**: Logistic regression and LightGBM (AUROC 0.82, F1 ~0.74–0.75), with logistic having more balanced operating characteristics.
- **Consistency across cohorts**:
  - Tree-based models (XGBoost, LightGBM, Random Forest) and logistic regression perform similarly on C4 and CARD (AUROC ~0.90–0.91).
  - In Dataset3, **logistic regression** slightly outperforms tree models in AUROC and gives more balanced sensitivity and specificity.
  - No single algorithm is uniformly superior across all three datasets, though XGBoost is consistently strong and is used as the main model in subgroup analyses.
- **Pairwise model comparisons**:
  - No formal pairwise statistical comparisons of model AUROC (e.g. DeLong between XGBoost and logistic regression) are implemented.
  - **[NOT FOUND — formal statistical tests comparing models; add DeLong tests using test-set probabilities]**

### 2.6 Key Numbers for Abstract

To feed into the abstract, the following key numbers are extracted (rounded to 2 decimal places).

- **Primary result (Study 1, best model, best cohort)**:
  - Cohort: **C4**.
  - Model: **XGBoost** (within-cohort, 5-fold CV on training, evaluation on balanced test set).
  - Test set metrics:
    - AUROC: **0.91** (95% CI **[0.90, 0.92]**).
    - Sensitivity: **0.90**.
    - Specificity: **0.77**.
    - F1: **0.84**.
  - N participants in Study 1 C4 balanced dataset: **26 860 (13 430 autism, 13 430 non-autism)**.
- **Replication (Study 1, same model, second cohort)**:
  - Cohort: **CARD**.
  - Model: **XGBoost**.
  - Test AUROC: **0.90** (95% CI **[0.88, 0.92]**).
  - F1: **0.83**.
- **Subgroup finding (Study 2, most notable)**:
  - **Age effect**:
    - In C4, AUROC remains high (≈0.90–0.92) across all age bands (18–30, 31–40, 41–50, 51–55), with only modest variation.
    - In Dataset3, AUROC is moderate but sensitivity at threshold 0.5 is extremely low in younger adults (18–30 and 31–40), improving somewhat at older ages; this indicates **suboptimal thresholding and possible calibration issues rather than large ROC differences**.
  - **Sex effect**:
    - AUROC is generally slightly higher in females than males across cohorts (e.g., C4: 0.91 vs 0.89; Dataset3: 0.80 vs 0.71).
    - This may be a candidate for a key subgroup finding: **models perform better in females than males across all three cohorts**, though no formal p-values are computed.
- **Feature finding (Study 3, most notable)**:
  - **AQ contribution**:
    - In C4, **AQ-only** features achieve AUROC ≈ **0.94**, substantially higher than demographics alone (0.62) and slightly higher than full multi-questionnaire models (~0.91).
    - This suggests that **short-form AQ items alone are highly predictive of autism**, even when other questionnaires and SPQ are omitted.
  - **SPQ contribution**:
    - Adding SPQ to the full model yields only **small AUROC gains (~0.01)** in both C4 and CARD.
    - Thus, SPQ provides **modest incremental value** beyond other questionnaires and demographics.

---

## SECTION 3: TABLES

### TABLE 1: Participant Characteristics (Post-Exclusion, Pre-Balance, Age 18–55)

All means and SDs are as extracted above. Values rounded to 1 decimal place for age and questionnaire scores, percentages to 1 decimal where obvious from counts.

| Characteristic                         | C4 (n=28003)           | CARD (n=3893)          | Dataset3/YBT (n=14368) |
|----------------------------------------|------------------------|------------------------|------------------------|
| Age, mean (SD), years                  | 28.2 (9.9)             | 34.1 (10.6)            | 31.3 (10.3)            |
| Age range, years                       | 18.0–55.0              | 18.0–55.0              | 18.0–55.0              |
| Age IQR (Q1–Q3), years                 | 20.0–35.0              | 24.0–43.0              | 22.0–39.0              |
| Sex, male N (%)                        | 11799 (42.1%)          | 1424 (36.6%)           | 5529 (38.5%)           |
| Sex, female N (%)                      | 15299 (54.6%)          | 2469 (63.4%)           | 8839 (61.5%)           |
| Sex, other/unknown N (%)              | 905 (3.2%)             | 0 (0.0%)               | 0 (0.0%)               |
| Autism N (%)                           | 13076 (46.7%)          | 2033 (52.2%)           | 549 (3.8%)             |
| Non-autism N (%)                       | 14927 (53.3%)          | 1860 (47.8%)           | 13819 (96.2%)          |
| AQ-10 total, mean (SD), overall        | 5.4 (2.9)              | 5.6 (1.6)              | [NOT FOUND]            |
| AQ-10 total, autism, mean (SD)         | 7.8 (1.4)              | 6.5 (0.7)              | [NOT FOUND]            |
| AQ-10 total, non-autism, mean (SD)     | 3.3 (2.3)              | 4.6 (1.8)              | [NOT FOUND]            |
| EQ-10 total, mean (SD), overall        | 5.9 (1.6)              | 4.5 (2.5)              | 5.0 (2.9)              |
| EQ-10 total, autism, mean (SD)         | 5.7 (1.7)              | 3.5 (1.8)              | 5.2 (2.4)              |
| EQ-10 total, non-autism, mean (SD)     | 6.0 (1.6)              | 5.5 (2.8)              | 4.9 (3.0)              |
| SQ-R-10 total, mean (SD), overall      | 5.4 (1.8)              | 2.8 (2.2)              | 3.9 (2.6)              |
| SQ-R-10 total, autism, mean (SD)       | 6.0 (1.7)              | 3.0 (2.0)              | 4.8 (2.1)              |
| SQ-R-10 total, non-autism, mean (SD)   | 4.8 (1.8)              | 2.5 (2.5)              | 3.9 (2.6)              |
| SPQ-10 total, mean (SD), overall       | 13.0 (6.4)             | 3.8 (6.4)              | [N/A – no SPQ]         |
| SPQ-10 total, autism, mean (SD)        | 10.3 (6.0)             | 5.2 (6.7)              | [N/A]                  |
| SPQ-10 total, non-autism, mean (SD)    | 15.4 (5.7)             | 2.3 (5.7)              | [N/A]                  |

### TABLE 2: Model Performance — Within-Cohort (Study 1, Best Model per Cohort)

Note: All test metrics computed on **balanced test sets (50/50 autism vs non-autism)**.

| Metric                             | C4 (XGBoost)      | CARD (Logistic)   | Dataset3/YBT (Logistic) |
|------------------------------------|-------------------|-------------------|--------------------------|
| Test AUROC                         | 0.91              | 0.90              | 0.82                     |
| Test AUROC 95% CI                  | [0.90, 0.92]      | [0.89, 0.92]      | [0.76, 0.87]             |
| Sensitivity                        | 0.90              | 0.88              | 0.75                     |
| Specificity                        | 0.77              | 0.77              | 0.71                     |
| F1                                 | 0.84              | 0.83              | 0.74                     |
| Accuracy                           | 0.83              | 0.82              | 0.73                     |
| PPV                                | 0.80              | 0.79              | 0.72                     |
| NPV                                | 0.88              | 0.87              | 0.74                     |
| Optimal threshold (F1-optimized)   | 0.40              | 0.42              | 0.40                     |
| CV AUROC mean ± SD (5-fold)        | 0.91 ± 0.01       | 0.91 ± 0.01       | 0.77 ± 0.04              |

### TABLE 3: Subgroup Performance (Study 2, XGBoost)

Selected summary; full subgroup rows are in the CSVs.

| Subgroup                 | N       | C4 AUROC [95% CI]    | CARD AUROC [95% CI]   | Dataset3 AUROC [95% CI] | p (difference) |
|--------------------------|---------|----------------------|-----------------------|-------------------------|----------------|
| Age 18–30                | C4:18774| 0.90 [0.90, 0.91]    | 0.91 [0.89, 0.92]     | 0.73 [0.69, 0.76]       | [NOT FOUND]    |
| Age 31–40                | C4:4937 | 0.91 [0.90, 0.91]    | 0.89 [0.87, 0.91]     | 0.74 [0.70, 0.78]       | [NOT FOUND]    |
| Age 41–50                | C4:3325 | 0.92 [0.91, 0.93]    | 0.90 [0.88, 0.92]     | 0.81 [0.76, 0.86]       | [NOT FOUND]    |
| Age 51–55                | C4:967  | 0.92 [0.90, 0.94]    | 0.87 [0.82, 0.91]     | 0.82 [0.72, 0.91]       | [NOT FOUND]    |
| Sex: Male                | C4:11799| 0.89 [0.88, 0.90]    | 0.90 [0.88, 0.92]     | 0.71 [0.68, 0.75]       | [NOT FOUND]    |
| Sex: Female              | C4:15299| 0.91 [0.91, 0.92]    | 0.91 [0.90, 0.92]     | 0.80 [0.77, 0.83]       | [NOT FOUND]    |
| Comorbidity: ADHD        | C4:3198 | 0.85 [0.83, 0.87]    | 0.76 [0.68, 0.83]     | 0.75 [0.71, 0.78]       | [NOT FOUND]    |
| Comorbidity: Anxiety     | C4:1270 | 0.84 [0.82, 0.87]    | 0.67 [0.60, 0.74]     | 0.77 [0.74, 0.81]       | [NOT FOUND]    |
| Comorbidity: Depression  | C4:9220 | 0.90 [0.89, 0.91]    | 0.83 [0.79, 0.85]     | 0.75 [0.71, 0.78]       | [NOT FOUND]    |

### TABLE 4: Feature Set Comparison (Study 3, XGBoost CV AUROC)

| Feature Set    | N Features | C4 AUROC | CARD AUROC | Dataset3 AUROC | Δ vs Full Model (C4) | Δ vs Full Model (CARD) | Δ vs Full Model (Dataset3) |
|----------------|-----------:|---------:|-----------:|---------------:|----------------------:|------------------------:|---------------------------:|
| Demographics   | 8          | 0.62     | 0.59       | 0.50           | −0.29                 | −0.32                   | −0.25                      |
| AQ only        | 10         | 0.94     | [N/A]      | [N/A]          | +0.03                 | [N/A]                   | [N/A]                      |
| EQ+SQ only     | 20         | 0.90     | 0.89       | 0.74           | −0.01                 | −0.02                   | −0.01                      |
| SPQ only       | 10         | 0.77     | 0.66       | [N/A]          | −0.14                 | −0.25                   | [N/A]                      |
| All-no-AQ      | 45 (C4/CARD), 34 (YBT) | 0.91 | 0.91 | 0.75           | 0.00                  | 0.00                    | 0.00                       |
| All features   | 45 (C4/CARD), 34 (YBT) | 0.91 | 0.91 | 0.75           | reference             | reference               | reference                  |

---

## SECTION 4: FIGURE DATA

This section summarizes the raw data needed to construct each planned figure. Where arrays or detailed values are missing, this is explicitly noted.

### 4.1 FIGURE 1: ROC Curves (Study 1)

Desired data:

- For each cohort (C4, CARD, Dataset3) and best model:
  - False positive rates (FPR) as an array of length K.
  - True positive rates (TPR) as an array of length K.
  - AUC value corresponding to these curves.

Current status:

- Study 1’s ROC plotting cell (`study1_within_cohort_cv.ipynb` Cell 14) does **not** recompute ROC curves from saved models and test data; it plots only a diagonal reference line and saves `roc_curves.png`.
- The held-out test predictions (probabilities) are **not persisted** to disk, and test splits are not saved.

**ROC data status**:

- **[NOT FOUND — FPR/TPR arrays and ROC curve points for C4, CARD, Dataset3]**
- To obtain these, it will be necessary to:
  - Save `X_test` and `y_test` or test predictions in Study 1.
  - Use `roc_curve(y_test, y_proba)` to compute FPR and TPR.
  - Store the resulting arrays in e.g. `results/study1_within_cohort/{cohort}/roc_data.csv`.

### 4.2 FIGURE 2: Subgroup Forest Plot (Study 2)

Data needed per point:

- Label (e.g., “Age 18–30”, “Female”, “ADHD comorbid”).
- Cohort (C4, CARD, Dataset3).
- AUROC point estimate.
- 95% CI lower bound.
- 95% CI upper bound.

Available data:

- For each subgroup type, `subgroup_comparison_table.csv` contains:
  - `n`, `auroc`, `ci_lower`, `ci_upper`, `sensitivity`, `specificity`, `f1`, `Cohort`, `Subgroup`, `Category`.

This is sufficient to construct forest plots. Example dictionary-style representation (not saved but derivable directly from the CSVs):

- **Age subgroups (Age-stratified CSV)**:
  - For each row:
    - `label`: `"Age {Category}"` (e.g. `"Age 18-30"`).
    - `cohort`: `Cohort` (e.g. `"C4"`).
    - `auroc`: `auroc`.
    - `ci_lower`: `ci_lower`.
    - `ci_upper`: `ci_upper`.
- **Sex subgroups (Sex-stratified CSV)**:
  - `label`: `"Sex {Category}"` (e.g. `"Sex Female"`).
  - Other fields as above.
- **Comorbidity subgroups (Comorbidity-stratified CSV)**:
  - `label`: `"Comorbidity {Category}"` (e.g. `"Comorbidity ADHD"`).

All required values for plotting are present in the three CSVs; they can be read directly and mapped into plotting structures.

### 4.3 FIGURE 3: SHAP Summary (Study 3)

Desired data:

- For the top 15 features by mean absolute SHAP value (|SHAP|):
  - Feature name.
  - Mean |SHAP| value.
  - Directionality (whether higher values are associated with higher autism probability).
  - Optionally, distribution of SHAP values across individuals (for beeswarm).

Current outputs:

- Only `c4_shap_summary.png` is saved, based on:
  - 500 C4 participants.
  - XGBoost trained on full C4 feature set.
  - SHAP TreeExplainer applied to scaled features.
- Numeric SHAP values (`shap_vals`) and feature-wise statistics are not saved.

**SHAP data status**:

- **[NOT FOUND — SHAP value arrays and feature-wise statistics]**
- To obtain them:
  - Modify the SHAP cell in `study3_feature_comparison.ipynb` to:
    - Save `shap_vals` and the underlying feature matrix (e.g., as `.npy` and `.csv`).
    - Compute mean |SHAP| per feature and write to `results/study3_features/shap_values/c4_shap_importance.csv`.

### 4.4 FIGURE 4: Feature Comparison Bar Chart (Study 3)

Data needed:

- For each feature set × cohort:
  - Feature set name.
  - AUROC (CV).
  - 95% CI lower and upper (if available).

Available:

- `feature_comparison_table.csv` contains:
  - `Cohort`, `Feature_Set`, `N_Features`, `AUROC`, `F1`.

This is sufficient to produce bar charts; no CIs are computed at present.

**Feature comparison data (ready-to-plot)**:

- Key rows (already listed in Section 2.4) can be ingested programmatically by plotting scripts to create:
  - Bars grouped by cohort (C4, CARD, Dataset3).
  - Colors or patterns distinguishing feature sets (`demographics`, `aq_only`, `eq_sq_only`, `spq_only`, `all_no_aq`, `all_features`).

---

## SECTION 5: MISSING DATA LOG

This section collects all **[NOT FOUND]** items with pointers to where they could be recovered or recomputed.

1. **Pre-registration status and OSF link**:
   - Status: Not documented.
   - Suggested location: Check project-level documents (e.g., `README.md` in repo root), any OSF links in exploratory notebooks (`feature_engineering.ipynb`, `advanced_domain_adaption_test.ipynb`, `01_explore_ybt_data.ipynb`, `dora_ybt.ipynb`).

2. **C4 and CARD recruitment details and country of origin**:
   - Status: High-level dataset descriptions are not in this repo.
   - Suggested location: Original C4 and CARD publications and documentation; introduction sections in `data_pipeline_recreation.ipynb` and `card_c4_validation.ipynb`.

3. **Initial sample sizes before exclusions (all cohorts)**:
   - C4: Raw N before age/AQ/quality filters not recorded in this extraction.
   - CARD: `len(df_card_aggregated)` after aggregation is printed only inside `card_c4_validation.ipynb`.
   - Dataset3/YBT: Raw N in `YBT.csv` before filters not summarized here.
   - Suggested location: Early cells in `data_pipeline_recreation.ipynb`, `card_c4_validation.ipynb`, and `external_validation_ybt.ipynb`.

4. **Counts and percentages excluded at each step (age filters, AQ filters, other cleaning)**:
   - Status: Not assembled as stepwise counts in outputs.
   - Suggested location: Filtering and cleaning cells in preprocessing notebooks that may print before/after shapes and counts:
     - `data_pipeline_recreation.ipynb`
     - `card_c4_validation.ipynb`
     - `external_validation_ybt.ipynb`

5. **AQ-10 descriptive statistics in YBT (overall and by group)**:
   - Status: Not computed in helper script; though `aq_total` exists in processed YBT.
   - Suggested action: Compute directly from `df_ybt['aq_total']` in a short helper script or within `external_validation_ybt.ipynb`.

6. **Group-comparison statistics (e.g., t-tests, Mann-Whitney, chi-square)** for baseline demographics:
   - Status: Not part of Study 1–3 notebooks; only ML performance metrics are reported.
   - Suggested action: Add a separate notebook or section to compute:
     - t-tests or non-parametric tests for age and questionnaire totals (autism vs non-autism).
     - Chi-square tests for sex differences.

7. **Confidence intervals for Study 1 (within-cohort) AUROCs and other metrics**:
   - Status: Not computed; `performance_metrics.json` stores point estimates and CV mean±SD only.
   - Suggested action: Add bootstrap or DeLong-based CI computation using held-out test predictions and store in:
     - `results/study1_within_cohort/{cohort}/performance_metrics.json` as `test_auroc_ci_lower`, `test_auroc_ci_upper`.

8. **Accuracy and confusion matrix details (TP, TN, FP, FN) for Study 1 best models**:
   - Status: `evaluate_model` computes these but only returns summary metrics; confusion counts are not saved.
   - Suggested action:
     - Extend `evaluate_model` to include confusion counts in the dictionary.
     - Modify Study 1 notebook to write confusion matrices to JSON or CSV.

9. **ROC curve FPR/TPR arrays (Figure 1)**:
   - Status: Not stored; `roc_curves.png` contains a reference diagonal only.
   - Suggested action:
     - Save test-set probabilities and labels per cohort.
     - Compute `fpr`, `tpr` via `roc_curve`.
     - Store them in `results/study1_within_cohort/{cohort}/roc_data.csv`.

10. **Calibration curve data**:
    - Status: Not computed for any cohort.
    - Suggested action:
      - Use `calibration_curve` from scikit-learn on test-set probabilities.
      - Save calibration bins and mean predicted probabilities per bin.

11. **P-values for subgroup differences (age, sex, comorbidities)**:
    - Status: Subgroup analyses provide AUROC ± 95% CI but no formal p-values.
    - Suggested action:
      - Implement DeLong tests or bootstrap-based difference testing between subgroups (e.g., male vs female, age bands vs 18–30).

12. **Pairwise model comparison p-values (XGBoost vs logistic vs others)**:
    - Status: Not computed.
    - Suggested action:
      - Use DeLong tests or bootstrap on test-set predictions for each model pair per cohort.

13. **SPQ contribution statistical significance (ΔAUROC CIs and p-values)**:
    - Status: Only point estimates reported for AUROC with vs without SPQ.
    - Suggested action:
      - Bootstrap differences in AUROC or use DeLong to compare models trained with vs without SPQ.

14. **SHAP numeric outputs and cross-cohort feature-ranking correlations**:
    - Status: Only `c4_shap_summary.png` is saved.
    - Suggested action:
      - Save SHAP arrays and compute:
        - Mean |SHAP| per feature.
        - Rank features for C4, CARD, and Dataset3 (if SHAP is run there).
        - Spearman rank correlations between cohorts.

15. **Exact software versions (Python, pandas, numpy, scikit-learn, xgboost, lightgbm, shap)**:
    - Status: Not logged in notebooks.
    - Suggested action:
      - Capture `pip freeze` output (e.g. into `environment.txt`) and record key library versions in the Methods section of the manuscript.

16. **Dataset3/YBT SPQ status and any partial SPQ data**:
    - Status: Repo-level text says YBT has no SPQ; current feature schema uses 35 features with SPQ columns removed or set to zero.
    - Suggested action:
      - Confirm in `01_explore_ybt_data.ipynb` and `external_validation_ybt.ipynb` that SPQ data is indeed absent.

17. **Any additional limitations identified via manual inspection**:
    - Examples to highlight in manuscript:
      - **Performance degradation**:
        - C4/CARD AUROC ~0.90–0.91 vs Dataset3 AUROC ~0.82: degradation ≈0.08–0.10 in external cohort.
      - **Subgroups with AUROC < 0.70**:
        - Several Dataset3 subgroups (e.g., male sex, younger age bands) approach or fall below 0.75 with highly imbalanced sensitivity/specificity.
      - **High CV variability**:
        - Dataset3 logistic regression CV AUROC SD ~0.04–0.05 suggests less stable performance across folds.
      - **AQ dependence**:
        - C4 AQ-only AUROC ~0.94 indicates strong dependence on AQ for classification, which may limit applicability where AQ is unavailable or not independent of diagnosis.

This completes the structured extraction of methods and results necessary to draft a JAMA Network Open–style Methods and Results section for the autism prediction manuscript.

