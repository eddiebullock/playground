# ML study audit: completeness, write-up readiness, robustness

## Are the three studies done?

Yes. All three notebooks are implemented end-to-end and produce the outputs described in `ML_STUDY_README.md`:

- **Study 1**: Loads C4, CARD, YBT; stratified 80/20 train/test; 5-fold CV on train; evaluates four models on held-out test; saves metrics, models, scaler, feature importance; comparison table and optional ROC figure.
- **Study 2**: Loads same cohorts (no balance); age, sex, and comorbidity subgroups; 5-fold CV per subgroup with bootstrap 95% CI; saves subgroup tables and forest plot.
- **Study 3**: Feature-set comparison (demographics, aq_only, eq_sq_only, spq_only, all_no_aq, all_features) and SPQ contribution; optional SHAP. Saves feature_comparison_table.csv, spq_contribution_analysis.json, SHAP plots.

You can write them up. A few consistency and reporting details are below.

---

## Robustness and validity

### What is done correctly

1. **No target leakage in main models**  
   The 45-feature schema (and 35 for YBT) **excludes AQ items** from model input. Diagnosis is autism vs non-autism; AQ is used only for filtering (AQ >= 6 in cases) and in Study 3 as an explicit “aq_only” feature set to show it is highly predictive. So the main AUROC (~0.91 C4/CARD, ~0.81 YBT) is not inflated by putting the screening tool in the feature set.

2. **Train/test and scaling**  
   Study 1: one stratified 80/20 split per cohort; scaler is fit on the training set only and reused for all models; test is never used for fitting or threshold choice. Optimal F1 threshold is chosen from CV out-of-fold predictions on the **training** set, then applied to the held-out test. No leakage.

3. **External-like cohort**  
   Dataset3 (YBT) is a different study/schema (no SPQ, 35 features). Lower test AUROC (~0.81) and different subgroup behaviour (see below) are expected and support that results are not purely same-source overfitting.

4. **Subgroup analysis**  
   Study 2 uses 5-fold CV within each subgroup and bootstrap 95% CIs for AUROC. No separate test set per subgroup, which is normal when subgroup N is limited; CIs reflect uncertainty.

### Why results look “suspiciously good”

1. **Task and features align**  
   You are predicting autism vs non-autism from questionnaires designed to capture autism-related traits (EQ, SQ-R, SPQ, and in one analysis AQ). High discriminative performance is expected; it is construct validity, not leakage.

2. **50/50 balance (Study 1 and 3)**  
   Balanced classes improve AUROC and F1 compared to natural prevalence. For write-up, report that you used balanced cohorts and discuss how performance would be interpreted under real-world prevalence (e.g. lower PPV if prevalence is low).

3. **Same-cohort evaluation in Study 1**  
   Train and test are from the same cohort (C4, CARD, or YBT), so distribution shift is limited. The only “external” view is Dataset3 (YBT) with lower AUROC; that is the right place to stress-test.

4. **Study 3 “aq_only” AUROC ~0.94 (C4)**  
   AQ is the autism screening questionnaire. Showing that it alone gives very high AUROC is expected and is reported explicitly as a feature set, not as the main model. No bug.

### Issues and caveats to fix or report

1. **Age range inconsistency**  
   - Study 1: `age_max=120` (effectively no upper bound).  
   - Study 2 and 3: `age_max=55`.  
   So Study 1 includes older adults; 2 and 3 do not. Either align age ranges across studies (e.g. all 18–55) and re-run, or clearly state in the manuscript that Study 1 uses a broader age range and report the difference.

2. **Dataset3 (YBT) subgroup performance in Study 2**  
   With `balance_50_50=False`, YBT subgroups have natural prevalence. You see very low sensitivity (e.g. 2–20% in age/sex) and high specificity: the model often predicts “non-autism”. Possible reasons: (i) class imbalance within subgroups, (ii) different feature distribution in YBT so the model is poorly calibrated there, (iii) fixed 0.5 threshold. This is a real finding: performance is not transportable to YBT subgroups in the same way as C4/CARD. Report it clearly; consider reporting prevalence and optional threshold tuning or calibration per cohort.

3. **Multiple comparisons**  
   You run several models (Study 1), many subgroups (Study 2), and several feature sets (Study 3). There is no correction for multiple comparisons. For write-up: either add a short disclaimer or apply a simple correction (e.g. Bonferroni or FDR) for key claims (e.g. “SPQ adds X AUROC”).

4. **CARD and “aq_only” in Study 3**  
   CARD aligned data do not expose AQ item-level columns in the loaded dataframe, so the “aq_only” row is missing for CARD. That is consistent with the current pipeline; if you want to report “aq_only” for CARD, the CARD alignment or load would need to include aq_1..aq_10 in the exported/loaded columns.

5. **Study 1 ROC curves**  
   The optional ROC cell does not reload the test set or predictions; it only draws axes. So the saved ROC figure does not show actual curves. Either (i) save test set (or test indices) and predictions when running Study 1 and plot them, or (ii) drop the ROC figure or label it as a placeholder.

---

## Summary

- **Done**: Yes; all three studies are complete and can be written up.
- **Robust**: No target leakage; train/test and scaling are correct; Dataset3 gives a plausible external-like drop.
- **“Too good”**: Largely explained by (i) predicting autism from autism-related questionnaires, (ii) 50/50 balance, (iii) same-cohort evaluation except YBT.
- **Before write-up**: Align or justify age ranges (Study 1 vs 2/3), report YBT subgroup sensitivity/specificity and prevalence, and fix or clarify the ROC plot and optional multiple-comparison caveat.
