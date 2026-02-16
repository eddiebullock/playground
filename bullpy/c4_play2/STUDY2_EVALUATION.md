# Study 2 (Subgroup Analysis) Evaluation

Evaluation of the Study 2 run: methodology, outputs, and interpretation.

---

## 1. Did it work?

**Yes.** Study 2 ran correctly and produced the intended outputs.

- **Outputs present:** `results/study2_subgroups/` contains:
  - `age_stratified/subgroup_comparison_table.csv`
  - `sex_stratified/subgroup_comparison_table.csv`
  - `comorbidity_stratified/subgroup_comparison_table.csv`
  - `subgroup_comparison_table.csv` (combined)

- **All three cohorts** appear in the tables where expected: **C4**, **CARD**, **Dataset3 (YBT)**.

- **CARD comorbidity** is now included: CARD has rows for ADHD (n=598), Anxiety (n=820), and Depression (n=1729) with non-zero counts and plausible AUROCs. The “search all columns” parsing in `card_c4_validation.ipynb` is working.

---

## 2. Summary of results

### Age-stratified (18-30, 31-40, 41-50, 51-55)

| Cohort   | Age band | n     | AUROC (95% CI)   | Sensitivity | Specificity |
|----------|----------|-------|------------------|-------------|-------------|
| C4       | 18-30    | 18,774| 0.90 (0.90–0.91) | 0.85        | 0.80        |
| C4       | 31-40    | 4,937 | 0.91 (0.90–0.91) | 0.79        | 0.86        |
| C4       | 41-50    | 3,325 | 0.92 (0.91–0.93) | 0.79        | 0.88        |
| C4       | 51-55    | 967   | 0.92 (0.90–0.94) | 0.74        | 0.90        |
| CARD     | 18-30    | 2,111 | 0.92 (0.90–0.93) | 0.84        | 0.83        |
| CARD     | 31-40    | 1,359 | 0.88 (0.86–0.90) | 0.81        | 0.78        |
| CARD     | 41-50    | 1,290 | 0.88 (0.86–0.90) | 0.81        | 0.80        |
| CARD     | 51-55    | 381   | 0.89 (0.85–0.92) | 0.85        | 0.77        |
| Dataset3 | all bands| 7651–779 | 0.48–0.55   | 0.00        | 1.00        |

C4 and CARD show stable, good discrimination across age bands. Dataset3 is at chance (model predicts all negative in these subgroups).

### Sex-stratified (Male, Female)

- **C4:** Male n=11,799 AUROC 0.89; Female n=15,299 AUROC 0.91.
- **CARD:** Male n=1,946 AUROC 0.89; Female n=3,195 AUROC 0.91.
- **Dataset3:** Male/Female AUROC ~0.44–0.54, sensitivity 0, specificity 1 (no useful discrimination).

### Comorbidity-stratified (ADHD, Anxiety, Depression)

- **C4:** ADHD n=3,198 AUROC 0.85; Anxiety n=1,270 AUROC 0.84; Depression n=9,220 AUROC 0.90.
- **CARD:** ADHD n=598 AUROC 0.75; Anxiety n=820 AUROC 0.72; Depression n=1,729 AUROC 0.84.
- **Dataset3:** All three categories have AUROC ~0.54–0.57, sensitivity 0, specificity 1 (model predicts all negative).

---

## 3. Methodology check

- **No balance** in Study 2 load (balance_50_50=False) so subgroup sizes reflect cohort composition.
- **Per-subgroup evaluation:** For each (cohort, subgroup, category), the notebook filters to that subset, runs 5-fold stratified CV with XGBoost, and reports AUROC, 95% CI (bootstrap), sensitivity, specificity, F1. Threshold 0.5 is used for sensitivity/specificity.
- **Minimum n=30:** Subgroups with fewer than 30 participants are skipped (see `evaluate_subgroup`).
- **No leakage:** Each subgroup is evaluated only on that subgroup’s data; no test set from another cohort or subgroup is used.

---

## 4. Interpretation notes

1. **C4 and CARD**  
   Age, sex, and comorbidity subgroups show consistent, good performance (AUROC ~0.72–0.92). CARD comorbidity subgroups (especially ADHD, Anxiety) have smaller n and wider CIs but are interpretable.

2. **Dataset3 (YBT)**  
   Subgroup metrics are not meaningful: AUROC near 0.5 and sensitivity 0 indicate the model is effectively predicting “non-autism” for everyone in these subgroups, consistent with Study 1. Do not treat these as evidence of good or stable performance.

3. **CARD comorbidity**  
   Parsing from the full row text in `card_c4_validation.ipynb` is working; CARD now contributes to the comorbidity-stratified table with plausible counts and AUROCs.

---

## 5. Conclusion

Study 2 has run successfully. All three stratification types (age, sex, comorbidity) produced outputs for C4, CARD, and Dataset3. CARD comorbidity is now included. Use C4 and CARD subgroup results for interpretation; treat Dataset3 subgroup results as uninformative (model not discriminating on YBT).
