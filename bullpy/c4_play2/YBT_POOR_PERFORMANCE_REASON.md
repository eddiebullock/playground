# Why YBT (Dataset3) Scores Are So Poor

Short answer: **Lack of SPQ is one factor, but it is probably not the main reason.** The main evidence is that in YBT the model puts **zero importance on EQ and SQ-R**, which in C4 carry most of the signal. So the problem is not only “missing SPQ” but “the questionnaire features we do have don’t predict the YBT autism label.”

---

## 1. Role of SPQ in C4

In C4 (XGBoost feature importance):

- **EQ** (especially eq_1, eq_4, eq_10) carries the most weight (~0.27, ~0.12, ~0.11).
- **SPQ** (spq_1–spq_10, spq_total) contributes meaningfully (roughly ~10–15% of total importance).
- **SQ-R** and demographics contribute less than EQ but more than zero.

So in C4, **EQ is the main driver**, then SPQ, then SQ-R. Losing SPQ in YBT means losing a real part of the signal, but not the largest part.

---

## 2. What happens in YBT (Dataset3)

YBT is modelled with **35 features** (no SPQ): EQ-10, SQ-R-10, demographics, derived (d_score, eq_sqr_ratio, etc.). So we still have EQ and SQ-R.

In the **YBT-trained** XGBoost model:

- **All EQ items (eq_1–eq_10) and all SQ-R items (sqr_1–sqr_10) have importance 0.**
- The only non-zero importance is **age** (~0.23), **sex_num** (~0.19), and **age_group_31-45** (~0.58).

So the model is effectively using only age/sex/age band. It is **not** using the questionnaire scores to predict the YBT autism label. That is why performance is at chance (AUROC ~0.5, sensitivity 0, specificity 1): age/sex don’t carry the discrimination that EQ/SQ-R do in C4.

---

## 3. What that implies

- **Lack of SPQ**  
  Losing SPQ in YBT removes a real part of C4’s signal (~10–15%), so it **does** hurt. But in C4, EQ alone carries more weight than SPQ, and we have EQ in YBT. If the same relationship held in YBT, we’d expect EQ (and maybe SQ-R) to show non-zero importance and some discrimination. They don’t.

- **Primary issue**  
  The fact that **EQ and SQ-R have zero importance in YBT** suggests that in this dataset either:
  1. **Label definition / population:** Autism in YBT is defined from a free-text checklist (“diagnosis” contains autism/ASD/etc.). Prevalence and composition may differ from C4, and the link between EQ/SQ-R and this label may be weak.
  2. **Measurement / scaling:** YBT may use different scales, wording, or scoring for EQ/SQ-R, so the same numeric values don’t mean the same thing as in C4, and don’t relate to the YBT label.
  3. **Sample size / imbalance:** Smaller or more imbalanced YBT could make it harder to learn EQ/SQ-R effects, but that alone usually doesn’t force **all** questionnaire importance to zero; it more often widens CIs.

So the **primary** reason YBT scores are so poor is likely **not** “only lack of SPQ,” but rather that **the questionnaire features we do have (EQ, SQ-R) do not carry predictive signal for the YBT autism label** (plus possible label/population/measurement differences). Lack of SPQ is an additional negative.

---

## 4. Summary

| Factor | Role |
|--------|------|
| **No SPQ in YBT** | Real: we lose ~10–15% of C4-style signal. **Contributory**, not sufficient to explain near-zero use of EQ/SQ-R. |
| **EQ/SQ-R importance = 0 in YBT** | Model ignores the main C4 drivers. Suggests **different relationship** between questionnaires and autism in YBT (label, population, or measurement). |
| **Smaller sample / imbalance** | Can add variance and make learning harder; secondary to the above. |

**Bottom line:** Lack of SPQ hurts, but the main reason YBT scores are so poor is that **in YBT the model does not use EQ or SQ-R at all**; it only uses age/sex/age band, which don’t discriminate. That points to label/population/measurement differences (and possibly alignment issues) rather than to “only missing SPQ.”
