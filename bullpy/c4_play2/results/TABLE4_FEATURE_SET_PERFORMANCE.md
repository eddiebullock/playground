# Table 4: Feature Set Performance

5-fold cross-validated AUROC (XGBoost) on balanced datasets (age 18–55, AQ≥6 for autism cases, 50/50).  
**Marginal Gain** = AUROC − Demographics AUROC (C4 baseline).

| Feature Set        | C4 AUROC | CARD AUROC | YBT AUROC | Marginal Gain |
|--------------------|----------|------------|-----------|----------------|
| Demographics       | 0.63     | 0.58       | 0.47      | Baseline       |
| **Single questionnaires** | | | | |
| EQ-10              | 0.89     | 0.89       | 0.73      | +0.26          |
| SQ-R-10            | 0.77     | 0.74       | 0.62      | +0.14          |
| SPQ-10             | 0.77     | 0.66       | —         | +0.14          |
| AQ-10              | 0.94     | —          | —         | +0.32          |
| **Core combination** | | | | |
| EQ + SQ-R          | 0.90     | 0.89       | 0.74      | +0.27          |
| **Extended combinations** | | | | |
| EQ + SQ-R + SPQ    | 0.91     | 0.91       | 0.76      | +0.29          |
| EQ + SQ-R + AQ     | —        | —          | —         | —              |
| **Full model**     | | | | |
| All items + demo   | 0.91     | 0.91       | 0.76      | +0.29          |

Notes:
- CARD and YBT do not use AQ items as model input in the main pipeline, so AQ-10-only and "EQ + SQ-R + AQ" are not evaluated for those cohorts.
- YBT has no SPQ, so SPQ-10 and extended SPQ combinations are not applicable (—).
- Full model: C4/CARD = 45 features (demographics + SPQ-10 + EQ-10 + SQ-R-10 + derived, no AQ); YBT = 34 features (no SPQ).
- Source: `results/study3_features/feature_comparison_table.csv`; run `scripts/run_table4_feature_sets.py` to regenerate.
