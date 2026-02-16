# How to create the three cohort files for the study notebooks

The study notebooks (study1, study2, study3) expect these files under `data/processed/`:

| File | Role | How it’s created |
|------|------|-------------------|
| **data_c4_final_recreated_cleaned.csv** | C4 cohort (45 features, cleaned) | From **data_pipeline_recreation.ipynb** – no separate “c4_aligned” step. This file *is* the C4 dataset. |
| **card_aligned.csv** | CARD cohort (45 features, aligned to C4) | From **card_c4_validation.ipynb** – see section 1 below. |
| **ybt_aligned.csv** | YBT cohort (45 features, SPQ=0, aligned to C4) | From **external_validation_ybt.ipynb** – see section 2 below. |

**C4:** There is no `c4_aligned.csv`. Run **data_pipeline_recreation.ipynb** from the top through the pipeline (Steps A–F, balance, AQ filter, final clean). It writes `data_c4_final_recreated_cleaned.csv` to `data/processed/`. Start Jupyter from the project root (`c4_play2`) so `data/processed/` is `c4_play2/data/processed/`. If you already have that file, you’re set for C4.

---

## 0. Create `data_c4_final_recreated_cleaned.csv` (if missing)

**Notebook:** `notebooks/data_pipeline_recreation.ipynb`

- Run the notebook **from the top** (Steps A through F, then balance, AQ filter, final clean).
- It saves the cleaned file as `data/processed/data_c4_final_recreated_cleaned.csv` (path is **relative to the current working directory**).
- Start Jupyter from the **project root** (`c4_play2`) so that `data/processed/` is `c4_play2/data/processed/`. If you start from `notebooks/`, the file would be created under `notebooks/data/processed/` and the study notebooks would not find it.

---

## 1. Create `card_aligned.csv`

**Notebook:** `notebooks/card_c4_validation.ipynb`

**Prerequisites:**

- CARD raw file exists. The notebook looks for it in this order:
  - `CARD_PATH` (if you set it in the first cell)
  - `c4_play2/data/CARD_Nov2025(Sheet1).csv`
  - `~/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/CARD_Nov2025(Sheet1).csv`
- `c4_play2/models/cross_validation/feature_info_original.json` exists (created when you run the “Model saving for cross-dataset validation” section in `data_pipeline_recreation.ipynb`).

**Steps:**

1. Open `notebooks/card_c4_validation.ipynb`.
2. Run **all cells from the top** in order.
3. Stop when you see the message:  
   **`Saved CARD aligned dataset to .../data/processed/card_aligned.csv (... rows, 45 features + autism_target, age, aq_total)`**

That message is at the **end of the cell** that prints “Summary: all 45 C4 features (min, max, mean)”. So run up to and including that full cell.

4. Check the file exists:

   ```bash
   ls -la c4_play2/data/processed/card_aligned.csv
   ```

If the notebook errors before that (e.g. CARD file not found, or `feature_info_original.json` missing), fix the path or run the part of `data_pipeline_recreation.ipynb` that saves the models/feature info, then try again.

---

## 2. Create `ybt_aligned.csv`

**Notebook:** `notebooks/external_validation_ybt.ipynb`

**Prerequisites:**

- YBT raw file. The notebook uses:
  - `DATA_PATH = '/Users/eb2007/playground/bullpy/c4_play2/...'` or the path set in the second cell (often `~/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/YBT.csv`).
- `c4_play2/models/cross_validation/feature_info_original.json` exists (same as for CARD).

**Steps:**

1. Open `notebooks/external_validation_ybt.ipynb`.
2. Run **all cells from the top** in order.
3. Stop when you see:  
   **`Saved YBT aligned dataset to .../data/processed/ybt_aligned.csv (... rows, 45 features + autism_target, age, aq_total)`**

That message is in the **cell right after** the one that prints “Feature alignment complete - ready for scaling” (i.e. right after STEP 8: Feature alignment). Run up to and including that save cell.

4. Check the file exists:

   ```bash
   ls -la c4_play2/data/processed/ybt_aligned.csv
   ```

If the notebook errors (e.g. YBT file not found or wrong path), set `DATA_PATH` in the config cell to your actual YBT CSV path and re-run from the top.

---

## 3. Verify all three files

From the project root (`c4_play2`):

```bash
ls -la data/processed/data_c4_final_recreated_cleaned.csv data/processed/card_aligned.csv data/processed/ybt_aligned.csv
```

If all three exist, you can run the study notebooks (study1, study2, study3).
