# Autism Diagnosis Prediction with Machine Learning

## Project Structure

- `data/raw/` - Original, unmodified datasets
- `data/processed/` - Cleaned/processed datasets
- `notebooks/` - Jupyter notebooks for exploration and analysis
- `src/` - Source code (utilities, pipelines, etc.)
- `src/utils/` - Utility scripts
- `models/` - Saved models and related files

## Phase 2: Data Preparation

- Loads the main dataset (`data_c4_clean.csv`)
- Inspects shape, columns, dtypes
- Checks for missing values and suspicious columns
- Explores value distributions for key columns (AQ, EQ, SQ, SPQ, diagnosis, comorbidities)
- Outputs summary statistics
- (Optionally) saves a cleaned version if changes are made

See `notebooks/01_data_preparation.ipynb` for details. 