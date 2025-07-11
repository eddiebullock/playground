# Autism Diagnosis Prediction Project

This project aims to develop machine learning models to predict autism diagnoses from questionnaire data and demographic information.

## Project Structure

```
c4_play/
├── data/
│   ├── raw/                    # Original data files (never modified)
│   └── processed/              # Cleaned, preprocessed data
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory data analysis
│   ├── 02_prototype.ipynb     # Model prototyping
│   └── 03_reporting.ipynb     # Final figures/tables
├── src/
│   ├── data.py                 # Data loading, cleaning, preprocessing
│   ├── exploratory_analysis.py # EDA and visualization
│   ├── features.py             # Feature engineering
│   ├── models.py               # Model definitions
│   ├── train.py                # Training pipeline
│   └── evaluate.py             # Evaluation metrics
├── experiments/
│   ├── configs/                # YAML configuration files
│   ├── logs/                   # Experiment logs and results
│   └── outputs/                # Model checkpoints, predictions
├── results/
│   ├── figures/                # Plots for paper
│   └── tables/                 # Tables for paper
├── scripts/
│   ├── run_experiment.sh       # Shell script for experiments
│   └── submit_job.slurm        # SLURM job script for HPC
├── requirements.txt            # Python dependencies
└── README.md                  # This file
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Register Jupyter kernel
python -m ipykernel install --user --name=venv-c4play --display-name "Python (venv-c4play)"
```

### 2. Data Preprocessing

```bash
# Run data preprocessing pipeline
python src/data.py

# This will:
# - Load raw data from data/raw/
# - Clean and preprocess the data
# - Create engineered features
# - Split into train/test sets
# - Save processed data to data/processed/
```

### 3. Exploratory Analysis

```bash
# Run exploratory analysis
python src/exploratory_analysis.py

# This will:
# - Analyze dataset overview
# - Check missing values
# - Analyze target variable distribution
# - Create feature distribution plots
# - Generate correlation analysis
# - Save plots to results/figures/
# - Save analysis results to experiments/logs/
```

### 4. Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open notebooks in order:
# 1. notebooks/01_eda.ipynb - Review EDA results
# 2. notebooks/02_prototype.ipynb - Prototype models
# 3. notebooks/03_reporting.ipynb - Create final figures
```

## Data Preprocessing Pipeline

The `src/data.py` module provides a comprehensive data preprocessing pipeline:

### Key Features:
- **Data Loading**: Supports CSV and Excel files
- **Missing Value Handling**: 
  - Diagnosis columns: Drop rows with >2 missing values
  - Demographic columns: Impute with new category (0)
  - Questionnaire columns: Impute with mean if ≤2 missing, otherwise drop
- **Feature Engineering**:
  - Composite scores for questionnaire blocks (SPQ, EQ, SQR, AQ)
  - Diagnosis aggregations (count, binary flags)
  - Demographic features (age groups)
  - Interaction features
- **Data Splitting**: Stratified train/test split
- **Reproducibility**: Saves preprocessing information and random seeds

### Usage:

```python
from src.data import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor()

# Load and process data
df_raw = preprocessor.load_data("data/raw/your_data.csv")
df_clean = preprocessor.clean_data(df_raw)
df_features = preprocessor.create_features(df_clean)

# Prepare for modeling
X, y = preprocessor.prepare_modeling_data(df_features)
X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
```

## Exploratory Analysis

The `src/exploratory_analysis.py` module provides comprehensive EDA:

### Key Features:
- **Dataset Overview**: Shape, memory usage, data types, missing values
- **Missing Value Analysis**: Detailed missing value patterns and heatmaps
- **Target Variable Analysis**: Distribution, class imbalance analysis
- **Feature Distributions**: Numeric and categorical feature analysis
- **Correlation Analysis**: Feature correlations and target correlations
- **Specialized Analyses**: Questionnaire blocks, demographics
- **Visualization**: Automatic generation of plots and charts

### Usage:

```python
from src.exploratory_analysis import ExploratoryAnalyzer

# Initialize analyzer
analyzer = ExploratoryAnalyzer()

# Run comprehensive analysis
analyzer.analyze_dataset_overview(df)
analyzer.analyze_missing_values(df)
analyzer.analyze_target_variable(df, 'has_autism')
analyzer.analyze_feature_distributions(df, 'has_autism')
analyzer.analyze_correlations(df, 'has_autism')
analyzer.analyze_questionnaire_blocks(df)
analyzer.analyze_demographics(df)

# Save results
analyzer.save_analysis_results()
```

## Configuration

The project uses YAML configuration files for experiment parameters:

```yaml
# experiments/configs/data_config.yaml
data:
  raw_data_path: "data/raw/your_data_file.csv"
  processed_data_dir: "data/processed"

cleaning:
  missing_codes: [-1, -999, 999, '', 'NA', 'N/A', 'null']
  diagnosis_missing_threshold: 2
  demographic_impute_value: 0

features:
  questionnaire_blocks: ["SPQ", "EQ", "SQR", "AQ"]
  age_bins: [0, 18, 25, 35, 50, 100]

target:
  primary_target: "has_autism"

splitting:
  test_size: 0.2
  random_state: 42
  stratify: true
```

## Output Files

### Data Preprocessing Outputs:
- `data/processed/X_train.csv` - Training features
- `data/processed/X_test.csv` - Test features  
- `data/processed/y_train.csv` - Training targets
- `data/processed/y_test.csv` - Test targets
- `data/processed/features_full.csv` - Full dataset with features
- `experiments/logs/preprocessing_info.yaml` - Preprocessing documentation

### Exploratory Analysis Outputs:
- `results/figures/missing_values_heatmap.png` - Missing value visualization
- `results/figures/has_autism_distribution.png` - Target distribution
- `results/figures/correlation_heatmap.png` - Feature correlations
- `results/figures/questionnaire_analysis.png` - Questionnaire block analysis
- `results/figures/demographics_analysis.png` - Demographic analysis
- `experiments/logs/eda_results_YYYYMMDD_HHMMSS.yaml` - Analysis results

## Next Steps

1. **Update Configuration**: Modify `experiments/configs/data_config.yaml` with your actual data file path
2. **Run Preprocessing**: Execute `python src/data.py`
3. **Review EDA**: Run `python src/exploratory_analysis.py` and review results
4. **Prototype Models**: Use notebooks for initial model experimentation
5. **Scale to HPC**: Move to scripts for large-scale experiments

## Contributing

- Follow the established project structure
- Use configuration files for experiment parameters
- Document all preprocessing steps
- Save all results and random seeds for reproducibility
- Use version control for all code and configurations

