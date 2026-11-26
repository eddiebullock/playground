# Autism Screening Assessment Pipeline

A data processing and analysis pipeline for autism screening assessments (AQ, SQ, EQ).

## Setup

1. Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the data processing pipeline:
```bash
python src/data/pipeline.py
```

2. View the unified interactive dashboard:
```bash
# Option 1: Using the runner script (recommended)
python run_dashboard.py

# Option 2: Directly with streamlit
streamlit run src/visualization/unified_dashboard.py
```

The unified dashboard provides:
- Exploratory Data Analysis (EDA) with interactive visualizations
- Model evaluation metrics and comparisons
- Interactive plots for model performance
- Downloadable processed data

## Project Structure

```
bulldev/
├── data/
│   ├── raw/           # Original data files
│   └── processed/     # Processed data and visualizations
├── models/            # Trained model artifacts (not tracked in git)
│   ├── aq/           # AQ assessment models
│   ├── eq/           # EQ assessment models
│   └── sq/           # SQ assessment models
├── src/
│   ├── data/
│   │   ├── generation/    # Data generation code
│   │   ├── processing/    # Data processing code
│   │   ├── validation/    # Data validation code
│   │   ├── analysis/      # Analysis and visualization code
│   │   └── pipeline.py    # Main pipeline script
│   ├── models/
│   │   ├── base.py        # Base model class
│   │   └── tree_models.py # Tree-based model implementations
│   └── visualization/     # Dashboard code
├── tests/               # Unit tests
├── requirements.txt     # Project dependencies
└── README.md           # This file
```

## Model Training

The pipeline supports multiple model types:
- Decision Trees
- Random Forests
- Gradient Boosting
- Logistic Regression
- Support Vector Machines

Models are trained for each assessment type (AQ, SQ, EQ) and saved in the `models/` directory. Note that model artifacts are not tracked in git as they can be regenerated from the code.

## Development

- All code should be in the `src` directory
- Tests should be in the `tests` directory
- Data files should be in the `data` directory
- Model artifacts are stored in the `models` directory (not tracked in git)
- Use the virtual environment for all development work

## License

to come...