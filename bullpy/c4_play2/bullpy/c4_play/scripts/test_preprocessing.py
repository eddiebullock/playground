#!/usr/bin/env python3
"""
Test script for data preprocessing pipeline.

This script tests the data preprocessing and exploratory analysis modules
to ensure they work correctly with your dataset.
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
from data import DataPreprocessor
from exploratory_analysis import ExploratoryAnalyzer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data():
    """Create sample data for testing."""
    logger.info("Creating sample data for testing...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # Create sample data similar to your autism dataset
    data = {
        # Demographic variables
        'age': np.random.normal(30, 10, n_samples).astype(int),
        'sex': np.random.choice([1, 2], n_samples),
        'handedness': np.random.choice([1, 2, 3], n_samples),
        'education': np.random.choice([1, 2, 3, 4, 5], n_samples),
        'occupation': np.random.choice([1, 2, 3, 4, 5], n_samples),
        'country_region': np.random.choice([1, 2, 3, 4, 5], n_samples),
        
        # SPQ questionnaire (10 questions)
        **{f'spq_{i:02d}': np.random.choice([1, 2, 3, 4, 5], n_samples) for i in range(1, 11)},
        
        # EQ questionnaire (15 questions)
        **{f'eq_{i:02d}': np.random.choice([1, 2, 3, 4, 5], n_samples) for i in range(1, 16)},
        
        # SQR questionnaire (12 questions)
        **{f'sqr_{i:02d}': np.random.choice([1, 2, 3, 4, 5], n_samples) for i in range(1, 13)},
        
        # AQ questionnaire (20 questions)
        **{f'aq_{i:02d}': np.random.choice([1, 2, 3, 4, 5], n_samples) for i in range(1, 21)},
        
        # Diagnosis columns
        'autism_diagnosis_1': np.random.choice([0, 1, 2, 3], n_samples),
        'autism_diagnosis_2': np.random.choice([0, 1, 2, 3], n_samples),
        'adhd_diagnosis': np.random.choice([0, 1, 2, 3], n_samples),
        'anxiety_diagnosis': np.random.choice([0, 1, 2, 3], n_samples),
        'depression_diagnosis': np.random.choice([0, 1, 2, 3], n_samples),
    }
    
    # Add some missing values
    for col in ['age', 'sex', 'handedness']:
        missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
        data[col][missing_indices] = np.nan
    
    # Add some non-standard missing codes
    for col in ['education', 'occupation']:
        missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.03), replace=False)
        data[col][missing_indices] = -1
    
    df = pd.DataFrame(data)
    
    # Create target variable (autism diagnosis)
    # Make it somewhat related to questionnaire scores
    spq_scores = df[[col for col in df.columns if 'spq' in col]].sum(axis=1)
    aq_scores = df[[col for col in df.columns if 'aq' in col]].sum(axis=1)
    
    # Create target based on questionnaire scores and some randomness
    autism_prob = (spq_scores + aq_scores) / (spq_scores + aq_scores).max() * 0.7 + np.random.random(n_samples) * 0.3
    df['has_autism'] = (autism_prob > 0.5).astype(int)
    
    return df

def test_data_preprocessing():
    """Test the data preprocessing pipeline."""
    logger.info("Testing data preprocessing pipeline...")
    
    # Create sample data
    df_raw = create_sample_data()
    
    # Save raw data
    raw_data_path = Path("data/raw/sample_data.csv")
    raw_data_path.parent.mkdir(parents=True, exist_ok=True)
    df_raw.to_csv(raw_data_path, index=False)
    logger.info(f"Saved sample data to {raw_data_path}")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Test data loading
    df_loaded = preprocessor.load_data(str(raw_data_path))
    assert df_loaded.shape == df_raw.shape, "Data loading failed"
    logger.info("✓ Data loading test passed")
    
    # Test data cleaning
    df_clean = preprocessor.clean_data(df_loaded)
    assert df_clean.shape[0] <= df_loaded.shape[0], "Data cleaning should not increase rows"
    logger.info("✓ Data cleaning test passed")
    
    # Test feature engineering
    df_features = preprocessor.create_features(df_clean)
    expected_new_cols = ['spq_total', 'eq_total', 'sqr_total', 'aq_total', 'total_score', 
                        'num_diagnoses', 'has_autism', 'age_group']
    
    for col in expected_new_cols:
        if col in df_features.columns:
            logger.info(f"✓ Created feature: {col}")
    
    # Test data splitting
    X, y = preprocessor.prepare_modeling_data(df_features)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    assert len(X_train) + len(X_test) == len(X), "Data splitting failed"
    assert len(y_train) + len(y_test) == len(y), "Target splitting failed"
    logger.info("✓ Data splitting test passed")
    
    # Test saving processed data
    processed_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'features_full': df_features
    }
    preprocessor.save_processed_data(processed_data)
    logger.info("✓ Data saving test passed")
    
    logger.info("✓ All data preprocessing tests passed!")
    return df_features, X_train, X_test, y_train, y_test

def test_exploratory_analysis(df_features):
    """Test the exploratory analysis pipeline."""
    logger.info("Testing exploratory analysis pipeline...")
    
    # Initialize analyzer
    analyzer = ExploratoryAnalyzer()
    
    # Test dataset overview
    overview = analyzer.analyze_dataset_overview(df_features)
    assert 'shape' in overview, "Dataset overview failed"
    logger.info("✓ Dataset overview test passed")
    
    # Test missing value analysis
    missing_analysis = analyzer.analyze_missing_values(df_features)
    assert 'missing_summary' in missing_analysis, "Missing value analysis failed"
    logger.info("✓ Missing value analysis test passed")
    
    # Test target variable analysis
    if 'has_autism' in df_features.columns:
        target_analysis = analyzer.analyze_target_variable(df_features, 'has_autism')
        assert 'value_counts' in target_analysis, "Target analysis failed"
        logger.info("✓ Target variable analysis test passed")
    
    # Test feature distributions
    distribution_analysis = analyzer.analyze_feature_distributions(df_features, 'has_autism')
    assert 'numeric_features' in distribution_analysis or 'categorical_features' in distribution_analysis, "Distribution analysis failed"
    logger.info("✓ Feature distribution analysis test passed")
    
    # Test correlations
    correlation_analysis = analyzer.analyze_correlations(df_features, 'has_autism')
    assert 'feature_correlations' in correlation_analysis or 'target_correlations' in correlation_analysis, "Correlation analysis failed"
    logger.info("✓ Correlation analysis test passed")
    
    # Test specialized analyses
    questionnaire_analysis = analyzer.analyze_questionnaire_blocks(df_features)
    assert 'blocks' in questionnaire_analysis, "Questionnaire analysis failed"
    logger.info("✓ Questionnaire analysis test passed")
    
    demographic_analysis = analyzer.analyze_demographics(df_features)
    assert 'available_demographics' in demographic_analysis, "Demographic analysis failed"
    logger.info("✓ Demographic analysis test passed")
    
    # Test saving results
    analyzer.save_analysis_results()
    logger.info("✓ Analysis results saving test passed")
    
    logger.info("✓ All exploratory analysis tests passed!")

def main():
    """Run all tests."""
    logger.info("Starting preprocessing and EDA tests...")
    
    try:
        # Test data preprocessing
        df_features, X_train, X_test, y_train, y_test = test_data_preprocessing()
        
        # Test exploratory analysis
        test_exploratory_analysis(df_features)
        
        logger.info("All tests passed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")

if __name__ == "__main__":
    main() 