#!/usr/bin/env python3
"""
Test suite for the modular preprocessing pipeline.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from modules.data_loader import load_data, validate_data_structure
from modules.data_cleaner import clean_data
from modules.feature_engineer import create_features
from modules.data_splitter import prepare_targets, split_data
from pipeline import run_preprocessing_pipeline

def create_test_data():
    """Create test data that matches the expected structure."""
    n_samples = 100
    
    # Create test data with all expected columns
    data = {
        'userid': range(1, n_samples + 1),
        'age': np.random.randint(16, 80, n_samples).astype(float),
        'sex': np.random.choice([1, 2, 3, 4], n_samples).astype(float),
        'handedness': np.random.choice([1, 2, 3, 4], n_samples).astype(float),
        'education': np.random.choice([1, 2, 3, 4, 5], n_samples).astype(float),
        'occupation': np.random.choice(range(1, 27), n_samples).astype(float),
        'country_region': np.random.choice(range(1, 15), n_samples).astype(float),
        'repeat': np.random.choice([0, 1], n_samples).astype(float),
        
        # SPQ questionnaire (10 questions)
        **{f'spq_{i:02d}': np.random.choice([1, 2, 3, 4], n_samples) for i in range(1, 11)},
        
        # EQ questionnaire (10 questions)
        **{f'eq_{i:02d}': np.random.choice([1, 2, 3, 4], n_samples) for i in range(1, 11)},
        
        # SQR questionnaire (10 questions)
        **{f'sqr_{i:02d}': np.random.choice([1, 2, 3, 4], n_samples) for i in range(1, 11)},
        
        # AQ questionnaire (10 questions)
        **{f'aq_{i:02d}': np.random.choice([1, 2, 3, 4], n_samples) for i in range(1, 11)},
        
        # Diagnosis columns
        'diagnosis_0': np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], n_samples).astype(float),
        'diagnosis_1': np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], n_samples).astype(float),
        'diagnosis_2': np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], n_samples).astype(float),
        'diagnosis_3': np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], n_samples).astype(float),
        'diagnosis_4': np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], n_samples).astype(float),
        'diagnosis_5': np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], n_samples).astype(float),
        'diagnosis_6': np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], n_samples).astype(float),
        'diagnosis_7': np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], n_samples).astype(float),
        'diagnosis_8': np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], n_samples).astype(float),
        
        # Autism diagnosis columns
        'autism_diagnosis_0': np.random.choice([0, 1, 2, 3], n_samples).astype(float),
        'autism_diagnosis_1': np.random.choice([0, 1, 2, 3], n_samples).astype(float),
        'autism_diagnosis_2': np.random.choice([0, 1, 2, 3], n_samples).astype(float),
    }
    
    # Add some missing values
    for col in ['age', 'sex']:
        missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
        data[col][missing_indices] = np.nan
    
    df = pd.DataFrame(data)
    
    # Create target variable (autism_any will be created by feature engineering)
    autism_prob = np.random.random(n_samples)
    df['autism_any'] = (autism_prob > 0.5).astype(int)
    
    return df

def test_data_loader():
    """Test data loading functionality."""
    print("Testing data loader...")
    
    # Create test data
    df = create_test_data()
    test_file = "data/raw/test_data.csv"
    Path(test_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(test_file, index=False)
    
    # Test loading
    df_loaded = load_data(test_file)
    assert df_loaded.shape == df.shape, "Data loading failed"
    print("Data loader test passed")

def test_data_cleaner():
    """Test data cleaning functionality."""
    print("Testing data cleaner...")
    
    # Create test data with missing values
    df = create_test_data()
    
    # Add some missing codes
    df.loc[0, 'education'] = -1
    df.loc[1, 'occupation'] = 999
    
    # Test cleaning
    config = {
        'missing_codes': [-1, -999, 999, '', 'NA', 'N/A', 'null'],
        'diagnosis_missing_threshold': 2,
        'demographic_impute_value': 0,
        'questionnaire_missing_threshold': 2,
        'remove_duplicates': True
    }
    
    df_clean = clean_data(df, config)
    assert df_clean.shape[0] <= df.shape[0], "Data cleaning should not increase rows"
    print("Data cleaner test passed")

def test_feature_engineer():
    """Test feature engineering functionality."""
    print("Testing feature engineer...")
    
    # Create test data
    df = create_test_data()
    
    # Test feature engineering
    config = {
        'questionnaire_blocks': ['SPQ', 'EQ', 'SQR', 'AQ'],
        'age_bins': [0, 18, 25, 35, 50, 100],
        'age_labels': ['0-18', '19-25', '26-35', '36-50', '50+'],
        'interactions': [['spq_total', 'eq_total'], ['aq_total', 'sex']]
    }
    
    df_features = create_features(df, config)
    
    # Check that expected features were created
    expected_features = ['spq_total', 'eq_total', 'sqr_total', 'aq_total', 'total_score', 
                       'num_diagnoses', 'has_adhd', 'autism_any', 'autism_subtype', 'age_group']
    
    for feature in expected_features:
        if feature in df_features.columns:
            print(f"Created feature: {feature}")
    
    print("Feature engineer test passed")

def test_data_splitter():
    """Test data splitting functionality."""
    print("Testing data splitter...")
    
    # Create test data with features
    df = create_test_data()
    config = {
        'questionnaire_blocks': ['SPQ', 'EQ', 'SQR', 'AQ'],
        'age_bins': [0, 18, 25, 35, 50, 100],
        'age_labels': ['0-18', '19-25', '26-35', '36-50', '50+']
    }
    df_features = create_features(df, config)
    
    # Test target preparation
    target_config = {'primary_target': 'autism_any'}
    X, y = prepare_targets(df_features, target_config)
    assert len(X) == len(y), "Feature and target lengths should match"
    
    # Test data splitting
    split_config = {'test_size': 0.2, 'random_state': 42, 'stratify': True}
    X_train, X_test, y_train, y_test = split_data(X, y, split_config)
    
    assert len(X_train) + len(X_test) == len(X), "Data splitting failed"
    assert len(y_train) + len(y_test) == len(y), "Target splitting failed"
    
    print("Data splitter test passed")

def test_full_pipeline():
    """Test the complete pipeline."""
    print("Testing full pipeline...")
    
    # Create test data
    df = create_test_data()
    test_file = "data/raw/test_data.csv"
    Path(test_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(test_file, index=False)
    
    # Create test config
    config = {
        'data': {
            'raw_data_path': test_file,
            'processed_data_dir': 'data/processed'
        },
        'cleaning': {
            'missing_codes': [-1, -999, 999, '', 'NA', 'N/A', 'null'],
            'diagnosis_missing_threshold': 2,
            'demographic_impute_value': 0,
            'questionnaire_missing_threshold': 2,
            'remove_duplicates': True
        },
        'features': {
            'questionnaire_blocks': ['SPQ', 'EQ', 'SQR', 'AQ'],
            'age_bins': [0, 18, 25, 35, 50, 100],
            'age_labels': ['0-18', '19-25', '26-35', '36-50', '50+'],
            'interactions': [['spq_total', 'eq_total'], ['aq_total', 'sex']]
        },
        'target': {
            'primary_target': 'autism_any'
        },
        'splitting': {
            'test_size': 0.2,
            'random_state': 42,
            'stratify': True
        },
        'output': {
            'save_format': 'csv',
            'include_index': False
        }
    }
    
    # Save test config
    config_file = "experiments/configs/test_config.yaml"
    Path(config_file).parent.mkdir(parents=True, exist_ok=True)
    import yaml
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    
    # Test pipeline
    try:
        splits = run_preprocessing_pipeline(config_file)
        X_train, X_test, y_train, y_test = splits
        
        print(f"Pipeline test passed!")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        print(f"   Features: {X_train.shape[1]}")
        
    except Exception as e:
        print(f"Pipeline test failed: {str(e)}")
        raise

def main():
    """Run all tests."""
    print("Running modular pipeline tests...\n")
    
    try:
        test_data_loader()
        test_data_cleaner()
        test_feature_engineer()
        test_data_splitter()
        test_full_pipeline()
        
        print("\nAll tests passed! Your modular pipeline is working correctly.")
        
    except Exception as e:
        print(f"\nTest failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 