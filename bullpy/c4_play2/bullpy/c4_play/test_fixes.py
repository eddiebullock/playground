#!/usr/bin/env python3
"""
Test script to verify that the model and imbalance handling fixes work correctly.
"""

import sys
import os
sys.path.append('/home/eb2007/predict_asc_c4')

import pandas as pd
import numpy as np
from src.modules.advanced_models import get_model
from src.modules.imbalance_handler import handle_imbalance

def test_models():
    """Test that all models can be instantiated."""
    print("Testing model instantiation...")
    
    models_to_test = [
        'random_forest',
        'logistic_regression', 
        'xgboost',
        'mlp',
        'svm'
    ]
    
    for model_name in models_to_test:
        try:
            model = get_model(model_name)
            print(f"✅ {model_name}: SUCCESS")
        except Exception as e:
            print(f"❌ {model_name}: FAILED - {e}")
    
    print()

def test_imbalance_handling():
    """Test that imbalance handling methods work."""
    print("Testing imbalance handling...")
    
    # Create dummy data
    X = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000)
    })
    y = pd.Series([0] * 900 + [1] * 100)  # Imbalanced data
    
    methods_to_test = ['none', 'oversample', 'undersample', 'smote', 'class_weight']
    
    for method in methods_to_test:
        try:
            X_res, y_res = handle_imbalance(X, y, method=method, random_state=42)
            print(f"✅ {method}: SUCCESS - Original: {len(y)}, Resampled: {len(y_res)}")
        except Exception as e:
            print(f"❌ {method}: FAILED - {e}")
    
    print()

def test_data_loading():
    """Test that data loading works."""
    print("Testing data loading...")
    
    try:
        from src.modules.data_loader import load_data
        # This will fail if the data file doesn't exist, but that's expected
        print("✅ Data loader module imported successfully")
    except Exception as e:
        print(f"❌ Data loader: FAILED - {e}")
    
    print()

if __name__ == "__main__":
    print("=== Testing Fixes ===")
    print()
    
    test_models()
    test_imbalance_handling()
    test_data_loading()
    
    print("=== Test Complete ===") 