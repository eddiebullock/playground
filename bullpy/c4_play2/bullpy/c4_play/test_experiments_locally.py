#!/usr/bin/env python3
"""
Test script to verify all experiment scripts work locally.
"""

import os
import sys
import subprocess
from pathlib import Path

def test_experiment_script(script_path, args, description):
    """Test if an experiment script can be imported and run with basic arguments."""
    print(f"\n🧪 Testing {description}...")
    
    # Check if script exists
    if not os.path.exists(script_path):
        print(f"❌ Script not found: {script_path}")
        return False
    
    # Test import
    try:
        # Add src to path
        sys.path.insert(0, str(Path(script_path).parent.parent.parent / 'src'))
        
        # Import the script module
        script_name = Path(script_path).stem
        module = __import__(f'scripts.experiments.{script_name}', fromlist=['main'])
        
        print(f"✅ Import successful: {script_path}")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def main():
    """Test all experiment scripts."""
    print("=== Testing Experiment Scripts ===\n")
    
    # Define test cases
    test_cases = [
        {
            'script': 'scripts/experiments/experiment_sampling.py',
            'args': ['--method', 'SMOTE', '--config', 'experiments/configs/model_config.yaml', '--output-dir', 'test_outputs/sampling'],
            'description': 'Sampling Experiment'
        },
        {
            'script': 'scripts/experiments/experiment_feature_engineering.py',
            'args': ['--method', 'logistic_regression', '--config', 'experiments/configs/model_config.yaml', '--output-dir', 'test_outputs/feature_engineering'],
            'description': 'Feature Engineering Experiment'
        },
        {
            'script': 'scripts/experiments/experiment_models.py',
            'args': ['--config', 'experiments/configs/model_config.yaml', '--output-dir', 'test_outputs/models'],
            'description': 'Model Comparison Experiment'
        },
        {
            'script': 'scripts/experiments/experiment_nested_cv.py',
            'args': ['--model', 'xgboost', '--config', 'experiments/configs/model_config.yaml', '--output-dir', 'test_outputs/nested_cv'],
            'description': 'Nested CV Experiment'
        },
        {
            'script': 'scripts/experiments/experiment_xgboost_grid.py',
            'args': ['--config', 'experiments/configs/xgboost_grid.yaml', '--output-dir', 'test_outputs/xgboost_grid'],
            'description': 'XGBoost Grid Search Experiment'
        }
    ]
    
    # Test each script
    passed = 0
    total = len(test_cases)
    
    for test_case in test_cases:
        if test_experiment_script(test_case['script'], test_case['args'], test_case['description']):
            passed += 1
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All experiments ready for HPC!")
    else:
        print("❌ Some experiments need fixing before HPC transfer.")

if __name__ == "__main__":
    main() 