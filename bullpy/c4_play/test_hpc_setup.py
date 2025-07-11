#!/usr/bin/env python3
"""
Test script to verify HPC setup before running full experiments.
"""
import sys
import os
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import sklearn
        print("✅ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"❌ Scikit-learn import failed: {e}")
        return False
    
    try:
        import yaml
        print("✅ PyYAML imported successfully")
    except ImportError as e:
        print(f"❌ PyYAML import failed: {e}")
        return False
    
    return True

def test_data_access():
    """Test that data file can be accessed."""
    print("\nTesting data access...")
    
    data_path = "data/processed/features_full.csv"
    if os.path.exists(data_path):
        print(f"✅ Data file found: {data_path}")
        return True
    else:
        print(f"❌ Data file not found: {data_path}")
        print("Available files in data/processed/:")
        data_dir = Path("data/processed/")
        if data_dir.exists():
            for file in data_dir.iterdir():
                print(f"  - {file.name}")
        else:
            print("  data/processed/ directory not found")
        return False

def test_model_training_import():
    """Test that model training module can be imported."""
    print("\nTesting model training import...")
    
    try:
        sys.path.append(str(Path(__file__).parent / 'src'))
        from model_training import ClinicalModelTrainer
        print("✅ Model training module imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Model training import failed: {e}")
        return False

def test_experiment_scripts():
    """Test that experiment scripts exist."""
    print("\nTesting experiment scripts...")
    
    scripts = [
        "scripts/experiments/experiment_smote_improved.py",
        "scripts/experiments/experiment_feature_engineering_improved.py",
        "scripts/experiments/experiment_hyperparameter_tuning_improved.py",
        "run_all_experiments_improved.py",
        "extract_all_experiment_results.py"
    ]
    
    all_exist = True
    for script in scripts:
        if os.path.exists(script):
            print(f"✅ {script} exists")
        else:
            print(f"❌ {script} not found")
            all_exist = False
    
    return all_exist

def main():
    print("=" * 60)
    print("HPC SETUP TEST")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_data_access,
        test_model_training_import,
        test_experiment_scripts
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed! HPC setup is ready.")
        print("You can now run: sbatch run_experiments_hpc.slurm")
    else:
        print("❌ Some tests failed. Please fix the issues before running experiments.")
    
    return passed == total

if __name__ == "__main__":
    main() 