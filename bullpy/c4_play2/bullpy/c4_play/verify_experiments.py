#!/usr/bin/env python3
"""
Verification script to check that all experiment scripts exist and are properly configured.
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and is readable."""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} - MISSING")
        return False

def check_directory_structure():
    """Check the directory structure for experiments."""
    print("=== Checking Directory Structure ===")
    
    required_dirs = [
        "scripts/experiments",
        "experiments/configs", 
        "experiments/outputs",
        "experiments/logs",
        "src/modules",
        "data/processed"
    ]
    
    all_good = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ Directory: {dir_path}")
        else:
            print(f"❌ Directory: {dir_path} - MISSING")
            all_good = False
    
    return all_good

def check_experiment_scripts():
    """Check that all experiment scripts exist."""
    print("\n=== Checking Experiment Scripts ===")
    
    experiment_scripts = [
        ("scripts/experiments/experiment_sampling.py", "Sampling Experiment"),
        ("scripts/experiments/experiment_feature_engineering.py", "Feature Engineering Experiment"),
        ("scripts/experiments/experiment_models.py", "Models Experiment"),
        ("scripts/experiments/experiment_nested_cv.py", "Nested CV Experiment"),
        ("scripts/experiments/experiment_xgboost_grid.py", "XGBoost Grid Search Experiment")
    ]
    
    all_good = True
    for script_path, description in experiment_scripts:
        if not check_file_exists(script_path, description):
            all_good = False
    
    return all_good

def check_config_files():
    """Check that configuration files exist."""
    print("\n=== Checking Configuration Files ===")
    
    config_files = [
        ("experiments/configs/data_config.yaml", "Data Configuration"),
        ("experiments/configs/model_config.yaml", "Model Configuration"),
        ("experiments/configs/test_config.yaml", "Test Configuration"),
        ("experiments/configs/xgboost_grid.yaml", "XGBoost Grid Configuration")
    ]
    
    all_good = True
    for config_path, description in config_files:
        if not check_file_exists(config_path, description):
            all_good = False
    
    return all_good

def check_data_files():
    """Check that data files exist."""
    print("\n=== Checking Data Files ===")
    
    data_files = [
        ("data/processed/features_full.csv", "Processed Features Data")
    ]
    
    all_good = True
    for data_path, description in data_files:
        if not check_file_exists(data_path, description):
            all_good = False
    
    return all_good

def check_slurm_scripts():
    """Check that SLURM scripts exist."""
    print("\n=== Checking SLURM Scripts ===")
    
    slurm_scripts = [
        ("optimized_experiments_array.slurm", "Optimized Experiments Array Script"),
        ("rerun_failed_experiments.slurm", "Rerun Failed Experiments Script")
    ]
    
    all_good = True
    for script_path, description in slurm_scripts:
        if not check_file_exists(script_path, description):
            all_good = False
    
    return all_good

def main():
    """Run all verification checks."""
    print("=== Experiment Verification Script ===\n")
    
    checks = [
        check_directory_structure,
        check_experiment_scripts,
        check_config_files,
        check_data_files,
        check_slurm_scripts
    ]
    
    all_passed = True
    for check in checks:
        if not check():
            all_passed = False
        print()
    
    if all_passed:
        print("🎉 ALL CHECKS PASSED - Ready to run experiments!")
        print("\nTo submit the optimized array job:")
        print("sbatch optimized_experiments_array.slurm")
    else:
        print("⚠️  SOME CHECKS FAILED - Please fix missing files/directories before running experiments.")
    
    return all_passed

if __name__ == "__main__":
    main() 