#!/usr/bin/env python3
"""
Fix import issues and rerun failed experiments.
"""

import os
import sys
import subprocess
from pathlib import Path

def fix_import_issues():
    """Fix the import issues in experiment scripts."""
    print("🔧 Fixing import issues...")
    
    # The issue is that experiment scripts are trying to import from 'model_training'
    # but the function is in 'src.model_training'
    
    experiment_scripts = [
        "scripts/experiments/experiment_sampling.py",
        "scripts/experiments/experiment_feature_engineering.py", 
        "scripts/experiments/experiment_xgboost_grid.py"
    ]
    
    for script in experiment_scripts:
        if os.path.exists(script):
            print(f"Fixing {script}...")
            
            # Read the file
            with open(script, 'r') as f:
                content = f.read()
            
            # Fix the import
            old_import = "from model_training import run_modular_training_pipeline"
            new_import = "from src.model_training import run_modular_training_pipeline"
            
            if old_import in content:
                content = content.replace(old_import, new_import)
                
                # Write back
                with open(script, 'w') as f:
                    f.write(content)
                print(f"  ✅ Fixed import in {script}")
            else:
                print(f"  ⚠️  Import not found in {script}")

def fix_nested_cv_arguments():
    """Fix the nested CV script arguments."""
    print("\n🔧 Fixing nested CV arguments...")
    
    script = "scripts/experiments/experiment_nested_cv.py"
    if os.path.exists(script):
        print(f"Fixing {script}...")
        
        with open(script, 'r') as f:
            content = f.read()
        
        # Check if it has the right argument parsing
        if '--config' in content and '--output-dir' in content:
            print("  ✅ Arguments already present")
        else:
            print("  ⚠️  Need to check argument parsing")

def run_single_experiment(experiment_name, command):
    """Run a single experiment."""
    print(f"\n🚀 Running {experiment_name}...")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"  ✅ {experiment_name} completed successfully!")
            print(f"  Output: {result.stdout}")
        else:
            print(f"  ❌ {experiment_name} failed!")
            print(f"  Error: {result.stderr}")
            
    except Exception as e:
        print(f"  ❌ Error running {experiment_name}: {e}")

def main():
    """Main function to fix and rerun experiments."""
    print("🔧 FIXING AND RERUNNING FAILED EXPERIMENTS")
    print("=" * 50)
    
    # Fix import issues
    fix_import_issues()
    fix_nested_cv_arguments()
    
    print("\n📋 EXPERIMENTS TO RERUN:")
    print("1. Sampling (SMOTE)")
    print("2. Feature Engineering") 
    print("3. Nested CV")
    print("4. XGBoost Grid Search")
    
    print("\n🚀 Ready to run experiments!")
    print("To run them manually:")
    print("cd /home/eb2007/predict_asc_c4")
    print("python scripts/experiments/experiment_sampling.py --method SMOTE --config experiments/configs/model_config.yaml --output_dir experiments/outputs/sampling_smote")
    print("python scripts/experiments/experiment_feature_engineering.py --model logistic_regression --config experiments/configs/model_config.yaml --output_dir experiments/outputs/feature_engineering_lr")
    print("python scripts/experiments/experiment_nested_cv.py --model xgboost --outer_folds 3 --inner_folds 3 --config experiments/configs/model_config.yaml --output_dir experiments/outputs/nested_cv_xgb")
    print("python scripts/experiments/experiment_xgboost_grid.py --config experiments/configs/model_config.yaml --output_dir experiments/outputs/xgboost_grid")

if __name__ == "__main__":
    main() 