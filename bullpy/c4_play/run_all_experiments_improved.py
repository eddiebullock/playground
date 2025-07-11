#!/usr/bin/env python3
"""
Master script to run all improved experiments for autism prediction.
Each experiment is optimized for F1 score and saves results to its own directory.
"""
import os
import sys
import subprocess
import time
from pathlib import Path
import argparse

def run_experiment(script_path, args, experiment_name):
    """Run a single experiment and capture output."""
    print(f"\n{'='*80}")
    print(f"RUNNING EXPERIMENT: {experiment_name}")
    print(f"{'='*80}")
    
    cmd = [sys.executable, script_path] + args
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        end_time = time.time()
        print(f"✅ {experiment_name} completed successfully in {end_time - start_time:.1f} seconds")
        print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {experiment_name} failed")
        print(f"Error: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run all improved experiments for autism prediction')
    parser.add_argument('--output-base-dir', type=str, default='experiments/outputs',
                       help='Base directory for experiment outputs')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip experiments that already have results')
    args = parser.parse_args()
    
    # Create base output directory
    Path(args.output_base_dir).mkdir(parents=True, exist_ok=True)
    
    # Define all experiments to run
    experiments = [
        # SMOTE Experiments
        {
            'name': 'smote_basic',
            'script': 'scripts/experiments/experiment_smote_improved.py',
            'args': ['--smote-method', 'smote', '--output-dir', f'{args.output_base_dir}/smote_basic']
        },
        {
            'name': 'smote_borderline',
            'script': 'scripts/experiments/experiment_smote_improved.py',
            'args': ['--smote-method', 'borderline_smote', '--output-dir', f'{args.output_base_dir}/smote_borderline']
        },
        {
            'name': 'smote_adasyn',
            'script': 'scripts/experiments/experiment_smote_improved.py',
            'args': ['--smote-method', 'adasyn', '--output-dir', f'{args.output_base_dir}/smote_adasyn']
        },
        
        # Feature Engineering Experiments
        {
            'name': 'feature_engineering_polynomial',
            'script': 'scripts/experiments/experiment_feature_engineering_improved.py',
            'args': ['--method', 'polynomial', '--output-dir', f'{args.output_base_dir}/feature_engineering_polynomial']
        },
        {
            'name': 'feature_engineering_interaction',
            'script': 'scripts/experiments/experiment_feature_engineering_improved.py',
            'args': ['--method', 'interaction', '--output-dir', f'{args.output_base_dir}/feature_engineering_interaction']
        },
        {
            'name': 'feature_engineering_aggregate',
            'script': 'scripts/experiments/experiment_feature_engineering_improved.py',
            'args': ['--method', 'aggregate', '--output-dir', f'{args.output_base_dir}/feature_engineering_aggregate']
        },
        
        # Hyperparameter Tuning Experiments
        {
            'name': 'hyperparameter_xgboost',
            'script': 'scripts/experiments/experiment_hyperparameter_tuning_improved.py',
            'args': ['--model', 'xgboost', '--tuning-method', 'grid', '--output-dir', f'{args.output_base_dir}/hyperparameter_xgboost']
        },
        {
            'name': 'hyperparameter_logistic_regression',
            'script': 'scripts/experiments/experiment_hyperparameter_tuning_improved.py',
            'args': ['--model', 'logistic_regression', '--tuning-method', 'grid', '--output-dir', f'{args.output_base_dir}/hyperparameter_logistic_regression']
        },
        {
            'name': 'hyperparameter_random_forest',
            'script': 'scripts/experiments/experiment_hyperparameter_tuning_improved.py',
            'args': ['--model', 'random_forest', '--tuning-method', 'grid', '--output-dir', f'{args.output_base_dir}/hyperparameter_random_forest']
        }
    ]
    
    # Track results
    successful_experiments = []
    failed_experiments = []
    
    print(f"Starting {len(experiments)} experiments...")
    print(f"Output base directory: {args.output_base_dir}")
    
    for i, experiment in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Running {experiment['name']}...")
        
        # Check if experiment already exists and should be skipped
        output_dir = experiment['args'][-1]  # Last argument is output directory
        if args.skip_existing and Path(output_dir).exists():
            results_file = Path(output_dir) / 'model_results.yaml'
            if results_file.exists():
                print(f"⏭️  Skipping {experiment['name']} (results already exist)")
                successful_experiments.append(experiment['name'])
                continue
        
        # Run experiment
        success = run_experiment(experiment['script'], experiment['args'], experiment['name'])
        
        if success:
            successful_experiments.append(experiment['name'])
        else:
            failed_experiments.append(experiment['name'])
    
    # Summary
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"Total experiments: {len(experiments)}")
    print(f"Successful: {len(successful_experiments)}")
    print(f"Failed: {len(failed_experiments)}")
    
    if successful_experiments:
        print(f"\n✅ Successful experiments:")
        for exp in successful_experiments:
            print(f"  - {exp}")
    
    if failed_experiments:
        print(f"\n❌ Failed experiments:")
        for exp in failed_experiments:
            print(f"  - {exp}")
    
    print(f"\n📁 All results saved to: {args.output_base_dir}")
    print("💡 Run 'python extract_all_experiment_results.py' to analyze all results")

if __name__ == "__main__":
    main() 