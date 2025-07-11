#!/usr/bin/env python3
"""
Results Analysis Script for Autism Prediction Experiments

This script helps you understand:
1. Where results are stored
2. How to compare different experiments
3. What the YAML files mean
4. How to interpret model performance

Usage:
    python analyze_results.py
    python analyze_results.py --experiment sampling_smote
    python analyze_results.py --compare-all
"""

import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
from typing import Dict, List, Any

def find_experiment_results():
    """Find all experiment results and their locations."""
    print("="*80)
    print("EXPERIMENT RESULTS LOCATIONS")
    print("="*80)
    
    # Check experiments/outputs directory
    outputs_dir = Path("experiments/outputs")
    if outputs_dir.exists():
        print(f"\n📁 Experiment Outputs: {outputs_dir}")
        for exp_dir in outputs_dir.iterdir():
            if exp_dir.is_dir():
                print(f"  📂 {exp_dir.name}")
                # Check for temp_config.yaml (shows what was actually run)
                config_file = exp_dir / "temp_config.yaml"
                if config_file.exists():
                    print(f"    📄 Config: {config_file}")
                
                # Check for other result files
                for file in exp_dir.iterdir():
                    if file.name != "temp_config.yaml":
                        print(f"    📄 {file.name}")
    
    # Check experiments/logs directory
    logs_dir = Path("experiments/logs")
    if logs_dir.exists():
        print(f"\n📁 Experiment Logs: {logs_dir}")
        for log_file in logs_dir.iterdir():
            if log_file.is_file():
                print(f"  📄 {log_file.name}")
    
    # Check experiments/models directory
    models_dir = Path("experiments/models")
    if models_dir.exists():
        print(f"\n📁 Trained Models: {models_dir}")
        for item in models_dir.iterdir():
            if item.is_file():
                print(f"  📄 {item.name}")
            elif item.is_dir():
                print(f"  📂 {item.name}/")
                for subitem in item.iterdir():
                    print(f"    📄 {subitem.name}")

def analyze_experiment_config(exp_name: str):
    """Analyze the configuration for a specific experiment."""
    config_file = Path(f"experiments/outputs/{exp_name}/temp_config.yaml")
    
    if not config_file.exists():
        print(f"❌ Config file not found: {config_file}")
        return
    
    print(f"\n🔍 Analyzing experiment: {exp_name}")
    print("="*50)
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract key settings
    print("📋 Experiment Configuration:")
    print(f"  • Imbalance handling: {config.get('imbalance_handling', {}).get('method', 'none')}")
    print(f"  • Feature engineering: {config.get('feature_engineering', {}).get('engineering_methods', [])}")
    print(f"  • Models: {list(config.get('models', {}).keys())}")
    print(f"  • Hyperparameter tuning: {config.get('hyperparameter_tuning', {}).get('method', 'none')}")
    print(f"  • CV folds: {config.get('hyperparameter_tuning', {}).get('cv', 'N/A')}")
    
    # Show what this experiment was testing
    if exp_name == "sampling_smote":
        print("\n🎯 This experiment tested SMOTE oversampling to handle class imbalance")
    elif exp_name == "feature_engineering_lr":
        print("\n🎯 This experiment tested feature engineering with logistic regression")
    elif exp_name == "model_comparison":
        print("\n🎯 This experiment compared multiple models (LR, RF, XGBoost)")
    elif exp_name == "nested_cv_xgb":
        print("\n🎯 This experiment used nested cross-validation with XGBoost")
    elif exp_name == "xgboost_grid":
        print("\n🎯 This experiment performed grid search hyperparameter tuning for XGBoost")

def find_model_results():
    """Find and analyze model results."""
    print("\n" + "="*80)
    print("MODEL RESULTS ANALYSIS")
    print("="*80)
    
    # Check main results file
    results_file = Path("experiments/logs/model_results.yaml")
    if results_file.exists():
        print(f"\n📊 Main Results File: {results_file}")
        try:
            with open(results_file, 'r') as f:
                results = yaml.safe_load(f)
            
            if results:
                print("\n📈 Model Performance Summary:")
                for model_name, metrics in results.items():
                    if isinstance(metrics, dict):
                        print(f"\n  🤖 {model_name.upper()}:")
                        for metric, value in metrics.items():
                            if isinstance(value, (int, float)):
                                print(f"    • {metric}: {value:.4f}")
                            else:
                                print(f"    • {metric}: {value}")
        except Exception as e:
            print(f"⚠️  Could not parse results file (contains numpy objects): {e}")
            print("💡 Check the file manually or look at individual experiment logs")
    
    # Check for individual experiment logs
    logs_dir = Path("experiments/logs")
    if logs_dir.exists():
        print(f"\n📄 Individual Experiment Logs:")
        for log_file in logs_dir.iterdir():
            if log_file.name.endswith('.log'):
                print(f"  📄 {log_file.name}")

def compare_experiments():
    """Compare results across different experiments."""
    print("\n" + "="*80)
    print("EXPERIMENT COMPARISON")
    print("="*80)
    
    # This would require parsing individual experiment results
    # For now, show what experiments were run
    outputs_dir = Path("experiments/outputs")
    if outputs_dir.exists():
        experiments = [d.name for d in outputs_dir.iterdir() if d.is_dir()]
        print(f"\n🔬 Experiments Run: {len(experiments)}")
        for exp in experiments:
            print(f"  • {exp}")
    
    print("\n💡 To compare experiments:")
    print("  1. Check individual log files in experiments/logs/")
    print("  2. Look for F1 scores, precision, recall in each experiment")
    print("  3. Compare the 'best' model from each experiment")

def explain_yaml_files():
    """Explain what the YAML files mean and how to use them."""
    print("\n" + "="*80)
    print("YAML FILES EXPLANATION")
    print("="*80)
    
    print("\n📋 temp_config.yaml files:")
    print("  • These show the ACTUAL configuration used for each experiment")
    print("  • They're created by the experiment scripts")
    print("  • They show what settings were tested")
    
    print("\n📋 experiments/configs/ files:")
    print("  • These are the TEMPLATE configurations")
    print("  • They define the base settings for experiments")
    print("  • The experiment scripts modify these for specific tests")
    
    print("\n🔧 How to optimize models:")
    print("  1. Look at the best performing experiment")
    print("  2. Check its temp_config.yaml for the winning settings")
    print("  3. Apply those settings to your main config files")
    print("  4. Re-run with the optimized settings")

def show_performance_metrics():
    """Show how to interpret performance metrics."""
    print("\n" + "="*80)
    print("PERFORMANCE METRICS GUIDE")
    print("="*80)
    
    print("\n📊 Key Metrics to Look For:")
    print("  • F1 Score: Best overall measure (precision + recall)")
    print("  • Precision: How many predicted positives were actually positive")
    print("  • Recall: How many actual positives were caught")
    print("  • ROC AUC: Overall model discrimination ability")
    print("  • Balanced Accuracy: Good for imbalanced datasets")
    
    print("\n🎯 For Autism Prediction:")
    print("  • High Recall is important (don't miss cases)")
    print("  • But Precision matters too (avoid false alarms)")
    print("  • F1 balances both concerns")
    print("  • ROC AUC > 0.8 is good, > 0.9 is excellent")

def main():
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument('--experiment', type=str, help='Analyze specific experiment')
    parser.add_argument('--compare-all', action='store_true', help='Compare all experiments')
    parser.add_argument('--explain', action='store_true', help='Explain YAML files and metrics')
    
    args = parser.parse_args()
    
    # Always show results locations
    find_experiment_results()
    
    # Analyze specific experiment if requested
    if args.experiment:
        analyze_experiment_config(args.experiment)
    
    # Find model results
    find_model_results()
    
    # Compare experiments if requested
    if args.compare_all:
        compare_experiments()
    
    # Explain YAML files and metrics
    if args.explain:
        explain_yaml_files()
        show_performance_metrics()
    else:
        print("\n💡 Run with --explain to understand YAML files and metrics")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. Check experiments/logs/ for detailed results")
    print("2. Look at experiments/models/ for trained models and plots")
    print("3. Compare F1 scores across experiments to find the best approach")
    print("4. Use the best settings in your main configuration files")

if __name__ == "__main__":
    main() 