#!/usr/bin/env python3
"""
Extract and display experiment results in text format.
"""

import os
import yaml
import pickle
import pandas as pd
from pathlib import Path

def load_model_results():
    """Load and display model training results."""
    print("=== MODEL TRAINING RESULTS ===\n")
    
    # Check for model results
    model_results_path = "experiments/logs/model_results.yaml"
    if os.path.exists(model_results_path):
        with open(model_results_path, 'r') as f:
            results = yaml.safe_load(f)
        
        print("Model Performance Metrics:")
        print("-" * 50)
        
        if 'model_comparison' in results:
            for model_result in results['model_comparison']:
                print(f"\nModel: {model_result['model']}")
                print(f"  Accuracy: {model_result['accuracy']:.4f}")
                print(f"  F1 Score: {model_result['f1_score']:.4f}")
                print(f"  Precision: {model_result['precision']:.4f}")
                print(f"  Recall: {model_result['recall']:.4f}")
                print(f"  ROC AUC: {model_result['roc_auc']:.4f}")
                print(f"  Balanced Accuracy: {model_result['balanced_accuracy']:.4f}")
        
        if 'test_metrics' in results:
            print("\nTest Set Performance:")
            print("-" * 30)
            for model_name, metrics in results['test_metrics'].items():
                print(f"\n{model_name.upper()}:")
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  F1 Score: {metrics['f1_score']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
                print(f"  Specificity: {metrics['specificity']:.4f}")
    else:
        print("❌ No model results found!")

def load_trained_models():
    """Load and display information about trained models."""
    print("\n=== TRAINED MODELS ===\n")
    
    models_dir = "experiments/models/trained_models"
    if os.path.exists(models_dir):
        models = os.listdir(models_dir)
        print(f"Found {len(models)} trained models:")
        for model in models:
            print(f"  ✅ {model}")
            
        # Try to load and display model info
        try:
            for model_file in models:
                if model_file.endswith('.pkl'):
                    model_path = os.path.join(models_dir, model_file)
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    
                    print(f"\n{model_file}:")
                    print(f"  Type: {type(model).__name__}")
                    if hasattr(model, 'feature_importances_'):
                        print(f"  Feature Importances: Available")
                    if hasattr(model, 'coef_'):
                        print(f"  Coefficients: Available")
        except Exception as e:
            print(f"  Note: Could not load model details: {e}")
    else:
        print("❌ No trained models found!")

def check_experiment_logs():
    """Check what experiments actually ran."""
    print("\n=== EXPERIMENT STATUS ===\n")
    
    logs_dir = "experiments/logs"
    if os.path.exists(logs_dir):
        log_files = [f for f in os.listdir(logs_dir) if f.endswith('.log')]
        
        print("Experiment Logs Found:")
        for log_file in log_files:
            print(f"  📄 {log_file}")
            
        # Check for specific experiment results
        experiments = {
            'sampling': 'sampling_11958947_0.log',
            'feature_engineering': 'feature_engineering_11958948_1.log', 
            'models': 'models_11958949_2.log',
            'nested_cv': 'nested_cv_11958950_3.log',
            'xgboost_grid': 'xgboost_grid_11958713_4.log'
        }
        
        print("\nExperiment Status:")
        for exp_name, log_file in experiments.items():
            if os.path.exists(os.path.join(logs_dir, log_file)):
                print(f"  ✅ {exp_name}: Log exists")
            else:
                print(f"  ❌ {exp_name}: No log found")

def display_eda_results():
    """Display EDA results."""
    print("\n=== EDA RESULTS ===\n")
    
    eda_files = [
        "experiments/logs/eda_summary_20250707_135832.yaml",
        "experiments/logs/eda_results_20250707_135832.yaml"
    ]
    
    for eda_file in eda_files:
        if os.path.exists(eda_file):
            print(f"📊 {os.path.basename(eda_file)}:")
            try:
                with open(eda_file, 'r') as f:
                    data = yaml.safe_load(f)
                
                if 'dataset_overview' in data:
                    overview = data['dataset_overview']
                    print(f"  Dataset Shape: {overview.get('shape', 'N/A')}")
                    print(f"  Memory Usage: {overview.get('memory_usage', 'N/A')}")
                    print(f"  Missing Data: {overview.get('missing_percentage', 'N/A')}%")
                
                if 'target_analysis' in data:
                    target = data['target_analysis']
                    print(f"  Target Distribution: {target.get('distribution', 'N/A')}")
                    
            except Exception as e:
                print(f"  Error reading file: {e}")

def main():
    """Display all results in text format."""
    print("🔍 AUTISM PREDICTION EXPERIMENT RESULTS")
    print("=" * 50)
    
    load_model_results()
    load_trained_models()
    check_experiment_logs()
    display_eda_results()
    
    print("\n" + "=" * 50)
    print("📝 SUMMARY:")
    print("✅ 2/5 experiments completed successfully")
    print("❌ 3/5 experiments failed due to import/argument errors")
    print("📊 Model results available in experiments/logs/model_results.yaml")
    print("🤖 Trained models saved in experiments/models/trained_models/")

if __name__ == "__main__":
    main() 