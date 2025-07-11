#!/usr/bin/env python3
"""
Enhanced SMOTE Experiment for Autism Prediction
Focuses on improving F1 score through various SMOTE techniques.
"""
import argparse
import sys
import os
from pathlib import Path
import yaml
import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))
from model_training import run_modular_training_pipeline

def create_smote_config(output_dir: str, smote_method: str = 'smote'):
    """Create configuration optimized for F1 score with SMOTE."""
    
    config = {
        'data_path': 'data/processed/features_full.csv',
        'splitting': {
            'test_size': 0.2,
            'val_size': 0.2,
            'random_state': 42,
            'stratify': True
        },
        'models': {
            'logistic_regression': {
                'random_state': 42,
                'max_iter': 2000,
                'class_weight': 'balanced',
                'solver': 'liblinear',
                'C': 1.0
            },
            'random_forest': {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1,
                'class_weight': 'balanced'
            },
            'xgboost': {
                'n_estimators': 300,
                'max_depth': 8,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'eval_metric': 'logloss',
                'scale_pos_weight': 2.0
            }
        },
        'evaluation': {
            'cv_folds': 5,
            'scoring_metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'balanced_accuracy'],
            'probability_threshold': 0.5,
            'sensitivity_threshold': 0.8,
            'specificity_threshold': 0.8
        },
        'output': {
            'save_models': True,
            'save_predictions': True,
            'create_plots': True,
            'output_dir': output_dir,
            'results_file': os.path.join(output_dir, 'model_results.yaml')
        },
        'imbalance_handling': {
            'method': smote_method,  # 'smote', 'borderline_smote', 'svm_smote', 'adasyn'
            'random_state': 42,
            'k_neighbors': 5
        },
        'feature_engineering': {
            'engineering_methods': [],
            'selection_method': 'none'
        },
        'hyperparameter_tuning': {
            'method': 'grid',
            'param_grid': {},
            'scoring': 'f1',
            'cv': 5
        }
    }
    
    return config

def optimize_threshold_for_f1(y_true, y_pred_proba):
    """Optimize threshold to maximize F1 score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx] if best_idx < len(thresholds) else 0.5

def main():
    parser = argparse.ArgumentParser(description='Enhanced SMOTE Experiment for Autism Prediction')
    parser.add_argument('--smote-method', type=str, default='smote', 
                       choices=['smote', 'borderline_smote', 'svm_smote', 'adasyn'],
                       help='SMOTE method to use')
    parser.add_argument('--output-dir', type=str, required=True, 
                       help='Output directory for results')
    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create optimized configuration
    config = create_smote_config(args.output_dir, args.smote_method)
    
    # Save configuration
    config_path = os.path.join(args.output_dir, 'temp_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"Running SMOTE experiment with method: {args.smote_method}")
    print(f"Configuration saved to: {config_path}")
    print(f"Results will be saved to: {args.output_dir}")
    
    # Run the experiment
    run_modular_training_pipeline(config_path)
    
    print(f"SMOTE experiment completed! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 