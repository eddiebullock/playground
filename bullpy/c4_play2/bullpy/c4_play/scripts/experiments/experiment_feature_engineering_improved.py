#!/usr/bin/env python3
"""
Enhanced Feature Engineering Experiment for Autism Prediction
Focuses on improving F1 score through various feature engineering techniques.
"""
import argparse
import sys
import os
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))
from model_training import run_modular_training_pipeline

def create_feature_engineering_config(output_dir: str, method: str = 'polynomial'):
    """Create configuration optimized for F1 score with feature engineering."""
    
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
                'C': 0.1  # Reduced C for better generalization with engineered features
            },
            'random_forest': {
                'n_estimators': 200,
                'max_depth': 12,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1,
                'class_weight': 'balanced'
            },
            'xgboost': {
                'n_estimators': 250,
                'max_depth': 6,
                'learning_rate': 0.05,
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
            'method': 'smote',
            'random_state': 42
        },
        'feature_engineering': {
            'engineering_methods': [method],
            'selection_method': 'importance',
            'max_features': 50
        },
        'hyperparameter_tuning': {
            'method': 'grid',
            'param_grid': {},
            'scoring': 'f1',
            'cv': 5
        }
    }
    
    return config

def create_interaction_features_config(output_dir: str):
    """Create configuration for interaction features experiment."""
    config = create_feature_engineering_config(output_dir, 'interaction')
    config['feature_engineering']['engineering_methods'] = ['interaction']
    config['feature_engineering']['interaction_pairs'] = [
        ['aq_5', 'aq_8'],
        ['eq_3', 'eq_6'],
        ['spq_1', 'spq_10'],
        ['sqr_4', 'sqr_5']
    ]
    return config

def create_polynomial_features_config(output_dir: str):
    """Create configuration for polynomial features experiment."""
    config = create_feature_engineering_config(output_dir, 'polynomial')
    config['feature_engineering']['polynomial_degree'] = 2
    config['feature_engineering']['max_features'] = 100
    return config

def create_aggregate_features_config(output_dir: str):
    """Create configuration for aggregate features experiment."""
    config = create_feature_engineering_config(output_dir, 'aggregate')
    config['feature_engineering']['aggregate_groups'] = {
        'aq_features': ['aq_5', 'aq_8'],
        'eq_features': ['eq_3', 'eq_6', 'eq_8'],
        'spq_features': ['spq_1', 'spq_8', 'spq_10'],
        'sqr_features': ['sqr_4', 'sqr_5']
    }
    return config

def main():
    parser = argparse.ArgumentParser(description='Enhanced Feature Engineering Experiment for Autism Prediction')
    parser.add_argument('--method', type=str, default='polynomial', 
                       choices=['polynomial', 'interaction', 'aggregate', 'selection'],
                       help='Feature engineering method to use')
    parser.add_argument('--output-dir', type=str, required=True, 
                       help='Output directory for results')
    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create optimized configuration based on method
    if args.method == 'interaction':
        config = create_interaction_features_config(args.output_dir)
    elif args.method == 'polynomial':
        config = create_polynomial_features_config(args.output_dir)
    elif args.method == 'aggregate':
        config = create_aggregate_features_config(args.output_dir)
    else:
        config = create_feature_engineering_config(args.output_dir, args.method)
    
    # Save configuration
    config_path = os.path.join(args.output_dir, 'temp_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"Running feature engineering experiment with method: {args.method}")
    print(f"Configuration saved to: {config_path}")
    print(f"Results will be saved to: {args.output_dir}")
    
    # Run the experiment
    run_modular_training_pipeline(config_path)
    
    print(f"Feature engineering experiment completed! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 