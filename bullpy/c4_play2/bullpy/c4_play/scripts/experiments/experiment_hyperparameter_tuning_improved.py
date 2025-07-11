#!/usr/bin/env python3
"""
Enhanced Hyperparameter Tuning Experiment for Autism Prediction
Focuses on improving F1 score through comprehensive hyperparameter optimization.
"""
import argparse
import sys
import os
from pathlib import Path
import yaml
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))
from model_training import run_modular_training_pipeline

def create_hyperparameter_config(output_dir: str, tuning_method: str = 'grid', model: str = 'xgboost'):
    """Create configuration optimized for F1 score with hyperparameter tuning."""
    
    # Define comprehensive parameter grids for each model
    param_grids = {
        'logistic_regression': {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'class_weight': ['balanced', None],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000, 2000]
        },
        'random_forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', 'balanced_subsample', None]
        },
        'xgboost': {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'scale_pos_weight': [1.0, 2.0, 3.0, 5.0]
        }
    }
    
    config = {
        'data_path': 'data/processed/features_full.csv',
        'splitting': {
            'test_size': 0.2,
            'val_size': 0.2,
            'random_state': 42,
            'stratify': True
        },
        'models': {
            model: {}  # Will be filled with best parameters
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
            'engineering_methods': [],
            'selection_method': 'importance',
            'max_features': 30
        },
        'hyperparameter_tuning': {
            'method': tuning_method,
            'param_grid': param_grids[model],
            'scoring': 'f1',
            'cv': 5,
            'n_iter': 50 if tuning_method == 'random' else None
        }
    }
    
    return config

def create_xgboost_optimization_config(output_dir: str):
    """Create configuration for XGBoost hyperparameter optimization."""
    config = create_hyperparameter_config(output_dir, 'grid', 'xgboost')
    config['hyperparameter_tuning']['param_grid'] = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 8],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'scale_pos_weight': [2.0, 3.0, 5.0]
    }
    return config

def create_logistic_regression_optimization_config(output_dir: str):
    """Create configuration for Logistic Regression hyperparameter optimization."""
    config = create_hyperparameter_config(output_dir, 'grid', 'logistic_regression')
    config['hyperparameter_tuning']['param_grid'] = {
        'C': [0.01, 0.1, 1.0, 10.0],
        'class_weight': ['balanced', None],
        'solver': ['liblinear'],
        'max_iter': [2000]
    }
    return config

def create_random_forest_optimization_config(output_dir: str):
    """Create configuration for Random Forest hyperparameter optimization."""
    config = create_hyperparameter_config(output_dir, 'grid', 'random_forest')
    config['hyperparameter_tuning']['param_grid'] = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    return config

def main():
    parser = argparse.ArgumentParser(description='Enhanced Hyperparameter Tuning Experiment for Autism Prediction')
    parser.add_argument('--model', type=str, default='xgboost', 
                       choices=['xgboost', 'logistic_regression', 'random_forest'],
                       help='Model to optimize')
    parser.add_argument('--tuning-method', type=str, default='grid', 
                       choices=['grid', 'random'],
                       help='Hyperparameter tuning method')
    parser.add_argument('--output-dir', type=str, required=True, 
                       help='Output directory for results')
    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create optimized configuration based on model
    if args.model == 'xgboost':
        config = create_xgboost_optimization_config(args.output_dir)
    elif args.model == 'logistic_regression':
        config = create_logistic_regression_optimization_config(args.output_dir)
    elif args.model == 'random_forest':
        config = create_random_forest_optimization_config(args.output_dir)
    else:
        config = create_hyperparameter_config(args.output_dir, args.tuning_method, args.model)
    
    # Update tuning method
    config['hyperparameter_tuning']['method'] = args.tuning_method
    
    # Save configuration
    config_path = os.path.join(args.output_dir, 'temp_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"Running hyperparameter tuning experiment for {args.model} with {args.tuning_method} search")
    print(f"Configuration saved to: {config_path}")
    print(f"Results will be saved to: {args.output_dir}")
    
    # Run the experiment
    run_modular_training_pipeline(config_path)
    
    print(f"Hyperparameter tuning experiment completed! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 