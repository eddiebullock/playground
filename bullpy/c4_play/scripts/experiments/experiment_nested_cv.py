#!/usr/bin/env python3
"""
Experiment: Run nested cross-validation using the modular pipeline.
Usage:
    python experiment_nested_cv.py --outer-folds 5 --inner-folds 3 --model XGBoost --config ../../experiments/configs/model_config.yaml --output-dir ../../experiments/outputs/nested_cv_XGBoost
"""
import argparse
import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))
from model_training import ClinicalModelTrainer
from sklearn.model_selection import StratifiedKFold

def run_nested_cv(config, outer_folds, inner_folds, model_name, output_dir):
    import numpy as np
    import pandas as pd
    from model_training import run_modular_training_pipeline
    # Load data
    data_path = config.get('data_path', 'data/processed/features_full.csv')
    trainer = ClinicalModelTrainer(output_dir=output_dir)
    X, y = trainer.load_data(data_path)
    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=42)
    results = []
    for i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        # Save split to temp files
        split_dir = Path(output_dir) / f'fold_{i+1}'
        split_dir.mkdir(parents=True, exist_ok=True)
        X_train.to_csv(split_dir / 'X_train.csv', index=False)
        X_test.to_csv(split_dir / 'X_test.csv', index=False)
        y_train.to_csv(split_dir / 'y_train.csv', index=False)
        y_test.to_csv(split_dir / 'y_test.csv', index=False)
        # Update config for this fold
        fold_config = config.copy()
        fold_config['data_path'] = str(split_dir / 'X_train.csv')  # Use only train for inner CV
        fold_config['models'] = {model_name: config['models'].get(model_name, {})}
        fold_config['output']['output_dir'] = str(split_dir)
        # Save fold config
        fold_config_path = split_dir / 'fold_config.yaml'
        with open(fold_config_path, 'w') as f:
            yaml.dump(fold_config, f)
        # Run modular training pipeline (inner CV handled by pipeline's tuning)
        run_modular_training_pipeline(str(fold_config_path))
        # Optionally, evaluate on X_test/y_test here
        # ...
        results.append({'fold': i+1, 'output_dir': str(split_dir)})
    print('Nested CV complete. Results:', results)

def main():
    parser = argparse.ArgumentParser(description='Experiment: Nested Cross-Validation')
    parser.add_argument('--outer-folds', type=int, default=5, help='Number of outer CV folds')
    parser.add_argument('--inner-folds', type=int, default=3, help='Number of inner CV folds (for tuning)')
    parser.add_argument('--model', type=str, required=True, help='Model to use')
    parser.add_argument('--config', type=str, required=True, help='Path to model config YAML')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for results')
    args = parser.parse_args()

    # Load config and update output_dir
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config['hyperparameter_tuning']['cv'] = args.inner_folds
    config['output']['output_dir'] = args.output_dir

    run_nested_cv(config, args.outer_folds, args.inner_folds, args.model, args.output_dir)

if __name__ == "__main__":
    main() 