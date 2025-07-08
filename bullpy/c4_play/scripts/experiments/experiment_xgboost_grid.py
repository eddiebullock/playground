#!/usr/bin/env python3
"""
Experiment: XGBoost hyperparameter grid search using the modular pipeline.
Usage:
    python experiment_xgboost_grid.py --config ../../experiments/configs/model_config.yaml --param-grid ../../experiments/configs/xgboost_grid.yaml --output-dir ../../experiments/outputs/xgb_grid
"""
import argparse
import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))
from model_training import run_modular_training_pipeline

def main():
    parser = argparse.ArgumentParser(description='Experiment: XGBoost Grid Search')
    parser.add_argument('--config', type=str, required=True, help='Path to model config YAML')
    parser.add_argument('--param-grid', type=str, required=True, help='Path to YAML file with XGBoost param grid')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for results')
    args = parser.parse_args()

    # Load config and update for XGBoost grid search
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    with open(args.param_grid, 'r') as f:
        param_grid = yaml.safe_load(f)
    config['models'] = {'xgboost': config['models'].get('xgboost', {})}
    config['hyperparameter_tuning']['method'] = 'grid'
    config['hyperparameter_tuning']['param_grid'] = param_grid
    config['output']['output_dir'] = args.output_dir

    # Save modified config to a temp file
    temp_config_path = Path(args.output_dir) / 'temp_model_config.yaml'
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)

    # Run modular training pipeline
    run_modular_training_pipeline(str(temp_config_path))

if __name__ == "__main__":
    main() 