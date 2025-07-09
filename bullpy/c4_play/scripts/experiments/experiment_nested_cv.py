#!/usr/bin/env python3
"""
Experiment: Run nested cross-validation using the modular pipeline.
Usage:
    python experiment_nested_cv.py --model xgboost --config ../../experiments/configs/model_config.yaml --output-dir ../../experiments/outputs/nested_cv_xgb
"""
import argparse
import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))
from model_training import run_modular_training_pipeline

def main():
    parser = argparse.ArgumentParser(description='Experiment: Nested Cross-Validation')
    parser.add_argument('--model', type=str, required=True, help='Model to use for nested CV')
    parser.add_argument('--config', type=str, required=True, help='Path to model config YAML')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--outer-folds', type=int, default=5, help='Number of outer CV folds')
    parser.add_argument('--inner-folds', type=int, default=3, help='Number of inner CV folds')
    args = parser.parse_args()

    # Load config and update for nested CV
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure hyperparameter_tuning section exists
    if 'hyperparameter_tuning' not in config:
        config['hyperparameter_tuning'] = {}
    config['hyperparameter_tuning']['cv'] = args.inner_folds
    
    # Filter to only use the specified model
    if 'models' in config:
        config['models'] = {args.model: config['models'].get(args.model, {})}
    
    # Ensure output section exists
    if 'output' not in config:
        config['output'] = {}
    config['output']['output_dir'] = args.output_dir

    # Save modified config to a temp file
    temp_config_path = Path(args.output_dir) / 'temp_nested_cv_config.yaml'
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)

    # Run modular training pipeline
    run_modular_training_pipeline(str(temp_config_path))

if __name__ == "__main__":
    main() 