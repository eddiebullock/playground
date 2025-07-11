#!/usr/bin/env python3
"""
Experiment: Compare different feature engineering strategies using the modular pipeline.
Usage:
    python experiment_feature_engineering.py --method logistic_regression --config ../../experiments/configs/model_config.yaml --output-dir ../../experiments/outputs/feature_engineering_lr
"""
import argparse
import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))
from model_training import run_modular_training_pipeline

def main():
    parser = argparse.ArgumentParser(description='Experiment: Feature Engineering Strategies')
    parser.add_argument('--method', type=str, required=True, help='Feature engineering method to use')
    parser.add_argument('--config', type=str, required=True, help='Path to model config YAML')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for results')
    args = parser.parse_args()

    # Load config and update feature engineering method and output_dir
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure feature_engineering section exists
    if 'feature_engineering' not in config:
        config['feature_engineering'] = {}
    config['feature_engineering']['method'] = args.method
    
    # Ensure output section exists
    if 'output' not in config:
        config['output'] = {}
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