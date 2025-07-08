#!/usr/bin/env python3
"""
Experiment: Compare advanced models using the modular pipeline.
Usage:
    python experiment_models.py --model XGBoost --config ../../experiments/configs/model_config.yaml --output-dir ../../experiments/outputs/model_XGBoost
"""
import argparse
import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))
from model_training import run_modular_training_pipeline

def main():
    parser = argparse.ArgumentParser(description='Experiment: Advanced Models')
    parser.add_argument('--model', type=str, required=True, help='Model to use (e.g., XGBoost, LightGBM, CatBoost, MLP, TabNet, SVM, etc.)')
    parser.add_argument('--config', type=str, required=True, help='Path to model config YAML')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for results')
    args = parser.parse_args()

    # Load config and update models and output_dir
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    # Only keep the selected model in the config
    if args.model.lower() in config['models']:
        config['models'] = {args.model.lower(): config['models'][args.model.lower()]}
    else:
        config['models'] = {args.model: {}}
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