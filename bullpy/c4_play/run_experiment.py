#!/usr/bin/env python3

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def run_sampling_experiment(method, config_path, output_dir):
    """Run sampling experiment."""
    from model_training import run_modular_training_pipeline
    import yaml
    
    # Load and modify config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure sections exist
    if 'imbalance_handling' not in config:
        config['imbalance_handling'] = {}
    config['imbalance_handling']['method'] = method
    
    if 'output' not in config:
        config['output'] = {}
    config['output']['output_dir'] = output_dir
    
    # Save temp config
    temp_config = Path(output_dir) / 'temp_config.yaml'
    temp_config.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)
    
    # Run pipeline
    run_modular_training_pipeline(str(temp_config))

def run_feature_engineering_experiment(method, config_path, output_dir):
    """Run feature engineering experiment."""
    from model_training import run_modular_training_pipeline
    import yaml
    
    # Load and modify config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure sections exist
    if 'feature_engineering' not in config:
        config['feature_engineering'] = {}
    config['feature_engineering']['method'] = method
    
    if 'output' not in config:
        config['output'] = {}
    config['output']['output_dir'] = output_dir
    
    # Save temp config
    temp_config = Path(output_dir) / 'temp_config.yaml'
    temp_config.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)
    
    # Run pipeline
    run_modular_training_pipeline(str(temp_config))

def run_model_comparison_experiment(config_path, output_dir):
    """Run model comparison experiment."""
    from model_training import run_modular_training_pipeline
    import yaml
    
    # Load and modify config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure output section exists
    if 'output' not in config:
        config['output'] = {}
    config['output']['output_dir'] = output_dir
    
    # Save temp config
    temp_config = Path(output_dir) / 'temp_config.yaml'
    temp_config.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)
    
    # Run pipeline
    run_modular_training_pipeline(str(temp_config))

def run_nested_cv_experiment(model, config_path, output_dir):
    """Run nested CV experiment."""
    from model_training import run_modular_training_pipeline
    import yaml
    
    # Load and modify config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Filter to only use specified model
    if 'models' in config:
        config['models'] = {model: config['models'].get(model, {})}
    
    # Ensure hyperparameter_tuning section exists
    if 'hyperparameter_tuning' not in config:
        config['hyperparameter_tuning'] = {}
    config['hyperparameter_tuning']['cv'] = 3  # Inner CV
    
    if 'output' not in config:
        config['output'] = {}
    config['output']['output_dir'] = output_dir
    
    # Save temp config
    temp_config = Path(output_dir) / 'temp_config.yaml'
    temp_config.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)
    
    # Run pipeline
    run_modular_training_pipeline(str(temp_config))

def run_xgboost_grid_experiment(config_path, output_dir):
    """Run XGBoost grid search experiment."""
    from model_training import run_modular_training_pipeline
    import yaml
    
    # Load and modify config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure output section exists
    if 'output' not in config:
        config['output'] = {}
    config['output']['output_dir'] = output_dir
    
    # Save temp config
    temp_config = Path(output_dir) / 'temp_config.yaml'
    temp_config.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)
    
    # Run pipeline
    run_modular_training_pipeline(str(temp_config))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_experiment.py <experiment_type> [args]")
        sys.exit(1)
    
    experiment_type = sys.argv[1]
    
    if experiment_type == "sampling":
        if len(sys.argv) != 5:
            print("Usage: python run_experiment.py sampling <method> <config> <output_dir>")
            sys.exit(1)
        run_sampling_experiment(sys.argv[2], sys.argv[3], sys.argv[4])
    
    elif experiment_type == "feature_engineering":
        if len(sys.argv) != 5:
            print("Usage: python run_experiment.py feature_engineering <method> <config> <output_dir>")
            sys.exit(1)
        run_feature_engineering_experiment(sys.argv[2], sys.argv[3], sys.argv[4])
    
    elif experiment_type == "models":
        if len(sys.argv) != 4:
            print("Usage: python run_experiment.py models <config> <output_dir>")
            sys.exit(1)
        run_model_comparison_experiment(sys.argv[2], sys.argv[3])
    
    elif experiment_type == "nested_cv":
        if len(sys.argv) != 5:
            print("Usage: python run_experiment.py nested_cv <model> <config> <output_dir>")
            sys.exit(1)
        run_nested_cv_experiment(sys.argv[2], sys.argv[3], sys.argv[4])
    
    elif experiment_type == "xgboost_grid":
        if len(sys.argv) != 4:
            print("Usage: python run_experiment.py xgboost_grid <config> <output_dir>")
            sys.exit(1)
        run_xgboost_grid_experiment(sys.argv[2], sys.argv[3])
    
    else:
        print(f"Unknown experiment type: {experiment_type}")
        sys.exit(1)
