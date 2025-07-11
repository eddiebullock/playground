"""
Main pipeline orchestrator for autism diagnosis prediction project.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

from modules.data_loader import load_data, validate_data_structure
from modules.data_cleaner import clean_data
from modules.feature_engineer import create_features
from modules.data_splitter import prepare_targets, prepare_linear_features, split_data, save_splits

logger = logging.getLogger(__name__)

def run_preprocessing_pipeline(config_path: str) -> Tuple:
    """
    Run the complete preprocessing pipeline.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info("Starting preprocessing pipeline...")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    data_config = config.get('data', {})
    df = load_data(data_config['raw_data_path'])
    
    # Validate data structure
    validate_data_structure(df)
    
    # Clean data
    cleaning_config = config.get('cleaning', {})
    df_clean = clean_data(df, cleaning_config)
    
    # Create features
    features_config = config.get('features', {})
    df_features = create_features(df_clean, features_config)
    # Save full features for EDA
    df_features.to_csv(str(data_config['processed_data_dir']) + '/features_full.csv', index=False)
    
    # Prepare targets for tree-based models
    target_config = config.get('target', {})
    X_tree, y = prepare_targets(df_features, target_config)
    
    # Prepare targets for linear models
    X_linear, _ = prepare_linear_features(df_features, target_config)
    
    # Split data
    split_config = config.get('splitting', {})
    splits_tree = split_data(X_tree, y, split_config)
    splits_linear = split_data(X_linear, y, split_config)
    
    # Save results
    output_config = config.get('output', {})
    save_splits(splits_tree, data_config['processed_data_dir'], output_config)
    
    # Save linear features separately
    X_train_linear, X_test_linear, y_train, y_test = splits_linear
    output_path = Path(data_config['processed_data_dir'])
    X_train_linear.to_csv(output_path / "X_train_linear.csv", index=False)
    X_test_linear.to_csv(output_path / "X_test_linear.csv", index=False)
    
    logger.info("Preprocessing pipeline completed successfully!")
    
    return splits_tree

def main():
    """Main function for command-line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run preprocessing pipeline')
    parser.add_argument('--config', required=True, help='Path to configuration file')
    args = parser.parse_args()
    
    try:
        splits = run_preprocessing_pipeline(args.config)
        print("Pipeline completed successfully!")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 