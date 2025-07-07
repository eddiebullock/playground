#!/usr/bin/env python3
"""
Command-line interface for model training pipeline.

Usage:
    python run_model_training.py --config experiments/configs/model_config.yaml
    python run_model_training.py --config experiments/configs/model_config.yaml --verbose
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from model_training import ClinicalModelTrainer

def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('experiments/logs/model_training.log')
        ]
    )

def main():
    """Main function for CLI."""
    parser = argparse.ArgumentParser(description='Run model training pipeline')
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Path to model configuration file'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/processed/features_full.csv',
        help='Path to processed features file'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='experiments/models',
        help='Output directory for models and results'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate config file
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    # Validate data file
    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        logger.error("Run data preprocessing first: python src/pipeline.py --config experiments/configs/data_config.yaml")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info("Starting model training pipeline...")
        
        # Initialize trainer
        trainer = ClinicalModelTrainer(output_dir=str(output_dir))
        
        # Load data
        logger.info(f"Loading data from {data_path}")
        X, y = trainer.load_data(str(data_path))
        
        # Load configuration
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Split data
        split_config = config.get('splitting', {})
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(
            X, y,
            test_size=split_config.get('test_size', 0.2),
            val_size=split_config.get('val_size', 0.2),
            random_state=split_config.get('random_state', 42)
        )
        
        # Initialize models
        trainer.initialize_models()
        
        # Train models
        logger.info("Training models...")
        val_results = trainer.train_models(X_train, y_train, X_val, y_val)
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_results = trainer.evaluate_on_test_set(X_test, y_test)
        
        # Create plots
        if config.get('output', {}).get('create_plots', True):
            logger.info("Creating evaluation plots...")
            trainer.create_evaluation_plots(X_test, y_test, test_results)
        
        # Save results
        results_file = config.get('output', {}).get('results_file', 'experiments/logs/model_results.yaml')
        trainer.save_results(test_results, results_file)
        
        # Print summary
        summary = trainer.create_summary_report(test_results)
        print("\n" + "="*80)
        print("MODEL TRAINING SUMMARY")
        print("="*80)
        print(summary)
        print("="*80)
        
        logger.info("Model training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in model training pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 