#!/usr/bin/env python3
"""
Command-line interface for running the preprocessing pipeline.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from pipeline import run_preprocessing_pipeline

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description='Run preprocessing pipeline')
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Run pipeline
        splits = run_preprocessing_pipeline(args.config)
        
        # Print summary
        X_train, X_test, y_train, y_test = splits
        print("Preprocessing pipeline completed successfully!")
        print(f"\nSummary:")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        print(f"   Features: {X_train.shape[1]}")
        print(f"   Target distribution: {y_train.value_counts().to_dict()}")
        
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 