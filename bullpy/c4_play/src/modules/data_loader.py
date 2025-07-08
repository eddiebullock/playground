"""
Data loading module for autism diagnosis prediction project.
"""

import pandas as pd
import logging
from pathlib import Path 
from typing import Optional 

logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from file.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        Loaded DataFrame
    """
    logger.info(f"Loading data from {file_path}")

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Load data based on file extension
    if file_path.suffix.lower() == '.csv':
        df = pd.read_csv(file_path)
    elif file_path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    logger.info(f"Loaded data shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")

    return df

def validate_data_structure(df: pd.DataFrame, expected_columns: Optional[list] = None) -> bool:
    """
    Validate that the data has the expected structure.
    
    Args:
        df: DataFrame to validate
        expected_columns: List of expected column names
        
    Returns:
        True if validation passes
    """
    logger.info("Validating data structure...")

    if df.empty:
        raise ValueError("DataFrame is empty")

    if expected_columns:
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing expected columns: {missing_cols}")
        
    logger.info("Data structure validation passed")
    return True

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/eb2007/predict_asc_c4/data/data_c4_raw.csv', help='Path to the data file')
    parser.add_argument('--sample_size', type=int, default=None, help='Number of samples to load')
    parser.add_argument('--output_dir', type=str, default='test_output', help='Output directory')
    args = parser.parse_args()

    df = load_data(args.data_path)
    if args.sample_size:
        df = df.sample(n=args.sample_size, random_state=42)
    
    # Save to output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'loaded_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} samples to {output_path}")
