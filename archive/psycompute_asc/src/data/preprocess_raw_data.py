"""
Universal preprocessing script for converting any downloaded raw assessment data into the expected format.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob
import os
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.data_preparation import DataPreparator

def generate_column_mapping(df: pd.DataFrame, prefix: str) -> dict:
    """
    Automatically generate column mapping based on a prefix.
    
    Args:
        df: DataFrame containing raw data
        prefix: Prefix for the expected column names (e.g., 'A' for AQ, 'E' for EQ, 'S' for SQ)
    
    Returns:
        Dictionary mapping original column names to expected names
    """
    print(f"generate_column_mapping called with prefix: {prefix}")
    
    # Handle test data format (AQ_1, AQ_2, etc.)
    test_cols = [col for col in df.columns if col.startswith(f'{prefix}Q_')]
    if test_cols:
        return {col: col for col in test_cols}  # Keep original names for test data
    
    # Handle original formats
    if prefix == 'A':
        # Accept various AQ formats
        aq_formats = [
            [f'A{i}_Score' for i in range(1, 51)],  # A1_Score...A50_Score
            [f'A{i}' for i in range(1, 51)],        # A1...A50
            [f'AQ_{i}' for i in range(1, 51)],      # AQ_1...AQ_50
            [f'Q{i}' for i in range(1, 51)]         # Q1...Q50
        ]
        for format_cols in aq_formats:
            if all(col in df.columns for col in format_cols):
                return dict(zip(format_cols, [f'AQ_{i}' for i in range(1, 51)]))
    
    elif prefix == 'E':
        # Accept various EQ formats
        eq_formats = [
            [f'E{i}_Score' for i in range(1, 61)],  # E1_Score...E60_Score
            [f'E{i}' for i in range(1, 61)],        # E1...E60
            [f'EQ_{i}' for i in range(1, 61)],      # EQ_1...EQ_60
            [f'Q{i}' for i in range(1, 61)]         # Q1...Q60
        ]
        for format_cols in eq_formats:
            if all(col in df.columns for col in format_cols):
                return dict(zip(format_cols, [f'EQ_{i}' for i in range(1, 61)]))
    
    elif prefix == 'S':
        # Accept various SQ formats
        sq_formats = [
            [f'S{i}_Score' for i in range(1, 41)],  # S1_Score...S40_Score
            [f'S{i}' for i in range(1, 41)],        # S1...S40
            [f'SQ_{i}' for i in range(1, 41)],      # SQ_1...SQ_40
            [f'Q{i}' for i in range(1, 41)]         # Q1...Q40
        ]
        for format_cols in sq_formats:
            if all(col in df.columns for col in format_cols):
                return dict(zip(format_cols, [f'SQ_{i}' for i in range(1, 41)]))
    
    # If no exact match found, try partial matching
    original_cols = [col for col in df.columns if col.startswith(prefix)]
    if original_cols:
        # Sort columns to ensure consistent mapping
        original_cols.sort()
    expected_cols = [f'{prefix}Q_{i+1}' for i in range(len(original_cols))]
    return dict(zip(original_cols, expected_cols))
    
    return {}

def generate_response_mapping(df: pd.DataFrame, columns: list) -> dict:
    """
    Automatically generate response mapping based on unique values in the data.
    
    Args:
        df: DataFrame containing raw data
        columns: List of columns to consider for response mapping
    
    Returns:
        Dictionary mapping original response values to 1-4 scale
    """
    # For test data (already in 0-4 scale), return identity mapping
    values = df[columns].values.flatten()
    unique_values = np.unique(values[~pd.isna(values)])  # Remove NaN values
    
    # Common response scales
    if set(unique_values).issubset({0, 1, 2, 3, 4}):
        return {i: i for i in range(5)}  # Identity mapping for 0-4 scale
    
    if set(unique_values).issubset({1, 2, 3, 4}):
        return {i: i for i in range(1, 5)}  # Identity mapping for 1-4 scale
    
    if set(unique_values).issubset({0, 1}):
        return {0: 1, 1: 2}  # Map 0->1 (disagree) and 1->2 (agree)
    
    if set(unique_values).issubset({1, 2}):
        return {1: 1, 2: 2}  # Map 1->1 (disagree) and 2->2 (agree)
    
    if set(unique_values).issubset({1, 2, 3, 4, 5}):
        return {i: i for i in range(1, 6)}  # Identity mapping for 1-5 scale
    
    if set(unique_values).issubset({0, 1, 2, 3, 4, 5}):
        return {i: i for i in range(6)}  # Identity mapping for 0-5 scale
    
    # For other scales, map to 1-4
    return {val: i+1 for i, val in enumerate(sorted(unique_values))}

def preprocess_raw_data(raw_file_path: str, assessment_type: str, output_dir: str = 'data/raw'):
    """
    Preprocess raw assessment data from any downloaded dataset.
    
    Args:
        raw_file_path: Path to the raw CSV file
        assessment_type: Type of assessment ('AQ', 'SQ', or 'EQ')
        output_dir: Directory to save the processed file
    """
    try:
        # Use pandas' delimiter auto-detection
        df = pd.read_csv(raw_file_path, sep=None, engine='python', encoding='utf-8')
        print(f"Actual columns in {raw_file_path}: {df.columns.tolist()}")
        
        # Determine prefix based on assessment type
        prefix = assessment_type[0]  # 'A' for AQ, 'E' for EQ, 'S' for SQ
        
        # Generate column mapping
        column_mapping = generate_column_mapping(df, prefix)
        if not column_mapping:
            raise ValueError(f"No valid column mapping found for {assessment_type}")
        
        # Generate response mapping
        response_mapping = generate_response_mapping(df, list(column_mapping.keys()))
        
        # Prepare data using DataPreparator
        preparator = DataPreparator()
        df_prepared = preparator.prepare_data(df, assessment_type, column_mapping, response_mapping)
        
        # Handle missing values
        df_prepared = df_prepared.fillna(method='ffill').fillna(method='bfill')
        
        # Generate output filename based on input filename
        input_filename = os.path.basename(raw_file_path)
        output_filename = f"{assessment_type.lower()}_{os.path.splitext(input_filename)[0]}_processed"
        
        # Save prepared data
        preparator.save_prepared_data(df_prepared, assessment_type, output_dir, output_filename)
        print(f"Successfully processed {raw_file_path}")
        
    except Exception as e:
        print(f"Error processing {raw_file_path}: {str(e)}")
        raise

def process_directory(directory_path: str, assessment_type: str, output_dir: str = 'data/raw'):
    """
    Process all CSV files in a directory.
    
    Args:
        directory_path: Path to directory containing raw CSV files
        assessment_type: Type of assessment ('AQ', 'SQ', or 'EQ')
        output_dir: Directory to save the processed files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {directory_path}")
        return
    
    print(f"Found {len(csv_files)} CSV files in {directory_path}")
    
    # Process each file
    for file_path in csv_files:
        print(f"\nProcessing {file_path}...")
        preprocess_raw_data(file_path, assessment_type, output_dir)

def merge_processed_files(output_dir: str, assessment_type: str):
    """
    Merge all processed files for a given assessment type into a single file.
    
    Args:
        output_dir: Directory containing processed files
        assessment_type: Type of assessment ('AQ', 'SQ', or 'EQ')
    """
    import glob
    import pandas as pd
    from pathlib import Path

    # Get all processed files for the assessment type
    pattern = f"{assessment_type.lower()}_*_processed.csv"
    processed_files = glob.glob(str(Path(output_dir) / pattern))

    if not processed_files:
        print(f"No processed files found for {assessment_type}.")
        return

    # Read and concatenate all processed files
    dfs = [pd.read_csv(file) for file in processed_files]
    merged_df = pd.concat(dfs, ignore_index=True)

    # Save the merged file
    merged_file_path = Path(output_dir) / f"{assessment_type.lower()}_data_processed.csv"
    merged_df.to_csv(merged_file_path, index=False)
    print(f"Merged {len(processed_files)} files into {merged_file_path}")

if __name__ == "__main__":
    # Process test data files
    test_data_dirs = {
        'AQ': '/Users/eb2007/bulldev/data/raw/Data-downloads/kaggle/AQ',
        'EQ': '/Users/eb2007/bulldev/data/raw/Data-downloads/kaggle/EQ',
        'SQ': '/Users/eb2007/bulldev/data/raw/Data-downloads/kaggle/SQ'
    }
    
    # Process each assessment type
    for assessment_type, directory in test_data_dirs.items():
        print(f"\nProcessing {assessment_type} test data...")
        process_directory(directory, assessment_type)
        merge_processed_files('data/raw', assessment_type) 