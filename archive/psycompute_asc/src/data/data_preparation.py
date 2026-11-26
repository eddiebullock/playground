"""
Data preparation utility for converting real assessment data to the expected format.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path

class DataPreparator:
    """Prepares real assessment data for processing."""
    
    def __init__(self):
        """Initialize the data preparator."""
        self.assessment_types = ['AQ', 'SQ', 'EQ']
        self.expected_columns = {
            'AQ': [f'AQ_{i}' for i in range(1, 11)],  # AQ has 10 questions
            'SQ': [f'SQ_{i}' for i in range(1, 41)],  # SQ has 40 questions
            'EQ': [f'EQ_{i}' for i in range(1, 61)]   # EQ has 60 questions
        }
    
    def prepare_data(self, df: pd.DataFrame, assessment_type: str, 
                    column_mapping: Optional[Dict[str, str]] = None,
                    response_mapping: Optional[Dict[int, int]] = None) -> pd.DataFrame:
        """
        Prepare real data for processing by converting to expected format.
        
        Args:
            df: Input DataFrame with real data
            assessment_type: Type of assessment ('AQ', 'SQ', or 'EQ')
            column_mapping: Dictionary mapping original column names to expected names
            response_mapping: Dictionary mapping original response values to 1-4 scale
            
        Returns:
            DataFrame in the expected format
        """
        import warnings
        if assessment_type not in self.assessment_types:
            raise ValueError(f"Invalid assessment type. Must be one of {self.assessment_types}")
        
        # Create a copy to avoid modifying original
        df_prepared = df.copy()
        
        # Get the expected columns for this assessment type
        expected_cols = self.expected_columns[assessment_type]
        
        # Rename columns if mapping provided
        if column_mapping:
            # Only rename columns that exist
            cols_to_rename = {k: v for k, v in column_mapping.items() if k in df_prepared.columns}
            df_prepared = df_prepared.rename(columns=cols_to_rename)
        
        # Find which expected columns are present and which are missing
        present_cols = [col for col in expected_cols if col in df_prepared.columns]
        missing_cols = [col for col in expected_cols if col not in df_prepared.columns]
        
        if len(present_cols) == 0:
            raise ValueError(f"None of the expected columns for {assessment_type} are present in the data.")
        if missing_cols:
            warnings.warn(f"Missing expected columns for {assessment_type}: {missing_cols}. Filling with NaN.")
        for col in missing_cols:
            df_prepared[col] = np.nan
        
        # Get all columns that are not in expected_cols
        other_cols = [c for c in df_prepared.columns if c not in expected_cols]
        
        # Reorder columns to match expected order
        df_prepared = df_prepared[expected_cols + other_cols]
        
        # Convert response values if mapping provided
        if response_mapping:
            for col in present_cols:
                df_prepared[col] = df_prepared[col].map(response_mapping)
        
        # Ensure all responses are numeric and in 1-4 range
        for col in present_cols:
            df_prepared[col] = pd.to_numeric(df_prepared[col], errors='coerce')
            df_prepared[col] = df_prepared[col].apply(
                lambda x: x if pd.isna(x) or (1 <= x <= 4) else np.nan
            )
        
        return df_prepared
    
    def save_prepared_data(self, df: pd.DataFrame, assessment_type: str,
                          output_dir: str = 'data/raw', output_filename: Optional[str] = None) -> None:
        """
        Save prepared data to CSV file.
        
        Args:
            df: Prepared DataFrame
            assessment_type: Type of assessment
            output_dir: Directory to save the file
            output_filename: Optional custom filename (without extension)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if output_filename:
            file_path = output_path / f'{output_filename}.csv'
        else:
            file_path = output_path / f'{assessment_type.lower()}_data.csv'
            
        df.to_csv(file_path, index=False)
        print(f"Saved prepared data to {file_path}")

def main():
    """Example usage of the data preparator."""
    # Example column mapping (adjust based on your real data)
    column_mapping = {
        'question_1': 'AQ_1',
        'question_2': 'AQ_2',
        # ... add mappings for all questions
    }
    
    # Example response mapping (adjust based on your real data)
    response_mapping = {
        0: 1,  # Strongly disagree
        1: 2,  # Slightly disagree
        2: 3,  # Slightly agree
        3: 4   # Strongly agree
    }
    
    # Example usage
    preparator = DataPreparator()
    
    # Load your real data
    # df = pd.read_csv('your_real_data.csv')
    
    # Prepare the data
    # df_prepared = preparator.prepare_data(
    #     df, 
    #     'AQ',
    #     column_mapping=column_mapping,
    #     response_mapping=response_mapping
    # )
    
    # Save prepared data
    # preparator.save_prepared_data(df_prepared, 'AQ')

if __name__ == "__main__":
    main() 