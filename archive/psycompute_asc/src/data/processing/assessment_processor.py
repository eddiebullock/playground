"""
Data processing module for autism screening questionnaires.
Handles ingestion, validation, and preprocessing of assessment data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path

class AssessmentProcessor:
    """Processes and validates autism screening questionnaire data."""
    
    def __init__(self):
        """Initialize the assessment processor."""
        self.valid_assessments = ['AQ', 'SQ', 'EQ']
        self.required_columns = {
            'AQ': [f'AQ_{i}' for i in range(1, 51)],  # 50 questions
            'SQ': [f'SQ_{i}' for i in range(1, 41)],  # 40 questions
            'EQ': [f'EQ_{i}' for i in range(1, 61)]   # 60 questions
        }
        
    def load_data(self, file_path: Union[str, Path], assessment_type: str) -> pd.DataFrame:
        """
        Load assessment data from a file.
        
        Args:
            file_path: Path to the data file
            assessment_type: Type of assessment ('AQ', 'SQ', or 'EQ')
            
        Returns:
            DataFrame containing the assessment data
        """
        if assessment_type not in self.valid_assessments:
            raise ValueError(f"Invalid assessment type. Must be one of {self.valid_assessments}")
            
        try:
            df = pd.read_csv(file_path)
            self._validate_columns(df, assessment_type)
            return df
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def _validate_columns(self, df: pd.DataFrame, assessment_type: str) -> None:
        """
        Validate that the DataFrame contains at least one required column.
        Args:
            df: DataFrame to validate
            assessment_type: Type of assessment
        """
        required = self.required_columns[assessment_type]
        present = [col for col in required if col in df.columns]
        if not present:
            raise ValueError(f"No required columns for {assessment_type} found in data.")
    
    def validate_responses(self, df: pd.DataFrame, assessment_type: str) -> pd.DataFrame:
        """
        Validate and clean response data.
        Args:
            df: DataFrame containing responses
            assessment_type: Type of assessment
        Returns:
            DataFrame with validated responses
        """
        # Copy to avoid modifying original
        df_valid = df.copy()
        # Validate response values (assuming 1-4 scale)
        for col in self.required_columns[assessment_type]:
            if col in df_valid.columns:
                # Replace invalid values with NaN
                df_valid[col] = pd.to_numeric(df_valid[col], errors='coerce')
                df_valid[col] = df_valid[col].apply(
                    lambda x: x if pd.isna(x) or (1 <= x <= 4) else np.nan
                )
        return df_valid
    
    def calculate_scores(self, df: pd.DataFrame, assessment_type: str) -> pd.DataFrame:
        """
        Calculate total scores and subscale scores for the assessment.
        Args:
            df: DataFrame containing validated responses
            assessment_type: Type of assessment
        Returns:
            DataFrame with added score columns
        """
        df_scored = df.copy()
        available_cols = [col for col in self.required_columns[assessment_type] if col in df_scored.columns]
        if assessment_type == 'AQ':
            df_scored['AQ_Total'] = df_scored[available_cols].sum(axis=1)
        elif assessment_type == 'SQ':
            df_scored['SQ_Total'] = df_scored[available_cols].sum(axis=1)
        elif assessment_type == 'EQ':
            df_scored['EQ_Total'] = df_scored[available_cols].sum(axis=1)
        return df_scored
    
    def generate_statistics(self, df: pd.DataFrame, assessment_type: str) -> Dict:
        """
        Generate statistical summaries of the assessment data.
        
        Args:
            df: DataFrame containing scored responses
            assessment_type: Type of assessment
            
        Returns:
            Dictionary containing statistical summaries
        """
        stats = {
            'total_score': {
                'mean': df[f'{assessment_type}_Total'].mean(),
                'std': df[f'{assessment_type}_Total'].std(),
                'min': df[f'{assessment_type}_Total'].min(),
                'max': df[f'{assessment_type}_Total'].max(),
                'median': df[f'{assessment_type}_Total'].median()
            },
            'missing_data': {
                'total_missing': df.isnull().sum().sum(),
                'missing_by_question': df.isnull().sum().to_dict()
            }
        }
        
        return stats 