"""
Data validation module for quality control of assessment data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class ValidationResult:
    """Container for validation results."""
    is_valid: bool
    issues: List[str]
    warnings: List[str]

class DataValidator:
    """Validates assessment data for quality and consistency."""
    
    def __init__(self):
        """Initialize the data validator."""
        self.valid_response_ranges = {
            'AQ': (1, 4),
            'SQ': (1, 4),
            'EQ': (1, 4)
        }
        self.max_missing_percentage = 0.15  # Allow up to 15% missing values
        self.straight_line_threshold = 0.8  # Require 80% same responses to flag straight-lining
        self.subscale_correlation_threshold = 0.3  # Minimum expected correlation between items in same subscale
        
        # Define subscales for each assessment type
        self.subscales = {
            'AQ': {
                'social': list(range(1, 11)),
                'attention_switch': list(range(11, 21)),
                'attention_detail': list(range(21, 31)),
                'communication': list(range(31, 41)),
                'imagination': list(range(41, 51))
            },
            'EQ': {
                'cognitive': list(range(1, 21)),
                'emotional': list(range(21, 41)),
                'social': list(range(41, 61))
            },
            'SQ': {
                'patterns': list(range(1, 17)),
                'details': list(range(17, 33)),
                'systems': list(range(33, 51))
            }
        }
    
    def validate_dataset(self, df: pd.DataFrame, assessment_type: str) -> ValidationResult:
        """
        Perform comprehensive validation of the dataset.
        
        Args:
            df: DataFrame to validate
            assessment_type: Type of assessment
            
        Returns:
            ValidationResult containing validation status and issues
        """
        issues = []
        warnings = []
        
        # Check for missing values
        missing_check = self._check_missing_values(df)
        issues.extend(missing_check[0])
        warnings.extend(missing_check[1])
        
        # Check response ranges
        range_check = self._check_response_ranges(df, assessment_type)
        issues.extend(range_check[0])
        warnings.extend(range_check[1])
        
        # Check for duplicate entries
        duplicate_check = self._check_duplicates(df)
        issues.extend(duplicate_check[0])
        warnings.extend(duplicate_check[1])
        
        # Check for response patterns
        pattern_check = self._check_response_patterns(df, assessment_type)
        issues.extend(pattern_check[0])
        warnings.extend(pattern_check[1])
        
        # Check subscale correlations
        correlation_check = self._check_subscale_correlations(df, assessment_type)
        issues.extend(correlation_check[0])
        warnings.extend(correlation_check[1])
        
        # Consider dataset valid if there are no critical issues
        is_valid = len([i for i in issues if 'critical' in i.lower()]) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            warnings=warnings
        )
    
    def _check_missing_values(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Check for missing values in the dataset."""
        issues = []
        warnings = []
        
        # Calculate missing percentage for each column
        missing_percentages = df.isnull().mean() * 100
        
        # Check for columns with high missing percentages
        high_missing_cols = missing_percentages[missing_percentages > self.max_missing_percentage * 100]
        if not high_missing_cols.empty:
            for col, percentage in high_missing_cols.items():
                if percentage > 30:  # Critical if more than 30% missing
                    issues.append(f"Critical: Column {col} has {percentage:.1f}% missing values")
                else:
                    warnings.append(f"Column {col} has {percentage:.1f}% missing values")
        
        # Check total missing values
        total_missing = df.isnull().sum().sum()
        if total_missing > 0:
            warnings.append(f"Total missing values: {total_missing}")
        
        return issues, warnings
    
    def _check_response_ranges(self, df: pd.DataFrame, assessment_type: str) -> Tuple[List[str], List[str]]:
        """Check if response values are within valid ranges."""
        issues = []
        warnings = []
        
        min_val, max_val = self.valid_response_ranges[assessment_type]
        question_cols = [col for col in df.columns if col.startswith(f'{assessment_type}_')]
        
        for col in question_cols:
            invalid_values = df[col][~df[col].between(min_val, max_val)]
            if not invalid_values.empty:
                issues.append(f"Invalid values found in {col}: {invalid_values.unique().tolist()}")
        
        return issues, warnings
    
    def _check_duplicates(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Check for duplicate entries in the dataset."""
        issues = []
        warnings = []
        
        # Check for duplicate participant IDs
        if 'participant_id' in df.columns:
            duplicates = df[df.duplicated(subset=['participant_id'], keep=False)]
            if not duplicates.empty:
                issues.append(f"Found {len(duplicates)} duplicate participant IDs")
        
        # Check for duplicate responses
        question_cols = [col for col in df.columns if any(col.startswith(prefix) for prefix in ['AQ_', 'SQ_', 'EQ_'])]
        if question_cols:
            duplicates = df[df.duplicated(subset=question_cols, keep=False)]
            if not duplicates.empty:
                warnings.append(f"Found {len(duplicates)} duplicate response patterns")
        
        return issues, warnings
    
    def _check_response_patterns(self, df: pd.DataFrame, assessment_type: str) -> Tuple[List[str], List[str]]:
        """Check for suspicious response patterns."""
        issues = []
        warnings = []
        
        question_cols = [col for col in df.columns if col.startswith(f'{assessment_type}_')]
        if not question_cols:
            return issues, warnings
        
        # Check for straight-lining (same response for many questions)
        for idx, row in df.iterrows():
            responses = row[question_cols].dropna()
            if len(responses) > 0:
                most_common = responses.mode().iloc[0]
                same_responses = (responses == most_common).mean()
                if same_responses >= self.straight_line_threshold:
                    warnings.append(f"Possible straight-lining detected in row {idx}")
        
        # Check for response distribution
        for col in question_cols:
            value_counts = df[col].value_counts(normalize=True)
            if len(value_counts) < 3:  # Less than 3 different responses
                warnings.append(f"Limited response variation in {col}")
        
        return issues, warnings
    
    def _check_subscale_correlations(self, df: pd.DataFrame, assessment_type: str) -> Tuple[List[str], List[str]]:
        """Check correlations between items within subscales."""
        issues = []
        warnings = []
        
        if assessment_type not in self.subscales:
            return issues, warnings
        
        for subscale_name, items in self.subscales.items():
            # Get columns for this subscale (e.g., AQ_1, AQ_2, etc.)
            subscale_cols = [f'{assessment_type}_{i}' for i in items]
            
            # Verify all columns exist
            missing_cols = [col for col in subscale_cols if col not in df.columns]
            if missing_cols:
                warnings.append(f"Missing columns for {subscale_name} subscale: {missing_cols}")
                continue
            
            # Calculate correlations between items
            try:
                corr_matrix = df[subscale_cols].corr()
                
                # Check if correlations are too low
                low_corr_pairs = []
                for i in range(len(subscale_cols)):
                    for j in range(i+1, len(subscale_cols)):
                        if abs(corr_matrix.iloc[i,j]) < self.subscale_correlation_threshold:
                            low_corr_pairs.append((subscale_cols[i], subscale_cols[j]))
                
                if low_corr_pairs:
                    warnings.append(f"Low correlations found in {subscale_name} subscale: {len(low_corr_pairs)} pairs")
            except Exception as e:
                warnings.append(f"Error calculating correlations for {subscale_name} subscale: {str(e)}")
        
        return issues, warnings 