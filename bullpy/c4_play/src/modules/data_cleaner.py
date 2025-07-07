"""
Data cleaning module for autism diagnosis prediction project.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def replace_missing_codes(df: pd.DataFrame, missing_codes: List) -> pd.DataFrame:
    """
    Replace non-standard missing codes with NaN.
    
    Args:
        df: DataFrame to clean
        missing_codes: List of codes to replace with NaN
        
    Returns:
        DataFrame with missing codes replaced
    """
    logger.info("Replacing non-standard missing codes...")
    
    df_clean = df.copy()
    
    for col in df_clean.columns:
        for code in missing_codes:
            df_clean[col] = df_clean[col].replace(code, np.nan)
    
    return df_clean

def hybrid_impute_drop(df: pd.DataFrame, cols: List[str], max_missing: int = 2, strategy: str = 'mean') -> pd.DataFrame:
    missing_counts = df[cols].isnull().sum(axis=1)
    to_impute = df[missing_counts <= max_missing].copy()
    to_drop = df[missing_counts > max_missing].copy()
    if strategy == 'mean':
        to_impute[cols] = to_impute[cols].apply(lambda x: x.fillna(x.mean()), axis=0)
    elif strategy == 'median':
        to_impute[cols] = to_impute[cols].apply(lambda x: x.fillna(x.median()), axis=0)
    return pd.concat([to_impute], axis=0)

def handle_missing_values(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Handle missing values according to configuration.
    
    Args:
        df: DataFrame to clean
        config: Configuration dictionary
        
    Returns:
        DataFrame with missing values handled
    """
    logger.info("Handling missing values...")
    
    df_clean = df.copy()
    
    # Questionnaire columns
    spq_cols = [col for col in df_clean.columns if col.startswith('spq_')]
    eq_cols = [col for col in df_clean.columns if col.startswith('eq_')]
    sqr_cols = [col for col in df_clean.columns if col.startswith('sqr_')]
    aq_cols = [col for col in df_clean.columns if col.startswith('aq_')]
    # Hybrid impute/drop for each block
    for block_cols in [spq_cols, eq_cols, sqr_cols, aq_cols]:
        if block_cols:
            df_clean = hybrid_impute_drop(df_clean, block_cols, max_missing=config.get('questionnaire_missing_threshold', 2), strategy='mean')
    
    # Demographics: impute with 0
    demo_cols = ['sex', 'handedness', 'education', 'occupation', 'country_region', 'repeat', 'age']
    for col in demo_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0)
    
    return df_clean

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows.
    
    Args:
        df: DataFrame to clean
        
    Returns:
        DataFrame with duplicates removed
    """
    logger.info("Checking for duplicates...")
    
    duplicates = df.duplicated().sum()
    logger.info(f"Found {duplicates} duplicate rows")
    
    if duplicates > 0:
        df_clean = df.drop_duplicates()
        logger.info(f"Removed {duplicates} duplicate rows")
        return df_clean
    
    return df

def validate_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and correct data types.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        DataFrame with corrected data types
    """
    logger.info("Validating data types...")
    
    df_clean = df.copy()
    
    # Ensure numeric columns are numeric
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    return df_clean

def clean_data(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Complete data cleaning pipeline.
    
    Args:
        df: Raw DataFrame
        config: Configuration dictionary
        
    Returns:
        Cleaned DataFrame
    """
    logger.info("Starting data cleaning pipeline...")
    
    # Replace missing codes
    df_clean = replace_missing_codes(df, config.get('missing_codes', [-1, -999, 999, '', 'NA', 'N/A', 'null']))
    
    # Handle missing values
    df_clean = handle_missing_values(df_clean, config)
    
    # Remove duplicates
    if config.get('remove_duplicates', True):
        df_clean = remove_duplicates(df_clean)
    
    # Validate data types
    df_clean = validate_data_types(df_clean)
    
    logger.info(f"Data cleaning complete. Shape: {df_clean.shape}")
    
    return df_clean 