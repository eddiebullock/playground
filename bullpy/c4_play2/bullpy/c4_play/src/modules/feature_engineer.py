"""
Feature engineering module for autism diagnosis prediction project.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def create_questionnaire_scores(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Create composite scores for questionnaire blocks.
    
    Args:
        df: DataFrame with questionnaire data
        config: Configuration dictionary
        
    Returns:
        DataFrame with questionnaire scores added
    """
    logger.info("Creating questionnaire scores...")
    
    df_features = df.copy()
    
    # Questionnaire blocks
    spq_cols = [col for col in df_features.columns if col.startswith('spq_')]
    eq_cols = [col for col in df_features.columns if col.startswith('eq_')]
    sqr_cols = [col for col in df_features.columns if col.startswith('sqr_')]
    aq_cols = [col for col in df_features.columns if col.startswith('aq_')]
    
    # Composite scores
    if spq_cols:
        df_features['spq_total'] = df_features[spq_cols].sum(axis=1)
    if eq_cols:
        df_features['eq_total'] = df_features[eq_cols].sum(axis=1)
    if sqr_cols:
        df_features['sqr_total'] = df_features[sqr_cols].sum(axis=1)
    if aq_cols:
        df_features['aq_total'] = df_features[aq_cols].sum(axis=1)
    
    # Overall total
    total_cols = ['spq_total', 'eq_total', 'sqr_total', 'aq_total']
    if all(col in df_features.columns for col in total_cols):
        df_features['total_score'] = df_features[total_cols].sum(axis=1)
    
    return df_features

def create_diagnosis_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features from diagnosis columns.
    
    Args:
        df: DataFrame with diagnosis data
        
    Returns:
        DataFrame with diagnosis features added
    """
    logger.info("Creating diagnosis features...")
    
    df_features = df.copy()
    
    # Diagnosis columns
    diagnosis_cols = [col for col in df_features.columns if col.startswith('diagnosis_')]
    autism_diag_cols = [col for col in df_features.columns if col.startswith('autism_diagnosis_')]
    
    # num_diagnoses
    if diagnosis_cols:
        df_features['num_diagnoses'] = df_features[diagnosis_cols].notnull().sum(axis=1)
    
    # has_adhd
    if diagnosis_cols:
        df_features['has_adhd'] = df_features[diagnosis_cols].apply(lambda row: 1 if 1 in row.values else 0, axis=1)
    
    # autism_any (target: any autism diagnosis)
    if autism_diag_cols:
        df_features['autism_any'] = df_features[autism_diag_cols].apply(
            lambda row: int(any(x in [1.0, 2.0, 3.0] for x in row if not pd.isnull(x))), axis=1
        )
        # autism_subtype (most specific subtype)
        def get_first_autism_subtype(row):
            for x in row:
                if x in [1.0, 2.0, 3.0]:
                    return int(x)
            return 0
        df_features['autism_subtype'] = df_features[autism_diag_cols].apply(get_first_autism_subtype, axis=1)
        # one-hot for each subtype
        for subtype in [1.0, 2.0, 3.0]:
            df_features[f'autism_subtype_{int(subtype)}'] = df_features[autism_diag_cols].apply(
                lambda row: int(subtype in row.values), axis=1
            )
    
    return df_features

def create_demographic_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Create demographic features.
    
    Args:
        df: DataFrame with demographic data
        config: Configuration dictionary
        
    Returns:
        DataFrame with demographic features added
    """
    logger.info("Creating demographic features...")
    
    df_features = df.copy()
    
    # Age group
    if 'age' in df_features.columns:
        age_bins = config.get('age_bins', [0, 18, 25, 35, 50, 100])
        age_labels = config.get('age_labels', ['0-18', '19-25', '26-35', '36-50', '50+'])
        
        df_features['age_group'] = pd.cut(df_features['age'], bins=age_bins, labels=age_labels)
        logger.info("Created age_group feature")
    
    # Convert demographics to category codes
    cat_cols = ['sex', 'handedness', 'education', 'occupation', 'country_region', 'repeat']
    for col in cat_cols:
        if col in df_features.columns:
            df_features[col] = df_features[col].astype('category').cat.codes
    
    return df_features

def create_interaction_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Create interaction features.
    
    Args:
        df: DataFrame with features
        config: Configuration dictionary
        
    Returns:
        DataFrame with interaction features added
    """
    logger.info("Creating interaction features...")
    
    df_features = df.copy()
    
    # Create interactions based on configuration
    interactions = config.get('interactions', [])
    
    for interaction in interactions:
        if len(interaction) == 2:
            col1, col2 = interaction
            if col1 in df_features.columns and col2 in df_features.columns:
                interaction_name = f"{col1}_{col2}_interaction"
                df_features[interaction_name] = df_features[col1] * df_features[col2]
                logger.info(f"Created {interaction_name} feature")
    
    return df_features

def create_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.
    
    Args:
        df: Cleaned DataFrame
        config: Configuration dictionary
        
    Returns:
        DataFrame with engineered features
    """
    logger.info("Starting feature engineering pipeline...")
    
    # Create questionnaire scores
    df_features = create_questionnaire_scores(df, config)
    
    # Create diagnosis features
    df_features = create_diagnosis_features(df_features)
    
    # Create demographic features
    df_features = create_demographic_features(df_features, config)
    
    # Create interaction features
    df_features = create_interaction_features(df_features, config)
    
    # One-hot encoding for linear models (optional, not returned by default)
    if config.get('one_hot', False):
        cat_cols = ['sex', 'handedness', 'education', 'occupation', 'country_region', 'repeat']
        df_features = pd.get_dummies(df_features, columns=cat_cols, drop_first=True)
    
    logger.info(f"Feature engineering complete. Final shape: {df_features.shape}")
    
    return df_features 