"""
Data splitting module for autism diagnosis prediction project.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

def prepare_targets(df: pd.DataFrame, target_config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for modeling.
    
    Args:
        df: DataFrame with features and target
        target_config: Target configuration
        
    Returns:
        Tuple of (features, target)
    """
    logger.info("Preparing targets for modeling...")
    # Use autism_any as the main target
    target_col = target_config.get('primary_target', 'autism_any')
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    # Drop all autism diagnosis and subtype columns from features to prevent leakage
    cols_to_drop = [
        'autism_any', 'userid',
        'autism_subtype', 'autism_subtype_1', 'autism_subtype_2', 'autism_subtype_3',
        'autism_diagnosis_0', 'autism_diagnosis_1', 'autism_diagnosis_2'
    ]
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    y = df[target_col]
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target distribution:\n{y.value_counts()}")
    return X, y

def prepare_linear_features(df: pd.DataFrame, target_config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]:
    logger.info("Preparing features for linear models...")
    # Create one-hot encoded version for linear models
    cat_cols = ['sex', 'handedness', 'education', 'occupation', 'country_region', 'repeat']
    df_onehot = pd.get_dummies(df, columns=[col for col in cat_cols if col in df.columns], drop_first=True)
    # Use same target preparation logic
    return prepare_targets(df_onehot, target_config)

def split_data(X: pd.DataFrame, y: pd.Series, split_config: Dict[str, Any]) -> Tuple:
    """
    Split data into train/test sets.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        split_config: Splitting configuration
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info("Splitting data into train/test sets...")
    
    test_size = split_config.get('test_size', 0.2)
    random_state = split_config.get('random_state', 42)
    stratify = y if split_config.get('stratify', True) else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    
    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

def save_splits(splits: Tuple, output_dir: str, config: Dict[str, Any]) -> None:
    """
    Save train/test splits to files.
    
    Args:
        splits: Tuple of (X_train, X_test, y_train, y_test)
        output_dir: Output directory
        config: Output configuration
    """
    from pathlib import Path
    
    X_train, X_test, y_train, y_test = splits
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save format
    save_format = config.get('save_format', 'csv')
    include_index = config.get('include_index', False)
    
    # Save splits
    X_train.to_csv(output_path / f"X_train.{save_format}", index=include_index)
    X_test.to_csv(output_path / f"X_test.{save_format}", index=include_index)
    y_train.to_csv(output_path / f"y_train.{save_format}", index=include_index)
    y_test.to_csv(output_path / f"y_test.{save_format}", index=include_index)
    
    logger.info(f"Saved data splits to {output_path}") 