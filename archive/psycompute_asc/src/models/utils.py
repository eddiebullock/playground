"""
Utility functions for model training and evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any
import json
from pathlib import Path

def prepare_data(df: pd.DataFrame, assessment_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for training.
    
    Args:
        df: DataFrame containing the data
        assessment_type: Type of assessment ('AQ', 'SQ', or 'EQ')
        
    Returns:
        Tuple of (X, y) where X is the feature matrix and y is the target vector
    """
    # Get question columns
    question_cols = [col for col in df.columns 
                    if col.startswith(f'{assessment_type}_') 
                    and not col.endswith('_score') 
                    and col != 'asd_diagnosis']
    
    # Fill missing values with median
    df[question_cols] = df[question_cols].fillna(df[question_cols].median())
    
    # Prepare features and target
    X = df[question_cols].values
    y = df['asd_diagnosis'].values.astype(int)
    
    return X, y

def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and test sets.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of data to use for testing
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    return train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=42, 
        stratify=y
    )

def scale_features(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Scale features using StandardScaler.
    
    Args:
        X_train: Training feature matrix
        X_test: Test feature matrix
        
    Returns:
        Tuple of (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def save_metrics(metrics: Dict[str, Any], output_dir: str, model_name: str):
    """
    Save metrics to a JSON file.
    
    Args:
        metrics: Dictionary containing metrics
        output_dir: Directory to save metrics
        model_name: Name of the model
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / f"{model_name}_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4)

def load_metrics(input_dir: str, model_name: str) -> Dict[str, Any]:
    """
    Load metrics from a JSON file.
    
    Args:
        input_dir: Directory containing metrics
        model_name: Name of the model
        
    Returns:
        Dictionary containing metrics
    """
    input_path = Path(input_dir)
    with open(input_path / f"{model_name}_metrics.json", 'r') as f:
        return json.load(f)

def get_feature_names(df: pd.DataFrame, assessment_type: str) -> list:
    """
    Get feature names for a given assessment type.
    
    Args:
        df: DataFrame containing the data
        assessment_type: Type of assessment ('AQ', 'SQ', or 'EQ')
        
    Returns:
        List of feature names
    """
    return [col for col in df.columns 
            if col.startswith(f'{assessment_type}_') 
            and not col.endswith('_score') 
            and col != 'asd_diagnosis'] 