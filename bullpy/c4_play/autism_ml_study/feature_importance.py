#!/usr/bin/env python3
"""
Feature importance analysis script for autism prediction project.
Analyzes feature importance using multiple methods.
"""

import pandas as pd
import numpy as np
import argparse
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Machine learning imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_prepare_data(data_path: str, target_col: str = 'diagnosis') -> Tuple[pd.DataFrame, pd.Series]:
    """Load and prepare data for feature importance analysis."""
    logger.info(f"Loading data from {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"Data shape: {df.shape}")
    
    # Separate features and target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available columns: {list(df.columns)}")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y

def preprocess_for_importance(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """Preprocess data for feature importance analysis."""
    logger.info("Preprocessing data for feature importance analysis...")
    
    preprocessing_info = {}
    
    # Handle missing values
    missing_cols = X.columns[X.isnull().sum() > 0].tolist()
    if missing_cols:
        logger.info(f"Filling missing values in {len(missing_cols)} columns")
        for col in missing_cols:
            if X[col].dtype in ['int64', 'float64']:
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown')
    
    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
    
    preprocessing_info['encoders'] = encoders
    preprocessing_info['categorical_cols'] = categorical_cols
    
    # Scale numeric features
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        preprocessing_info['scaler'] = scaler
        preprocessing_info['numeric_cols'] = numeric_cols
    
    # Encode target variable
    if y.dtype == 'object':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
        preprocessing_info['target_encoder'] = le_target
    
    logger.info(f"Preprocessing completed. Final shape: {X.shape}")
    return X, y, preprocessing_info

def calculate_correlation_importance(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Calculate feature importance based on correlation with target."""
    logger.info("Calculating correlation-based feature importance...")
    
    correlations = {}
    for col in X.columns:
        if X[col].dtype in ['int64', 'float64']:
            corr = abs(X[col].corr(y))
            correlations[col] = corr
    
    correlation_importance = pd.Series(correlations).sort_values(ascending=False)
    return correlation_importance

def calculate_permutation_importance(X: pd.DataFrame, y: pd.Series, 
                                   random_state: int = 42) -> Tuple[pd.Series, pd.Series]:
    """Calculate permutation importance using Random Forest."""
    logger.info("Calculating permutation importance...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    rf.fit(X_train, y_train)
    
    # Calculate permutation importance
    perm_importance = permutation_importance(
        rf, X_test, y_test, n_repeats=10, random_state=random_state
    )
    
    # Create Series
    perm_importance_mean = pd.Series(
        perm_importance.importances_mean, index=X.columns
    ).sort_values(ascending=False)
    
    perm_importance_std = pd.Series(
        perm_importance.importances_std, index=X.columns
    )
    
    return perm_importance_mean, perm_importance_std

def calculate_xgboost_importance(X: pd.DataFrame, y: pd.Series, 
                               random_state: int = 42) -> pd.Series:
    """Calculate feature importance using XGBoost."""
    logger.info("Calculating XGBoost feature importance...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )
    
    # Train XGBoost
    xgb_model = xgb.XGBClassifier(random_state=random_state, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    
    # Get feature importance
    importance = xgb_model.feature_importances_
    xgb_importance = pd.Series(importance, index=X.columns).sort_values(ascending=False)
    
    return xgb_importance

def analyze_feature_groups(X: pd.DataFrame) -> Dict:
    """Analyze feature importance by groups (e.g., questionnaire blocks)."""
    logger.info("Analyzing feature importance by groups...")
    
    # Group features by common prefixes (assuming questionnaire structure)
    feature_groups = {}
    
    for col in X.columns:
        # Extract prefix (e.g., 'Q1_' from 'Q1_age')
        parts = col.split('_')
        if len(parts) > 1:
            prefix = parts[0]
            if prefix not in feature_groups:
                feature_groups[prefix] = []
            feature_groups[prefix].append(col)
        else:
            # Single word features
            if 'other' not in feature_groups:
                feature_groups['other'] = []
            feature_groups['other'].append(col)
    
    logger.info(f"Feature groups found: {list(feature_groups.keys())}")
    for group, features in feature_groups.items():
        logger.info(f"  {group}: {len(features)} features")
    
    return feature_groups

def create_importance_visualizations(importance_results: Dict, output_dir: str):
    """Create visualizations for feature importance analysis."""
    logger.info("Creating feature importance visualizations...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up plotting
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Top features comparison
    methods = ['correlation', 'permutation', 'xgboost']
    top_n = 20
    
    fig, axes = plt.subplots(len(methods), 1, figsize=(12, 4*len(methods)))
    if len(methods) == 1:
        axes = [axes]
    
    for i, method in enumerate(methods):
        if method in importance_results:
            importance = importance_results[method]
            top_features = importance.head(top_n)
            
            bars = axes[i].barh(range(len(top_features)), top_features.values)
            axes[i].set_yticks(range(len(top_features)))
            axes[i].set_yticklabels(top_features.index)
            axes[i].set_title(f'{method.replace("_", " ").title()} Importance')
            axes[i].set_xlabel('Importance Score')
            
            # Add value labels
            for j, (bar, value) in enumerate(zip(bars, top_features.values)):
                axes[i].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                           f'{value:.3f}', ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Correlation heatmap for top features
    if 'correlation' in importance_results:
        top_features = importance_results['correlation'].head(15).index
        correlation_matrix = X[top_features].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, fmt='.2f')
        plt.title('Correlation Matrix of Top Features')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_features_correlation.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

def save_importance_results(importance_results: Dict, feature_groups: Dict, output_dir: str):
    """Save feature importance results."""
    logger.info("Saving feature importance results...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual importance scores
    for method, importance in importance_results.items():
        importance_df = importance.reset_index()
        importance_df.columns = ['feature', 'importance']
        importance_df.to_csv(os.path.join(output_dir, f'{method}_importance.csv'), index=False)
    
    # Save summary report
    summary = []
    summary.append("# Feature Importance Analysis Report")
    summary.append("=" * 50)
    summary.append("")
    
    # Top features by method
    for method, importance in importance_results.items():
        summary.append(f"## Top 10 Features by {method.replace('_', ' ').title()}")
        top_10 = importance.head(10)
        for i, (feature, score) in enumerate(top_10.items(), 1):
            summary.append(f"{i}. {feature}: {score:.3f}")
        summary.append("")
    
    # Feature group analysis
    summary.append("## Feature Groups Analysis")
    for group, features in feature_groups.items():
        summary.append(f"### {group.upper()} ({len(features)} features)")
        if group in importance_results:
            group_importance = importance_results['permutation'][features].mean()
            summary.append(f"Average importance: {group_importance:.3f}")
        summary.append("")
    
    # Save summary
    summary_path = os.path.join(output_dir, 'feature_importance_report.md')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary))
    
    logger.info(f"Results saved in: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Analyze feature importance for autism prediction')
    parser.add_argument('--data_path', type=str, default='/home/eb2007/predict_asc_c4/data/data_c4_raw.csv',
                       help='Path to the data file')
    parser.add_argument('--output_dir', type=str, default='feature_importance_results',
                       help='Output directory for results')
    parser.add_argument('--target_col', type=str, default='diagnosis',
                       help='Name of the target variable column')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Load and prepare data
    X, y = load_and_prepare_data(args.data_path, args.target_col)
    
    # Preprocess data
    X_processed, y_processed, preprocessing_info = preprocess_for_importance(X, y)
    
    # Calculate different types of importance
    importance_results = {}
    
    # Correlation importance
    correlation_importance = calculate_correlation_importance(X_processed, y_processed)
    importance_results['correlation'] = correlation_importance
    
    # Permutation importance
    perm_importance_mean, perm_importance_std = calculate_permutation_importance(
        X_processed, y_processed, args.random_state
    )
    importance_results['permutation'] = perm_importance_mean
    
    # XGBoost importance
    xgb_importance = calculate_xgboost_importance(X_processed, y_processed, args.random_state)
    importance_results['xgboost'] = xgb_importance
    
    # Analyze feature groups
    feature_groups = analyze_feature_groups(X_processed)
    
    # Create visualizations
    create_importance_visualizations(importance_results, args.output_dir)
    
    # Save results
    save_importance_results(importance_results, feature_groups, args.output_dir)
    
    logger.info("Feature importance analysis completed successfully!")
    logger.info(f"Results saved in: {args.output_dir}")

if __name__ == "__main__":
    main() 