#!/usr/bin/env python3
"""
Comprehensive data exploration script for autism prediction project.
Analyzes data quality, missing values, distributions, and provides insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_explore_data(data_path: str) -> pd.DataFrame:
    """Load data and perform initial exploration."""
    logger.info(f"Loading data from {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    return df

def analyze_missing_values(df: pd.DataFrame) -> Dict:
    """Analyze missing values in the dataset."""
    logger.info("Analyzing missing values...")
    
    missing_info = {
        'total_missing': df.isnull().sum().sum(),
        'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
        'columns_with_missing': df.columns[df.isnull().sum() > 0].tolist(),
        'missing_by_column': df.isnull().sum().to_dict()
    }
    
    logger.info(f"Total missing values: {missing_info['total_missing']}")
    logger.info(f"Missing percentage: {missing_info['missing_percentage']:.2f}%")
    logger.info(f"Columns with missing values: {len(missing_info['columns_with_missing'])}")
    
    return missing_info

def analyze_data_types(df: pd.DataFrame) -> Dict:
    """Analyze data types and categorical variables."""
    logger.info("Analyzing data types...")
    
    type_info = {
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
        'datetime_columns': df.select_dtypes(include=['datetime']).columns.tolist(),
        'unique_counts': {col: df[col].nunique() for col in df.columns}
    }
    
    logger.info(f"Numeric columns: {len(type_info['numeric_columns'])}")
    logger.info(f"Categorical columns: {len(type_info['categorical_columns'])}")
    
    return type_info

def analyze_target_variable(df: pd.DataFrame, target_col: str = 'diagnosis') -> Dict:
    """Analyze the target variable distribution."""
    logger.info(f"Analyzing target variable: {target_col}")
    
    if target_col not in df.columns:
        logger.warning(f"Target column '{target_col}' not found. Available columns: {list(df.columns)}")
        return {}
    
    target_info = {
        'value_counts': df[target_col].value_counts().to_dict(),
        'missing_count': df[target_col].isnull().sum(),
        'class_balance': df[target_col].value_counts(normalize=True).to_dict()
    }
    
    logger.info(f"Target distribution: {target_info['value_counts']}")
    logger.info(f"Class balance: {target_info['class_balance']}")
    
    return target_info

def create_visualizations(df: pd.DataFrame, output_dir: str, target_col: str = 'diagnosis'):
    """Create exploratory visualizations."""
    logger.info("Creating visualizations...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Missing values heatmap
    plt.figure(figsize=(12, 8))
    missing_data = df.isnull()
    sns.heatmap(missing_data, cbar=True, yticklabels=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'missing_values_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Target variable distribution
    if target_col in df.columns:
        plt.figure(figsize=(8, 6))
        df[target_col].value_counts().plot(kind='bar')
        plt.title(f'Distribution of {target_col}')
        plt.xlabel(target_col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'target_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Numeric variables distributions
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                df[col].hist(ax=axes[i], bins=30, alpha=0.7)
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
        
        # Hide empty subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'numeric_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Correlation matrix for numeric variables
    if len(numeric_cols) > 1:
        plt.figure(figsize=(12, 10))
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5)
        plt.title('Correlation Matrix of Numeric Variables')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()

def generate_report(df: pd.DataFrame, missing_info: Dict, type_info: Dict, 
                   target_info: Dict, output_dir: str) -> str:
    """Generate a comprehensive exploration report."""
    logger.info("Generating exploration report...")
    
    report = []
    report.append("# Data Exploration Report")
    report.append("=" * 50)
    report.append("")
    
    # Basic information
    report.append("## Basic Information")
    report.append(f"- Dataset shape: {df.shape}")
    report.append(f"- Number of features: {df.shape[1]}")
    report.append(f"- Number of samples: {df.shape[0]}")
    report.append("")
    
    # Missing values
    report.append("## Missing Values Analysis")
    report.append(f"- Total missing values: {missing_info['total_missing']}")
    report.append(f"- Missing percentage: {missing_info['missing_percentage']:.2f}%")
    report.append(f"- Columns with missing values: {len(missing_info['columns_with_missing'])}")
    if missing_info['columns_with_missing']:
        report.append("### Columns with missing values:")
        for col in missing_info['columns_with_missing']:
            missing_count = missing_info['missing_by_column'][col]
            missing_pct = (missing_count / len(df)) * 100
            report.append(f"- {col}: {missing_count} ({missing_pct:.1f}%)")
    report.append("")
    
    # Data types
    report.append("## Data Types")
    report.append(f"- Numeric columns: {len(type_info['numeric_columns'])}")
    report.append(f"- Categorical columns: {len(type_info['categorical_columns'])}")
    report.append(f"- Datetime columns: {len(type_info['datetime_columns'])}")
    report.append("")
    
    # Target variable
    if target_info:
        report.append("## Target Variable Analysis")
        report.append(f"- Target column: {list(target_info['value_counts'].keys())[0] if target_info['value_counts'] else 'N/A'}")
        report.append(f"- Missing target values: {target_info['missing_count']}")
        report.append("### Class Distribution:")
        for class_name, count in target_info['value_counts'].items():
            percentage = target_info['class_balance'].get(class_name, 0) * 100
            report.append(f"- {class_name}: {count} ({percentage:.1f}%)")
        report.append("")
    
    # Recommendations
    report.append("## Recommendations")
    report.append("### Data Preprocessing:")
    if missing_info['missing_percentage'] > 5:
        report.append("- High missing values detected. Consider imputation strategies.")
    if target_info and len(target_info['value_counts']) == 2:
        balance = list(target_info['class_balance'].values())
        if max(balance) - min(balance) > 0.1:
            report.append("- Class imbalance detected. Consider resampling techniques.")
    
    report.append("### Feature Engineering:")
    if len(type_info['categorical_columns']) > 0:
        report.append("- Categorical variables detected. Consider encoding strategies.")
    if len(type_info['numeric_columns']) > 10:
        report.append("- Many numeric features. Consider feature selection.")
    
    # Save report
    report_path = os.path.join(output_dir, 'exploration_report.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    logger.info(f"Report saved to {report_path}")
    return report_path

def main():
    parser = argparse.ArgumentParser(description='Comprehensive data exploration for autism prediction')
    parser.add_argument('--data_path', type=str, default='/home/eb2007/predict_asc_c4/data/data_c4_raw.csv',
                       help='Path to the data file')
    parser.add_argument('--output_dir', type=str, default='exploration_results',
                       help='Output directory for results')
    parser.add_argument('--target_col', type=str, default='diagnosis',
                       help='Name of the target variable column')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Number of samples to analyze (for large datasets)')
    
    args = parser.parse_args()
    
    # Load data
    df = load_and_explore_data(args.data_path)
    
    # Sample if specified
    if args.sample_size and args.sample_size < len(df):
        df = df.sample(n=args.sample_size, random_state=42)
        logger.info(f"Sampled {args.sample_size} rows for analysis")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Perform analyses
    missing_info = analyze_missing_values(df)
    type_info = analyze_data_types(df)
    target_info = analyze_target_variable(df, args.target_col)
    
    # Create visualizations
    create_visualizations(df, args.output_dir, args.target_col)
    
    # Generate report
    report_path = generate_report(df, missing_info, type_info, target_info, args.output_dir)
    
    logger.info("Data exploration completed successfully!")
    logger.info(f"Results saved in: {args.output_dir}")
    logger.info(f"Report saved as: {report_path}")

if __name__ == "__main__":
    main() 