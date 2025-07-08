#!/usr/bin/env python3
"""
Baseline model training script for autism prediction project.
Implements simple models with basic evaluation metrics.
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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
from sklearn.neural_network import MLPClassifier

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_prepare_data(data_path: str, target_col: str = 'diagnosis') -> Tuple[pd.DataFrame, pd.Series]:
    """Load and prepare data for modeling."""
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

def preprocess_features(X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Basic preprocessing for baseline models."""
    logger.info("Preprocessing features...")
    
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
    
    logger.info(f"Preprocessing completed. Final shape: {X.shape}")
    return X, preprocessing_info

def train_baseline_models(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, 
                         random_state: int = 42) -> Dict:
    """Train baseline models and evaluate performance."""
    logger.info("Training baseline models...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Training set: {X_train.shape}")
    logger.info(f"Test set: {X_test.shape}")
    
    # Define models
    models = {
        'logistic_regression': LogisticRegression(random_state=random_state, max_iter=1000),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
        'xgboost': xgb.XGBClassifier(random_state=random_state, eval_metric='logloss'),
        'mlp': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100, random_state=random_state)
    }
    
    results = {}
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'accuracy': (y_pred == y_test).mean(),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
        metrics['cv_f1_mean'] = cv_scores.mean()
        metrics['cv_f1_std'] = cv_scores.std()
        
        results[name] = {
            'model': model,
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        logger.info(f"{name} - F1: {metrics['f1_score']:.3f}, CV F1: {metrics['cv_f1_mean']:.3f} ± {metrics['cv_f1_std']:.3f}")
    
    return results

def save_results(results: Dict, output_dir: str):
    """Save model results and performance metrics."""
    logger.info("Saving results...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save performance summary
    summary = []
    summary.append("Model Performance Summary")
    summary.append("=" * 40)
    summary.append("")
    
    for name, result in results.items():
        metrics = result['metrics']
        summary.append(f"## {name.replace('_', ' ').title()}")
        summary.append(f"- Accuracy: {metrics['accuracy']:.3f}")
        summary.append(f"- F1 Score: {metrics['f1_score']:.3f}")
        summary.append(f"- CV F1 Score: {metrics['cv_f1_mean']:.3f} ± {metrics['cv_f1_std']:.3f}")
        if 'roc_auc' in metrics:
            summary.append(f"- ROC AUC: {metrics['roc_auc']:.3f}")
        summary.append("")
    
    # Save summary
    summary_path = os.path.join(output_dir, 'model_performance_summary.txt')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary))
    
    # Save detailed results
    for name, result in results.items():
        # Save predictions
        pred_df = pd.DataFrame({
            'actual': result.get('actual', []),
            'predicted': result['predictions']
        })
        if result['probabilities'] is not None:
            pred_df['probability'] = result['probabilities']
        
        pred_path = os.path.join(output_dir, f'{name}_predictions.csv')
        pred_df.to_csv(pred_path, index=False)
        
        # Save classification report
        report = result['metrics']['classification_report']
        report_df = pd.DataFrame(report).transpose()
        report_path = os.path.join(output_dir, f'{name}_classification_report.csv')
        report_df.to_csv(report_path)
    
    logger.info(f"Results saved in: {output_dir}")

def create_visualizations(results: Dict, output_dir: str):
    """Create visualization plots for model comparison."""
    logger.info("Creating visualizations...")
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set up plotting
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Model comparison plot
    model_names = list(results.keys())
    f1_scores = [results[name]['metrics']['f1_score'] for name in model_names]
    cv_scores = [results[name]['metrics']['cv_f1_mean'] for name in model_names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # F1 scores comparison
    bars1 = ax1.bar(model_names, f1_scores, alpha=0.7)
    ax1.set_title('F1 Scores Comparison')
    ax1.set_ylabel('F1 Score')
    ax1.set_ylim(0, 1)
    for bar, score in zip(bars1, f1_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    # Cross-validation scores
    bars2 = ax2.bar(model_names, cv_scores, alpha=0.7)
    ax2.set_title('Cross-Validation F1 Scores')
    ax2.set_ylabel('CV F1 Score')
    ax2.set_ylim(0, 1)
    for bar, score in zip(bars2, cv_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Confusion matrices
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    if n_models == 1:
        axes = [axes]
    
    for i, (name, result) in enumerate(results.items()):
        cm = confusion_matrix(result.get('actual', []), result['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'{name.replace("_", " ").title()}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train baseline models for autism prediction')
    parser.add_argument('--data_path', type=str, default='data/processed/features_full.csv',
                       help='Path to the data file')
    parser.add_argument('--output_dir', type=str, default='results/baseline',
                       help='Output directory for results')
    parser.add_argument('--target_col', type=str, default='autism_any',
                       help='Name of the target variable column')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Proportion of data for testing')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Load and prepare data
    X, y = load_and_prepare_data(args.data_path, args.target_col)
    
    # Preprocess features
    X_processed, preprocessing_info = preprocess_features(X)
    
    # Train baseline models
    results = train_baseline_models(X_processed, y, args.test_size, args.random_state)
    
    # Save results
    save_results(results, args.output_dir)
    
    # Create visualizations
    create_visualizations(results, args.output_dir)
    
    logger.info("Baseline model training completed successfully!")
    logger.info(f"Results saved in: {args.output_dir}")

if __name__ == "__main__":
    main() 