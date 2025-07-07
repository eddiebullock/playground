"""
Model training pipeline for autism diagnosis prediction project.

This module handles:
- Model training with cross-validation
- Baseline models (Random Forest, XGBoost, Logistic Regression)
- Clinical evaluation metrics
- Feature importance analysis
- Model comparison and selection
"""

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List
from datetime import datetime
import pickle
import json

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb
from imblearn.over_sampling import SMOTE

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ClinicalModelTrainer:
    """
    Handles model training with clinical evaluation metrics.
    """
    
    def __init__(self, output_dir: str = "experiments/models"):
        """
        Initialize the model trainer.
        
        Args:
            output_dir: Directory to save models and results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        
        # Initialize imputer
        self.imputer = SimpleImputer(strategy='median')
        
    def load_data(self, data_path: str, subset: int = None, random_state: int = 42) -> tuple:
        """
        Load processed data and prepare features/target. Optionally use a random subset for local runs.
        """
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Loaded data shape: {df.shape}")
        if subset is not None and subset < len(df):
            df = df.sample(n=subset, random_state=random_state)
            logger.info(f"Using random subset of {subset} samples for local experimentation.")
        # Prepare target (autism_any)
        target_col = 'autism_any'
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        cols_to_drop = [
            'autism_any', 'userid',
            'autism_subtype', 'autism_subtype_1', 'autism_subtype_2', 'autism_subtype_3',
            'autism_diagnosis_0', 'autism_diagnosis_1', 'autism_diagnosis_2'
        ]
        X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        y = df[target_col]
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            logger.info(f"Dropping categorical columns: {categorical_cols.tolist()}")
            X = X.drop(columns=categorical_cols)
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X = X.dropna(axis=1, how='all')
        logger.info(f"Features shape after cleaning: {X.shape}")
        logger.info(f"Target distribution:\n{y.value_counts()}")
        return X, y
    
    def add_missingness_indicators(self, X: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
        """
        Add missingness indicator columns for features with >threshold missing values.
        """
        X_new = X.copy()
        n = len(X_new)
        for col in X_new.columns:
            missing_pct = X_new[col].isnull().mean()
            if missing_pct > threshold:
                X_new[f'{col}_missing'] = X_new[col].isnull().astype(int)
        return X_new

    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2, val_size: float = 0.2,
                   random_state: int = 42, add_missingness: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 
                                                   pd.Series, pd.Series, pd.Series]:
        """
        Split data into train/validation/test sets.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            test_size: Proportion for test set
            val_size: Proportion for validation set (of remaining data)
            random_state: Random seed for reproducibility
            add_missingness: Whether to add missingness indicators
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("Splitting data into train/validation/test sets...")
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train vs validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
        )
        
        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Validation set: {X_val.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        
        # Check class distribution in each split
        for name, y_split in [("Train", y_train), ("Validation", y_val), ("Test", y_test)]:
            logger.info(f"{name} target distribution: {y_split.value_counts().to_dict()}")
        
        # Add missingness indicators if requested
        if add_missingness:
            X_train = self.add_missingness_indicators(X_train)
            X_val = self.add_missingness_indicators(X_val)
            X_test = self.add_missingness_indicators(X_test)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def initialize_models(self, only_logistic: bool = False) -> dict:
        """
        Initialize baseline models. If only_logistic is True, only initialize Logistic Regression.
        """
        logger.info("Initializing baseline models...")
        models = {}
        if not only_logistic:
            models['random_forest'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                scale_pos_weight=1,
                eval_metric='logloss'
            )
        models['logistic_regression'] = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced',
            solver='liblinear'
        )
        self.models = models
        logger.info(f"Initialized {len(models)} models: {list(models.keys())}")
        return models
    
    def train_models(self, X_train, y_train, X_val, y_val, use_smote=True, only_logistic=False):
        logger.info("Training baseline models with local options...")
        results = {}
        # Impute missing values
        logger.info("Imputing missing values...")
        X_train_imputed = pd.DataFrame(self.imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_val_imputed = pd.DataFrame(self.imputer.transform(X_val), columns=X_val.columns, index=X_val.index)
        # SMOTE oversampling (skip if only_logistic or use_smote is False)
        if use_smote and not only_logistic:
            logger.info("Applying SMOTE oversampling to training set...")
            smote = SMOTE(random_state=42)
            X_train_imputed, y_train = smote.fit_resample(X_train_imputed, y_train)
            logger.info(f"After SMOTE: {dict(pd.Series(y_train).value_counts())}")
        # Set scale_pos_weight for XGBoost
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        if 'xgboost' in self.models:
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
            self.models['xgboost'].set_params(scale_pos_weight=scale_pos_weight)
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train_imputed, y_train)
            y_pred_proba = model.predict_proba(X_val_imputed)[:, 1]
            # Threshold optimization
            best_f1, best_thresh = 0, 0.5
            precisions, recalls, thresholds = precision_recall_curve(y_val, y_pred_proba)
            for t in thresholds:
                y_pred = (y_pred_proba >= t).astype(int)
                f1 = f1_score(y_val, y_pred)
                if f1 > best_f1:
                    best_f1, best_thresh = f1, t
            logger.info(f"{name} best F1 on validation: {best_f1:.4f} at threshold {best_thresh:.3f}")
            y_pred = (y_pred_proba >= best_thresh).astype(int)
            metrics = self._calculate_clinical_metrics(y_val, y_pred, y_pred_proba)
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = dict(zip(X_train.columns, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                self.feature_importance[name] = dict(zip(X_train.columns, np.abs(model.coef_[0])))
            results[name] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'best_threshold': best_thresh
            }
            logger.info(f"{name} validation metrics:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
        self.results = results
        return results
    
    def _calculate_clinical_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                                  y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate clinical evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary of clinical metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Clinical metrics
        metrics['sensitivity'] = recall_score(y_true, y_pred, zero_division=0)  # Same as recall
        metrics['specificity'] = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        
        # AUC metrics
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        
        # Calculate precision-recall AUC
        try:
            from sklearn.metrics import average_precision_score
            metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba)
        except ImportError:
            metrics['pr_auc'] = np.nan
        
        # Balanced accuracy (important for imbalanced clinical data)
        metrics['balanced_accuracy'] = (metrics['sensitivity'] + metrics['specificity']) / 2
        
        # Positive and negative predictive values
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive predictive value
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative predictive value
        
        return metrics
    
    def evaluate_on_test_set(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate all trained models on the test set.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary with test set results
        """
        logger.info("Evaluating models on test set with optimized thresholds...")
        
        test_results = {}
        
        # Impute test set
        X_test_imputed = pd.DataFrame(
            self.imputer.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        for name, result in self.results.items():
            model = result['model']
            
            # Predictions on test set
            best_thresh = result.get('best_threshold', 0.5)
            y_pred_proba = model.predict_proba(X_test_imputed)[:, 1]
            y_pred = (y_pred_proba >= best_thresh).astype(int)
            
            # Calculate test metrics
            test_metrics = self._calculate_clinical_metrics(y_test, y_pred, y_pred_proba)
            
            test_results[name] = {
                'metrics': test_metrics,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'best_threshold': best_thresh
            }
            
            logger.info(f"{name} test metrics:")
            for metric, value in test_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
        
        return test_results
    
    def create_evaluation_plots(self, X_test: pd.DataFrame, y_test: pd.Series,
                              test_results: Dict[str, Any]) -> None:
        """
        Create comprehensive evaluation plots.
        
        Args:
            X_test: Test features
            y_test: Test target
            test_results: Test set results
        """
        logger.info("Creating evaluation plots...")
        
        # 1. ROC Curves
        plt.figure(figsize=(10, 6))
        for name, result in test_results.items():
            fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
            auc = result['metrics']['roc_auc']
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Test Set')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Precision-Recall Curves
        plt.figure(figsize=(10, 6))
        for name, result in test_results.items():
            precision, recall, _ = precision_recall_curve(y_test, result['probabilities'])
            pr_auc = result['metrics']['pr_auc']
            plt.plot(recall, precision, label=f'{name} (PR-AUC = {pr_auc:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves - Test Set')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.output_dir / 'precision_recall_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Confusion Matrices
        fig, axes = plt.subplots(1, len(test_results), figsize=(5*len(test_results), 4))
        if len(test_results) == 1:
            axes = [axes]
        
        for i, (name, result) in enumerate(test_results.items()):
            cm = result['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{name} - Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Feature Importance (top 20 features)
        fig, axes = plt.subplots(1, len(self.feature_importance), figsize=(6*len(self.feature_importance), 8))
        if len(self.feature_importance) == 1:
            axes = [axes]
        
        for i, (name, importance_dict) in enumerate(self.feature_importance.items()):
            # Sort by importance and take top 20
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:20]
            features, importances = zip(*sorted_features)
            
            axes[i].barh(range(len(features)), importances)
            axes[i].set_yticks(range(len(features)))
            axes[i].set_yticklabels(features)
            axes[i].set_xlabel('Importance')
            axes[i].set_title(f'{name} - Feature Importance')
            axes[i].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, test_results: Dict[str, Any], output_file: str = None) -> None:
        """
        Save training results and model comparison.
        
        Args:
            test_results: Test set results
            output_file: Output file path
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"experiments/logs/model_results_{timestamp}.yaml"
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare results for saving
        results_summary = {
            'timestamp': datetime.now().isoformat(),
            'test_metrics': {},
            'feature_importance': {},
            'model_comparison': {}
        }
        
        # Test metrics
        for name, result in test_results.items():
            results_summary['test_metrics'][name] = result['metrics']
        
        # Feature importance
        for name, importance_dict in self.feature_importance.items():
            # Sort by importance and take top 20
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:20]
            results_summary['feature_importance'][name] = dict(sorted_features)
        
        # Model comparison
        comparison_data = []
        for name, result in test_results.items():
            metrics = result['metrics']
            comparison_data.append({
                'model': name,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'roc_auc': metrics['roc_auc'],
                'balanced_accuracy': metrics['balanced_accuracy'],
                'sensitivity': metrics['sensitivity'],
                'specificity': metrics['specificity']
            })
        
        results_summary['model_comparison'] = comparison_data
        
        # Save as YAML
        with open(output_path, 'w') as f:
            yaml.dump(results_summary, f, default_flow_style=False)
        
        # Save models
        models_dir = self.output_dir / 'trained_models'
        models_dir.mkdir(exist_ok=True)
        
        for name, result in self.results.items():
            model_path = models_dir / f'{name}_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(result['model'], f)
        
        logger.info(f"Results saved to {output_path}")
        logger.info(f"Models saved to {models_dir}")
    
    def create_summary_report(self, test_results: Dict[str, Any]) -> str:
        """
        Create a summary report of model performance.
        
        Args:
            test_results: Test set results
            
        Returns:
            Summary report string
        """
        report = f"""
# Model Training Summary Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Test Set Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Balanced Acc | Sensitivity | Specificity |
|-------|----------|-----------|--------|----------|---------|--------------|-------------|-------------|
"""
        
        for name, result in test_results.items():
            metrics = result['metrics']
            report += f"| {name} | {metrics['accuracy']:.3f} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1_score']:.3f} | {metrics['roc_auc']:.3f} | {metrics['balanced_accuracy']:.3f} | {metrics['sensitivity']:.3f} | {metrics['specificity']:.3f} |\n"
        
        report += "\n## Top Features by Model\n"
        
        for name, importance_dict in self.feature_importance.items():
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
            report += f"\n### {name.title()}\n"
            for feature, importance in sorted_features:
                report += f"- {feature}: {importance:.4f}\n"
        
        return report

def main():
    """Main function to run model training pipeline."""
    
    # Initialize trainer
    trainer = ClinicalModelTrainer()
    
    # Load data
    try:
        X, y = trainer.load_data("data/processed/features_full.csv")
    except FileNotFoundError:
        logger.error("Processed data not found. Run data preprocessing first.")
        return
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(X, y)
    
    # Initialize models
    trainer.initialize_models()
    
    # Train models
    val_results = trainer.train_models(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    test_results = trainer.evaluate_on_test_set(X_test, y_test)
    
    # Create plots
    trainer.create_evaluation_plots(X_test, y_test, test_results)
    
    # Save results
    trainer.save_results(test_results)
    
    # Print summary
    summary = trainer.create_summary_report(test_results)
    print(summary)
    
    logger.info("Model training pipeline completed successfully!")

if __name__ == "__main__":
    main() 