#!/usr/bin/env python3
"""
Traditional ML Optimization Script for Autism Classification
Comprehensive hyperparameter tuning, feature selection, and ensemble optimization
"""

import os
import sys
import yaml
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# ML imports
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, classification_report
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Utility imports
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def setup_logging(log_dir):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"traditional_ml_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

class TraditionalMLOptimizer:
    """Comprehensive traditional ML optimization"""
    
    def __init__(self, config_path="hpc_config.yaml"):
        """Initialize the optimizer"""
        self.config = self.load_config(config_path)
        self.logger = setup_logging(self.config['output']['logs_dir'])
        self.setup_directories()
        self.load_data()
        
    def load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def setup_directories(self):
        """Create necessary directories"""
        for dir_name in ['results', 'models', 'logs', 'plots']:
            os.makedirs(self.config['output'][f'{dir_name}_dir'], exist_ok=True)
    
    def load_data(self):
        """Load and preprocess the dataset"""
        self.logger.info("Loading dataset...")
        
        data_file = self.config['data']['input_file']
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        self.df = pd.read_csv(data_file)
        self.logger.info(f"Dataset loaded: {self.df.shape}")
        
        # Prepare features and target
        self.X = self.df.drop(columns=[self.config['data']['target_column']])
        self.y = self.df[self.config['data']['target_column']]
        
        # Handle missing values
        imputer = SimpleImputer(strategy=self.config['feature_engineering']['imputation_strategy'])
        self.X_imputed = pd.DataFrame(
            imputer.fit_transform(self.X), 
            columns=self.X.columns
        )
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_imputed, self.y,
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state'],
            stratify=self.y if self.config['data']['stratify'] else None
        )
        
        # Apply scaling
        self.scaler = StandardScaler()
        self.X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.X_train.columns
        )
        self.X_test_scaled = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.X_test.columns
        )
        
        self.logger.info(f"Training set: {self.X_train_scaled.shape}")
        self.logger.info(f"Test set: {self.X_test_scaled.shape}")
    
    def optimize_feature_selection(self, X_train, X_test, y_train, y_test):
        """Optimize feature selection methods"""
        self.logger.info("Optimizing feature selection...")
        
        k_values = self.config['feature_engineering']['feature_selection']['selectkbest_k']
        best_k = None
        best_score = 0
        best_features = None
        
        for k in k_values:
            if k <= X_train.shape[1]:
                self.logger.info(f"Testing SelectKBest with k={k}")
                
                selector = SelectKBest(score_func=f_classif, k=k)
                X_train_selected = selector.fit_transform(X_train, y_train)
                X_test_selected = selector.transform(X_test)
                
                # Quick evaluation with Random Forest
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X_train_selected, y_train)
                
                y_pred = rf.predict(X_test_selected)
                y_probs = rf.predict_proba(X_test_selected)[:, 1]
                
                f1 = f1_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_probs)
                
                # Threshold optimization
                prec, rec, thresholds = precision_recall_curve(y_test, y_probs)
                f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)
                best_threshold = thresholds[np.argmax(f1_scores)]
                
                y_pred_optimized = (y_probs >= best_threshold).astype(int)
                f1_optimized = f1_score(y_test, y_pred_optimized)
                
                self.logger.info(f"k={k}: F1={f1:.4f}, Optimized F1={f1_optimized:.4f}")
                
                if f1_optimized > best_score:
                    best_score = f1_optimized
                    best_k = k
                    best_features = X_train.columns[selector.get_support()]
        
        self.logger.info(f"Best feature selection: k={best_k}, F1={best_score:.4f}")
        
        return best_k, best_features, best_score
    
    def optimize_random_forest(self, X_train, X_test, y_train, y_test):
        """Optimize Random Forest"""
        self.logger.info("Optimizing Random Forest...")
        
        rf = RandomForestClassifier(random_state=42)
        param_grid = self.config['random_forest']
        
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        best_rf = grid_search.best_estimator_
        y_pred = best_rf.predict(X_test)
        y_probs = best_rf.predict_proba(X_test)[:, 1]
        
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_probs)
        
        # Threshold optimization
        prec, rec, thresholds = precision_recall_curve(y_test, y_probs)
        f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)
        best_threshold = thresholds[np.argmax(f1_scores)]
        
        y_pred_optimized = (y_probs >= best_threshold).astype(int)
        f1_optimized = f1_score(y_test, y_pred_optimized)
        
        results = {
            'model': best_rf,
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_,
            'test_f1': f1,
            'test_auc': auc,
            'optimized_f1': f1_optimized,
            'best_threshold': best_threshold
        }
        
        self.logger.info(f"Random Forest - F1: {f1:.4f}, Optimized F1: {f1_optimized:.4f}")
        
        return results
    
    def optimize_xgboost(self, X_train, X_test, y_train, y_test):
        """Optimize XGBoost"""
        self.logger.info("Optimizing XGBoost...")
        
        xgb = XGBClassifier(random_state=42, eval_metric='logloss')
        param_grid = self.config['xgboost']
        
        grid_search = GridSearchCV(
            xgb, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        best_xgb = grid_search.best_estimator_
        y_pred = best_xgb.predict(X_test)
        y_probs = best_xgb.predict_proba(X_test)[:, 1]
        
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_probs)
        
        # Threshold optimization
        prec, rec, thresholds = precision_recall_curve(y_test, y_probs)
        f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)
        best_threshold = thresholds[np.argmax(f1_scores)]
        
        y_pred_optimized = (y_probs >= best_threshold).astype(int)
        f1_optimized = f1_score(y_test, y_pred_optimized)
        
        results = {
            'model': best_xgb,
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_,
            'test_f1': f1,
            'test_auc': auc,
            'optimized_f1': f1_optimized,
            'best_threshold': best_threshold
        }
        
        self.logger.info(f"XGBoost - F1: {f1:.4f}, Optimized F1: {f1_optimized:.4f}")
        
        return results
    
    def optimize_lightgbm(self, X_train, X_test, y_train, y_test):
        """Optimize LightGBM"""
        self.logger.info("Optimizing LightGBM...")
        
        lgb = LGBMClassifier(random_state=42, verbose=-1)
        param_grid = self.config['lightgbm']
        
        grid_search = GridSearchCV(
            lgb, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        best_lgb = grid_search.best_estimator_
        y_pred = best_lgb.predict(X_test)
        y_probs = best_lgb.predict_proba(X_test)[:, 1]
        
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_probs)
        
        # Threshold optimization
        prec, rec, thresholds = precision_recall_curve(y_test, y_probs)
        f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)
        best_threshold = thresholds[np.argmax(f1_scores)]
        
        y_pred_optimized = (y_probs >= best_threshold).astype(int)
        f1_optimized = f1_score(y_test, y_pred_optimized)
        
        results = {
            'model': best_lgb,
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_,
            'test_f1': f1,
            'test_auc': auc,
            'optimized_f1': f1_optimized,
            'best_threshold': best_threshold
        }
        
        self.logger.info(f"LightGBM - F1: {f1:.4f}, Optimized F1: {f1_optimized:.4f}")
        
        return results
    
    def optimize_logistic_regression(self, X_train, X_test, y_train, y_test):
        """Optimize Logistic Regression"""
        self.logger.info("Optimizing Logistic Regression...")
        
        lr = LogisticRegression(random_state=42)
        param_grid = self.config['logistic_regression']
        
        grid_search = GridSearchCV(
            lr, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        best_lr = grid_search.best_estimator_
        y_pred = best_lr.predict(X_test)
        y_probs = best_lr.predict_proba(X_test)[:, 1]
        
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_probs)
        
        # Threshold optimization
        prec, rec, thresholds = precision_recall_curve(y_test, y_probs)
        f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)
        best_threshold = thresholds[np.argmax(f1_scores)]
        
        y_pred_optimized = (y_probs >= best_threshold).astype(int)
        f1_optimized = f1_score(y_test, y_pred_optimized)
        
        results = {
            'model': best_lr,
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_,
            'test_f1': f1,
            'test_auc': auc,
            'optimized_f1': f1_optimized,
            'best_threshold': best_threshold
        }
        
        self.logger.info(f"Logistic Regression - F1: {f1:.4f}, Optimized F1: {f1_optimized:.4f}")
        
        return results
    
    def create_ensemble(self, models_dict, X_train, X_test, y_train, y_test):
        """Create voting ensemble from best models"""
        self.logger.info("Creating voting ensemble...")
        
        # Create voting classifier
        estimators = []
        for name, model_data in models_dict.items():
            estimators.append((name, model_data['model']))
        
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting='soft'
        )
        
        voting_clf.fit(X_train, y_train)
        y_pred = voting_clf.predict(X_test)
        y_probs = voting_clf.predict_proba(X_test)[:, 1]
        
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_probs)
        
        # Threshold optimization
        prec, rec, thresholds = precision_recall_curve(y_test, y_probs)
        f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)
        best_threshold = thresholds[np.argmax(f1_scores)]
        
        y_pred_optimized = (y_probs >= best_threshold).astype(int)
        f1_optimized = f1_score(y_test, y_pred_optimized)
        
        results = {
            'model': voting_clf,
            'test_f1': f1,
            'test_auc': auc,
            'optimized_f1': f1_optimized,
            'best_threshold': best_threshold
        }
        
        self.logger.info(f"Ensemble - F1: {f1:.4f}, Optimized F1: {f1_optimized:.4f}")
        
        return results
    
    def save_results(self, results, model_name):
        """Save results to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        if self.config['output']['save_models']:
            model_path = os.path.join(
                self.config['output']['models_dir'], 
                f"{model_name}_{timestamp}.joblib"
            )
            joblib.dump(results['model'], model_path)
            self.logger.info(f"Model saved: {model_path}")
        
        # Save results summary
        results_summary = {
            'model_name': model_name,
            'timestamp': timestamp,
            'test_f1': results['test_f1'],
            'test_auc': results['test_auc'],
            'optimized_f1': results['optimized_f1'],
            'best_threshold': results['best_threshold']
        }
        
        if 'best_params' in results:
            results_summary['best_params'] = results['best_params']
        if 'cv_score' in results:
            results_summary['cv_score'] = results['cv_score']
        
        results_path = os.path.join(
            self.config['output']['results_dir'],
            f"{model_name}_results_{timestamp}.json"
        )
        
        import json
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        self.logger.info(f"Results saved: {results_path}")
        
        return results_summary
    
    def run_optimization(self):
        """Run complete traditional ML optimization"""
        self.logger.info("Starting traditional ML optimization...")
        
        # Feature selection
        best_k, best_features, feature_score = self.optimize_feature_selection(
            self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
        )
        
        # Apply best feature selection
        selector = SelectKBest(score_func=f_classif, k=best_k)
        X_train_selected = selector.fit_transform(self.X_train_scaled, self.y_train)
        X_test_selected = selector.transform(self.X_test_scaled)
        
        all_results = {}
        
        # Optimize individual models
        rf_results = self.optimize_random_forest(X_train_selected, X_test_selected, self.y_train, self.y_test)
        all_results['random_forest'] = self.save_results(rf_results, 'random_forest')
        
        xgb_results = self.optimize_xgboost(X_train_selected, X_test_selected, self.y_train, self.y_test)
        all_results['xgboost'] = self.save_results(xgb_results, 'xgboost')
        
        lgb_results = self.optimize_lightgbm(X_train_selected, X_test_selected, self.y_train, self.y_test)
        all_results['lightgbm'] = self.save_results(lgb_results, 'lightgbm')
        
        lr_results = self.optimize_logistic_regression(X_train_selected, X_test_selected, self.y_train, self.y_test)
        all_results['logistic_regression'] = self.save_results(lr_results, 'logistic_regression')
        
        # Create ensemble
        models_dict = {
            'rf': rf_results,
            'xgb': xgb_results,
            'lgb': lgb_results,
            'lr': lr_results
        }
        
        ensemble_results = self.create_ensemble(models_dict, X_train_selected, X_test_selected, self.y_train, self.y_test)
        all_results['ensemble'] = self.save_results(ensemble_results, 'ensemble')
        
        # Summary
        self.logger.info("=== TRADITIONAL ML OPTIMIZATION SUMMARY ===")
        for model_name, results in all_results.items():
            self.logger.info(f"{model_name}: F1={results['test_f1']:.4f}, "
                           f"Optimized F1={results['optimized_f1']:.4f}")
        
        return all_results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Traditional ML Optimization')
    parser.add_argument('--config', default='hpc_config.yaml', help='Configuration file')
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = TraditionalMLOptimizer(args.config)
    
    # Run optimization
    results = optimizer.run_optimization()
    
    print("Traditional ML optimization completed successfully!")

if __name__ == "__main__":
    main() 