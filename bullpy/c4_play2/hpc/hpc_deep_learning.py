#!/usr/bin/env python3
"""
HPC Deep Learning Optimization Script for Autism Classification
This script performs comprehensive MLP hyperparameter optimization.
"""

import os
import sys
import yaml
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# ML imports
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, classification_report
from sklearn.feature_selection import SelectKBest, f_classif

# Utility imports
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def setup_logging(log_dir):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"hpc_deep_learning_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

class HPCDeepLearningOptimizer:
    """Main class for HPC deep learning optimization"""
    
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
        
        # Apply scaling (critical for neural networks)
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
    
    def feature_selection(self, k=100):
        """Perform feature selection"""
        self.logger.info(f"Performing feature selection with k={k}")
        
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train_selected = selector.fit_transform(self.X_train_scaled, self.y_train)
        X_test_selected = selector.transform(self.X_test_scaled)
        
        selected_features = self.X_train_scaled.columns[selector.get_support()]
        self.logger.info(f"Selected {len(selected_features)} features")
        
        return X_train_selected, X_test_selected, selected_features
    
    def optimize_mlp_architectures(self, X_train, X_test, y_train, y_test):
        """Optimize MLP architectures"""
        self.logger.info("Optimizing MLP architectures...")
        
        # Create parameter grid for MLP
        param_grid = {
            'hidden_layer_sizes': self.config['neural_network']['hidden_layer_sizes'],
            'activation': self.config['neural_network']['activation'],
            'solver': self.config['neural_network']['solver'],
            'alpha': self.config['neural_network']['alpha'],
            'learning_rate': self.config['neural_network']['learning_rate'],
            'learning_rate_init': self.config['neural_network']['learning_rate_init'],
            'max_iter': self.config['neural_network']['max_iter']
        }
        
        # Initialize MLP
        mlp = MLPClassifier(
            early_stopping=self.config['neural_network']['early_stopping'],
            validation_fraction=self.config['neural_network']['validation_fraction'],
            random_state=42
        )
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=mlp,
            param_grid=param_grid,
            scoring='f1',
            cv=self.config['cross_validation']['inner_cv'],
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the grid search
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_mlp = grid_search.best_estimator_
        
        # Evaluate on test set
        y_pred = best_mlp.predict(X_test)
        y_probs = best_mlp.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_probs)
        
        # Threshold optimization
        prec, rec, thresholds = precision_recall_curve(y_test, y_probs)
        f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)
        best_threshold = thresholds[np.argmax(f1_scores)]
        
        y_pred_optimized = (y_probs >= best_threshold).astype(int)
        f1_optimized = f1_score(y_test, y_pred_optimized)
        
        results = {
            'model': best_mlp,
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_,
            'test_f1': f1,
            'test_auc': auc,
            'optimized_f1': f1_optimized,
            'best_threshold': best_threshold
        }
        
        self.logger.info(f"MLP - F1: {f1:.4f}, Optimized F1: {f1_optimized:.4f}")
        
        return results
    
    def optimize_deep_ensemble(self, X_train, X_test, y_train, y_test):
        """Optimize deep ensemble with multiple MLP architectures"""
        self.logger.info("Optimizing Deep Ensemble...")
        
        # Define different MLP architectures
        architectures = [
            {'hidden_layer_sizes': (100, 50), 'name': 'Small_MLP'},
            {'hidden_layer_sizes': (200, 100, 50), 'name': 'Medium_MLP'},
            {'hidden_layer_sizes': (300, 200, 100), 'name': 'Large_MLP'},
            {'hidden_layer_sizes': (500, 300, 200, 100), 'name': 'Deep_MLP'},
            {'hidden_layer_sizes': (1000, 500, 250, 100), 'name': 'Very_Deep_MLP'}
        ]
        
        trained_models = {}
        predictions = {}
        probabilities = {}
        
        # Train different architectures
        for arch in architectures:
            self.logger.info(f"Training {arch['name']}...")
            
            mlp = MLPClassifier(
                hidden_layer_sizes=arch['hidden_layer_sizes'],
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42
            )
            
            mlp.fit(X_train, y_train)
            trained_models[arch['name']] = mlp
            
            # Get predictions and probabilities
            pred = mlp.predict(X_test)
            prob = mlp.predict_proba(X_test)[:, 1]
            
            predictions[arch['name']] = pred
            probabilities[arch['name']] = prob
        
        # Optimize ensemble weights
        weight_combinations = [
            [0.2, 0.2, 0.2, 0.2, 0.2],  # Equal weights
            [0.3, 0.25, 0.2, 0.15, 0.1],  # Favor smaller models
            [0.1, 0.15, 0.2, 0.25, 0.3],  # Favor larger models
            [0.4, 0.3, 0.2, 0.08, 0.02],  # Heavy small models
            [0.02, 0.08, 0.2, 0.3, 0.4],  # Heavy large models
        ]
        
        best_weights = None
        best_score = 0
        
        for weights in weight_combinations:
            # Weighted average of probabilities
            weighted_probs = np.zeros(len(y_test))
            for i, (name, _) in enumerate(trained_models.items()):
                weighted_probs += weights[i] * probabilities[name]
            
            # Apply threshold optimization
            prec, rec, thresholds = precision_recall_curve(y_test, weighted_probs)
            f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)
            best_threshold = thresholds[np.argmax(f1_scores)]
            
            y_pred_optimized = (weighted_probs >= best_threshold).astype(int)
            f1_optimized = f1_score(y_test, y_pred_optimized)
            
            if f1_optimized > best_score:
                best_score = f1_optimized
                best_weights = weights
        
        # Create deep ensemble class
        class DeepEnsemble:
            def __init__(self, models, weights):
                self.models = models
                self.weights = weights
            
            def predict_proba(self, X):
                probs = np.zeros((X.shape[0], 2))
                for i, (name, model) in enumerate(self.models.items()):
                    model_probs = model.predict_proba(X)
                    probs += self.weights[i] * model_probs
                return probs
            
            def predict(self, X):
                probs = self.predict_proba(X)[:, 1]
                return (probs >= 0.5).astype(int)
        
        deep_ensemble = DeepEnsemble(trained_models, best_weights)
        
        # Final evaluation
        y_pred = deep_ensemble.predict(X_test)
        y_probs = deep_ensemble.predict_proba(X_test)[:, 1]
        
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_probs)
        
        # Threshold optimization
        prec, rec, thresholds = precision_recall_curve(y_test, y_probs)
        f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)
        best_threshold = thresholds[np.argmax(f1_scores)]
        
        y_pred_optimized = (y_probs >= best_threshold).astype(int)
        f1_optimized = f1_score(y_test, y_pred_optimized)
        
        results = {
            'model': deep_ensemble,
            'weights': best_weights,
            'test_f1': f1,
            'test_auc': auc,
            'optimized_f1': f1_optimized,
            'best_threshold': best_threshold
        }
        
        self.logger.info(f"Deep Ensemble - F1: {f1:.4f}, Optimized F1: {f1_optimized:.4f}")
        
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
        
        # Add model-specific parameters
        if 'best_params' in results:
            results_summary['best_params'] = results['best_params']
        if 'weights' in results:
            results_summary['weights'] = results['weights']
        
        results_path = os.path.join(
            self.config['output']['results_dir'],
            f"{model_name}_results_{timestamp}.json"
        )
        
        import json
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        self.logger.info(f"Results saved: {results_path}")
        
        return results_summary
    
    def run_deep_learning_optimization(self, k=100):
        """Run complete deep learning optimization"""
        self.logger.info("Starting deep learning optimization...")
        
        # Feature selection
        X_train_selected, X_test_selected, selected_features = self.feature_selection(k)
        
        all_results = {}
        
        # Single MLP optimization
        mlp_results = self.optimize_mlp_architectures(
            X_train_selected, X_test_selected, self.y_train, self.y_test
        )
        all_results['mlp_optimized'] = self.save_results(mlp_results, 'mlp_optimized')
        
        # Deep ensemble optimization
        ensemble_results = self.optimize_deep_ensemble(
            X_train_selected, X_test_selected, self.y_train, self.y_test
        )
        all_results['deep_ensemble'] = self.save_results(ensemble_results, 'deep_ensemble')
        
        # Summary
        self.logger.info("=== DEEP LEARNING OPTIMIZATION SUMMARY ===")
        for model_name, results in all_results.items():
            self.logger.info(f"{model_name}: F1={results['test_f1']:.4f}, "
                           f"Optimized F1={results['optimized_f1']:.4f}")
        
        return all_results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='HPC Deep Learning Optimization')
    parser.add_argument('--config', default='hpc_config.yaml', help='Configuration file')
    parser.add_argument('--k', type=int, default=100, help='Number of features to select')
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = HPCDeepLearningOptimizer(args.config)
    
    # Run optimization
    results = optimizer.run_deep_learning_optimization(k=args.k)
    
    print("Deep learning optimization completed successfully!")

if __name__ == "__main__":
    main() 