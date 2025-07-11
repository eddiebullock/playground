#!/usr/bin/env python3
"""
Improved Deep Learning Benchmark Script for Autism Diagnosis Prediction

- Handles class imbalance with weighted loss, SMOTE, and focal loss
- Includes advanced architectures: MLP, 1D CNN, Autoencoder, TabNet-style models
- Better hyperparameter tuning and regularization
- Comprehensive evaluation metrics
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# Helper: Set random seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Helper: Drop all columns that could cause data leakage
def drop_leakage_columns(df):
    cols_to_drop = [col for col in df.columns if (
        col.startswith('diagnosis_') or
        col.startswith('autism_diagnosis') or
        col.startswith('autism_subtype') or
        col == 'autism_any' or
        col == 'userid')]
    return df.drop(columns=cols_to_drop, errors='ignore')

# Focal Loss for class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

# Improved MLP with batch normalization and residual connections
class ImprovedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*layers)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.feature_layers(x)
        return self.output_layer(features)

# TabNet-style model (simplified)
class TabNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_decision_steps=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_decision_steps = num_decision_steps
        
        # Feature transformers
        self.feature_transformers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ) for _ in range(num_decision_steps)
        ])
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim, input_dim)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        total_entropy = torch.zeros(batch_size, self.input_dim).to(x.device)
        
        # Decision steps
        for step in range(self.num_decision_steps):
            # Feature transformation
            features = self.feature_transformers[step](x)
            
            # Attention mechanism
            attention_weights = torch.softmax(self.attention(features), dim=1)
            total_entropy += attention_weights
            
            # Apply attention
            x = x * attention_weights
        
        # Final prediction
        final_features = self.feature_transformers[-1](x)
        return self.output_layer(final_features)

# Autoencoder for feature learning
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, mode='classify'):
        encoded = self.encoder(x)
        if mode == 'reconstruct':
            return self.decoder(encoded)
        else:
            return self.classifier(encoded)

# Improved 1D CNN with attention
class AttentionCNN(nn.Module):
    def __init__(self, input_dim, num_filters=32):
        super().__init__()
        self.conv1 = nn.Conv1d(1, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(num_filters, num_filters*2, kernel_size=3, padding=1)
        self.attention = nn.MultiheadAttention(num_filters*2, num_heads=4, batch_first=True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(num_filters*2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, features)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        
        # Attention mechanism
        x = x.transpose(1, 2)  # (batch, features, channels)
        x, _ = self.attention(x, x, x)
        x = x.transpose(1, 2)  # (batch, channels, features)
        
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)

# Training loop with class imbalance handling
def train_model_improved(model, train_loader, val_loader, device, epochs=50, lr=1e-3, 
                        use_focal_loss=True, class_weights=None):
    if use_focal_loss:
        criterion = FocalLoss(alpha=1, gamma=2)
    elif class_weights is not None:
        criterion = nn.BCELoss(weight=class_weights.to(device))
    else:
        criterion = nn.BCELoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)
    
    best_val_auc = 0
    best_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb).squeeze()
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb).squeeze()
                val_preds.append(out.cpu().numpy())
                val_targets.append(yb.cpu().numpy())
        
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_auc = roc_auc_score(val_targets, val_preds)
        
        scheduler.step(val_auc)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= 10:  # Early stopping
            break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# Evaluation with more metrics
def evaluate_model_improved(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb).squeeze()
            preds.append(out.cpu().numpy())
            targets.append(yb.cpu().numpy())
    
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    pred_labels = (preds > 0.5).astype(int)
    
    # Calculate class weights for balanced accuracy
    pos_weight = (targets == 0).sum() / (targets == 1).sum()
    
    return {
        'accuracy': accuracy_score(targets, pred_labels),
        'roc_auc': roc_auc_score(targets, preds),
        'f1': f1_score(targets, pred_labels),
        'f1_weighted': f1_score(targets, pred_labels, average='weighted'),
        'precision': precision_score(targets, pred_labels),
        'recall': recall_score(targets, pred_labels),
        'report': classification_report(targets, pred_labels, digits=3),
        'pos_weight': pos_weight
    }

def main():
    parser = argparse.ArgumentParser(description='Improved Deep Learning Benchmark for Autism Diagnosis Prediction')
    parser.add_argument('--data_path', type=str, default='data/processed/features_full.csv', help='Path to processed data')
    parser.add_argument('--output_dir', type=str, default='experiments/logs/deep_learning_improved', help='Directory to save results')
    parser.add_argument('--target_col', type=str, default='autism_any', help='Target column name')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set proportion')
    parser.add_argument('--val_size', type=float, default=0.2, help='Validation set proportion (of train)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_smote', action='store_true', help='Use SMOTE for class balancing')
    parser.add_argument('--use_focal_loss', action='store_true', help='Use focal loss')
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(args.data_path)
    if args.target_col not in df.columns:
        raise ValueError(f"Target column {args.target_col} not found in data.")
    
    y = df[args.target_col].values.astype(np.float32)
    X = drop_leakage_columns(df)
    X = pd.get_dummies(X, drop_first=True)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    # Split data
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_scaled, y, test_size=args.test_size, stratify=y, random_state=args.seed)
    val_relative = args.val_size / (1 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_relative, stratify=y_trainval, random_state=args.seed)

    # Handle class imbalance
    if args.use_smote:
        smote = SMOTE(random_state=args.seed)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    else:
        X_train_balanced, y_train_balanced = X_train, y_train

    # Calculate class weights
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    class_weights = torch.tensor([1.0, pos_weight])

    # Convert to PyTorch tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train_t = torch.tensor(X_train_balanced, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_balanced, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=args.batch_size)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=args.batch_size)

    results = {}

    # Improved MLP
    print("\nTraining Improved MLP...")
    mlp = ImprovedMLP(X_train.shape[1]).to(device)
    mlp = train_model_improved(mlp, train_loader, val_loader, device, epochs=args.epochs, 
                              use_focal_loss=args.use_focal_loss, class_weights=class_weights)
    results['Improved_MLP'] = evaluate_model_improved(mlp, test_loader, device)

    # TabNet-style model
    print("\nTraining TabNet-style model...")
    tabnet = TabNetModel(X_train.shape[1]).to(device)
    tabnet = train_model_improved(tabnet, train_loader, val_loader, device, epochs=args.epochs,
                                 use_focal_loss=args.use_focal_loss, class_weights=class_weights)
    results['TabNet'] = evaluate_model_improved(tabnet, test_loader, device)

    # Attention CNN
    print("\nTraining Attention CNN...")
    cnn = AttentionCNN(X_train.shape[1]).to(device)
    cnn = train_model_improved(cnn, train_loader, val_loader, device, epochs=args.epochs,
                              use_focal_loss=args.use_focal_loss, class_weights=class_weights)
    results['Attention_CNN'] = evaluate_model_improved(cnn, test_loader, device)

    # Autoencoder
    print("\nTraining Autoencoder...")
    autoencoder = Autoencoder(X_train.shape[1]).to(device)
    autoencoder = train_model_improved(autoencoder, train_loader, val_loader, device, epochs=args.epochs,
                                     use_focal_loss=args.use_focal_loss, class_weights=class_weights)
    results['Autoencoder'] = evaluate_model_improved(autoencoder, test_loader, device)

    # Save results
    results_path = os.path.join(args.output_dir, 'deep_learning_improved_results.csv')
    pd.DataFrame(results).T.to_csv(results_path)
    
    # Save detailed results
    detailed_results = {}
    for model_name, metrics in results.items():
        detailed_results[model_name] = {
            'accuracy': metrics['accuracy'],
            'roc_auc': metrics['roc_auc'],
            'f1': metrics['f1'],
            'f1_weighted': metrics['f1_weighted'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'pos_weight': metrics['pos_weight']
        }
    
    detailed_path = os.path.join(args.output_dir, 'detailed_results.csv')
    pd.DataFrame(detailed_results).T.to_csv(detailed_path)
    
    print(f"\nResults saved to {results_path}")
    print(f"Detailed results saved to {detailed_path}")
    
    # Print summary
    for model, res in results.items():
        print(f"\n=== {model} ===")
        print(f"Accuracy: {res['accuracy']:.3f}")
        print(f"ROC-AUC: {res['roc_auc']:.3f}")
        print(f"F1: {res['f1']:.3f}")
        print(f"Precision: {res['precision']:.3f}")
        print(f"Recall: {res['recall']:.3f}")
        print(res['report'])

if __name__ == '__main__':
    main() 