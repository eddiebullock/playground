#!/usr/bin/env python3
"""
Deep Learning Benchmark Script for Autism Diagnosis Prediction

- Loads processed data (e.g., data/processed/features_full.csv)
- Drops all columns that could cause data leakage (diagnosis_, autism_diagnosis_, autism_any, autism_subtype, userid)
- Uses 'autism_any' as the target
- Splits data into train/val/test (stratified)
- Benchmarks several deep learning models (MLP, 1D CNN; TabNet/TabTransformer if available)
- Outputs metrics (accuracy, ROC-AUC, F1, etc.) and saves results
- All paths and parameters are configurable via argparse
- No interactive elements; suitable for local and HPC runs
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
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

# Simple MLP model
def build_mlp(input_dim, hidden_dim=128, dropout=0.3):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim//2),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim//2, 1),
        nn.Sigmoid()
    )

# Optionally: 1D CNN model for tabular data
def build_cnn(input_dim, dropout=0.3):
    class Tabular1DCNN(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.fc1 = nn.Linear(16 * input_dim, 64)
            self.fc2 = nn.Linear(64, 1)
        def forward(self, x):
            x = x.unsqueeze(1)  # (batch, 1, features)
            x = self.conv1(x)
            x = self.relu(x)
            x = x.view(x.size(0), -1)
            x = self.dropout(x)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return torch.sigmoid(x)
    return Tabular1DCNN(input_dim)

# Training loop
def train_model(model, train_loader, val_loader, device, epochs=20, lr=1e-3):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_auc = 0
    best_state = None
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb).squeeze()
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
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
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = model.state_dict()
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# Evaluation
def evaluate_model(model, loader, device):
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
    return {
        'accuracy': accuracy_score(targets, pred_labels),
        'roc_auc': roc_auc_score(targets, preds),
        'f1': f1_score(targets, pred_labels),
        'report': classification_report(targets, pred_labels, digits=3)
    }

def main():
    parser = argparse.ArgumentParser(description='Deep Learning Benchmark for Autism Diagnosis Prediction')
    parser.add_argument('--data_path', type=str, default='data/processed/features_full.csv', help='Path to processed data')
    parser.add_argument('--output_dir', type=str, default='experiments/logs/deep_learning', help='Directory to save results')
    parser.add_argument('--target_col', type=str, default='autism_any', help='Target column name')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set proportion')
    parser.add_argument('--val_size', type=float, default=0.2, help='Validation set proportion (of train)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(args.data_path)
    if args.target_col not in df.columns:
        raise ValueError(f"Target column {args.target_col} not found in data.")
    y = df[args.target_col].values.astype(np.float32)
    X = drop_leakage_columns(df)
    # Encode categorical columns
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

    # Convert to PyTorch tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=args.batch_size)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=args.batch_size)

    results = {}

    # MLP
    print("\nTraining MLP...")
    mlp = build_mlp(X_train.shape[1]).to(device)
    mlp = train_model(mlp, train_loader, val_loader, device, epochs=args.epochs)
    results['MLP'] = evaluate_model(mlp, test_loader, device)

    # 1D CNN
    print("\nTraining 1D CNN...")
    cnn = build_cnn(X_train.shape[1]).to(device)
    cnn = train_model(cnn, train_loader, val_loader, device, epochs=args.epochs)
    results['1D_CNN'] = evaluate_model(cnn, test_loader, device)

    # Optionally: Add TabNet/TabTransformer if available
    # (Not included here for simplicity and dependency reasons)

    # Save results
    results_path = os.path.join(args.output_dir, 'deep_learning_results.csv')
    pd.DataFrame(results).T.to_csv(results_path)
    print(f"\nResults saved to {results_path}")
    for model, res in results.items():
        print(f"\n=== {model} ===")
        print(f"Accuracy: {res['accuracy']:.3f}")
        print(f"ROC-AUC: {res['roc_auc']:.3f}")
        print(f"F1: {res['f1']:.3f}")
        print(res['report'])

if __name__ == '__main__':
    main() 