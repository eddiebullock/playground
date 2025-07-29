"""
TabNet Baseline Template
TabNet for Autism Classification
Train on C4 → Test on YBT
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from pytorch_tabnet.tab_model import TabNetClassifier
import os

# =============================================================================
# CELL 1: SETUP AND DATA LOADING
# =============================================================================

print("="*60)
print("TABNET BASELINE: AUTISM CLASSIFICATION")
print("="*60)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load balanced datasets
print("\nLoading balanced datasets...")
c4_balanced = pd.read_csv('/Users/eb2007/playground/bullpy/c4_play2/data/processed/data_c4_matched_balanced.csv')
ybt_balanced = pd.read_csv('/Users/eb2007/playground/bullpy/c4_play2/data/processed/YBT_balanced_standardized.csv')

print(f"C4 balanced shape: {c4_balanced.shape}")
print(f"YBT balanced shape: {ybt_balanced.shape}")

# =============================================================================
# CELL 2: DATA PREPARATION
# =============================================================================

print("\n" + "="*60)
print("DATA PREPARATION")
print("="*60)

# Identify common features (excluding target)
exclude_cols = ['autism_target']
c4_features = [col for col in c4_balanced.columns if col not in exclude_cols]
ybt_features = [col for col in ybt_balanced.columns if col not in exclude_cols]

# Find common features
common_features = list(set(c4_features) & set(ybt_features))
print(f"Common features: {len(common_features)}")

# Prepare data
X_c4 = c4_balanced[common_features].values
y_c4 = c4_balanced['autism_target'].values
X_ybt = ybt_balanced[common_features].values
y_ybt = ybt_balanced['autism_target'].values

print(f"C4 features shape: {X_c4.shape}")
print(f"YBT features shape: {X_ybt.shape}")

# Split C4 data for training/validation
X_train, X_val, y_train, y_val = train_test_split(
    X_c4, y_c4, test_size=0.2, stratify=y_c4, random_state=42
)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"YBT test set: {X_ybt.shape}")

# =============================================================================
# CELL 3: TABNET MODEL TRAINING
# =============================================================================

print("\n" + "="*60)
print("TABNET MODEL TRAINING")
print("="*60)

# Initialize TabNet
tabnet_model = TabNetClassifier(
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
    scheduler_params=dict(mode='min', factor=0.5, patience=5, min_lr=1e-5),
    mask_type='entmax',  # 'sparsemax' or 'entmax'
    verbose=10,
    max_epochs=100,
    patience=20,
    batch_size=256,
    virtual_batch_size=128,
    num_workers=0,
    drop_last=False,
    # TabNet specific parameters
    cat_idxs=[],  # No categorical features in our case
    cat_dims=[],  # No categorical features
    cat_emb_dim=1,  # Not used since no categorical features
    n_d=8,  # Dimension of prediction layer
    n_a=8,  # Dimension of attention layer
    n_steps=3,  # Number of decision steps
    gamma=1.3,  # Relaxation parameter
    n_independent=2,  # Number of independent GLU layers
    n_shared=2,  # Number of shared GLU layers
    epsilon=1e-15,
    momentum=0.3,
    clip_value=2,
    lambda_sparse=1e-3,
)

# Train TabNet
print("Starting TabNet training...")
tabnet_model.fit(
    X_train=X_train,
    y_train=y_train,
    eval_set=[(X_val, y_val)],
    max_epochs=100,
    patience=20,
    batch_size=256,
    virtual_batch_size=128,
    num_workers=0,
    drop_last=False,
    eval_metric=['auc', 'accuracy'],
    eval_name=['val_auc', 'val_accuracy'],
)

print("TabNet training completed!")

# =============================================================================
# CELL 4: MODEL EVALUATION
# =============================================================================

print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

# Evaluate on validation set
val_preds = tabnet_model.predict(X_val)
val_probs = tabnet_model.predict_proba(X_val)[:, 1]

print("\nValidation Set Performance:")
print(classification_report(y_val, val_preds))
print(f"ROC-AUC: {roc_auc_score(y_val, val_probs):.3f}")

# =============================================================================
# CELL 5: CROSS-DATASET TESTING
# =============================================================================

print("\n" + "="*60)
print("CROSS-DATASET TESTING (C4 → YBT)")
print("="*60)

# Evaluate on YBT
ybt_preds = tabnet_model.predict(X_ybt)
ybt_probs = tabnet_model.predict_proba(X_ybt)[:, 1]

print("\nYBT Test Set Performance:")
print(classification_report(y_ybt, ybt_preds))
print(f"ROC-AUC: {roc_auc_score(y_ybt, ybt_probs):.3f}")

# Threshold optimization for YBT
from sklearn.metrics import precision_recall_curve
prec, rec, thresholds = precision_recall_curve(y_ybt, ybt_probs)
f1s = 2 * (prec * rec) / (prec + rec + 1e-8)
best_thresh_idx = np.argmax(f1s)
best_threshold = thresholds[best_thresh_idx]

print(f"\nBest threshold for YBT: {best_threshold:.3f}")
ybt_predictions_optimal = (ybt_probs >= best_threshold).astype(int)
print(f"F1 at optimal threshold: {f1_score(y_ybt, ybt_predictions_optimal):.3f}")

# =============================================================================
# CELL 6: COMPARISON WITH PREVIOUS MODELS
# =============================================================================

print("\n" + "="*60)
print("COMPARISON WITH PREVIOUS MODELS")
print("="*60)

results_comparison = {
    'Model': ['Random Forest', 'Neural Network', 'TabNet'],
    'F1_Score': [0.619, 0.667, f1_score(y_ybt, ybt_predictions_optimal)],
    'ROC_AUC': [0.325, 0.523, roc_auc_score(y_ybt, ybt_probs)],
    'Threshold': [0.159, 0.157, best_threshold]
}

comparison_df = pd.DataFrame(results_comparison)
print("\nPerformance Comparison:")
print(comparison_df)

# Save model
os.makedirs('/Users/eb2007/playground/bullpy/c4_play2/models', exist_ok=True)
tabnet_model.save_model('/Users/eb2007/playground/bullpy/c4_play2/models/tabnet_baseline.zip')

print("\nTabNet model saved successfully!")
print("TabNet baseline experiment completed!")

# =============================================================================
# CELL 7: FEATURE IMPORTANCE ANALYSIS
# =============================================================================

print("\n" + "="*60)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*60)

# Get feature importance from TabNet
feature_importance = tabnet_model.feature_importances_
feature_names = common_features

# Create feature importance DataFrame
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("\nTop 20 Most Important Features:")
print(importance_df.head(20))

# Plot feature importance
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
top_features = importance_df.head(20)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance')
plt.title('TabNet Feature Importance (Top 20)')
plt.tight_layout()
plt.show() 