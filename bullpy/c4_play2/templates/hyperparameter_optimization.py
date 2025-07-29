"""
Hyperparameter Optimization Template
Using Optuna for Neural Network Hyperparameter Tuning
"""

import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import optuna
import wandb
import joblib
import os

# =============================================================================
# CELL 1: SETUP AND DATA LOADING
# =============================================================================

print("="*60)
print("HYPERPARAMETER OPTIMIZATION: AUTISM CLASSIFICATION")
print("="*60)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load balanced datasets
print("\nLoading balanced datasets...")
c4_balanced = pd.read_csv('/Users/eb2007/playground/bullpy/c4_play2/data/processed/c4_balanced_standardized.csv')
ybt_balanced = pd.read_csv('/Users/eb2007/playground/bullpy/c4_play2/data/processed/ybt_balanced_standardized.csv')

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
# CELL 3: OPTIMIZABLE NEURAL NETWORK
# =============================================================================

print("\n" + "="*60)
print("OPTIMIZABLE NEURAL NETWORK")
print("="*60)

class OptimizableAutismClassifier(L.LightningModule):
    def __init__(self, input_dim, hidden_dims, dropout_rate, learning_rate, 
                 batch_norm=True, activation='relu'):
        super().__init__()
        self.save_hyperparameters()
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            elif activation == 'elu':
                layers.append(nn.ELU())
            
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        self.learning_rate = learning_rate
        
    def forward(self, x):
        return self.network(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = nn.BCELoss()(y_hat, y.float())
        
        # Log training loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Calculate metrics
        y_pred = (y_hat > 0.5).float()
        f1 = f1_score(y.cpu(), y_pred.cpu(), average='weighted')
        self.log('train_f1', f1, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = nn.BCELoss()(y_hat, y.float())
        
        # Log validation loss
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        
        # Calculate metrics
        y_pred = (y_hat > 0.5).float()
        f1 = f1_score(y.cpu(), y_pred.cpu(), average='weighted')
        auc = roc_auc_score(y.cpu(), y_hat.detach().cpu())
        
        self.log('val_f1', f1, on_epoch=True, prog_bar=True)
        self.log('val_auc', auc, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

# =============================================================================
# CELL 4: DATA MODULES
# =============================================================================

print("\n" + "="*60)
print("DATA MODULES")
print("="*60)

class AutismDataModule(L.LightningDataModule):
    def __init__(self, X_train, X_val, y_train, y_val, batch_size=32):
        super().__init__()
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.batch_size = batch_size
    
    def setup(self, stage=None):
        # Convert to tensors
        self.train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(self.X_train),
            torch.LongTensor(self.y_train)
        )
        self.val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(self.X_val),
            torch.LongTensor(self.y_val)
        )
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=0
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=0
        )

# =============================================================================
# CELL 5: OPTUNA OBJECTIVE FUNCTION
# =============================================================================

print("\n" + "="*60)
print("OPTUNA OBJECTIVE FUNCTION")
print("="*60)

def objective(trial):
    # Suggest hyperparameters
    n_layers = trial.suggest_int('n_layers', 2, 5)
    hidden_dims = []
    
    for i in range(n_layers):
        hidden_dim = trial.suggest_int(f'hidden_dim_{i}', 32, 512)
        hidden_dims.append(hidden_dim)
    
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    batch_norm = trial.suggest_categorical('batch_norm', [True, False])
    activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'elu'])
    
    # Initialize model
    input_dim = len(common_features)
    model = OptimizableAutismClassifier(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        batch_norm=batch_norm,
        activation=activation
    )
    
    # Initialize data module
    data_module = AutismDataModule(X_train, X_val, y_train, y_val, batch_size=batch_size)
    
    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=50,  # Shorter for optimization
        accelerator='auto',
        devices=1,
        callbacks=[
            L.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                mode='min'
            )
        ],
        enable_progress_bar=False,  # Reduce output
        enable_logging=False,  # Reduce output
        logger=False  # Disable logging for optimization
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Get best validation F1 score
    best_f1 = trainer.callback_metrics.get('val_f1', 0.0)
    
    return best_f1

# =============================================================================
# CELL 6: HYPERPARAMETER OPTIMIZATION
# =============================================================================

print("\n" + "="*60)
print("HYPERPARAMETER OPTIMIZATION")
print("="*60)

# Create study
study = optuna.create_study(
    direction='maximize',
    study_name='autism_classification_optimization'
)

# Run optimization
print("Starting hyperparameter optimization...")
study.optimize(objective, n_trials=50, timeout=3600)  # 50 trials, 1 hour timeout

print(f"Best trial: {study.best_trial}")
print(f"Best F1 score: {study.best_value:.3f}")
print(f"Best parameters: {study.best_params}")

# =============================================================================
# CELL 7: TRAIN BEST MODEL
# =============================================================================

print("\n" + "="*60)
print("TRAIN BEST MODEL")
print("="*60)

# Get best parameters
best_params = study.best_params

# Build hidden dimensions from best parameters
hidden_dims = []
for i in range(best_params['n_layers']):
    hidden_dims.append(best_params[f'hidden_dim_{i}'])

# Initialize best model
input_dim = len(common_features)
best_model = OptimizableAutismClassifier(
    input_dim=input_dim,
    hidden_dims=hidden_dims,
    dropout_rate=best_params['dropout_rate'],
    learning_rate=best_params['learning_rate'],
    batch_norm=best_params['batch_norm'],
    activation=best_params['activation']
)

# Initialize data module
data_module = AutismDataModule(X_train, X_val, y_train, y_val, batch_size=best_params['batch_size'])

# Initialize trainer with full training
trainer = L.Trainer(
    max_epochs=100,
    accelerator='auto',
    devices=1,
    callbacks=[
        L.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min'
        ),
        L.callbacks.ModelCheckpoint(
            monitor='val_f1',
            mode='max',
            save_top_k=1,
            filename='optimized_model_{epoch:02d}_{val_f1:.3f}'
        )
    ],
    logger=L.loggers.WandbLogger(project="autism-deep-learning"),
    log_every_n_steps=10
)

# Train best model
print("Training best model with optimized hyperparameters...")
trainer.fit(best_model, data_module)

print("Training completed!")

# =============================================================================
# CELL 8: EVALUATE BEST MODEL
# =============================================================================

print("\n" + "="*60)
print("EVALUATE BEST MODEL")
print("="*60)

# Load best model
best_model_path = trainer.checkpoint_callback.best_model_path
print(f"Loading best model from: {best_model_path}")

# Load model
best_model = OptimizableAutismClassifier.load_from_checkpoint(best_model_path)
best_model.eval()

# Evaluate on validation set
val_predictions = []
val_probs = []

with torch.no_grad():
    for batch in data_module.val_dataloader():
        x, y = batch
        y_hat = best_model(x).squeeze()
        val_probs.extend(y_hat.cpu().numpy())
        val_predictions.extend((y_hat > 0.5).cpu().numpy())

val_probs = np.array(val_probs)
val_predictions = np.array(val_predictions)

# Get validation targets
val_targets = []
for batch in data_module.val_dataloader():
    x, y = batch
    val_targets.extend(y.cpu().numpy())
val_targets = np.array(val_targets)

print("\nValidation Set Performance:")
print(classification_report(val_targets, val_predictions))
print(f"ROC-AUC: {roc_auc_score(val_targets, val_probs):.3f}")

# =============================================================================
# CELL 9: CROSS-DATASET TESTING
# =============================================================================

print("\n" + "="*60)
print("CROSS-DATASET TESTING (C4 → YBT)")
print("="*60)

# Prepare YBT test data
X_ybt_tensor = torch.FloatTensor(X_ybt)
ybt_dataset = torch.utils.data.TensorDataset(X_ybt_tensor, torch.LongTensor(y_ybt))
ybt_dataloader = torch.utils.data.DataLoader(ybt_dataset, batch_size=best_params['batch_size'], shuffle=False)

# Evaluate on YBT
ybt_predictions = []
ybt_probs = []

best_model.eval()
with torch.no_grad():
    for batch in ybt_dataloader():
        x, y = batch
        y_hat = best_model(x).squeeze()
        ybt_probs.extend(y_hat.cpu().numpy())
        ybt_predictions.extend((y_hat > 0.5).cpu().numpy())

ybt_probs = np.array(ybt_probs)
ybt_predictions = np.array(ybt_predictions)

print("\nYBT Test Set Performance:")
print(classification_report(y_ybt, ybt_predictions))
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
# CELL 10: SAVE RESULTS
# =============================================================================

print("\n" + "="*60)
print("SAVE RESULTS")
print("="*60)

# Save optimized model
os.makedirs('/Users/eb2007/playground/bullpy/c4_play2/models', exist_ok=True)
torch.save(best_model.state_dict(), '/Users/eb2007/playground/bullpy/c4_play2/models/optimized_neural_network.pth')

# Save optimization results
optimization_results = {
    'best_f1_score': study.best_value,
    'best_parameters': study.best_params,
    'n_trials': len(study.trials),
    'optimization_history': [trial.value for trial in study.trials if trial.value is not None]
}

joblib.dump(optimization_results, '/Users/eb2007/playground/bullpy/c4_play2/models/optimization_results.pkl')

# Save study
joblib.dump(study, '/Users/eb2007/playground/bullpy/c4_play2/models/optuna_study.pkl')

print("Optimization results saved successfully!")
print("Hyperparameter optimization completed!")

# Close wandb
wandb.finish() 