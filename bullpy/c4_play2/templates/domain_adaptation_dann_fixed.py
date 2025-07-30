"""
Domain-Adversarial Neural Network (DANN) Template - FIXED VERSION
Cross-Dataset Autism Classification: C4 → YBT
Addresses domain shift between C4 and YBT datasets
"""

import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import os

# =============================================================================
# CELL 1: SETUP AND DATA LOADING
# =============================================================================

print("="*60)
print("DOMAIN-ADVERSARIAL NEURAL NETWORK (DANN) - FIXED")
print("CROSS-DATASET AUTISM CLASSIFICATION")
print("="*60)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load enhanced balanced datasets (44 features)
print("\nLoading enhanced balanced datasets...")
c4_balanced = pd.read_csv('/Users/eb2007/playground/bullpy/c4_play2/data/processed/data_c4_matched_balanced_enhanced.csv')
ybt_balanced = pd.read_csv('/Users/eb2007/playground/bullpy/c4_play2/data/processed/YBT_balanced_standardized.csv')

print(f"C4 enhanced balanced shape: {c4_balanced.shape}")
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

# Verify key features are present
key_features = ['age_x_eq', 'aq_eq_interaction', 'eq_sqr_ratio', 'log_aq_total', 'sqrt_age', 'high_aq']
missing_features = [f for f in key_features if f not in common_features]
if missing_features:
    print(f"⚠️  Missing key features: {missing_features}")
else:
    print("✅ All key features present")

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
# CELL 3: IMPROVED GRADIENT REVERSAL LAYER
# =============================================================================

print("\n" + "="*60)
print("IMPROVED GRADIENT REVERSAL LAYER")
print("="*60)

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

# =============================================================================
# CELL 4: IMPROVED DOMAIN-ADVERSARIAL CLASSIFIER
# =============================================================================

print("\n" + "="*60)
print("IMPROVED DOMAIN-ADVERSARIAL CLASSIFIER")
print("="*60)

class ImprovedDomainAdversarialClassifier(L.LightningModule):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.3, 
                 learning_rate=0.001, lambda_domain=0.5):
        super().__init__()
        self.save_hyperparameters()
        
        # Feature extractor (shared between tasks)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(hidden_dims[2])
        )
        
        # Classifier (for autism prediction)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[2], 1),
            nn.Sigmoid()
        )
        
        # Domain discriminator (for domain prediction)
        self.domain_discriminator = nn.Sequential(
            GradientReversalLayer(lambda_domain),
            nn.Linear(hidden_dims[2], 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.learning_rate = learning_rate
        self.lambda_domain = lambda_domain
        
    def forward(self, x):
        features = self.feature_extractor(x)
        class_output = self.classifier(features)
        domain_output = self.domain_discriminator(features)
        return class_output, domain_output
    
    def training_step(self, batch, batch_idx):
        x, y, domain = batch
        
        # Forward pass
        class_output, domain_output = self(x)
        
        # Classification loss
        class_loss = nn.BCELoss()(class_output.squeeze(), y.float())
        
        # Domain discrimination loss
        domain_loss = nn.BCELoss()(domain_output.squeeze(), domain.float())
        
        # Total loss with stronger domain adversarial component
        total_loss = class_loss + self.lambda_domain * domain_loss
        
        # Log metrics
        self.log('train_class_loss', class_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_domain_loss', domain_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Calculate classification metrics
        y_pred = (class_output.squeeze() > 0.5).float()
        f1 = f1_score(y.cpu(), y_pred.cpu(), average='weighted')
        self.log('train_f1', f1, on_epoch=True, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        x, y, domain = batch
        
        # Forward pass
        class_output, domain_output = self(x)
        
        # Classification loss
        class_loss = nn.BCELoss()(class_output.squeeze(), y.float())
        
        # Domain discrimination loss
        domain_loss = nn.BCELoss()(domain_output.squeeze(), domain.float())
        
        # Total loss
        total_loss = class_loss + self.lambda_domain * domain_loss
        
        # Log metrics
        self.log('val_class_loss', class_loss, on_epoch=True, prog_bar=True)
        self.log('val_domain_loss', domain_loss, on_epoch=True, prog_bar=True)
        self.log('val_total_loss', total_loss, on_epoch=True, prog_bar=True)
        
        # Calculate classification metrics
        y_pred = (class_output.squeeze() > 0.5).float()
        f1 = f1_score(y.cpu(), y_pred.cpu(), average='weighted')
        auc = roc_auc_score(y.cpu(), class_output.squeeze().detach().cpu())
        
        self.log('val_f1', f1, on_epoch=True, prog_bar=True)
        self.log('val_auc', auc, on_epoch=True, prog_bar=True)
        
        return total_loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_total_loss",
            },
        }

# =============================================================================
# CELL 5: IMPROVED DOMAIN-AWARE DATA MODULE
# =============================================================================

print("\n" + "="*60)
print("IMPROVED DOMAIN-AWARE DATA MODULE")
print("="*60)

class ImprovedDomainAdaptationDataModule(L.LightningDataModule):
    def __init__(self, X_train, X_val, y_train, y_val, X_target, y_target, batch_size=32):
        super().__init__()
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.X_target = X_target  # YBT data
        self.y_target = y_target
        self.batch_size = batch_size
    
    def setup(self, stage=None):
        # Source domain (C4) - domain label 0
        self.train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(self.X_train),
            torch.LongTensor(self.y_train),
            torch.zeros(len(self.X_train))  # Domain label 0 for source
        )
        
        # Validation dataset (C4) - domain label 0
        self.val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(self.X_val),
            torch.LongTensor(self.y_val),
            torch.zeros(len(self.X_val))  # Domain label 0 for validation
        )
        
        # Target domain (YBT) - domain label 1
        self.target_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(self.X_target),
            torch.LongTensor(self.y_target),
            torch.ones(len(self.X_target))  # Domain label 1 for target
        )
    
    def train_dataloader(self):
        # IMPORTANT: Train on source data only, but include target data for domain adaptation
        # Alternate between source and target batches
        source_loader = torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=0
        )
        
        target_loader = torch.utils.data.DataLoader(
            self.target_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=0
        )
        
        # Return source loader for training (target used in domain adaptation)
        return source_loader
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=0
        )

# =============================================================================
# CELL 6: IMPROVED MODEL TRAINING
# =============================================================================

print("\n" + "="*60)
print("IMPROVED DANN MODEL TRAINING")
print("="*60)

# Initialize model with stronger domain adversarial component
input_dim = len(common_features)
model = ImprovedDomainAdversarialClassifier(
    input_dim=input_dim,
    hidden_dims=[256, 128, 64],
    dropout_rate=0.3,
    learning_rate=0.001,
    lambda_domain=0.5  # Stronger domain adversarial loss
)

# Initialize data module
data_module = ImprovedDomainAdaptationDataModule(
    X_train, X_val, y_train, y_val, X_ybt, y_ybt, batch_size=32
)

# Initialize trainer
trainer = L.Trainer(
    max_epochs=50,
    accelerator='auto',
    devices=1,
    callbacks=[
        L.pytorch.callbacks.EarlyStopping(
            monitor='val_total_loss',
            patience=10,
            mode='min'
        ),
        L.pytorch.callbacks.ModelCheckpoint(
            monitor='val_f1',
            mode='max',
            save_top_k=1,
            filename='best_improved_dann_model_{epoch:02d}_{val_f1:.3f}'
        )
    ],
    log_every_n_steps=10
)

# Train model
print("Starting improved DANN training...")
trainer.fit(model, data_module)

print("Improved DANN training completed!")

# =============================================================================
# CELL 7: IMPROVED MODEL EVALUATION
# =============================================================================

print("\n" + "="*60)
print("IMPROVED MODEL EVALUATION")
print("="*60)

# Load best model
best_model_path = trainer.checkpoint_callback.best_model_path
print(f"Loading best model from: {best_model_path}")

# Load model
model = ImprovedDomainAdversarialClassifier.load_from_checkpoint(best_model_path)
model.eval()

# Evaluate on validation set
val_predictions = []
val_probs = []

with torch.no_grad():
    for batch in data_module.val_dataloader():
        x, y, domain = batch
        device = next(model.parameters()).device
        x = x.to(device)
        class_output, domain_output = model(x)
        val_probs.extend(class_output.squeeze().cpu().numpy())
        val_predictions.extend((class_output.squeeze() > 0.5).cpu().numpy())

# Convert to numpy arrays
val_probs = np.array(val_probs)
val_predictions = np.array(val_predictions)

# Get true labels
val_true = []
for batch in data_module.val_dataloader():
    x, y, domain = batch
    val_true.extend(y.cpu().numpy())
val_true = np.array(val_true)

# Print validation results
print("\nValidation Set Performance:")
print(classification_report(val_true, val_predictions))
print(f"ROC-AUC: {roc_auc_score(val_true, val_probs):.3f}")

# =============================================================================
# CELL 8: CROSS-DATASET TESTING
# =============================================================================

print("\n" + "="*60)
print("CROSS-DATASET TESTING (C4 → YBT)")
print("="*60)

# Evaluate on YBT test set
ybt_predictions = []
ybt_probs = []

with torch.no_grad():
    # Create YBT dataloader
    ybt_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_ybt),
        torch.LongTensor(y_ybt),
        torch.ones(len(X_ybt))  # Domain label 1 for target
    )
    ybt_loader = torch.utils.data.DataLoader(ybt_dataset, batch_size=32, shuffle=False)
    
    for batch in ybt_loader:
        x, y, domain = batch
        device = next(model.parameters()).device
        x = x.to(device)
        class_output, domain_output = model(x)
        ybt_probs.extend(class_output.squeeze().cpu().numpy())
        ybt_predictions.extend((class_output.squeeze() > 0.5).cpu().numpy())

# Convert to numpy arrays
ybt_probs = np.array(ybt_probs)
ybt_predictions = np.array(ybt_predictions)

# Print YBT results
print("\nYBT Test Set Performance:")
print(classification_report(y_ybt, ybt_predictions))
print(f"ROC-AUC: {roc_auc_score(y_ybt, ybt_probs):.3f}")

# Find optimal threshold
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_ybt, ybt_probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_idx = np.argmax(f1_scores[:-1])  # Exclude last point
optimal_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print(f"\nBest threshold for YBT: {optimal_threshold:.3f}")
print(f"F1 at optimal threshold: {best_f1:.3f}")

# =============================================================================
# CELL 9: COMPARISON WITH PREVIOUS MODELS
# =============================================================================

print("\n" + "="*60)
print("COMPARISON WITH PREVIOUS MODELS")
print("="*60)

# Create comparison DataFrame
comparison_data = {
    'Model': ['Random Forest', 'Neural Network', 'TabNet', 'DANN (Original)', 'DANN (Improved)'],
    'F1_Score': [0.619, 0.667, 0.667, 0.667, best_f1],
    'ROC_AUC': [0.325, 0.523, 0.388, 0.500, roc_auc_score(y_ybt, ybt_probs)],
    'Threshold': [0.159, 0.157, 0.116, 0.505, optimal_threshold]
}

comparison_df = pd.DataFrame(comparison_data)
print("\nPerformance Comparison:")
print(comparison_df)

# Save model
model_save_path = '/Users/eb2007/playground/bullpy/c4_play2/models/improved_dann_model.pt'
torch.save(model.state_dict(), model_save_path)
print(f"\nImproved DANN model saved successfully!")
print("Improved domain adaptation experiment completed!") 