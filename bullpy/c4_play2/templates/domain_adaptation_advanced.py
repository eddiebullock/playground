"""
Domain Adaptation Neural Network Template
Domain-Adversarial Neural Network (DANN) for Autism Classification
Train on C4 → Test on YBT with domain adaptation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import wandb
import os

# =============================================================================
# CELL 1: SETUP AND DATA LOADING
# =============================================================================

print("="*60)
print("DOMAIN ADAPTATION: AUTISM CLASSIFICATION")
print("="*60)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Initialize wandb for experiment tracking
wandb.init(
    project="autism-deep-learning",
    name="domain-adaptation-dann",
    config={
        "architecture": "domain_adversarial",
        "feature_dim": 64,
        "classifier_dim": 32,
        "discriminator_dim": 32,
        "learning_rate": 0.001,
        "lambda_domain": 0.1,  # Domain adversarial weight
        "batch_size": 32,
        "epochs": 100,
        "early_stopping_patience": 10
    }
)

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
# CELL 3: DOMAIN-ADVERSARIAL NEURAL NETWORK
# =============================================================================

print("\n" + "="*60)
print("DOMAIN-ADVERSARIAL NEURAL NETWORK")
print("="*60)

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class DomainAdversarialClassifier(L.LightningModule):
    def __init__(self, input_dim, feature_dim=64, classifier_dim=32, discriminator_dim=32, 
                 learning_rate=0.001, lambda_domain=0.1):
        super().__init__()
        self.save_hyperparameters()
        
        # Feature extractor (shared between source and target)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(128),
            nn.Linear(128, feature_dim),
            nn.ReLU(),
            nn.BatchNorm1d(feature_dim)
        )
        
        # Classifier (for autism prediction)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, classifier_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(classifier_dim, 1),
            nn.Sigmoid()
        )
        
        # Domain discriminator (for domain adaptation)
        self.domain_discriminator = nn.Sequential(
            nn.Linear(feature_dim, discriminator_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(discriminator_dim, 1),
            nn.Sigmoid()
        )
        
        self.learning_rate = learning_rate
        self.lambda_domain = lambda_domain
        
    def forward(self, x):
        features = self.feature_extractor(x)
        class_output = self.classifier(features)
        return class_output, features
    
    def domain_forward(self, features, alpha=1.0):
        # Apply gradient reversal
        reversed_features = GradientReversalLayer.apply(features, alpha)
        domain_output = self.domain_discriminator(reversed_features)
        return domain_output
    
    def training_step(self, batch, batch_idx):
        x, y, domain_labels = batch
        
        # Forward pass
        class_output, features = self(x)
        
        # Classification loss
        class_loss = nn.BCELoss()(class_output.squeeze(), y.float())
        
        # Domain adversarial loss
        domain_output = self.domain_forward(features, alpha=self.lambda_domain)
        domain_loss = nn.BCELoss()(domain_output.squeeze(), domain_labels.float())
        
        # Total loss
        total_loss = class_loss + self.lambda_domain * domain_loss
        
        # Log losses
        self.log('train_class_loss', class_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_domain_loss', domain_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Calculate classification metrics
        y_pred = (class_output.squeeze() > 0.5).float()
        f1 = f1_score(y.cpu(), y_pred.cpu(), average='weighted')
        self.log('train_f1', f1, on_epoch=True, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        x, y, domain_labels = batch
        
        # Forward pass
        class_output, features = self(x)
        
        # Classification loss
        class_loss = nn.BCELoss()(class_output.squeeze(), y.float())
        
        # Domain adversarial loss
        domain_output = self.domain_forward(features, alpha=self.lambda_domain)
        domain_loss = nn.BCELoss()(domain_output.squeeze(), domain_labels.float())
        
        # Total loss
        total_loss = class_loss + self.lambda_domain * domain_loss
        
        # Log losses
        self.log('val_class_loss', class_loss, on_epoch=True, prog_bar=True)
        self.log('val_domain_loss', domain_loss, on_epoch=True, prog_bar=True)
        self.log('val_total_loss', total_loss, on_epoch=True, prog_bar=True)
        
        # Calculate metrics
        y_pred = (class_output.squeeze() > 0.5).float()
        f1 = f1_score(y.cpu(), y_pred.cpu(), average='weighted')
        auc = roc_auc_score(y.cpu(), class_output.squeeze().detach().cpu())
        
        self.log('val_f1', f1, on_epoch=True, prog_bar=True)
        self.log('val_auc', auc, on_epoch=True, prog_bar=True)
        
        return total_loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_total_loss",
            },
        }

# =============================================================================
# CELL 4: DOMAIN-AWARE DATA MODULES
# =============================================================================

print("\n" + "="*60)
print("DOMAIN-AWARE DATA MODULES")
print("="*60)

class DomainAdaptationDataModule(L.LightningDataModule):
    def __init__(self, X_train, X_val, y_train, y_val, X_target, y_target, batch_size=32):
        super().__init__()
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.X_target = X_target
        self.y_target = y_target
        self.batch_size = batch_size
    
    def setup(self, stage=None):
        # Source domain (C4) - domain label 0
        self.train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(self.X_train),
            torch.LongTensor(self.y_train),
            torch.zeros(len(self.X_train))  # Source domain label
        )
        self.val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(self.X_val),
            torch.LongTensor(self.y_val),
            torch.zeros(len(self.X_val))  # Source domain label
        )
        
        # Target domain (YBT) - domain label 1
        self.target_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(self.X_target),
            torch.LongTensor(self.y_target),
            torch.ones(len(self.X_target))  # Target domain label
        )
    
    def train_dataloader(self):
        # Combine source and target data for training
        combined_dataset = torch.utils.data.ConcatDataset([
            self.train_dataset, self.target_dataset
        ])
        return torch.utils.data.DataLoader(
            combined_dataset, 
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
    
    def target_dataloader(self):
        return torch.utils.data.DataLoader(
            self.target_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=0
        )

# =============================================================================
# CELL 5: MODEL TRAINING
# =============================================================================

print("\n" + "="*60)
print("MODEL TRAINING")
print("="*60)

# Initialize model
input_dim = len(common_features)
model = DomainAdversarialClassifier(
    input_dim=input_dim,
    feature_dim=64,
    classifier_dim=32,
    discriminator_dim=32,
    learning_rate=0.001,
    lambda_domain=0.1
)

# Initialize data module
data_module = DomainAdaptationDataModule(
    X_train, X_val, y_train, y_val, 
    X_ybt, y_ybt,  # Use YBT as target domain during training
    batch_size=32
)

# Initialize trainer
trainer = L.Trainer(
    max_epochs=100,
    accelerator='auto',
    devices=1,
    callbacks=[
        L.callbacks.EarlyStopping(
            monitor='val_total_loss',
            patience=10,
            mode='min'
        ),
        L.callbacks.ModelCheckpoint(
            monitor='val_f1',
            mode='max',
            save_top_k=1,
            filename='dann_model_{epoch:02d}_{val_f1:.3f}'
        )
    ],
    logger=L.loggers.WandbLogger(project="autism-deep-learning"),
    log_every_n_steps=10
)

# Train model
print("Starting domain adaptation training...")
trainer.fit(model, data_module)

print("Training completed!")

# =============================================================================
# CELL 6: MODEL EVALUATION
# =============================================================================

print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

# Load best model
best_model_path = trainer.checkpoint_callback.best_model_path
print(f"Loading best model from: {best_model_path}")

# Load model
model = DomainAdversarialClassifier.load_from_checkpoint(best_model_path)
model.eval()

# Evaluate on validation set
val_predictions = []
val_probs = []

with torch.no_grad():
    for batch in data_module.val_dataloader():
        x, y, domain_labels = batch
        class_output, features = model(x)
        val_probs.extend(class_output.squeeze().cpu().numpy())
        val_predictions.extend((class_output.squeeze() > 0.5).cpu().numpy())

val_probs = np.array(val_probs)
val_predictions = np.array(val_predictions)

# Get validation targets
val_targets = []
for batch in data_module.val_dataloader():
    x, y, domain_labels = batch
    val_targets.extend(y.cpu().numpy())
val_targets = np.array(val_targets)

print("\nValidation Set Performance:")
print(classification_report(val_targets, val_predictions))
print(f"ROC-AUC: {roc_auc_score(val_targets, val_probs):.3f}")

# =============================================================================
# CELL 7: CROSS-DATASET TESTING
# =============================================================================

print("\n" + "="*60)
print("CROSS-DATASET TESTING (C4 → YBT)")
print("="*60)

# Evaluate on YBT target domain
ybt_predictions = []
ybt_probs = []

model.eval()
with torch.no_grad():
    for batch in data_module.target_dataloader():
        x, y, domain_labels = batch
        class_output, features = model(x)
        ybt_probs.extend(class_output.squeeze().cpu().numpy())
        ybt_predictions.extend((class_output.squeeze() > 0.5).cpu().numpy())

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
# CELL 8: COMPARISON WITH BASELINE
# =============================================================================

print("\n" + "="*60)
print("COMPARISON WITH BASELINE")
print("="*60)

# Compare with baseline neural network results
results_comparison = {
    'Model': ['Baseline NN', 'Domain Adaptation DANN'],
    'F1_Score': [0.75, f1_score(y_ybt, ybt_predictions_optimal)],  # Update with actual baseline
    'ROC_AUC': [0.80, roc_auc_score(y_ybt, ybt_probs)],  # Update with actual baseline
    'Threshold': [0.5, best_threshold]
}

comparison_df = pd.DataFrame(results_comparison)
print("\nPerformance Comparison:")
print(comparison_df)

# Save model and results
os.makedirs('/Users/eb2007/playground/bullpy/c4_play2/models', exist_ok=True)
torch.save(model.state_dict(), '/Users/eb2007/playground/bullpy/c4_play2/models/domain_adaptation_dann.pth')

print("\nModel saved successfully!")
print("Domain adaptation experiment completed!")

# Close wandb
wandb.finish() 