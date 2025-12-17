#!/usr/bin/env python3
"""
Prototypical Networks baseline for few-shot emotion recognition.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import MindreadingDataset
from src.data.create_splits import create_splits
from src.models.prototypical import PrototypicalNetwork, PrototypicalLoss
from src.utils.transforms import get_default_transform
from src.utils.seed import set_seed, worker_init_fn
from src.evaluation.metrics import compute_all_metrics, print_metrics, compare_to_random


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_embeddings = []
    all_labels = []
    
    for batch in dataloader:
        frames = batch['frames'].to(device)  # (B, T, C, H, W)
        labels = batch['label'].to(device)
        
        # Forward: extract embeddings
        optimizer.zero_grad()
        embeddings = model(frames)  # (B, embedding_dim)
        
        # Compute prototypes from current batch
        # (In practice, we compute from all training data, but for efficiency
        # we use batch prototypes during training)
        prototypes = model.compute_prototypes(embeddings, labels)
        
        # Compute loss
        loss = criterion(embeddings, labels, prototypes)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Store for metrics
        total_loss += loss.item()
        all_embeddings.append(embeddings.detach().cpu())
        all_labels.append(labels.cpu())
    
    avg_loss = total_loss / len(dataloader)
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return avg_loss, all_embeddings, all_labels


def evaluate(
    model,
    train_dataloader,
    eval_dataloader,
    criterion,
    device,
    num_classes: int,
):
    """
    Evaluate model using prototypical networks approach.
    
    For each query sample:
    1. Compute prototypes from training set
    2. Classify query by distance to prototypes
    """
    model.eval()
    
    # Compute prototypes from training set
    print("Computing prototypes from training set...")
    all_train_embeddings = []
    all_train_labels = []
    
    with torch.no_grad():
        for batch in train_dataloader:
            frames = batch['frames'].to(device)
            labels = batch['label'].to(device)
            
            embeddings = model(frames)
            all_train_embeddings.append(embeddings.cpu())
            all_train_labels.append(labels.cpu())
    
    all_train_embeddings = torch.cat(all_train_embeddings, dim=0)
    all_train_labels = torch.cat(all_train_labels, dim=0)
    
    # Compute prototypes
    prototypes = model.compute_prototypes(all_train_embeddings.to(device), all_train_labels.to(device))
    prototypes = prototypes.cpu()
    
    # Evaluate on validation/test set
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_pred_proba = []
    all_emotions = []
    
    with torch.no_grad():
        for batch in eval_dataloader:
            frames = batch['frames'].to(device)
            labels = batch['label'].to(device)
            
            # Extract embeddings
            embeddings = model(frames).cpu()
            
            # Classify by distance to prototypes
            logits = model.classify_by_distance(embeddings, prototypes)
            
            # Compute loss
            loss = criterion(embeddings.to(device), labels.to(device), prototypes.to(device))
            total_loss += loss.item()
            
            # Get predictions
            pred_proba = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1).numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_pred_proba.append(pred_proba.numpy())
            all_emotions.extend(batch['emotion'])
    
    avg_loss = total_loss / len(eval_dataloader)
    all_pred_proba = np.vstack(all_pred_proba)
    
    return avg_loss, all_preds, all_labels, all_pred_proba, all_emotions


def main():
    parser = argparse.ArgumentParser(description="Prototypical Networks baseline")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/mindreading_transporter_files/Mindreading emotions library/Emotions",
        help="Root directory of the dataset"
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        default="data/splits",
        help="Directory containing train/val/test splits"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=30,
        help="Number of epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=8,
        help="Number of frames per video"
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=512,
        help="Embedding dimension"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (auto-detect if not specified)"
    )
    parser.add_argument(
        "--create_splits",
        action="store_true",
        help="Create splits if they don't exist"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet50"],
        help="Backbone architecture"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate"
    )
    parser.add_argument(
        "--use_augmentation",
        action="store_true",
        help="Use data augmentation"
    )
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    print(f"Using device: {args.device}")
    
    # Set seed
    set_seed(args.seed)
    
    # Create splits if needed
    splits_dir = Path(args.splits_dir)
    if args.create_splits or not splits_dir.exists() or not (splits_dir / "train.csv").exists():
        print("Creating splits...")
        create_splits(args.data_root, args.splits_dir, seed=args.seed)
    
    # Load datasets
    print("Loading datasets...")
    train_transform = get_default_transform(augment=args.use_augmentation)
    val_test_transform = get_default_transform(augment=False)
    
    train_dataset = MindreadingDataset(
        data_root=args.data_root,
        split_file=str(splits_dir / "train.csv"),
        transform=train_transform,
        num_frames=args.num_frames,
    )
    
    val_dataset = MindreadingDataset(
        data_root=args.data_root,
        split_file=str(splits_dir / "val.csv"),
        transform=val_test_transform,
        num_frames=args.num_frames,
    )
    
    test_dataset = MindreadingDataset(
        data_root=args.data_root,
        split_file=str(splits_dir / "test.csv"),
        transform=val_test_transform,
        num_frames=args.num_frames,
    )
    
    print(f"Train: {len(train_dataset)} samples, {train_dataset.num_classes} classes")
    print(f"Val: {len(val_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    
    # Create model
    print("Creating model...")
    model = PrototypicalNetwork(
        num_classes=train_dataset.num_classes,
        embedding_dim=args.embedding_dim,
        backbone=args.backbone,
        pretrained=True,
        freeze_backbone=False,  # Fine-tune for better embeddings
        dropout=args.dropout,
    ).to(args.device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss and optimizer
    criterion = PrototypicalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    print("\nTraining...")
    best_val_acc = 0.0
    patience_counter = 0
    early_stop_patience = 10
    results_dir = Path("results/prototypical")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    train_history = []
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Train
        train_loss, _, _ = train_epoch(model, train_loader, criterion, optimizer, args.device)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss, val_preds, val_labels, val_pred_proba, _ = evaluate(
            model, train_loader, val_loader, criterion, args.device, train_dataset.num_classes
        )
        
        # Compute metrics
        class_names = [train_dataset.idx_to_emotion[i] for i in range(train_dataset.num_classes)]
        val_metrics = compute_all_metrics(
            np.array(val_labels),
            np.array(val_preds),
            val_pred_proba,
            class_names,
            top_k_values=(1, 5, 10, 20),
        )
        
        print_metrics(val_metrics, f"Val Metrics (Epoch {epoch+1})")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        train_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            **val_metrics,
        })
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0
            torch.save(model.state_dict(), results_dir / "best_model.pth")
            print(f"Saved best model (val_acc: {best_val_acc:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(results_dir / "best_model.pth"))
    test_loss, test_preds, test_labels, test_pred_proba, test_emotions = evaluate(
        model, train_loader, test_loader, criterion, args.device, train_dataset.num_classes
    )
    
    # Compute test metrics
    test_metrics = compute_all_metrics(
        np.array(test_labels),
        np.array(test_preds),
        test_pred_proba,
        class_names,
        top_k_values=(1, 5, 10, 20, 50),
    )
    
    print_metrics(test_metrics, "Test Metrics")
    
    # Compare to random baseline
    random_metrics = compare_to_random(train_dataset.num_classes, top_k_values=(1, 5, 10, 20, 50))
    print("\nRandom Baseline:")
    print("-" * 60)
    for k in [1, 5, 10, 20, 50]:
        if f'top_{k}_accuracy' in random_metrics:
            print(f"Top-{k} Accuracy:  {random_metrics[f'top_{k}_accuracy']:.4f} ({random_metrics[f'top_{k}_accuracy']*100:.2f}%)")
    print("-" * 60)
    
    # Save results
    pd.DataFrame(train_history).to_csv(results_dir / "train_history.csv", index=False)
    
    with open(results_dir / "test_results.txt", "w") as f:
        f.write("Test Set Results\n")
        f.write("=" * 60 + "\n\n")
        for key, value in test_metrics.items():
            f.write(f"{key}: {value:.4f}\n")
        f.write("\nRandom Baseline:\n")
        for key, value in random_metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()

