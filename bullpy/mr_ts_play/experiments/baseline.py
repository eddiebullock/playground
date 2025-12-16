#!/usr/bin/env python3
"""
Minimal baseline experiment to verify dataset loading and evaluation pipeline.
"""

import os
import sys
import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import MindreadingDataset
from src.data.create_splits import create_splits
from src.models.baseline import SimpleFrameClassifier
from src.utils.transforms import get_default_transform
from src.utils.seed import set_seed, worker_init_fn


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        frames = batch['frames'].to(device)  # (B, T, C, H, W)
        labels = batch['label'].to(device)
        
        # Forward
        optimizer.zero_grad()
        logits = model(frames)
        loss = criterion(logits, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_emotions = []
    
    with torch.no_grad():
        for batch in dataloader:
            frames = batch['frames'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(frames)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_emotions.extend(batch['emotion'])
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy, all_preds, all_labels, all_emotions


def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Baseline experiment")
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
        default=8,
        help="Batch size"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=8,
        help="Number of frames per video"
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
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--create_splits",
        action="store_true",
        help="Create splits if they don't exist"
    )
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create splits if needed
    splits_dir = Path(args.splits_dir)
    if args.create_splits or not splits_dir.exists() or not (splits_dir / "train.csv").exists():
        print("Creating splits...")
        create_splits(args.data_root, args.splits_dir, seed=args.seed)
    
    # Load datasets
    print("Loading datasets...")
    transform = get_default_transform()
    
    train_dataset = MindreadingDataset(
        data_root=args.data_root,
        split_file=str(splits_dir / "train.csv"),
        transform=transform,
        num_frames=args.num_frames,
    )
    
    val_dataset = MindreadingDataset(
        data_root=args.data_root,
        split_file=str(splits_dir / "val.csv"),
        transform=transform,
        num_frames=args.num_frames,
    )
    
    test_dataset = MindreadingDataset(
        data_root=args.data_root,
        split_file=str(splits_dir / "test.csv"),
        transform=transform,
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
    model = SimpleFrameClassifier(
        num_classes=train_dataset.num_classes,
        backbone="resnet18",
        pretrained=True,
        freeze_backbone=True,
    ).to(args.device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print("\nTraining...")
    best_val_acc = 0.0
    results_dir = Path("results/baseline")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    train_history = []
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, args.device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # Validate
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, args.device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        train_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
        })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), results_dir / "best_model.pth")
            print(f"Saved best model (val_acc: {val_acc:.4f})")
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(results_dir / "best_model.pth"))
    test_loss, test_acc, test_preds, test_labels, test_emotions = evaluate(
        model, test_loader, criterion, args.device
    )
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    # Classification report
    class_names = [train_dataset.idx_to_emotion[i] for i in range(train_dataset.num_classes)]
    report = classification_report(
        test_labels, test_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    
    # Save results
    pd.DataFrame(train_history).to_csv(results_dir / "train_history.csv", index=False)
    
    with open(results_dir / "test_results.txt", "w") as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(test_labels, test_preds, target_names=class_names))
    
    # Plot confusion matrix (sample of classes if too many)
    if train_dataset.num_classes <= 50:
        plot_confusion_matrix(
            test_labels, test_preds,
            class_names,
            results_dir / "confusion_matrix.png"
        )
    else:
        print(f"Skipping confusion matrix (too many classes: {train_dataset.num_classes})")
    
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()

