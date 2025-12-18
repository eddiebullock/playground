#!/usr/bin/env python3
"""
Baseline experiment for 6-7 basic emotion categories.
This should achieve much higher accuracy (60-80%) than fine-grained classification.
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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.basic_emotion_dataset import BasicEmotionDataset
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
        frames = batch['frames'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        logits = model(frames)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
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
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Basic Emotions')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Basic emotion classification baseline")
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
        "--mapping_file",
        type=str,
        default="data/basic_emotion_mapping.json",
        help="Path to emotion mapping file"
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
        default=20,
        help="Number of epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--backbone_lr",
        type=float,
        default=1e-5,
        help="Backbone learning rate (if fine-tuning)"
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
        default=None,
        help="Device to use (auto-detect if not specified)"
    )
    parser.add_argument(
        "--create_splits",
        action="store_true",
        help="Create splits if they don't exist"
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze backbone"
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=5,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--use_augmentation",
        action="store_true",
        help="Use data augmentation"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet50"],
        help="Backbone architecture"
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
    
    # Check if mapping file exists
    if not Path(args.mapping_file).exists():
        print(f"Emotion mapping file not found: {args.mapping_file}")
        print("Creating mapping file...")
        from src.data.create_basic_emotion_mapping import analyze_dataset_and_create_mapping
        analyze_dataset_and_create_mapping(args.data_root, args.mapping_file)
    
    # Create splits if needed
    splits_dir = Path(args.splits_dir)
    if args.create_splits or not splits_dir.exists() or not (splits_dir / "train.csv").exists():
        print("Creating splits...")
        create_splits(args.data_root, args.splits_dir, seed=args.seed)
    
    # Load datasets with basic emotion mapping
    print("Loading datasets with basic emotion mapping...")
    train_transform = get_default_transform(augment=args.use_augmentation)
    val_test_transform = get_default_transform(augment=False)
    
    train_dataset = BasicEmotionDataset(
        data_root=args.data_root,
        split_file=str(splits_dir / "train.csv"),
        transform=train_transform,
        num_frames=args.num_frames,
        mapping_file=args.mapping_file,
    )
    
    val_dataset = BasicEmotionDataset(
        data_root=args.data_root,
        split_file=str(splits_dir / "val.csv"),
        transform=val_test_transform,
        num_frames=args.num_frames,
        mapping_file=args.mapping_file,
    )
    
    test_dataset = BasicEmotionDataset(
        data_root=args.data_root,
        split_file=str(splits_dir / "test.csv"),
        transform=val_test_transform,
        num_frames=args.num_frames,
        mapping_file=args.mapping_file,
    )
    
    print(f"\nDataset Statistics:")
    print(f"Train: {len(train_dataset)} samples, {train_dataset.num_classes} basic emotion classes")
    print(f"Val: {len(val_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")
    
    # Show class distribution
    print(f"\nTrain set class distribution:")
    train_dist = train_dataset.get_class_distribution()
    for emotion, count in sorted(train_dist.items()):
        print(f"  {emotion:15s}: {count:4d} samples")
    
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
    
    # Create model (now only 7 classes!)
    print("\nCreating model...")
    model = SimpleFrameClassifier(
        num_classes=train_dataset.num_classes,  # Now 7, not 410!
        backbone=args.backbone,
        pretrained=True,
        freeze_backbone=args.freeze_backbone,
    ).to(args.device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    if args.freeze_backbone:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        classifier_params = [p for p in model.classifier.parameters() if p.requires_grad]
        
        if backbone_params:
            optimizer = optim.AdamW([
                {'params': backbone_params, 'lr': args.backbone_lr},
                {'params': classifier_params, 'lr': args.lr},
            ], weight_decay=1e-4)
        else:
            optimizer = optim.AdamW(classifier_params, lr=args.lr, weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Training loop
    print("\nTraining...")
    best_val_acc = 0.0
    patience_counter = 0
    results_dir = Path("results/basic_emotions")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    train_history = []
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, args.device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} ({train_acc*100:.2f}%)")
        
        # Validate
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, args.device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} ({val_acc*100:.2f}%)")
        
        scheduler.step(val_loss)
        
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
            patience_counter = 0
            torch.save(model.state_dict(), results_dir / "best_model.pth")
            print(f"Saved best model (val_acc: {best_val_acc:.4f} = {best_val_acc*100:.2f}%)")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.early_stop_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(results_dir / "best_model.pth"))
    test_loss, test_acc, test_preds, test_labels, test_emotions = evaluate(
        model, test_loader, criterion, args.device
    )
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
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
        f.write(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)\n")
        f.write(f"Test Loss: {test_loss:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(test_labels, test_preds, target_names=class_names))
    
    # Plot confusion matrix
    plot_confusion_matrix(
        test_labels, test_preds,
        class_names,
        results_dir / "confusion_matrix.png"
    )
    
    print(f"\nResults saved to {results_dir}")
    print(f"\n🎉 Final Test Accuracy: {test_acc*100:.2f}%")
    print(f"   (Random baseline: {100/train_dataset.num_classes:.2f}%)")
    print(f"   (Improvement: {test_acc / (1/train_dataset.num_classes):.1f}x better than random)")

if __name__ == "__main__":
    main()

