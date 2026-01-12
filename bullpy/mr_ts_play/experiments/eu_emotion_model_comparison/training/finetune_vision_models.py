#!/usr/bin/env python3
"""
Fine-tune vision models (ResNet, ViT, EfficientNet) on EU-Emotion dataset.

This script fine-tunes pretrained vision models for emotion recognition.
Models are pretrained on ImageNet, then fine-tuned for 27 emotion classes.

Usage:
    # Fine-tune ResNet50
    python experiments/eu_emotion_model_comparison/training/finetune_vision_models.py \
        --model resnet50 \
        --train_trials data/trial_definitions/eu_emotion_train.json \
        --val_trials data/trial_definitions/eu_emotion_val.json \
        --data_root /path/to/EU_emotions \
        --output_dir models/resnet50_emotion_finetuned \
        --num_epochs 20 \
        --batch_size 16 \
        --learning_rate 1e-4

    # Fine-tune ViT
    python experiments/eu_emotion_model_comparison/training/finetune_vision_models.py \
        --model vit_base \
        --train_trials data/trial_definitions/eu_emotion_train.json \
        --val_trials data/trial_definitions/eu_emotion_val.json \
        --data_root /path/to/EU_emotions \
        --output_dir models/vit_emotion_finetuned \
        --num_epochs 20
"""

import argparse
import json
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EUEmotionVideoDataset(Dataset):
    """Dataset for EU-Emotion videos with emotion labels."""
    
    def __init__(
        self,
        trial_file: str,
        data_root: str,
        num_frames: int = 8,
        frame_sampling: str = "uniform",
        transform=None,
    ):
        """
        Args:
            trial_file: Path to JSON file with trials
            data_root: Root directory of EU-Emotion dataset
            num_frames: Number of frames to extract per video
            frame_sampling: "uniform", "temporal", or "keyframe"
            transform: Image transforms
        """
        with open(trial_file, 'r') as f:
            data = json.load(f)
        self.trials = data['trials']
        self.data_root = Path(data_root)
        self.num_frames = num_frames
        self.frame_sampling = frame_sampling
        
        # Get all unique emotions
        self.emotions = sorted(set(t['correct_label'] for t in self.trials))
        self.emotion_to_idx = {e: i for i, e in enumerate(self.emotions)}
        self.idx_to_emotion = {i: e for i, e in enumerate(self.emotions)}
        self.num_classes = len(self.emotions)
        
        # Default transform if not provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transform
        
        logger.info(f"Loaded {len(self.trials)} trials with {self.num_classes} emotion classes")
    
    def __len__(self):
        return len(self.trials)
    
    def _extract_frames(self, video_path: Path) -> list:
        """Extract frames from video."""
        # Use the video_utils function
        from experiments.eu_emotion_model_comparison.models.video_utils import extract_frames
        try:
            frames = extract_frames(str(video_path), num_frames=self.num_frames, sampling=self.frame_sampling)
            return frames
        except Exception as e:
            logger.warning(f"Could not extract frames from {video_path}: {e}")
            return []
    
    def __getitem__(self, idx):
        trial = self.trials[idx]
        video_path = self.data_root / trial['stimulus_path']
        
        # Extract frames
        frames = self._extract_frames(video_path)
        
        if not frames:
            # Return black image if video can't be read
            frames = [Image.new('RGB', (224, 224), (0, 0, 0))]
        
        # Use first frame (or average multiple frames - for now use first)
        # TODO: Could average features from multiple frames
        frame = frames[0]
        frame_tensor = self.transform(frame)
        
        # Get label
        emotion = trial['correct_label']
        label = self.emotion_to_idx[emotion]
        
        return frame_tensor, label


def create_model(model_type: str, num_classes: int, device: str):
    """Create model for fine-tuning."""
    if model_type == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V2")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model.to(device)
    
    elif model_type == "resnet101":
        model = models.resnet101(weights="IMAGENET1K_V2")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model.to(device)
    
    elif model_type == "vit_base":
        # Use timm for ViT (more flexible)
        try:
            import timm
            model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
            return model.to(device)
        except ImportError:
            logger.error("timm not installed. Install with: pip install timm")
            raise
    
    elif model_type.startswith("efficientnet"):
        # efficientnet_b0, efficientnet_b1, etc.
        try:
            import timm
            model = timm.create_model(model_type, pretrained=True, num_classes=num_classes)
            return model.to(device)
        except ImportError:
            logger.error("timm not installed. Install with: pip install timm")
            raise
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    return train_loss / len(train_loader), 100 * correct / total


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    return val_loss / len(val_loader), 100 * correct / total


def finetune_vision_model(
    model_type: str,
    train_trials: str,
    val_trials: str,
    data_root: str,
    output_dir: str,
    num_epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    device: str = "auto",
    num_frames: int = 8,
    frame_sampling: str = "uniform",
    save_every: int = 5,
):
    """Fine-tune a vision model on EU-Emotion dataset."""
    
    # Setup device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = torch.device(device)
    logger.info(f"Using device: {device}")
    
    # Create datasets
    logger.info("Loading datasets...")
    train_dataset = EUEmotionVideoDataset(
        train_trials, data_root, num_frames=num_frames, frame_sampling=frame_sampling
    )
    val_dataset = EUEmotionVideoDataset(
        val_trials, data_root, num_frames=num_frames, frame_sampling=frame_sampling
    )
    
    num_classes = train_dataset.num_classes
    logger.info(f"Number of emotion classes: {num_classes}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True if device.type == "cuda" else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )
    
    # Create model
    logger.info(f"Creating {model_type} model...")
    model = create_model(model_type, num_classes, device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save emotion mapping
    emotion_mapping = {
        'emotions': train_dataset.emotions,
        'emotion_to_idx': train_dataset.emotion_to_idx,
        'idx_to_emotion': train_dataset.idx_to_emotion,
    }
    with open(output_dir / "emotion_mapping.json", 'w') as f:
        json.dump(emotion_mapping, f, indent=2)
    
    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    
    logger.info(f"Starting training for {num_epochs} epochs...")
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        logger.info("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'emotion_mapping': emotion_mapping,
            }, output_dir / "best_model.pth")
            logger.info(f"✅ Saved best model (Val Acc: {val_acc:.2f}%)")
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, output_dir / f"checkpoint_epoch_{epoch+1}.pth")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training completed!")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
    logger.info(f"Model saved to: {output_dir / 'best_model.pth'}")
    logger.info(f"{'='*60}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Fine-tune vision models on EU-Emotion")
    parser.add_argument('--model', type=str, required=True,
                       choices=['resnet50', 'resnet101', 'vit_base', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3'],
                       help='Model to fine-tune')
    parser.add_argument('--train_trials', type=str, required=True,
                       help='Path to train trials JSON file')
    parser.add_argument('--val_trials', type=str, required=True,
                       help='Path to validation trials JSON file')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of EU-Emotion dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for fine-tuned model')
    parser.add_argument('--num_epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to train on')
    parser.add_argument('--num_frames', type=int, default=8,
                       help='Number of frames to extract per video')
    parser.add_argument('--frame_sampling', type=str, default='uniform',
                       choices=['uniform', 'temporal', 'keyframe'],
                       help='Frame sampling strategy')
    parser.add_argument('--save_every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    finetune_vision_model(
        model_type=args.model,
        train_trials=args.train_trials,
        val_trials=args.val_trials,
        data_root=args.data_root,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        num_frames=args.num_frames,
        frame_sampling=args.frame_sampling,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
