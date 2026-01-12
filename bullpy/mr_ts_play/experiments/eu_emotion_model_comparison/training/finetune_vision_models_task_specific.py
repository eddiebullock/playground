#!/usr/bin/env python3
"""
Fine-tune vision models (ResNet, ViT, EfficientNet) on EU-Emotion dataset using TASK-SPECIFIC approach.

This matches how CLIP was fine-tuned - using 4-option forced-choice format.
Each video is paired with 4 candidate labels (1 correct + 3 foils).
Loss is cross-entropy over the 4 options (task-specific).

This is BETTER than standard classification because it matches the evaluation format.

Usage:
    # Fine-tune ResNet50 (task-specific)
    python experiments/eu_emotion_model_comparison/training/finetune_vision_models_task_specific.py \
        --model resnet50 \
        --train_trials data/trial_definitions/eu_emotion_train.json \
        --val_trials data/trial_definitions/eu_emotion_val.json \
        --data_root /path/to/EU_emotions \
        --output_dir models/resnet50_emotion_finetuned \
        --num_epochs 20 \
        --batch_size 8 \
        --learning_rate 1e-4
"""

import argparse
import json
import random
import sys
from pathlib import Path
from collections import defaultdict
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


class TaskSpecificEUEmotionDataset(Dataset):
    """
    Task-specific dataset for 4-option forced-choice fine-tuning.
    
    This matches the evaluation format: each video has 4 candidate labels.
    Returns: (frames, candidate_labels, correct_idx)
    """
    
    def __init__(
        self,
        trial_file: str,
        data_root: str,
        num_frames: int = 8,
        frame_sampling: str = "uniform",
        transform=None,
        use_augmentation: bool = False,
    ):
        """
        Args:
            trial_file: Path to JSON file with trials (must have candidate_labels)
            data_root: Root directory of EU-Emotion dataset
            num_frames: Number of frames to extract per video
            frame_sampling: "uniform", "temporal", or "keyframe"
            transform: Image transforms
            use_augmentation: Whether to apply data augmentation (for training)
        """
        with open(trial_file, 'r') as f:
            data = json.load(f)
        self.trials = data['trials']
        self.data_root = Path(data_root)
        self.num_frames = num_frames
        self.frame_sampling = frame_sampling
        self.use_augmentation = use_augmentation
        
        # Get all unique emotions for generating candidate labels
        all_emotions = set()
        for trial in self.trials:
            if 'correct_label' in trial:
                all_emotions.add(trial['correct_label'])
            elif 'emotion' in trial:
                all_emotions.add(trial['emotion'])
        self.all_emotions = sorted(list(all_emotions))
        
        # Calculate emotion frequencies for class weighting
        emotion_counts = defaultdict(int)
        for trial in self.trials:
            emotion = trial.get('correct_label', trial.get('emotion', 'unknown'))
            emotion_counts[emotion] += 1
        self.emotion_counts = dict(emotion_counts)
        total = sum(self.emotion_counts.values())
        # Calculate weights: inverse frequency (more weight for rare classes)
        self.emotion_weights = {
            emotion: total / (len(self.emotion_counts) * count)
            for emotion, count in self.emotion_counts.items()
        }
        
        # Default transform if not provided
        if transform is None:
            if use_augmentation:
                # Training: use augmentation
                self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop((224, 224)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            else:
                # Validation: no augmentation
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        else:
            self.transform = transform
        
        logger.info(f"Loaded {len(self.trials)} task-specific trials")
        logger.info(f"Found {len(self.all_emotions)} unique emotions")
        if use_augmentation:
            logger.info("Using data augmentation for training")
    
    def _extract_frames(self, video_path: Path) -> list:
        """Extract frames from video."""
        from experiments.eu_emotion_model_comparison.models.video_utils import extract_frames
        try:
            frames = extract_frames(str(video_path), num_frames=self.num_frames, sampling=self.frame_sampling)
            return frames
        except Exception as e:
            logger.warning(f"Could not extract frames from {video_path}: {e}")
            return []
    
    def __len__(self):
        return len(self.trials)
    
    def __getitem__(self, idx):
        trial = self.trials[idx]
        video_path = self.data_root / trial['stimulus_path']
        correct_label = trial.get('correct_label') or trial.get('emotion', 'neutral')
        
        # Get emotion weight for class weighting
        emotion_weight = self.emotion_weights.get(correct_label, 1.0)
        
        # Extract frames
        frames = self._extract_frames(video_path)
        
        if not frames:
            # Return black image if video can't be read
            frames = [Image.new('RGB', (224, 224), (0, 0, 0))]
        
        # Generate candidate_labels if not present
        if 'candidate_labels' in trial:
            candidate_labels = trial['candidate_labels']
        else:
            # Generate 4 candidate labels: correct + 3 random foils
            other_emotions = [e for e in self.all_emotions if e != correct_label]
            foils = random.sample(other_emotions, min(3, len(other_emotions)))
            candidate_labels = [correct_label] + foils
            random.shuffle(candidate_labels)
        
        # Find correct index
        try:
            correct_idx = candidate_labels.index(correct_label)
        except ValueError:
            logger.warning(f"Correct label '{correct_label}' not in candidate_labels for trial {idx}")
            correct_idx = 0  # Default to first option
        
        return frames, candidate_labels, correct_idx, emotion_weight


def collate_task_specific_batch(batch):
    """
    Custom collate function for task-specific batches.
    
    Returns:
        frames_batch: List[List[PIL.Image]] - batch_size x num_frames
        candidate_labels_batch: List[List[str]] - batch_size x 4
        correct_indices: Tensor - batch_size
        emotion_weights: Tensor - batch_size (for class weighting)
    """
    frames_batch = []
    candidate_labels_batch = []
    correct_indices = []
    emotion_weights = []
    
    for frames, candidate_labels, correct_idx, emotion_weight in batch:
        frames_batch.append(frames)
        candidate_labels_batch.append(candidate_labels)
        correct_indices.append(correct_idx)
        emotion_weights.append(emotion_weight)
    
    return (
        frames_batch,
        candidate_labels_batch,
        torch.tensor(correct_indices, dtype=torch.long),
        torch.tensor(emotion_weights, dtype=torch.float)
    )


def create_model(model_type: str, device: str):
    """Create model for task-specific fine-tuning."""
    # For task-specific, we don't need to know num_classes upfront
    # We'll use the model to score 4 candidate labels
    
    if model_type == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V2")
        # Keep the full model - we'll use features + a projection layer
        model.fc = nn.Identity()  # Remove final layer, use features
        return model.to(device)
    
    elif model_type == "resnet101":
        model = models.resnet101(weights="IMAGENET1K_V2")
        model.fc = nn.Identity()
        return model.to(device)
    
    elif model_type == "vit_base":
        try:
            import timm
            model = timm.create_model('vit_base_patch16_224', pretrained=True)
            # Remove classifier, use features
            if hasattr(model, 'head'):
                model.head = nn.Identity()
            return model.to(device)
        except ImportError:
            logger.error("timm not installed. Install with: pip install timm")
            raise
    
    elif model_type.startswith("efficientnet"):
        try:
            import timm
            model = timm.create_model(model_type, pretrained=True)
            # Remove classifier, use features
            if hasattr(model, 'classifier'):
                model.classifier = nn.Identity()
            return model.to(device)
        except ImportError:
            logger.error("timm not installed. Install with: pip install timm")
            raise
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_model_feature_dim(model_type: str, model) -> int:
    """Get feature dimension for a model."""
    if model_type.startswith("resnet"):
        # ResNet feature dim
        return 2048 if "101" in model_type else 2048 if "50" in model_type else 512
    elif model_type == "vit_base":
        return 768
    elif model_type.startswith("efficientnet"):
        # EfficientNet feature dim varies, check model
        if hasattr(model, 'num_features'):
            return model.num_features
        else:
            # Default for EfficientNet
            return 1280 if "b0" in model_type else 1536
    else:
        return 512  # Default


def train_epoch_task_specific(model, train_loader, criterion, optimizer, device, model_type: str, score_proj: nn.Module):
    """Train for one epoch (task-specific)."""
    model.train()
    score_proj.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for frames_batch, candidate_labels_batch, correct_indices, emotion_weights in pbar:
        batch_loss = 0
        batch_size_actual = len(frames_batch)
        
        # Process each video in the batch
        for video_idx in range(batch_size_actual):
            video_frames = frames_batch[video_idx]  # List of PIL Images
            candidate_labels = candidate_labels_batch[video_idx]  # List of 4 strings
            correct_idx = correct_indices[video_idx].item()  # Integer 0-3
            
            # Process frames
            frame_tensors = []
            for frame in video_frames:
                # Apply transform
                if model_type.startswith("resnet") or model_type.startswith("efficientnet"):
                    # Use standard ImageNet transform
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
                elif model_type == "vit_base":
                    # ViT might need different transform, but for now use same
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
                else:
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
                
                frame_t = transform(frame)
                frame_tensors.append(frame_t)
            
            if not frame_tensors:
                continue
            
            batch_frames = torch.stack(frame_tensors).to(device)
            
            # Get video features
            features = model(batch_frames)  # [num_frames, feature_dim]
            video_features = features.mean(dim=0)  # [feature_dim] - average over frames
            
            # Project video features directly to 4 scores (one per candidate option)
            emotion_scores = score_proj(video_features.unsqueeze(0))  # [1, 4]
            
            # Calculate accuracy (before backprop)
            predicted_idx = emotion_scores.argmax(dim=1).item()
            if predicted_idx == correct_idx:
                correct += 1
            total += 1
            
            # Cross-entropy loss over 4 options with class weighting
            target = torch.tensor([correct_idx], dtype=torch.long, device=device)
            loss = criterion(emotion_scores, target)
            # Apply emotion weight (higher weight for rare classes)
            emotion_weight = emotion_weights[video_idx].item()
            batch_loss += loss * emotion_weight
        
        # Average loss and backprop
        if batch_size_actual > 0:
            avg_batch_loss = batch_loss / batch_size_actual
            optimizer.zero_grad()
            avg_batch_loss.backward()
            optimizer.step()
            
            train_loss += avg_batch_loss.item()
    
    return train_loss / len(train_loader) if len(train_loader) > 0 else 0.0, 100 * correct / total if total > 0 else 0.0


def validate_task_specific(model, val_loader, device, model_type: str, score_proj: nn.Module):
    """Validate model (task-specific)."""
    model.eval()
    score_proj.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for frames_batch, candidate_labels_batch, correct_indices, emotion_weights in pbar:
            batch_size_actual = len(frames_batch)
            
            for video_idx in range(batch_size_actual):
                try:
                    video_frames = frames_batch[video_idx]
                    correct_idx = correct_indices[video_idx].item()
                    
                    # Process frames
                    frame_tensors = []
                    for frame in video_frames:
                        transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])
                        frame_t = transform(frame)
                        frame_tensors.append(frame_t)
                    
                    if not frame_tensors:
                        continue
                    
                    batch_frames = torch.stack(frame_tensors).to(device)
                    
                    # Get video features
                    features = model(batch_frames)  # [num_frames, feature_dim]
                    video_features = features.mean(dim=0)  # [feature_dim]
                    
                    # Project to 4 scores
                    emotion_scores = score_proj(video_features.unsqueeze(0))  # [1, 4]
                    predicted_idx = emotion_scores.argmax(dim=1).item()
                    
                    if predicted_idx == correct_idx:
                        correct += 1
                    total += 1
                except Exception as e:
                    logger.warning(f"Error in validation: {e}")
                    continue
    
    return 100 * correct / total if total > 0 else 0.0


def finetune_vision_model_task_specific(
    model_type: str,
    train_trials: str,
    val_trials: str,
    data_root: str,
    output_dir: str,
    num_epochs: int = 12,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    device: str = "auto",
    num_frames: int = 8,
    frame_sampling: str = "uniform",
    emotion_embedding_dim: int = 128,
):
    """
    Fine-tune vision model using task-specific approach (4-option forced-choice).
    
    This matches how CLIP was fine-tuned - better alignment with evaluation format.
    """
    
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
    logger.info("Loading task-specific datasets...")
    train_dataset = TaskSpecificEUEmotionDataset(
        train_trials, data_root, num_frames=num_frames, frame_sampling=frame_sampling,
        use_augmentation=True  # Use augmentation for training
    )
    val_dataset = TaskSpecificEUEmotionDataset(
        val_trials, data_root, num_frames=num_frames, frame_sampling=frame_sampling,
        use_augmentation=False  # No augmentation for validation
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_task_specific_batch,
        pin_memory=True if device.type == "cuda" else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_task_specific_batch,
        pin_memory=True if device.type == "cuda" else False,
    )
    
    # Create model
    logger.info(f"Creating {model_type} model...")
    model = create_model(model_type, device)
    
    # Create feature projection to 4 scores (one per candidate label)
    # Simplified approach: map video features directly to 4 scores
    feature_dim = get_model_feature_dim(model_type, model)
    # Project video features to 4 scores (one per candidate option)
    score_proj = nn.Linear(feature_dim, 4).to(device)
    
    # Setup training
    # Combine model and projection parameters
    optimizer = torch.optim.AdamW(
        [
            {'params': model.parameters()},
            {'params': score_proj.parameters()}
        ],
        lr=learning_rate,
        weight_decay=0.01
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_val_acc = 0.0
    
    logger.info(f"Starting task-specific fine-tuning for {num_epochs} epochs...")
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        logger.info("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch_task_specific(
            model, train_loader, criterion, optimizer, device, model_type, score_proj
        )
        
        # Validate
        val_acc = validate_task_specific(model, val_loader, device, model_type, score_proj)
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'score_proj_state_dict': score_proj.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, output_dir / "best_model.pth")
            logger.info(f"✅ Saved best model (Val Acc: {val_acc:.2f}%)")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training completed!")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    logger.info(f"Model saved to: {output_dir / 'best_model.pth'}")
    logger.info(f"{'='*60}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Fine-tune vision models (task-specific)")
    parser.add_argument('--model', type=str, required=True,
                       choices=['resnet50', 'resnet101', 'vit_base', 'efficientnet_b0'],
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
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size (smaller for task-specific)')
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
    
    args = parser.parse_args()
    
    finetune_vision_model_task_specific(
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
    )


if __name__ == "__main__":
    main()
