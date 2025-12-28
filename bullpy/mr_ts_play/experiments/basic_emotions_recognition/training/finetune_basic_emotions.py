#!/usr/bin/env python3
"""
Fine-tune CLIP on basic emotions (4-option forced-choice, matching complex emotion experiments).

This script fine-tunes CLIP to predict 7 basic emotions using 4-option forced-choice:
- happy, sad, angry, fear, surprise, disgust, neutral
- Each trial has 4 candidate labels: 1 correct + 3 foils
- Model selects from the 4 options (easier discrimination than 7-way)

Key change: Now uses 4-option forced-choice (like complex emotion experiments):
- Matches task structure of complex emotion experiments
- Easier discrimination task
- Better performance expected
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import task-specific dataset (reuse from cam_human_like)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "cam_human_like" / "training"))
from task_specific_dataset import TaskSpecificTrialDataset, collate_trial_batch

# Basic emotion categories
BASIC_EMOTIONS = ["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"]


class BasicEmotionDataset(TaskSpecificTrialDataset):
    """
    Dataset for basic emotion fine-tuning (4-option forced-choice).
    
    Extends TaskSpecificTrialDataset for forced-choice format:
    - Each trial has 4 candidate labels (1 correct + 3 foils)
    - Matches complex emotion experiment format
    """
    
    def __getitem__(self, idx):
        trial = self.trials[idx]
        
        # Get video path
        stimulus_path = trial['stimulus_path']
        
        # Handle both absolute and relative paths
        if Path(stimulus_path).is_absolute():
            video_path = Path(stimulus_path)
        else:
            video_path = self.data_root / stimulus_path
        
        # If path doesn't exist, try as absolute path from data_root
        if not video_path.exists():
            video_path = Path(stimulus_path)
            if not video_path.is_absolute():
                found_files = list(self.data_root.rglob(Path(stimulus_path).name))
                if found_files:
                    video_path = found_files[0]
        
        # Load frames - catch errors and return placeholder
        try:
            frames = self._load_video_frames(video_path)
        except (ValueError, FileNotFoundError, OSError) as e:
            # Return a random valid sample instead (like EUEmotionDataset does)
            print(f"Warning: Could not load {video_path.name}: {e}", flush=True)
            import random
            return self.__getitem__(random.randint(0, len(self.trials) - 1))
        
        # Apply transforms if provided
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        
        # Get candidate labels (should be 4 options: 1 correct + 3 foils)
        candidate_labels = trial.get('candidate_labels', [])
        
        # Validate we have 4 candidate labels
        if len(candidate_labels) != 4:
            # Fallback: create 4-option forced-choice from correct label
            correct_label = trial.get('correct_label', 'neutral')
            foils = [e for e in BASIC_EMOTIONS if e != correct_label][:3]
            candidate_labels = [correct_label] + foils
            import random
            random.shuffle(candidate_labels)
        
        # Get correct index (should be 0-3)
        correct_idx = trial.get('correct_idx', 0)
        
        # Validate correct_idx is in range
        if correct_idx < 0 or correct_idx >= 4:
            # Try to find correct label index
            correct_label = trial.get('correct_label', 'neutral')
            if correct_label in candidate_labels:
                correct_idx = candidate_labels.index(correct_label)
            else:
                correct_idx = 0
        
        return {
            'frames': frames,
            'candidate_labels': candidate_labels,
            'correct_idx': correct_idx,
            'trial_id': trial.get('trial_id', f'trial_{idx}'),
        }


def validate_basic_emotions(model, val_loader, processor, device):
    """Validate model on validation set (4-option forced-choice)."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for frames_batch, candidate_labels_batch, correct_indices in tqdm(val_loader, desc="Validating"):
            batch_size_actual = len(frames_batch)
            
            for video_idx in range(batch_size_actual):
                try:
                    video_frames = frames_batch[video_idx]
                    candidate_labels = candidate_labels_batch[video_idx]
                    correct_idx = correct_indices[video_idx].item()
                    
                    # Process frames
                    image_inputs = processor(images=video_frames, return_tensors="pt").to(device)
                    
                    # Process all 4 candidate labels with prompt templates
                    prompted_labels = [f"a photo of a person feeling {label}" for label in candidate_labels]
                    text_inputs = processor(
                        text=prompted_labels,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    ).to(device)
                    
                    # Get embeddings
                    image_features = model.get_image_features(**image_inputs)
                    text_features = model.get_text_features(**text_inputs)
                    
                    # Aggregate video features
                    video_features = image_features.mean(dim=0, keepdim=True)
                    
                    # Normalize
                    video_features = F.normalize(video_features, dim=-1)
                    text_features = F.normalize(text_features, dim=-1)
                    
                    # Compute similarity
                    logits = video_features @ text_features.t()
                    
                    # Get predicted index (0-3 for 4-option forced-choice)
                    predicted_idx = logits.argmax(dim=-1).item()
                    
                    if predicted_idx == correct_idx:
                        correct += 1
                    total += 1
                except Exception as e:
                    print(f"Error in validation: {e}", flush=True)
                    continue
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def finetune_basic_emotions(
    train_dataset,
    val_dataset,
    model_name="openai/clip-vit-base-patch32",
    output_dir="models/basic_emotions_finetuned",
    num_epochs=20,
    batch_size=4,
    learning_rate=5e-5,
    weight_decay=0.01,
    device="cpu",
    num_frames=16,
    use_lr_scheduler=True,
    warmup_steps=100,
    early_stopping_patience=5,
    early_stopping_min_delta=0.001,
):
    """
    Fine-tune CLIP for 4-option forced-choice basic emotion classification.
    
    Args:
        train_dataset: BasicEmotionDataset returning (frames, candidate_labels, correct_idx)
                      Each trial has 4 candidate labels (1 correct + 3 foils)
        val_dataset: Validation dataset (same format)
        model_name: CLIP model to fine-tune
        output_dir: Directory to save fine-tuned model
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: Weight decay
        device: Device to train on
        num_frames: Number of frames per video
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load CLIP
    print(f"Loading CLIP model: {model_name}...")
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model = model.to(device)
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_trial_batch,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_trial_batch,
    )
    
    # Setup learning rate scheduler
    if use_lr_scheduler:
        total_steps = len(train_loader) * num_epochs
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=learning_rate * 0.01)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
        print(f"Using LR scheduler: warmup={warmup_steps} steps, then cosine annealing")
    else:
        scheduler = None
    
    best_val_acc = 0.0
    epochs_without_improvement = 0
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        for batch_idx, (frames_batch, candidate_labels_batch, correct_indices) in enumerate(tqdm(train_loader, desc="Training")):
            try:
                batch_loss = 0
                batch_size_actual = len(frames_batch)
                
                # Process each video in the batch
                for video_idx in range(batch_size_actual):
                    try:
                        video_frames = frames_batch[video_idx]
                        candidate_labels = candidate_labels_batch[video_idx]
                        correct_idx = correct_indices[video_idx].item()
                        
                        # Ensure we have 4 candidate labels (forced-choice format)
                        if len(candidate_labels) != 4:
                            # Fallback: create 4-option forced-choice
                            trial = train_dataset.trials[video_idx] if hasattr(train_dataset, 'trials') else {}
                            correct_label = trial.get('correct_label', 'neutral')
                            foils = [e for e in BASIC_EMOTIONS if e != correct_label][:3]
                            candidate_labels = [correct_label] + foils
                            import random
                            random.shuffle(candidate_labels)
                            correct_idx = candidate_labels.index(correct_label)
                        
                        # Process frames
                        image_inputs = processor(images=video_frames, return_tensors="pt").to(device)
                        
                        # Process all 4 candidate labels with prompt templates
                        prompted_labels = [f"a photo of a person feeling {label}" for label in candidate_labels]
                        text_inputs = processor(
                            text=prompted_labels,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                        ).to(device)
                        
                        # Get embeddings
                        image_features = model.get_image_features(**image_inputs)
                        text_features = model.get_text_features(**text_inputs)
                        
                        # Aggregate video features
                        video_features = image_features.mean(dim=0, keepdim=True)
                        
                        # Normalize
                        video_features = F.normalize(video_features, dim=-1)
                        text_features = F.normalize(text_features, dim=-1)
                        
                        # Compute similarity: video_features @ text_features.t() -> (1, 4)
                        logits = video_features @ text_features.t()
                        
                        # Cross-entropy loss over 4 options (forced-choice)
                        target = torch.tensor([correct_idx], dtype=torch.long).to(device)
                        loss = nn.CrossEntropyLoss()(logits, target)
                        
                        batch_loss += loss
                    except Exception as e:
                        print(f"Error processing video {video_idx} in batch {batch_idx}: {e}", flush=True)
                        continue
                
                # Average loss over batch
                if batch_size_actual > 0:
                    avg_batch_loss = batch_loss / batch_size_actual
                    
                    # Backward
                    optimizer.zero_grad()
                    avg_batch_loss.backward()
                    optimizer.step()
                    
                    # Update learning rate scheduler
                    if scheduler is not None:
                        scheduler.step()
                    
                    total_loss += avg_batch_loss.item()
                    num_batches += 1
            except Exception as e:
                print(f"Error during training batch {batch_idx}: {e}", flush=True)
                continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"Average Loss: {avg_loss:.4f}")
        
        # Validate
        try:
            val_acc = validate_basic_emotions(model, val_loader, processor, device)
            print(f"Validation Accuracy: {val_acc:.2%}")
        except Exception as e:
            print(f"Error during validation: {e}", flush=True)
            val_acc = 0.0
        
        # Save checkpoint
        try:
            if (epoch + 1) % 1 == 0 or val_acc > best_val_acc:
                checkpoint_dir = output_dir / f"epoch_{epoch+1}"
                checkpoint_dir.mkdir(exist_ok=True)
                model.save_pretrained(str(checkpoint_dir))
                processor.save_pretrained(str(checkpoint_dir))
                print(f"Saved checkpoint to {checkpoint_dir}")
                
                if val_acc > best_val_acc + early_stopping_min_delta:
                    best_val_acc = val_acc
                    epochs_without_improvement = 0
                    # Save as best model
                    best_dir = output_dir / "best_model"
                    best_dir.mkdir(exist_ok=True)
                    model.save_pretrained(str(best_dir))
                    processor.save_pretrained(str(best_dir))
                    print(f"New best model! Saved to {best_dir}")
                else:
                    epochs_without_improvement += 1
        except Exception as e:
            print(f"Error saving checkpoint: {e}", flush=True)
        
        # Early stopping
        if epochs_without_improvement >= early_stopping_patience:
            print(f"\nEarly stopping: No improvement for {early_stopping_patience} epochs")
            break
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2%}")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune CLIP on basic emotions (7-way classification)"
    )
    parser.add_argument(
        '--dataset_type',
        type=str,
        choices=['cam', 'eu_emotion'],
        required=True,
        help='Dataset type: cam or eu_emotion'
    )
    parser.add_argument(
        '--train_trials',
        type=str,
        required=True,
        help='Path to train trial definitions JSON file'
    )
    parser.add_argument(
        '--val_trials',
        type=str,
        required=True,
        help='Path to validation/test trial definitions JSON file'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='Root directory of video files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for fine-tuned model'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=20,
        help='Number of training epochs (default: 20)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size (default: 4 for CPU)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=5e-5,
        help='Learning rate (default: 5e-5)'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.01,
        help='Weight decay (default: 0.01)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to train on: auto (detect), cpu, cuda, or mps (default: auto)'
    )
    parser.add_argument(
        '--num_frames',
        type=int,
        default=16,
        help='Number of frames per video (default: 16)'
    )
    parser.add_argument(
        '--early_stopping_patience',
        type=int,
        default=5,
        help='Early stopping patience (default: 5)'
    )
    parser.add_argument(
        '--early_stopping_min_delta',
        type=float,
        default=0.001,
        help='Early stopping minimum delta (default: 0.001)'
    )
    
    args = parser.parse_args()
    
    # Auto-detect device if 'auto' or not specified
    if args.device == 'auto' or args.device is None:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
        elif torch.cuda.is_available():
            args.device = 'cuda'
        else:
            args.device = 'cpu'
        print(f"Auto-detected device: {args.device}")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = BasicEmotionDataset(
        data_root=args.data_root,
        trial_definitions_file=args.train_trials,
        num_frames=args.num_frames,
    )
    
    val_dataset = BasicEmotionDataset(
        data_root=args.data_root,
        trial_definitions_file=args.val_trials,
        num_frames=args.num_frames,
    )
    
    print(f"Train dataset: {len(train_dataset)} trials")
    print(f"Val dataset: {len(val_dataset)} trials")
    
    # Fine-tune
    print("\nStarting fine-tuning...")
    model = finetune_basic_emotions(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        num_frames=args.num_frames,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )
    
    print(f"\nFine-tuning complete! Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

