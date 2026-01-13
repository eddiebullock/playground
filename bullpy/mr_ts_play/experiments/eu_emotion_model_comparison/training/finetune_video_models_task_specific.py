#!/usr/bin/env python3
"""
Fine-tune video models (I3D, TimeSformer) on EU-Emotion dataset using TASK-SPECIFIC approach.

This matches how CLIP was fine-tuned - using 4-option forced-choice format.
Each video is paired with 4 candidate labels (1 correct + 3 foils).
Loss is cross-entropy over the 4 options (task-specific).

Usage:
    # Fine-tune TimeSformer (task-specific)
    python experiments/eu_emotion_model_comparison/training/finetune_video_models_task_specific.py \
        --model timesformer \
        --train_trials data/trial_definitions/eu_emotion_train.json \
        --val_trials data/trial_definitions/eu_emotion_val.json \
        --data_root /path/to/EU_emotions \
        --output_dir models/timesformer_emotion_finetuned \
        --num_epochs 20 \
        --batch_size 4 \
        --learning_rate 1e-4

    # Fine-tune I3D (task-specific)
    python experiments/eu_emotion_model_comparison/training/finetune_video_models_task_specific.py \
        --model i3d \
        --train_trials data/trial_definitions/eu_emotion_train.json \
        --val_trials data/trial_definitions/eu_emotion_val.json \
        --data_root /path/to/EU_emotions \
        --output_dir models/i3d_emotion_finetuned \
        --num_epochs 20 \
        --batch_size 4 \
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
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskSpecificVideoDataset(Dataset):
    """
    Task-specific dataset for 4-option forced-choice fine-tuning of video models.
    
    This matches the evaluation format: each video has 4 candidate labels.
    Returns: (frames, candidate_labels, correct_idx, emotion_weight)
    """
    
    def __init__(
        self,
        trial_file: str,
        data_root: str,
        num_frames: int = 16,
        frame_sampling: str = "uniform",
    ):
        """
        Args:
            trial_file: Path to JSON file with trials (must have candidate_labels)
            data_root: Root directory of EU-Emotion dataset
            num_frames: Number of frames to extract per video
            frame_sampling: "uniform", "temporal", or "keyframe"
        """
        with open(trial_file, 'r') as f:
            data = json.load(f)
        self.trials = data['trials']
        self.data_root = Path(data_root)
        self.num_frames = num_frames
        self.frame_sampling = frame_sampling
        
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
            emotion: total / count if count > 0 else 1.0
            for emotion, count in emotion_counts.items()
        }
        # Normalize weights
        max_weight = max(self.emotion_weights.values()) if self.emotion_weights else 1.0
        self.emotion_weights = {
            emotion: weight / max_weight
            for emotion, weight in self.emotion_weights.items()
        }
    
    def _extract_frames(self, video_path: Path) -> list:
        """Extract frames from video (called in __getitem__ for parallelization)."""
        try:
            frames = extract_frames_for_video(str(video_path), num_frames=self.num_frames, sampling=self.frame_sampling)
            return frames
        except Exception as e:
            logger.warning(f"Could not extract frames from {video_path}: {e}")
            return []
    
    def __len__(self):
        return len(self.trials)
    
    def __getitem__(self, idx):
        trial = self.trials[idx]
        
        # Get video path
        video_path = trial['stimulus_path']
        full_path = self.data_root / video_path
        
        # Extract frames (this will be parallelized by DataLoader workers)
        frames = self._extract_frames(full_path)
        
        # Get candidate labels (should be in trial)
        if 'candidate_labels' in trial:
            candidate_labels = trial['candidate_labels']
        else:
            # Generate candidate labels if not present
            correct_label = trial.get('correct_label', trial.get('emotion', 'unknown'))
            candidate_labels = [correct_label]
            # Add 3 random foils
            foils = [e for e in self.all_emotions if e != correct_label]
            random.shuffle(foils)
            candidate_labels.extend(foils[:3])
            random.shuffle(candidate_labels)
        
        # Find correct index
        correct_label = trial.get('correct_label', trial.get('emotion', 'unknown'))
        correct_idx = candidate_labels.index(correct_label) if correct_label in candidate_labels else 0
        
        # Get emotion weight
        emotion_weight = self.emotion_weights.get(correct_label, 1.0)
        
        return {
            'frames': frames,  # Pre-extracted frames
            'candidate_labels': candidate_labels,
            'correct_idx': correct_idx,
            'emotion_weight': emotion_weight,
        }


def extract_frames_for_video(video_path: str, num_frames: int = 16, sampling: str = "uniform"):
    """Extract frames from video for video models."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return []
    
    # Avoid very first and last frames (often black)
    skip_start = max(1, int(total_frames * 0.02))
    skip_end = max(1, int(total_frames * 0.02))
    usable_start = skip_start
    usable_end = total_frames - skip_end - 1
    
    if usable_end <= usable_start:
        usable_start = 0
        usable_end = total_frames - 1
    
    frames = []
    
    if sampling == "uniform":
        frame_indices = np.linspace(usable_start, usable_end, num_frames, dtype=int)
    elif sampling == "temporal":
        if num_frames >= 3:
            mid_point = (usable_start + usable_end) // 2
            indices = [usable_start, mid_point, usable_end]
            if num_frames > 3:
                extra = num_frames - 3
                step = (usable_end - usable_start) // (extra + 1)
                indices.extend([usable_start + i * step for i in range(1, extra + 1)])
            frame_indices = sorted(indices[:num_frames])
        else:
            frame_indices = np.linspace(usable_start, usable_end, num_frames, dtype=int)
    else:
        frame_indices = np.linspace(usable_start, usable_end, num_frames, dtype=int)
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    
    cap.release()
    return frames[:num_frames]


def collate_video_batch(batch):
    """Collate function for video batch."""
    frames_batch = [item['frames'] for item in batch]
    candidate_labels = [item['candidate_labels'] for item in batch]
    correct_indices = [item['correct_idx'] for item in batch]
    emotion_weights = [item['emotion_weight'] for item in batch]
    
    return (
        frames_batch,  # List of frame lists (already extracted)
        candidate_labels,
        torch.tensor(correct_indices, dtype=torch.long),
        torch.tensor(emotion_weights, dtype=torch.float)
    )


def create_video_model(model_type: str, device: str):
    """Create video model for task-specific fine-tuning."""
    if model_type == "i3d":
        try:
            import pytorchvideo
            from pytorchvideo.models.hub import i3d_r50
            model = i3d_r50(pretrained=True)
            # Remove final classification layer - I3D structure varies, try common patterns
            if hasattr(model, 'blocks') and len(model.blocks) > 0:
                if hasattr(model.blocks[-1], 'proj'):
                    model.blocks[-1].proj = nn.Identity()
            elif hasattr(model, 'head'):
                model.head = nn.Identity()
            model = model.to(device)
            model.train()
            return model, False
        except ImportError:
            logger.error("pytorchvideo not installed. Install with: pip install pytorchvideo")
            raise
        except Exception as e:
            logger.error(f"Failed to load I3D model: {e}")
            raise
    
    elif model_type == "timesformer":
        try:
            from transformers import TimesformerForVideoClassification, VideoMAEImageProcessor
            model_name = "facebook/timesformer-base-finetuned-k400"
            model = TimesformerForVideoClassification.from_pretrained(model_name)
            # Replace classifier for task-specific scoring
            # We'll use the model's features and add our own projection
            model.classifier = nn.Identity()
            processor = VideoMAEImageProcessor.from_pretrained(model_name)
            model = model.to(device)
            model.train()
            return (model, processor), True
        except ImportError:
            logger.error("transformers not installed or version doesn't support TimeSformer")
            raise
        except Exception as e:
            logger.error(f"Failed to load TimeSformer model: {e}")
            raise
    
    else:
        raise ValueError(f"Unknown video model type: {model_type}")


def get_model_feature_dim(model_type: str, model):
    """Get feature dimension for a video model."""
    if model_type == "i3d":
        # I3D ResNet-50 feature dimension
        return 2048
    elif model_type == "timesformer":
        # TimeSformer base feature dimension
        return 768
    else:
        return 512


def preprocess_i3d_video(frames: list, device: str):
    """
    Preprocess frames for I3D model.
    
    I3D expects: (B, C, T, H, W) tensor with values in [0, 1]
    """
    if not frames:
        return None
    
    # Resize frames to 224x224
    frames_resized = []
    for frame in frames:
        frame_pil = Image.fromarray(frame)
        frame_pil = frame_pil.resize((224, 224))
        frame_np = np.array(frame_pil).astype(np.float32) / 255.0
        # Convert to (H, W, C) -> (C, H, W)
        frame_np = frame_np.transpose(2, 0, 1)
        frames_resized.append(frame_np)
    
    # Stack to (T, C, H, W)
    video_tensor = np.stack(frames_resized, axis=0)
    # Convert to (1, C, T, H, W) for batch
    video_tensor = video_tensor.transpose(1, 0, 2, 3)[np.newaxis, :, :, :, :]
    
    return torch.from_numpy(video_tensor).float().to(device)


def train_epoch_video(
    model, train_loader, criterion, optimizer, device, model_type: str, score_proj: nn.Module, 
    processor=None, use_amp=False, scaler=None, gradient_accumulation_steps=1
):
    """Train for one epoch (task-specific video models) with optimizations."""
    model.train()
    score_proj.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    accumulation_loss = 0.0
    steps_done = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (frames_batch, candidate_labels_batch, correct_indices, emotion_weights) in enumerate(pbar):
        batch_loss = 0
        batch_size_actual = len(frames_batch)
        
        # Process each video in the batch
        for video_idx in range(batch_size_actual):
            frames = frames_batch[video_idx]  # Frames already extracted by dataset
            correct_idx = correct_indices[video_idx].item()
            
            try:
                if not frames:
                    continue
                
                # Process video based on model type with mixed precision
                if model_type == "i3d":
                    video_tensor = preprocess_i3d_video(frames, device)
                    if video_tensor is None:
                        continue
                    
                    # Forward pass through I3D with mixed precision
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            features = model(video_tensor)  # [B, feature_dim]
                            video_features = features.squeeze(0)  # [feature_dim]
                            emotion_scores = score_proj(video_features.unsqueeze(0))  # [1, 4]
                    else:
                        features = model(video_tensor)
                        video_features = features.squeeze(0)
                        emotion_scores = score_proj(video_features.unsqueeze(0))
                
                elif model_type == "timesformer":
                    # Process frames with TimeSformer processor
                    inputs = processor(images=frames, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # Forward pass through TimeSformer with mixed precision
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = model(**inputs, output_hidden_states=True)
                            hidden_states = outputs.hidden_states[-1]  # [B, T, D]
                            video_features = hidden_states.mean(dim=1).squeeze(0)  # [D]
                            emotion_scores = score_proj(video_features.unsqueeze(0))  # [1, 4]
                    else:
                        outputs = model(**inputs, output_hidden_states=True)
                        hidden_states = outputs.hidden_states[-1]
                        video_features = hidden_states.mean(dim=1).squeeze(0)
                        emotion_scores = score_proj(video_features.unsqueeze(0))
                
                else:
                    continue
                
                # Calculate accuracy
                predicted_idx = emotion_scores.argmax(dim=1).item()
                if predicted_idx == correct_idx:
                    correct += 1
                total += 1
                
                # Cross-entropy loss over 4 options
                target = torch.tensor([correct_idx], dtype=torch.long, device=device)
                loss = criterion(emotion_scores, target)
                # Apply emotion weight and normalize by accumulation steps
                emotion_weight = emotion_weights[video_idx].item()
                loss = loss * emotion_weight / gradient_accumulation_steps
                accumulation_loss += loss.item()
                batch_loss += loss
            
            except Exception as e:
                logger.warning(f"Error processing video in batch: {e}")
                continue
        
        # Gradient accumulation: backprop every N steps
        if batch_size_actual > 0 and total > 0 and batch_loss > 0:
            if use_amp:
                scaler.scale(batch_loss).backward()
            else:
                batch_loss.backward()
            
            # Step optimizer every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(score_proj.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(score_proj.parameters(), max_norm=1.0)
                    optimizer.step()
                
                optimizer.zero_grad()
                steps_done += 1
            
            train_loss += accumulation_loss
            pbar.set_postfix({
                'loss': accumulation_loss,
                'acc': f'{100*correct/total:.1f}%' if total > 0 else '0%',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
            accumulation_loss = 0.0
    
    return train_loss / len(train_loader) if len(train_loader) > 0 else 0.0, 100 * correct / total if total > 0 else 0.0, steps_done


def validate_video(model, val_loader, device, model_type: str, score_proj: nn.Module, processor=None):
    """Validate video model (task-specific)."""
    model.eval()
    score_proj.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for frames_batch, candidate_labels_batch, correct_indices, emotion_weights in tqdm(val_loader, desc="Validation"):
            batch_size_actual = len(frames_batch)
            
            for video_idx in range(batch_size_actual):
                frames = frames_batch[video_idx]  # Frames already extracted by dataset
                correct_idx = correct_indices[video_idx].item()
                
                try:
                    if not frames:
                        continue
                    
                    # Process video based on model type
                    if model_type == "i3d":
                        video_tensor = preprocess_i3d_video(frames, device)
                        if video_tensor is None:
                            continue
                        features = model(video_tensor)
                        video_features = features.squeeze(0)
                    
                    elif model_type == "timesformer":
                        inputs = processor(images=frames, return_tensors="pt")
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        outputs = model(**inputs, output_hidden_states=True)
                        hidden_states = outputs.hidden_states[-1]
                        video_features = hidden_states.mean(dim=1).squeeze(0)
                    
                    else:
                        continue
                    
                    # Project to 4 scores
                    emotion_scores = score_proj(video_features.unsqueeze(0))
                    
                    # Calculate accuracy
                    predicted_idx = emotion_scores.argmax(dim=1).item()
                    if predicted_idx == correct_idx:
                        correct += 1
                    total += 1
                
                except Exception as e:
                    logger.warning(f"Error in validation: {e}")
                    continue
    
    return 100 * correct / total if total > 0 else 0.0


def finetune_video_model_task_specific(
    model_type: str,
    train_trials: str,
    val_trials: str,
    data_root: str,
    output_dir: str,
    num_epochs: int = 12,
    batch_size: int = 8,
    learning_rate: float = 2e-4,
    device: str = "auto",
    num_frames: int = 16,
    frame_sampling: str = "uniform",
    use_mixed_precision: bool = True,
    early_stopping_patience: int = 4,
    gradient_accumulation_steps: int = 1,
    num_workers: int = 2,
    validate_every_n_epochs: int = 1,
):
    """
    Fine-tune video model using task-specific approach (4-option forced-choice).
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
    logger.info("Loading task-specific video datasets...")
    train_dataset = TaskSpecificVideoDataset(
        train_trials, data_root, num_frames=num_frames, frame_sampling=frame_sampling
    )
    val_dataset = TaskSpecificVideoDataset(
        val_trials, data_root, num_frames=num_frames, frame_sampling=frame_sampling
    )
    
    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_video_batch,
        pin_memory=True if device.type == "cuda" else False,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_video_batch,
        pin_memory=True if device.type == "cuda" else False,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    # Create model
    logger.info(f"Creating {model_type} model...")
    model_result = create_video_model(model_type, device)
    if isinstance(model_result, tuple) and len(model_result) == 2:
        # TimeSformer case: returns ((model, processor), True)
        if model_result[1]:  # use_processor is True
            model_tuple, _ = model_result
            model, processor = model_tuple
        else:
            # I3D case: returns (model, False)
            model = model_result[0]
            processor = None
    else:
        model = model_result
        processor = None
    
    # Create feature projection to 4 scores
    feature_dim = get_model_feature_dim(model_type, model)
    score_proj = nn.Linear(feature_dim, 4).to(device)
    
    # Setup training
    # Get actual model for optimizer
    actual_model = model
    optimizer = torch.optim.AdamW(
        [
            {'params': actual_model.parameters()},
            {'params': score_proj.parameters()}
        ],
        lr=learning_rate,
        weight_decay=0.01
    )
    criterion = nn.CrossEntropyLoss()
    
    # Use cosine annealing with warmup for faster convergence
    # Calculate total steps properly
    steps_per_epoch = len(train_loader) // gradient_accumulation_steps
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = min(100, total_steps // 10)
    
    # Use ReduceLROnPlateau for simplicity (works better with early stopping)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    
    # Mixed precision training for speed (only on CUDA)
    use_amp = use_mixed_precision and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        logger.info("Using mixed precision training (AMP)")
    elif use_mixed_precision and device.type != "cuda":
        logger.info("Mixed precision only supported on CUDA, using FP32")
    
    # Early stopping
    best_val_acc = 0.0
    patience_counter = 0
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting task-specific fine-tuning for {num_epochs} epochs...")
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    logger.info(f"Batch size: {batch_size}, Gradient accumulation: {gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    
    global_step = 0
    
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        logger.info("-" * 60)
        
        # Train
        train_loss, train_acc, steps_done = train_epoch_video(
            model, train_loader, criterion, optimizer, device, model_type, score_proj, processor,
            use_amp=use_amp, scaler=scaler, gradient_accumulation_steps=gradient_accumulation_steps
        )
        
        global_step += steps_done
        
        # Validate (can skip some epochs for speed)
        if (epoch + 1) % validate_every_n_epochs == 0 or epoch == 0:
            val_acc = validate_video(model, val_loader, device, model_type, score_proj, processor)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Acc: {val_acc:.2f}%")
            
            # Learning rate scheduling
            scheduler.step(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                save_dict = {
                    'epoch': epoch + 1,
                    'score_proj_state_dict': score_proj.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'model_type': model_type,
                }
                # Save model state
                save_dict['model_state_dict'] = model.state_dict()
                
                torch.save(save_dict, output_dir / "best_model.pth")
                logger.info(f"✅ Saved best model (Val Acc: {val_acc:.2f}%)")
            else:
                patience_counter += 1
                logger.info(f"No improvement ({patience_counter}/{early_stopping_patience})")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        else:
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info("Skipping validation this epoch")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training completed!")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    logger.info(f"Model saved to: {output_dir / 'best_model.pth'}")
    logger.info(f"{'='*60}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Fine-tune video models (task-specific)")
    parser.add_argument('--model', type=str, required=True,
                       choices=['i3d', 'timesformer'],
                       help='Video model to fine-tune')
    parser.add_argument('--train_trials', type=str, required=True,
                       help='Path to train trials JSON file')
    parser.add_argument('--val_trials', type=str, required=True,
                       help='Path to validation trials JSON file')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of EU-Emotion dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for fine-tuned model')
    parser.add_argument('--num_epochs', type=int, default=12,
                       help='Number of training epochs (reduced for speed)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size (increased for speed)')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                       help='Learning rate (increased for faster convergence)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to train on')
    parser.add_argument('--num_frames', type=int, default=16,
                       help='Number of frames to extract per video')
    parser.add_argument('--frame_sampling', type=str, default='uniform',
                       choices=['uniform', 'temporal', 'keyframe'],
                       help='Frame sampling strategy')
    parser.add_argument('--use_mixed_precision', action='store_true', default=True,
                       help='Use mixed precision training (faster on GPU)')
    parser.add_argument('--early_stopping_patience', type=int, default=4,
                       help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Gradient accumulation steps (effective batch size = batch_size * this)')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of data loading workers')
    parser.add_argument('--validate_every_n_epochs', type=int, default=1,
                       help='Validate every N epochs (1 = every epoch)')
    
    args = parser.parse_args()
    
    finetune_video_model_task_specific(
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
        use_mixed_precision=args.use_mixed_precision,
        early_stopping_patience=args.early_stopping_patience,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_workers=args.num_workers,
        validate_every_n_epochs=args.validate_every_n_epochs,
    )


if __name__ == "__main__":
    main()
