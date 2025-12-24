#!/usr/bin/env python3
"""
Fine-tune CLIP on emotion recognition data.

This script fine-tunes CLIP specifically for emotion recognition, which should
improve CAM performance from 37% (zero-shot) to 60-75% (fine-tuned).

Usage:
    # Fine-tune on CAM train split (best alignment)
    python experiments/cam_human_like/training/finetune_clip_emotions.py \
        --train_data data/splits/train.csv \
        --val_data data/splits/val.csv \
        --data_root "/path/to/cam/stimuli" \
        --output_dir models/clip_emotion_finetuned \
        --num_epochs 10

    # Fine-tune on FER2013 (if you have it)
    python experiments/cam_human_like/training/finetune_clip_emotions.py \
        --fer2013_dir fer2013/ \
        --output_dir models/clip_emotion_finetuned \
        --num_epochs 10
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.data.dataset import MindreadingDataset

# Import FER2013 and EU-Emotion datasets
sys.path.insert(0, str(Path(__file__).parent))
from fer2013_dataset import FER2013Dataset
from eu_emotion_dataset import EUEmotionDataset
from task_specific_dataset import TaskSpecificTrialDataset, collate_trial_batch


class EmotionCLIPDataset(Dataset):
    """
    Dataset for fine-tuning CLIP on emotion recognition.
    
    Adapts MindreadingDataset to work with CLIP fine-tuning:
    - Loads video frames
    - Returns (frames, emotion_label) pairs
    """
    
    def __init__(self, data_root, split_file, num_frames=8):
        self.base_dataset = MindreadingDataset(
            data_root=data_root,
            split_file=split_file,
            modality='V',  # Face trials only
            num_frames=num_frames,
        )
        self.num_frames = num_frames
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        
        # Get frames and emotion label
        frames = sample['frames']  # (num_frames, H, W, C)
        emotion = sample['emotion']
        
        # Convert frames to PIL Images
        frame_images = []
        for frame in frames:
            frame_images.append(Image.fromarray(frame))
        
        # Return all frames (not just middle frame)
        # The collate function will handle averaging or selection
        return frame_images, emotion


def collate_pil_images(batch, use_multiframe: bool = True):
    """
    Custom collate function that handles PIL Images and multiple frames.
    
    Each item in batch is (frames_list, emotion) where frames_list is a list of PIL Images.
    
    If use_multiframe=True: Returns all frames for each video (for multi-frame processing)
    If use_multiframe=False: Returns middle frame only (faster, single-frame processing)
    
    Returns:
        - images: List of lists of PIL Images (one list per video) OR list of PIL Images
        - emotions: List of emotion labels (repeated for each frame if use_multiframe)
    """
    frames_lists, emotions = zip(*batch)
    
    if use_multiframe:
        # Return all frames for each video
        # This allows averaging features across frames during training
        all_images = []
        all_emotions = []
        for frames_list, emotion in zip(frames_lists, emotions):
            if isinstance(frames_list, list):
                all_images.extend(frames_list)
                all_emotions.extend([emotion] * len(frames_list))
            elif isinstance(frames_list, Image.Image):
                all_images.append(frames_list)
                all_emotions.append(emotion)
        return all_images, all_emotions
    else:
        # Use middle frame only (faster, backward compatible)
        images = []
        for frames_list in frames_lists:
            if isinstance(frames_list, list) and len(frames_list) > 0:
                images.append(frames_list[len(frames_list) // 2])
            elif isinstance(frames_list, Image.Image):
                images.append(frames_list)
            else:
                images.append(frames_list[0] if len(frames_list) > 0 else None)
        images = [img for img in images if img is not None]
        return images, list(emotions)


def finetune_clip(
    train_dataset,
    val_dataset,
    model_name="openai/clip-vit-base-patch32",
    output_dir="models/clip_emotion_finetuned",
    num_epochs=10,
    batch_size=16,
    learning_rate=1e-5,
    device="cpu",
    save_every=2,
    use_multiframe=True,
):
    """
    Fine-tune CLIP on emotion recognition.
    
    Args:
        train_dataset: Dataset returning (image, emotion_label) tuples
        val_dataset: Validation dataset
        model_name: CLIP model to fine-tune
        output_dir: Directory to save fine-tuned model
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on
        save_every: Save checkpoint every N epochs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load CLIP
    print(f"Loading CLIP model: {model_name}...")
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model = model.to(device)
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Create collate function with multi-frame setting
    collate_fn = lambda batch: collate_pil_images(batch, use_multiframe=use_multiframe)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        collate_fn=collate_fn,  # Custom collate for PIL Images
    )
    
    best_val_acc = 0.0
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        for batch_idx, (images, emotion_labels) in enumerate(tqdm(train_loader, desc="Training")):
            # Images and emotion_labels are already lists from collate_pil_images
            # Ensure they're lists (should already be from collate function)
            if not isinstance(images, list):
                images = [images] if images is not None else []
            if not isinstance(emotion_labels, list):
                emotion_labels = list(emotion_labels) if emotion_labels is not None else []
            
            # Process images
            image_inputs = processor(images=images, return_tensors="pt").to(device)
            
            # Process text (emotion labels)
            text_inputs = processor(
                text=emotion_labels,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)
            
            # Get embeddings
            image_features = model.get_image_features(**image_inputs)
            text_features = model.get_text_features(**text_inputs)
            
            # If using multi-frame, average features for frames from the same video
            if use_multiframe and len(images) > len(set(emotion_labels)):
                # Group frames by emotion (assuming same emotion = same video)
                # Average features for frames with the same emotion in this batch
                unique_emotions = list(set(emotion_labels))
                emotion_to_indices = {emotion: [] for emotion in unique_emotions}
                for idx, emotion in enumerate(emotion_labels):
                    emotion_to_indices[emotion].append(idx)
                
                # Average features for each unique emotion
                averaged_features = []
                averaged_labels = []
                for emotion in unique_emotions:
                    indices = emotion_to_indices[emotion]
                    if len(indices) > 0:
                        # Average the features for frames from the same video
                        video_features = image_features[indices].mean(dim=0, keepdim=True)
                        averaged_features.append(video_features)
                        averaged_labels.append(emotion)
                
                if len(averaged_features) > 0:
                    image_features = torch.cat(averaged_features, dim=0)
                    emotion_labels = averaged_labels
                    # Re-process text for unique emotions only
                    text_inputs = processor(
                        text=emotion_labels,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    ).to(device)
                    text_features = model.get_text_features(**text_inputs)
            
            # Normalize
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            
            # Compute similarity (logits)
            logits_per_image = image_features @ text_features.t()
            logits_per_text = logits_per_image.t()
            
            # Labels: diagonal (image i matches text i)
            labels = torch.arange(len(image_features)).to(device)
            
            # Contrastive loss
            loss_img = nn.CrossEntropyLoss()(logits_per_image, labels)
            loss_txt = nn.CrossEntropyLoss()(logits_per_text, labels)
            loss = (loss_img + loss_txt) / 2
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"Average Loss: {avg_loss:.4f}")
        
        # Validate
        val_acc = validate(model, val_dataset, processor, device, batch_size, use_multiframe)
        print(f"Validation Accuracy: {val_acc:.2%}")
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0 or val_acc > best_val_acc:
            checkpoint_dir = output_dir / f"epoch_{epoch+1}"
            checkpoint_dir.mkdir(exist_ok=True)
            model.save_pretrained(str(checkpoint_dir))
            processor.save_pretrained(str(checkpoint_dir))
            print(f"Saved checkpoint to {checkpoint_dir}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # Save as best model
                best_dir = output_dir / "best_model"
                best_dir.mkdir(exist_ok=True)
                model.save_pretrained(str(best_dir))
                processor.save_pretrained(str(best_dir))
                print(f"New best model! Saved to {best_dir}")
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2%}")
    return model


def finetune_clip_task_specific(
    train_dataset,
    val_dataset,
    model_name="openai/clip-vit-base-patch32",
    output_dir="models/clip_emotion_finetuned",
    num_epochs=10,
    batch_size=8,
    learning_rate=1e-5,
    weight_decay=0.01,
    device="cpu",
    num_frames=8,
    use_lr_scheduler=True,
    warmup_steps=100,
):
    """
    Fine-tune CLIP for task-specific 4-option forced-choice emotion recognition.
    
    This function trains CLIP specifically for the 4-option forced-choice task:
    - Each video is paired with 4 candidate labels (1 correct + 3 foils)
    - Loss is cross-entropy over the 4 options (task-specific)
    - Multi-frame processing: extracts multiple frames, averages features
    
    Args:
        train_dataset: TaskSpecificTrialDataset returning (frames, candidate_labels, correct_idx)
        val_dataset: Validation dataset (same format)
        model_name: CLIP model to fine-tune
        output_dir: Directory to save fine-tuned model
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
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
    
    # Setup learning rate scheduler with warmup (after DataLoader is created)
    if use_lr_scheduler:
        # Calculate total steps
        total_steps = len(train_loader) * num_epochs
        # Cosine annealing with warmup
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=learning_rate * 0.01)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
        print(f"Using LR scheduler: warmup={warmup_steps} steps, then cosine annealing over {total_steps - warmup_steps} steps")
    else:
        scheduler = None
    
    best_val_acc = 0.0
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        for batch_idx, (frames_batch, candidate_labels_batch, correct_indices) in enumerate(tqdm(train_loader, desc="Training")):
            try:
                # frames_batch: List[List[PIL.Image]] - batch_size x num_frames
                # candidate_labels_batch: List[List[str]] - batch_size x 4
                # correct_indices: Tensor - batch_size
                
                batch_loss = 0
                batch_size_actual = len(frames_batch)
                
                # Process each video in the batch
                for video_idx in range(batch_size_actual):
                    try:
                        video_frames = frames_batch[video_idx]  # List of num_frames PIL Images
                        candidate_labels = candidate_labels_batch[video_idx]  # List of 4 strings
                        correct_idx = correct_indices[video_idx].item()  # Integer 0-3
                        
                        # Process all frames
                        image_inputs = processor(images=video_frames, return_tensors="pt").to(device)
                        
                        # Process all 4 candidate labels with prompt templates
                        # Better prompts help CLIP understand the emotion recognition task
                        # Use prompt template: "a photo of a person feeling [emotion]"
                        prompted_labels = [f"a photo of a person feeling {label}" for label in candidate_labels]
                        
                        text_inputs = processor(
                            text=prompted_labels,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                        ).to(device)
                        
                        # Get embeddings
                        image_features = model.get_image_features(**image_inputs)  # (num_frames, hidden_dim)
                        text_features = model.get_text_features(**text_inputs)  # (4, hidden_dim)
                        
                        # Aggregate video features: average across frames (multi-frame architecture)
                        video_features = image_features.mean(dim=0, keepdim=True)  # (1, hidden_dim)
                        
                        # Normalize
                        video_features = F.normalize(video_features, dim=-1)
                        text_features = F.normalize(text_features, dim=-1)
                        
                        # Compute similarity: video_features @ text_features.t() -> (1, 4)
                        logits = video_features @ text_features.t()  # (1, 4)
                        
                        # Cross-entropy loss over 4 options (task-specific)
                        target = torch.tensor([correct_idx], dtype=torch.long).to(device)
                        loss = nn.CrossEntropyLoss()(logits, target)
                        
                        batch_loss += loss
                    except Exception as e:
                        print(f"Error processing video {video_idx} in training batch {batch_idx}: {e}", flush=True)
                        import traceback
                        traceback.print_exc()
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
                print(f"Error during training batch {batch_idx} in epoch {epoch+1}: {e}", flush=True)
                import traceback
                traceback.print_exc()
                continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"Average Loss: {avg_loss:.4f}")
        
        # Validate
        try:
            val_acc = validate_task_specific(model, val_loader, processor, device)
            print(f"Validation Accuracy: {val_acc:.2%}")
        except Exception as e:
            print(f"Error during validation in epoch {epoch+1}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            val_acc = 0.0  # Set to 0 to avoid saving a bad model
        
        # Save checkpoint
        try:
            if (epoch + 1) % 1 == 0 or val_acc > best_val_acc:
                checkpoint_dir = output_dir / f"epoch_{epoch+1}"
                checkpoint_dir.mkdir(exist_ok=True)
                model.save_pretrained(str(checkpoint_dir))
                processor.save_pretrained(str(checkpoint_dir))
                print(f"Saved checkpoint to {checkpoint_dir}")
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    # Save as best model
                    best_dir = output_dir / "best_model"
                    best_dir.mkdir(exist_ok=True)
                    model.save_pretrained(str(best_dir))
                    processor.save_pretrained(str(best_dir))
                    print(f"New best model! Saved to {best_dir}")
        except Exception as e:
            print(f"Error saving checkpoint in epoch {epoch+1}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            # Continue training even if checkpoint save fails
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2%}")
    return model


def validate_task_specific(model, val_loader, processor, device):
    """Validate task-specific fine-tuned model."""
    model.eval()
    correct = 0
    total = 0
    errors = 0
    
    with torch.no_grad():
        try:
            for batch_idx, batch_data in enumerate(tqdm(val_loader, desc="Validating")):
                try:
                    frames_batch, candidate_labels_batch, correct_indices = batch_data
                    batch_size_actual = len(frames_batch)
                    
                    for video_idx in range(batch_size_actual):
                        try:
                            video_frames = frames_batch[video_idx]
                            candidate_labels = candidate_labels_batch[video_idx]
                            correct_idx = correct_indices[video_idx].item()
                            
                            # Process
                            image_inputs = processor(images=video_frames, return_tensors="pt").to(device)
                            
                            # Use prompt templates for better text understanding
                            # Use prompt template: "a photo of a person feeling [emotion]"
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
                            logits = video_features @ text_features.t()  # (1, 4)
                            
                            # Prediction: argmax over 4 options
                            predicted_idx = logits.argmax(dim=1).item()
                            
                            if predicted_idx == correct_idx:
                                correct += 1
                            total += 1
                        except Exception as e:
                            errors += 1
                            print(f"Warning: Error processing video {video_idx} in batch {batch_idx}: {e}", flush=True)
                            continue
                except Exception as e:
                    errors += 1
                    print(f"Warning: Error processing batch {batch_idx}: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    continue
        except Exception as e:
            print(f"Error in validation loop: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise
    
    if errors > 0:
        print(f"Warning: {errors} errors encountered during validation", flush=True)
    
    return correct / total if total > 0 else 0.0


def validate(model, val_dataset, processor, device, batch_size=32, use_multiframe=True):
    """Validate fine-tuned model."""
    model.eval()
    correct = 0
    total = 0
    
    # Create collate function with multi-frame setting
    collate_fn = lambda batch: collate_pil_images(batch, use_multiframe=use_multiframe)
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        collate_fn=collate_fn,  # Custom collate for PIL Images
    )
    
    with torch.no_grad():
        for images, emotion_labels in tqdm(val_loader, desc="Validating"):
            # Images and emotion_labels are already lists from collate_pil_images
            # Ensure they're lists (should already be from collate function)
            if not isinstance(images, list):
                images = [images] if images is not None else []
            if not isinstance(emotion_labels, list):
                emotion_labels = list(emotion_labels) if emotion_labels is not None else []
            
            # Process
            image_inputs = processor(images=images, return_tensors="pt").to(device)
            text_inputs = processor(
                text=emotion_labels,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)
            
            # Get embeddings
            image_features = model.get_image_features(**image_inputs)
            text_features = model.get_text_features(**text_inputs)
            
            # If using multi-frame, average features for frames from the same video
            if use_multiframe and len(images) > len(set(emotion_labels)):
                # Group frames by emotion (assuming same emotion = same video)
                unique_emotions = list(set(emotion_labels))
                emotion_to_indices = {emotion: [] for emotion in unique_emotions}
                for idx, emotion in enumerate(emotion_labels):
                    emotion_to_indices[emotion].append(idx)
                
                # Average features for each unique emotion
                averaged_features = []
                averaged_labels = []
                for emotion in unique_emotions:
                    indices = emotion_to_indices[emotion]
                    if len(indices) > 0:
                        video_features = image_features[indices].mean(dim=0, keepdim=True)
                        averaged_features.append(video_features)
                        averaged_labels.append(emotion)
                
                if len(averaged_features) > 0:
                    image_features = torch.cat(averaged_features, dim=0)
                    emotion_labels = averaged_labels
                    # Re-process text for unique emotions only
                    text_inputs = processor(
                        text=emotion_labels,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    ).to(device)
                    text_features = model.get_text_features(**text_inputs)
            
            # Normalize
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            
            # Compute similarity
            logits = image_features @ text_features.t()
            
            # Predictions: argmax over text labels
            predictions = logits.argmax(dim=1)
            labels = torch.arange(len(image_features)).to(device)
            
            correct += (predictions == labels).sum().item()
            total += len(image_features)
    
    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune CLIP on emotion recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fine-tune on EU-Emotion (external dataset - 20 emotions, best for CAM)
  python finetune_clip_emotions.py \\
      --eu_emotion_dir data/eu_emotion \\
      --output_dir models/clip_eu_emotion_finetuned \\
      --num_epochs 10

  # Fine-tune on FER2013 (external dataset - 7 basic emotions)
  python finetune_clip_emotions.py \\
      --fer2013_dir fer2013/ \\
      --output_dir models/clip_fer2013_finetuned \\
      --num_epochs 10

  # Fine-tune on CAM train split (task-specific - higher performance)
  python finetune_clip_emotions.py \\
      --train_data data/splits/train.csv \\
      --val_data data/splits/val.csv \\
      --data_root "/path/to/cam/stimuli" \\
      --output_dir models/clip_cam_finetuned \\
      --num_epochs 10

  # Two-stage: EU-Emotion → CAM (best performance + rigor)
  # Stage 1: python finetune_clip_emotions.py --eu_emotion_dir data/eu_emotion --output_dir models/clip_eu_emotion
  # Stage 2: python finetune_clip_emotions.py --train_data ... --model_name models/clip_eu_emotion/best_model
        """
    )
    parser.add_argument('--train_data', type=str, help='Path to CAM train split CSV')
    parser.add_argument('--val_data', type=str, help='Path to CAM val split CSV')
    parser.add_argument('--data_root', type=str, help='Root directory of CAM stimuli (required if using train_data)')
    parser.add_argument('--fer2013_dir', type=str, help='Path to FER2013 dataset directory (alternative to train_data)')
    parser.add_argument('--eu_emotion_dir', type=str, help='Path to EU-Emotion dataset directory (alternative to train_data)')
    parser.add_argument('--eu_emotion_modality', type=str, default='face', choices=['face', 'voice', 'body', 'all'], help='Modality to use for EU-Emotion (default: face)')
    parser.add_argument('--dataset_type', type=str, choices=['cam', 'eu_emotion'], help='Dataset type for task-specific training (requires --train_trials and --val_trials)')
    parser.add_argument('--train_trials', type=str, help='Path to train trial definitions JSON (for task-specific training)')
    parser.add_argument('--val_trials', type=str, help='Path to validation trial definitions JSON (for task-specific training)')
    parser.add_argument('--task_specific', action='store_true', help='Use task-specific 4-option forced-choice training')
    parser.add_argument('--output_dir', type=str, default='models/clip_emotion_finetuned', help='Output directory')
    parser.add_argument('--model_name', type=str, default='openai/clip-vit-base-patch32', help='CLIP model to fine-tune (or path to previously fine-tuned model)')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizer')
    parser.add_argument('--use_lr_scheduler', action='store_true', help='Use cosine annealing LR scheduler with warmup (default: True)')
    parser.add_argument('--warmup_steps', type=int, default=100, help='Number of warmup steps for LR scheduler')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu, cuda, mps)')
    parser.add_argument('--num_frames', type=int, default=8, help='Frames per video (for CAM dataset)')
    parser.add_argument('--use_multiframe', action='store_true', default=True, help='Use multiple frames per video (average features)')
    parser.add_argument('--single_frame', action='store_true', help='Use only middle frame (faster, disables multi-frame)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Handle single_frame flag (overrides use_multiframe)
    use_multiframe = args.use_multiframe and not args.single_frame
    
    # Validate arguments
    if args.task_specific:
        # Task-specific training requires trial definitions
        if not args.train_trials or not args.val_trials:
            parser.error("--train_trials and --val_trials are required for task-specific training")
        if not args.data_root:
            parser.error("--data_root is required for task-specific training")
        if not args.dataset_type:
            parser.error("--dataset_type is required for task-specific training")
    else:
        # Standard training
        dataset_count = sum([
            bool(args.train_data),
            bool(args.fer2013_dir),
            bool(args.eu_emotion_dir),
        ])
        
        if dataset_count == 0:
            parser.error("Must provide one of: --train_data (CAM), --fer2013_dir (FER2013), or --eu_emotion_dir (EU-Emotion)")
        
        if dataset_count > 1:
            parser.error("Can only specify one dataset: --train_data, --fer2013_dir, or --eu_emotion_dir")
        
        if args.train_data and not args.data_root:
            parser.error("--data_root is required when using --train_data")
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create datasets
    print("Loading datasets...")
    if args.task_specific:
        # Task-specific training with trial definitions
        print(f"Using task-specific 4-option forced-choice training")
        print(f"Dataset type: {args.dataset_type}")
        print(f"Train trials: {args.train_trials}")
        print(f"Val trials: {args.val_trials}")
        
        train_dataset = TaskSpecificTrialDataset(
            data_root=args.data_root,
            trial_definitions_file=args.train_trials,
            num_frames=args.num_frames,
        )
        val_dataset = TaskSpecificTrialDataset(
            data_root=args.data_root,
            trial_definitions_file=args.val_trials,
            num_frames=args.num_frames,
        )
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        
        # Fine-tune with task-specific method
        print(f"\nMulti-frame processing: Enabled ({args.num_frames} frames per video)")
        model = finetune_clip_task_specific(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            model_name=args.model_name,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=args.device,
            num_frames=args.num_frames,
            use_lr_scheduler=getattr(args, 'use_lr_scheduler', True),
            warmup_steps=args.warmup_steps,
        )
        
        print(f"\nFine-tuning complete! Model saved to {args.output_dir}")
        return
    
    if args.eu_emotion_dir:
        # EU-Emotion dataset (external, 20 emotions - best match for CAM)
        print("Using EU-Emotion dataset (external emotion recognition dataset)")
        print(f"  Modality: {args.eu_emotion_modality}")
        print("  Note: 20 complex emotions - perfect match for CAM's 20 concepts")
        print("  Note: This is more rigorous - no data leakage with CAM test set")
        train_dataset = EUEmotionDataset(args.eu_emotion_dir, split='train', modality=args.eu_emotion_modality, num_frames=args.num_frames)
        val_dataset = EUEmotionDataset(args.eu_emotion_dir, split='test', modality=args.eu_emotion_modality, num_frames=args.num_frames)
    elif args.fer2013_dir:
        # FER2013 dataset (external, no overlap with CAM)
        print("Using FER2013 dataset (external emotion recognition dataset)")
        train_dataset = FER2013Dataset(args.fer2013_dir, split='train')
        val_dataset = FER2013Dataset(args.fer2013_dir, split='test')
        print("Note: This is more rigorous - no data leakage with CAM test set")
    else:
        # CAM dataset (task-specific fine-tuning)
        print("Using CAM train split (task-specific fine-tuning)")
        print("Note: This gives better performance but model sees CAM data during training")
        train_dataset = EmotionCLIPDataset(args.data_root, args.train_data, args.num_frames)
        val_dataset = EmotionCLIPDataset(args.data_root, args.val_data, args.num_frames)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Fine-tune
    print(f"\nMulti-frame processing: {'Enabled' if use_multiframe else 'Disabled (using middle frame only)'}")
    model = finetune_clip(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        use_multiframe=use_multiframe,
    )
    
    print(f"\nFine-tuning complete! Model saved to {args.output_dir}")
    print(f"Use this model in CAM experiment by setting:")
    print(f'  model.name: "{args.output_dir}/best_model"')


if __name__ == "__main__":
    import sys
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

