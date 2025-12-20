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

# Import FER2013 dataset
sys.path.insert(0, str(Path(__file__).parent))
from fer2013_dataset import FER2013Dataset


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
        
        # Use middle frame as representative (or average all frames)
        # For simplicity, use middle frame
        main_frame = frame_images[len(frame_images) // 2]
        
        return main_frame, emotion


def collate_pil_images(batch):
    """
    Custom collate function that handles PIL Images.
    
    Returns images as a list (not tensor) so CLIP processor can handle them.
    """
    images, emotions = zip(*batch)
    # Keep images as list of PIL Images (CLIP processor expects this)
    # Convert emotions to list
    return list(images), list(emotions)


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
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        collate_fn=collate_pil_images,  # Custom collate for PIL Images
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
            
            # Normalize
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            
            # Compute similarity (logits)
            logits_per_image = image_features @ text_features.t()
            logits_per_text = logits_per_image.t()
            
            # Labels: diagonal (image i matches text i)
            labels = torch.arange(len(images)).to(device)
            
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
        val_acc = validate(model, val_dataset, processor, device, batch_size)
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


def validate(model, val_dataset, processor, device, batch_size=32):
    """Validate fine-tuned model."""
    model.eval()
    correct = 0
    total = 0
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        collate_fn=collate_pil_images,  # Custom collate for PIL Images
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
            
            # Normalize
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            
            # Compute similarity
            logits = image_features @ text_features.t()
            
            # Predictions: argmax over text labels
            predictions = logits.argmax(dim=1)
            labels = torch.arange(len(images)).to(device)
            
            correct += (predictions == labels).sum().item()
            total += len(images)
    
    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune CLIP on emotion recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fine-tune on FER2013 (external dataset - recommended for rigor)
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

  # Two-stage: FER2013 then CAM (best performance)
  # Stage 1: python finetune_clip_emotions.py --fer2013_dir fer2013/ --output_dir models/clip_fer2013
  # Stage 2: python finetune_clip_emotions.py --train_data ... --model_name models/clip_fer2013/best_model
        """
    )
    parser.add_argument('--train_data', type=str, help='Path to CAM train split CSV')
    parser.add_argument('--val_data', type=str, help='Path to CAM val split CSV')
    parser.add_argument('--data_root', type=str, help='Root directory of CAM stimuli (required if using train_data)')
    parser.add_argument('--fer2013_dir', type=str, help='Path to FER2013 dataset directory (alternative to train_data)')
    parser.add_argument('--output_dir', type=str, default='models/clip_emotion_finetuned', help='Output directory')
    parser.add_argument('--model_name', type=str, default='openai/clip-vit-base-patch32', help='CLIP model to fine-tune (or path to previously fine-tuned model)')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu, cuda, mps)')
    parser.add_argument('--num_frames', type=int, default=8, help='Frames per video (for CAM dataset)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.train_data and not args.fer2013_dir:
        parser.error("Must provide either --train_data (for CAM) or --fer2013_dir (for FER2013)")
    
    if args.train_data and not args.data_root:
        parser.error("--data_root is required when using --train_data")
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create datasets
    print("Loading datasets...")
    if args.fer2013_dir:
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
    model = finetune_clip(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    
    print(f"\nFine-tuning complete! Model saved to {args.output_dir}")
    print(f"Use this model in CAM experiment by setting:")
    print(f'  model.name: "{args.output_dir}/best_model"')


if __name__ == "__main__":
    main()

