#!/usr/bin/env python3
"""
Fine-tune audio models (Wav2Vec2, Whisper) on EU-Emotion dataset using TASK-SPECIFIC approach.

This matches how vision models were fine-tuned - using 4-option forced-choice format.
Each audio file is paired with 4 candidate labels (1 correct + 3 foils).
Loss is cross-entropy over the 4 options (task-specific).

Usage:
    # Fine-tune Wav2Vec2-base (task-specific)
    python experiments/eu_emotion_audio_model_comparison/training/finetune_audio_models_task_specific.py \
        --model wav2vec2_base \
        --train_trials data/trial_definitions/eu_emotion_audio_train.json \
        --val_trials data/trial_definitions/eu_emotion_audio_val.json \
        --data_root /path/to/EU_emotions \
        --output_dir models/wav2vec2_emotion_finetuned \
        --num_epochs 20 \
        --batch_size 8 \
        --learning_rate 1e-4
"""

import argparse
import json
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.eu_emotion_audio_model_comparison.training.task_specific_audio_dataset import (
    TaskSpecificAudioDataset,
    collate_audio_batch,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_audio_model(model_type: str, device: str):
    """
    Create audio model for task-specific fine-tuning.
    
    Returns:
        model: Audio encoder model
        processor: Audio processor
        feature_dim: Dimension of audio embeddings
    """
    from transformers import Wav2Vec2Model, Wav2Vec2Processor, WhisperModel, WhisperProcessor
    
    if model_type == "wav2vec2_base":
        model_name = "facebook/wav2vec2-base"
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2Model.from_pretrained(model_name)
        feature_dim = 768  # Wav2Vec2-base output dimension
        return model.to(device), processor, feature_dim
    
    elif model_type == "wav2vec2_large":
        model_name = "facebook/wav2vec2-large"
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2Model.from_pretrained(model_name)
        feature_dim = 1024  # Wav2Vec2-large output dimension
        return model.to(device), processor, feature_dim
    
    elif model_type == "whisper_base":
        model_name = "openai/whisper-base"
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperModel.from_pretrained(model_name)
        feature_dim = 512  # Whisper-base encoder output dimension
        return model.to(device), processor, feature_dim
    
    elif model_type == "whisper_tiny":
        model_name = "openai/whisper-tiny"
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperModel.from_pretrained(model_name)
        feature_dim = 384  # Whisper-tiny encoder output dimension
        return model.to(device), processor, feature_dim
    
    elif model_type == "whisper_small":
        model_name = "openai/whisper-small"
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperModel.from_pretrained(model_name)
        feature_dim = 768  # Whisper-small encoder output dimension
        return model.to(device), processor, feature_dim
    
    else:
        raise ValueError(f"Unknown audio model type: {model_type}")


def extract_audio_features(model, processor, waveform: np.ndarray, model_type: str, device: str, training: bool = True) -> torch.Tensor:
    """
    Extract audio features from waveform.
    
    Args:
        model: Audio model
        processor: Audio processor
        waveform: Audio waveform as numpy array
        model_type: Type of model
        device: Device to run on
        training: Whether in training mode (affects gradient computation)
    """
    if model_type.startswith("wav2vec2"):
        # Process with Wav2Vec2
        inputs = processor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Don't use no_grad during training - we need gradients!
        if training:
            outputs = model(**inputs)
        else:
            with torch.no_grad():
                outputs = model(**inputs)
        
        # Use mean pooling over sequence dimension
        audio_features = outputs.last_hidden_state.mean(dim=1)  # [1, feature_dim]
        return audio_features.squeeze(0)  # [feature_dim]
    
    elif model_type.startswith("whisper"):
        # Process with Whisper
        inputs = processor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Don't use no_grad during training
        if training:
            encoder_outputs = model.encoder(**inputs)
        else:
            with torch.no_grad():
                encoder_outputs = model.encoder(**inputs)
        
        # Use mean pooling over sequence dimension
        audio_features = encoder_outputs.last_hidden_state.mean(dim=1)  # [1, feature_dim]
        return audio_features.squeeze(0)  # [feature_dim]
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_epoch_task_specific(
    model,
    processor,
    train_loader,
    criterion,
    optimizer,
    device,
    model_type: str,
    score_proj: nn.Module,
):
    """Train for one epoch (task-specific)."""
    model.train()
    score_proj.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for waveforms_batch, candidate_labels_batch, correct_indices, emotion_weights in pbar:
        batch_loss = 0
        batch_size_actual = len(waveforms_batch)
        
        # Process each audio in the batch
        for audio_idx in range(batch_size_actual):
            waveform = waveforms_batch[audio_idx]  # numpy array
            correct_idx = correct_indices[audio_idx].item()  # Integer 0-3
            
            try:
                # Extract audio features (training=True to enable gradients)
                audio_features = extract_audio_features(
                    model, processor, waveform, model_type, device, training=True
                )  # [feature_dim]
                
                # Project audio features directly to 4 scores (one per candidate option)
                emotion_scores = score_proj(audio_features.unsqueeze(0))  # [1, 4]
                
                # Calculate accuracy (before backprop)
                predicted_idx = emotion_scores.argmax(dim=1).item()
                if predicted_idx == correct_idx:
                    correct += 1
                total += 1
                
                # Cross-entropy loss over 4 options with class weighting
                target = torch.tensor([correct_idx], dtype=torch.long, device=device)
                loss = criterion(emotion_scores, target)
                # Apply emotion weight (higher weight for rare classes)
                emotion_weight = emotion_weights[audio_idx].item()
                batch_loss += loss * emotion_weight
            
            except Exception as e:
                logger.warning(f"Error processing audio {audio_idx}: {e}")
                continue
        
        # Average loss and backprop
        if batch_size_actual > 0 and batch_loss > 0:
            avg_batch_loss = batch_loss / batch_size_actual
            optimizer.zero_grad()
            avg_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(score_proj.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += avg_batch_loss.item()
    
    return train_loss / len(train_loader) if len(train_loader) > 0 else 0.0, 100 * correct / total if total > 0 else 0.0


def validate_task_specific(
    model,
    processor,
    val_loader,
    device,
    model_type: str,
    score_proj: nn.Module,
):
    """Validate model (task-specific)."""
    model.eval()
    score_proj.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for waveforms_batch, candidate_labels_batch, correct_indices, emotion_weights in pbar:
            batch_size_actual = len(waveforms_batch)
            
            for audio_idx in range(batch_size_actual):
                try:
                    waveform = waveforms_batch[audio_idx]
                    correct_idx = correct_indices[audio_idx].item()
                    
                    # Extract audio features (training=False for validation)
                    audio_features = extract_audio_features(
                        model, processor, waveform, model_type, device, training=False
                    )
                    
                    # Project to 4 scores
                    emotion_scores = score_proj(audio_features.unsqueeze(0))  # [1, 4]
                    predicted_idx = emotion_scores.argmax(dim=1).item()
                    
                    if predicted_idx == correct_idx:
                        correct += 1
                    total += 1
                
                except Exception as e:
                    logger.warning(f"Error processing audio {audio_idx}: {e}")
                    continue
    
    return 100 * correct / total if total > 0 else 0.0


def finetune_audio_model_task_specific(
    model_type: str,
    train_trials: str,
    val_trials: str,
    data_root: str,
    output_dir: str,
    num_epochs: int = 20,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    device: str = "auto",
    use_lr_scheduler: bool = True,
    warmup_steps: int = 100,
    early_stopping_patience: int = 5,
):
    """
    Fine-tune audio model for task-specific 4-option forced-choice emotion recognition.
    
    Args:
        model_type: Type of audio model (wav2vec2_base, whisper_base, etc.)
        train_trials: Path to training trial definitions JSON
        val_trials: Path to validation trial definitions JSON
        data_root: Root directory of EU-Emotion dataset
        output_dir: Directory to save fine-tuned model
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: Weight decay for optimizer
        device: Device to train on
        use_lr_scheduler: Whether to use learning rate scheduler
        warmup_steps: Number of warmup steps for scheduler
        early_stopping_patience: Patience for early stopping
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    logger.info(f"Using device: {device}")
    
    # Create datasets
    logger.info("Loading datasets...")
    train_dataset = TaskSpecificAudioDataset(
        trial_file=train_trials,
        data_root=data_root,
        target_sample_rate=16000,
        use_augmentation=True,  # Use augmentation for training
    )
    
    val_dataset = TaskSpecificAudioDataset(
        trial_file=val_trials,
        data_root=data_root,
        target_sample_rate=16000,
        use_augmentation=False,  # No augmentation for validation
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Audio loading can be tricky with multiprocessing
        collate_fn=collate_audio_batch,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_audio_batch,
    )
    
    # Create model
    logger.info(f"Loading {model_type} model...")
    model, processor, feature_dim = create_audio_model(model_type, device)
    
    # Create projection layer: audio_features -> 4 emotion scores
    score_proj = nn.Sequential(
        nn.Linear(feature_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 4),  # 4 options for forced-choice
    ).to(device)
    
    # Setup training
    # Fine-tune both the audio model and the projection layer
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(score_proj.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    # Learning rate scheduler
    if use_lr_scheduler:
        total_steps = len(train_loader) * num_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps
        )
    else:
        scheduler = None
    
    # Loss function (cross-entropy)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    
    logger.info("Starting training...")
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch_task_specific(
            model, processor, train_loader, criterion, optimizer, device, model_type, score_proj
        )
        
        # Validate
        val_acc = validate_task_specific(
            model, processor, val_loader, device, model_type, score_proj
        )
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        # Update learning rate
        if scheduler:
            scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save model
            model_save_dir = output_dir / "best_model"
            model_save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save audio model
            model.save_pretrained(str(model_save_dir))
            processor.save_pretrained(str(model_save_dir))
            
            # Save projection layer
            torch.save(
                score_proj.state_dict(),
                model_save_dir / "score_projection.pth"
            )
            
            # Save metadata
            metadata = {
                "model_type": model_type,
                "feature_dim": feature_dim,
                "val_accuracy": float(val_acc),
                "epoch": epoch + 1,
            }
            with open(model_save_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved best model (val_acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered (patience: {early_stopping_patience})")
                break
    
    logger.info(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")
    logger.info(f"Model saved to: {output_dir / 'best_model'}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune audio models for task-specific emotion recognition"
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['wav2vec2_base', 'wav2vec2_large', 'whisper_base', 'whisper_tiny', 'whisper_small'],
        help='Audio model to fine-tune'
    )
    parser.add_argument(
        '--train_trials',
        type=str,
        required=True,
        help='Path to training trial definitions JSON'
    )
    parser.add_argument(
        '--val_trials',
        type=str,
        required=True,
        help='Path to validation trial definitions JSON'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='Root directory of EU-Emotion dataset'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save fine-tuned model'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=20,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.01,
        help='Weight decay'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help='Device to train on'
    )
    
    args = parser.parse_args()
    
    finetune_audio_model_task_specific(
        model_type=args.model,
        train_trials=args.train_trials,
        val_trials=args.val_trials,
        data_root=args.data_root,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
    )


if __name__ == "__main__":
    main()
