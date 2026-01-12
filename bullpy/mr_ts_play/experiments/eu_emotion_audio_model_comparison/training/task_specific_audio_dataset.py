"""
Task-specific audio dataset for 4-option forced-choice fine-tuning.

This dataset loads audio trials with candidate labels and returns:
- Audio waveforms
- Candidate labels (4 options)
- Correct index (0-3)
"""

import sys
from pathlib import Path
from typing import List, Tuple, Optional
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.eu_emotion_audio_model_comparison.models.audio_utils import (
    load_audio_file,
    normalize_audio,
)

logger = logging.getLogger(__name__)


class TaskSpecificAudioDataset(Dataset):
    """
    Dataset for task-specific fine-tuning on 4-option forced-choice audio trials.
    
    Returns:
        - audio_waveform: numpy array of audio samples
        - candidate_labels: List of 4 emotion labels
        - correct_idx: Index of correct label (0-3)
        - emotion_weight: Weight for class balancing
    """
    
    AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.aac', '.flac'}
    
    def __init__(
        self,
        trial_file: str,
        data_root: str,
        target_sample_rate: int = 16000,
        max_duration: Optional[float] = None,
        use_augmentation: bool = False,
    ):
        """
        Initialize task-specific audio trial dataset.
        
        Args:
            trial_file: JSON file with trial definitions
            data_root: Root directory of audio files
            target_sample_rate: Target sample rate (default 16kHz for Wav2Vec2)
            max_duration: Maximum audio duration in seconds (None = no limit)
            use_augmentation: Whether to apply audio augmentation (for training)
        """
        self.data_root = Path(data_root)
        self.target_sample_rate = target_sample_rate
        self.max_duration = max_duration
        self.use_augmentation = use_augmentation
        
        # Load trial definitions
        with open(trial_file, 'r') as f:
            data = json.load(f)
        
        self.trials = data.get('trials', [])
        
        # Get all unique emotions for class weighting
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
        
        logger.info(f"Loaded {len(self.trials)} task-specific audio trials")
        logger.info(f"Found {len(self.all_emotions)} unique emotions")
        if use_augmentation:
            logger.info("Using audio augmentation for training")
    
    def _load_audio(self, audio_path: Path) -> np.ndarray:
        """Load and preprocess audio file."""
        # Check file size first
        if audio_path.exists():
            file_size = audio_path.stat().st_size
            if file_size < 1000:  # 1KB threshold for audio
                raise ValueError(f"Audio file too small (likely corrupted): {audio_path}")
        
        try:
            waveform, sr = load_audio_file(
                str(audio_path),
                target_sample_rate=self.target_sample_rate,
                max_duration=self.max_duration,
            )
            waveform = normalize_audio(waveform)
            return waveform
        except Exception as e:
            raise ValueError(f"Error loading audio {audio_path}: {e}")
    
    def _apply_augmentation(self, waveform: np.ndarray) -> np.ndarray:
        """Apply audio augmentation (for training)."""
        if not self.use_augmentation:
            return waveform
        
        # Simple augmentation: add noise, time shift, volume variation
        # Note: More sophisticated augmentation can be added later
        augmented = waveform.copy()
        
        # Add small amount of noise
        noise_level = 0.01
        noise = np.random.normal(0, noise_level, augmented.shape)
        augmented = augmented + noise
        
        # Volume variation
        volume_factor = np.random.uniform(0.8, 1.2)
        augmented = augmented * volume_factor
        
        # Clip to valid range
        augmented = np.clip(augmented, -1.0, 1.0)
        
        return augmented
    
    def __len__(self):
        return len(self.trials)
    
    def __getitem__(self, idx):
        trial = self.trials[idx]
        
        # Get audio path
        stimulus_path = trial['stimulus_path']
        
        # Handle both absolute and relative paths
        if Path(stimulus_path).is_absolute():
            audio_path = Path(stimulus_path)
        else:
            audio_path = self.data_root / stimulus_path
        
        # If path doesn't exist, try finding the file
        if not audio_path.exists():
            found_files = list(self.data_root.rglob(Path(stimulus_path).name))
            if found_files:
                audio_path = found_files[0]
            else:
                raise FileNotFoundError(f"Audio file not found: {stimulus_path}")
        
        # Load audio
        waveform = self._load_audio(audio_path)
        
        # Apply augmentation if training
        waveform = self._apply_augmentation(waveform)
        
        # Get candidate labels and correct index
        candidate_labels = trial['candidate_labels']
        correct_idx = trial.get('correct_idx', 0)
        correct_label = trial.get('correct_label', trial.get('emotion', 'neutral'))
        
        # Get emotion weight for class weighting
        emotion_weight = self.emotion_weights.get(correct_label, 1.0)
        
        return waveform, candidate_labels, correct_idx, emotion_weight


def collate_audio_batch(batch):
    """
    Custom collate function for audio batches.
    
    Returns:
        waveforms: List of numpy arrays (variable length, will be padded in model)
        candidate_labels_batch: List[List[str]] - batch_size x 4
        correct_indices: Tensor - batch_size
        emotion_weights: Tensor - batch_size
    """
    waveforms = []
    candidate_labels_batch = []
    correct_indices = []
    emotion_weights = []
    
    for waveform, candidate_labels, correct_idx, emotion_weight in batch:
        waveforms.append(waveform)
        candidate_labels_batch.append(candidate_labels)
        correct_indices.append(correct_idx)
        emotion_weights.append(emotion_weight)
    
    return (
        waveforms,
        candidate_labels_batch,
        torch.tensor(correct_indices, dtype=torch.long),
        torch.tensor(emotion_weights, dtype=torch.float32),
    )
