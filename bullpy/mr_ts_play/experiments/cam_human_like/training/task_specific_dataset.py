"""
Task-specific dataset for 4-option forced-choice fine-tuning.

This dataset loads trials with candidate labels and returns:
- Video frames (multiple frames per video)
- Candidate labels (4 options)
- Correct index (0-3)
"""

import sys
from pathlib import Path
from typing import List, Tuple, Optional
import json
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.cam_human_like.dataset import CAMDataset, CAMTrial


class TaskSpecificTrialDataset(Dataset):
    """
    Dataset for task-specific fine-tuning on 4-option forced-choice trials.
    
    Returns:
        - frames: List of PIL Images (num_frames frames from video)
        - candidate_labels: List of 4 emotion labels
        - correct_idx: Index of correct label (0-3)
    """
    
    VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.flv', '.wmv'}
    
    def __init__(
        self,
        data_root: str,
        trial_definitions_file: str,
        num_frames: int = 8,
        transform=None,
    ):
        """
        Initialize task-specific trial dataset.
        
        Args:
            data_root: Root directory of video files
            trial_definitions_file: JSON file with trial definitions
            num_frames: Number of frames to extract per video
            transform: Optional image transforms
        """
        self.data_root = Path(data_root)
        self.num_frames = num_frames
        self.transform = transform
        
        # Load trial definitions
        with open(trial_definitions_file, 'r') as f:
            data = json.load(f)
        
        self.trials = data.get('trials', [])
        print(f"Loaded {len(self.trials)} trials from {trial_definitions_file}")
    
    def _load_video_frames(self, video_path: Path) -> List[Image.Image]:
        """Extract frames from video."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Get total frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                raise ValueError(f"Video has no frames: {video_path}")
            
            # Sample frames uniformly
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            
            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame))
                else:
                    # If frame read fails, use last successful frame
                    if len(frames) > 0:
                        frames.append(frames[-1])
                    else:
                        # If no frames, create a black frame
                        frames.append(Image.new('RGB', (224, 224), (0, 0, 0)))
            
            cap.release()
            
            # Ensure we have exactly num_frames
            while len(frames) < self.num_frames:
                frames.append(frames[-1] if frames else Image.new('RGB', (224, 224), (0, 0, 0)))
            
            return frames[:self.num_frames]
        
        except Exception as e:
            print(f"Warning: Error loading video {video_path}: {e}")
            # Return black frames as fallback
            return [Image.new('RGB', (224, 224), (0, 0, 0)) for _ in range(self.num_frames)]
    
    def __len__(self):
        return len(self.trials)
    
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
            # Try finding the file in data_root
            video_path = Path(stimulus_path)
            if not video_path.is_absolute():
                # Search in data_root
                found_files = list(self.data_root.rglob(Path(stimulus_path).name))
                if found_files:
                    video_path = found_files[0]
        
        # Load frames
        frames = self._load_video_frames(video_path)
        
        # Apply transforms if provided
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        
        # Get candidate labels and correct index
        candidate_labels = trial['candidate_labels']
        correct_idx = trial['correct_idx']
        
        return {
            'frames': frames,
            'candidate_labels': candidate_labels,
            'correct_idx': correct_idx,
            'trial_id': trial.get('trial_id', f'trial_{idx}'),
        }


def collate_trial_batch(batch):
    """
    Custom collate function for task-specific trial batches.
    
    Returns:
        - frames: List of lists of PIL Images (batch_size x num_frames)
        - candidate_labels: List of lists of strings (batch_size x 4)
        - correct_indices: Tensor of correct indices (batch_size,)
    """
    frames = [item['frames'] for item in batch]
    candidate_labels = [item['candidate_labels'] for item in batch]
    correct_indices = torch.tensor([item['correct_idx'] for item in batch], dtype=torch.long)
    
    return frames, candidate_labels, correct_indices

