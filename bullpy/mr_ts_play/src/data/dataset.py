"""
Dataset loading module for Mindreading/CAM dataset.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from PIL import Image


class MindreadingDataset(Dataset):
    """
    PyTorch Dataset for Mindreading/CAM video clips.
    
    Each sample is a video clip with an emotion label.
    """
    
    def __init__(
        self,
        data_root: str,
        split_file: Optional[str] = None,
        video_list: Optional[List[str]] = None,
        transform=None,
        num_frames: int = 16,
        frame_interval: int = 1,
        modality: str = "V",  # "V" for visual, "T" for textual, "both" for both
    ):
        """
        Args:
            data_root: Root directory of the dataset
            split_file: Path to CSV file with columns [video_path, emotion, actor, scenario_id]
            video_list: List of video file paths (alternative to split_file)
            transform: Optional transform to apply to frames
            num_frames: Number of frames to sample from each video
            frame_interval: Interval between sampled frames
            modality: Which modality to use ("V", "T", or "both")
        """
        self.data_root = Path(data_root)
        self.transform = transform
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.modality = modality
        
        # Load data
        if split_file:
            self.df = pd.read_csv(split_file)
        elif video_list:
            # Create dataframe from video list
            self.df = self._create_df_from_list(video_list)
        else:
            raise ValueError("Either split_file or video_list must be provided")
        
        # Create label mapping
        self.emotions = sorted(self.df['emotion'].unique())
        self.emotion_to_idx = {emotion: idx for idx, emotion in enumerate(self.emotions)}
        self.idx_to_emotion = {idx: emotion for emotion, idx in self.emotion_to_idx.items()}
        self.num_classes = len(self.emotions)
        
        # Filter by modality if needed
        if modality != "both":
            self.df = self.df[self.df['modality'] == modality].reset_index(drop=True)
    
    def _create_df_from_list(self, video_list: List[str]) -> pd.DataFrame:
        """Create dataframe from list of video paths."""
        rows = []
        for video_path in video_list:
            parsed = self._parse_filename(Path(video_path).name)
            if parsed:
                parsed['video_path'] = video_path
                rows.append(parsed)
        return pd.DataFrame(rows)
    
    def _parse_filename(self, filename: str) -> Optional[Dict]:
        """Parse video filename to extract metadata."""
        base = filename.replace(".mov", "")
        
        # Extract emotion (last part after V/T, may contain hyphens)
        match = re.search(r'([VT])([a-z]+(?:-[a-z]+)*)$', base)
        if not match:
            return None
        
        modality = match.group(1)
        emotion = match.group(2)
        prefix = base[:match.start()]
        
        # Extract scenario ID (first 7 digits)
        scenario_match = re.match(r'^(\d{7})', prefix)
        if not scenario_match:
            return None
        
        scenario_id = scenario_match.group(1)
        actor_part = prefix[7:]
        
        # Extract actor and number
        actor_match = re.match(r'^([A-Z]+)(\d+)', actor_part)
        if actor_match:
            actor = actor_match.group(1)
            instance_num = actor_match.group(2)
        else:
            actor = actor_part[0] if actor_part else "?"
            instance_num = actor_part[1:] if len(actor_part) > 1 else "?"
        
        return {
            "scenario_id": scenario_id,
            "actor": actor,
            "instance_num": instance_num,
            "modality": modality,
            "emotion": emotion,
        }
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        row = self.df.iloc[idx]
        video_path_str = row['video_path']
        
        # Resolve full path
        video_path = Path(video_path_str)
        if not video_path.is_absolute():
            video_path = self.data_root / video_path
        
        # Load video frames
        frames = self._load_video_frames(video_path)
        
        # Apply transforms
        if self.transform:
            frames = self.transform(frames)
        
        # Get label
        emotion = row['emotion']
        label = self.emotion_to_idx[emotion]
        
        return {
            'frames': frames,
            'label': torch.tensor(label, dtype=torch.long),
            'emotion': emotion,
            'video_path': str(video_path),
            'actor': row['actor'],
            'scenario_id': row['scenario_id'],
        }
    
    def _load_video_frames(self, video_path: Path) -> np.ndarray:
        """
        Load and sample frames from video.
        
        Returns:
            frames: numpy array of shape (num_frames, H, W, C)
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            raise ValueError(f"Video has no frames: {video_path}")
        
        # Sample frame indices
        if total_frames <= self.num_frames * self.frame_interval:
            # If video is shorter than needed, sample uniformly
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            # Sample uniformly across video
            max_start = total_frames - (self.num_frames * self.frame_interval)
            start_idx = max_start // 2  # Start from middle
            indices = [start_idx + i * self.frame_interval for i in range(self.num_frames)]
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                # If frame read fails, duplicate last frame
                if frames:
                    frames.append(frames[-1])
                else:
                    # If no frames read, create black frame
                    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frames.append(np.zeros((h, w, 3), dtype=np.uint8))
        
        cap.release()
        
        # Ensure we have exactly num_frames
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
        
        frames = frames[:self.num_frames]
        
        return np.array(frames)

