"""
Dataset class that maps fine-grained emotions to basic emotion categories.
"""

import json
from pathlib import Path
from typing import Optional
import pandas as pd

from src.data.dataset import MindreadingDataset
from src.data.emotion_mapping import BASIC_EMOTIONS, EMOTION_NAMES


class BasicEmotionDataset(MindreadingDataset):
    """
    Dataset that maps fine-grained emotions to 6-7 basic emotion categories.
    """
    
    def __init__(
        self,
        data_root: str,
        split_file: Optional[str] = None,
        video_list: Optional[list] = None,
        transform=None,
        num_frames: int = 16,
        frame_interval: int = 1,
        modality: str = "V",
        mapping_file: str = "data/basic_emotion_mapping.json",
    ):
        """
        Args:
            mapping_file: Path to JSON file mapping fine-grained to basic emotions
        """
        # Load emotion mapping
        mapping_path = Path(mapping_file)
        if not mapping_path.exists():
            raise FileNotFoundError(
                f"Emotion mapping file not found: {mapping_file}\n"
                f"Run: python src/data/create_basic_emotion_mapping.py"
            )
        
        with open(mapping_path, 'r') as f:
            self.emotion_mapping = json.load(f)
        
        # Initialize parent dataset (this loads fine-grained emotions)
        super().__init__(
            data_root=data_root,
            split_file=split_file,
            video_list=video_list,
            transform=transform,
            num_frames=num_frames,
            frame_interval=frame_interval,
            modality=modality,
        )
        
        # Map fine-grained emotions to basic emotions
        self.df['basic_emotion'] = self.df['emotion'].map(self.emotion_mapping)
        
        # Handle any unmapped emotions (default to neutral)
        self.df['basic_emotion'] = self.df['basic_emotion'].fillna('neutral')
        
        # Create basic emotion label mapping
        self.basic_emotions = sorted(BASIC_EMOTIONS.keys())
        self.basic_emotion_to_idx = {emotion: BASIC_EMOTIONS[emotion] for emotion in self.basic_emotions}
        self.idx_to_basic_emotion = EMOTION_NAMES
        self.num_basic_classes = len(self.basic_emotions)
        
        # Update num_classes to use basic emotions
        self.num_classes = self.num_basic_classes
        
        # Store original fine-grained mappings for reference
        self.original_emotions = self.emotions
        self.original_emotion_to_idx = self.emotion_to_idx
        self.original_idx_to_emotion = self.idx_to_emotion
        
        # Update mappings to use basic emotions
        self.emotions = self.basic_emotions
        self.emotion_to_idx = self.basic_emotion_to_idx
        self.idx_to_emotion = self.idx_to_basic_emotion
    
    def __getitem__(self, idx):
        """Get item with basic emotion label."""
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
        
        # Get basic emotion label
        basic_emotion = row['basic_emotion']
        label = self.basic_emotion_to_idx[basic_emotion]
        
        return {
            'frames': frames,
            'label': label,  # Basic emotion label (0-6)
            'emotion': basic_emotion,  # Basic emotion name
            'fine_grained_emotion': row['emotion'],  # Original fine-grained emotion
            'video_path': str(video_path),
            'actor': row['actor'],
            'scenario_id': row['scenario_id'],
        }
    
    def get_class_distribution(self):
        """Get distribution of basic emotion classes."""
        return self.df['basic_emotion'].value_counts().to_dict()








