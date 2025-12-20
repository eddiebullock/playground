"""
EU-Emotion Stimulus Set Dataset loader for fine-tuning CLIP on emotion recognition.

EU-Emotion is from the Autism Research Centre and contains 20 complex emotions/mental states
with multiple modalities (face, voice, body, context). This dataset is ideal for two-stage
fine-tuning: EU-Emotion → CAM.

This loader is flexible and can handle various directory structures.
"""

import os
from pathlib import Path
from typing import Tuple, Optional, List
from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np


class EUEmotionDataset(Dataset):
    """
    Dataset loader for EU-Emotion Stimulus Set.
    
    Flexible loader that can handle various directory structures:
    
    Structure 1 (emotion-based):
    eu_emotion/
    ├── train/
    │   ├── emotion1/
    │   │   ├── video1.mp4
    │   │   ├── video2.mov
    │   │   └── ...
    │   ├── emotion2/
    │   └── ...
    ├── test/
    └── val/
    
    Structure 2 (modality-based):
    eu_emotion/
    ├── train/
    │   ├── face/
    │   │   ├── emotion1/
    │   │   └── emotion2/
    │   ├── voice/
    │   └── body/
    ├── test/
    └── val/
    
    Structure 3 (flat):
    eu_emotion/
    ├── train/
    │   ├── emotion1_video1.mp4
    │   ├── emotion1_video2.mp4
    │   └── ...
    ├── test/
    └── val/
    """
    
    # Supported video extensions
    VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.flv', '.wmv'}
    # Supported image extensions (if dataset has static images)
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    def __init__(
        self,
        eu_emotion_dir: str,
        split: str = "train",
        modality: str = "face",  # "face", "voice", "body", or "all"
        num_frames: int = 8,
        transform=None,
        auto_detect_structure: bool = True,
    ):
        """
        Initialize EU-Emotion dataset.
        
        Args:
            eu_emotion_dir: Root directory of EU-Emotion dataset
            split: "train", "test", or "val"
            modality: Which modality to use ("face", "voice", "body", or "all")
            num_frames: Number of frames to extract from videos (for video files)
            transform: Optional image transforms
            auto_detect_structure: Automatically detect directory structure
        """
        self.eu_emotion_dir = Path(eu_emotion_dir)
        self.split = split
        self.modality = modality
        self.num_frames = num_frames
        self.transform = transform
        
        if not self.eu_emotion_dir.exists():
            raise ValueError(f"EU-Emotion directory not found: {eu_emotion_dir}")
        
        # Load samples
        self.samples = []
        self.emotions = set()
        
        split_dir = self.eu_emotion_dir / split
        if not split_dir.exists():
            # Try alternative split names
            alt_splits = ['train', 'training', 'test', 'testing', 'val', 'validation', 'dev']
            for alt_split in alt_splits:
                alt_dir = self.eu_emotion_dir / alt_split
                if alt_dir.exists():
                    split_dir = alt_dir
                    print(f"Using alternative split directory: {alt_split}")
                    break
        
        if not split_dir.exists():
            raise ValueError(
                f"EU-Emotion {split} directory not found: {split_dir}\n"
                f"Available directories: {list(self.eu_emotion_dir.iterdir())}"
            )
        
        # Auto-detect structure and load samples
        if auto_detect_structure:
            self._auto_detect_and_load(split_dir)
        else:
            self._load_emotion_based(split_dir)
        
        # Sort emotions for consistent indexing
        self.emotions = sorted(self.emotions)
        self.emotion_to_idx = {emotion: i for i, emotion in enumerate(self.emotions)}
        
        print(f"Loaded {len(self.samples)} samples from EU-Emotion {split} split")
        print(f"Found {len(self.emotions)} emotion classes: {', '.join(self.emotions[:10])}{'...' if len(self.emotions) > 10 else ''}")
        
        if len(self.samples) == 0:
            raise ValueError(
                f"No samples found in {split_dir}\n"
                f"Please check the directory structure. Available files:\n"
                f"{list(split_dir.rglob('*'))[:20]}"
            )
    
    def _auto_detect_and_load(self, split_dir: Path):
        """Auto-detect directory structure and load samples."""
        # Strategy 1: Check if it's emotion-based (emotion folders directly in split)
        emotion_dirs = [d for d in split_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        if len(emotion_dirs) > 0:
            # Check if these look like emotion names (not modality names)
            sample_dir = emotion_dirs[0]
            has_videos = any(f.suffix.lower() in self.VIDEO_EXTENSIONS for f in sample_dir.rglob('*'))
            has_images = any(f.suffix.lower() in self.IMAGE_EXTENSIONS for f in sample_dir.rglob('*'))
            
            if has_videos or has_images:
                print(f"Detected emotion-based structure")
                self._load_emotion_based(split_dir)
                return
        
        # Strategy 2: Check if it's modality-based (modality folders, then emotion folders)
        modality_dirs = [d for d in split_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        if len(modality_dirs) > 0:
            sample_modality_dir = modality_dirs[0]
            emotion_subdirs = [d for d in sample_modality_dir.iterdir() if d.is_dir()]
            if len(emotion_subdirs) > 0:
                print(f"Detected modality-based structure")
                self._load_modality_based(split_dir)
                return
        
        # Strategy 3: Flat structure (files directly in split directory or subdirectories)
        print(f"Detected flat structure (searching recursively)")
        self._load_flat_structure(split_dir)
    
    def _load_emotion_based(self, split_dir: Path):
        """Load samples from emotion-based structure."""
        for emotion_dir in split_dir.iterdir():
            if not emotion_dir.is_dir() or emotion_dir.name.startswith('.'):
                continue
            
            emotion_name = emotion_dir.name.lower().replace('_', ' ').strip()
            self.emotions.add(emotion_name)
            
            # Look for videos/images in this emotion directory
            for ext in list(self.VIDEO_EXTENSIONS) + list(self.IMAGE_EXTENSIONS):
                for file_path in emotion_dir.rglob(f'*{ext}'):
                    if file_path.is_file():
                        self.samples.append((str(file_path), emotion_name))
    
    def _load_modality_based(self, split_dir: Path):
        """Load samples from modality-based structure."""
        # Filter by requested modality
        if self.modality != "all":
            modality_dirs = [split_dir / self.modality]
            if not modality_dirs[0].exists():
                # Try case-insensitive search
                for d in split_dir.iterdir():
                    if d.is_dir() and d.name.lower() == self.modality.lower():
                        modality_dirs = [d]
                        break
        else:
            modality_dirs = [d for d in split_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        for modality_dir in modality_dirs:
            if not modality_dir.exists():
                continue
            
            # Look for emotion subdirectories
            for emotion_dir in modality_dir.iterdir():
                if not emotion_dir.is_dir() or emotion_dir.name.startswith('.'):
                    continue
                
                emotion_name = emotion_dir.name.lower().replace('_', ' ').strip()
                self.emotions.add(emotion_name)
                
                # Look for videos/images
                for ext in list(self.VIDEO_EXTENSIONS) + list(self.IMAGE_EXTENSIONS):
                    for file_path in emotion_dir.rglob(f'*{ext}'):
                        if file_path.is_file():
                            self.samples.append((str(file_path), emotion_name))
    
    def _load_flat_structure(self, split_dir: Path):
        """Load samples from flat structure (recursive search)."""
        # Try to infer emotion from filename or parent directory
        for file_path in split_dir.rglob('*'):
            if not file_path.is_file():
                continue
            
            ext = file_path.suffix.lower()
            if ext not in self.VIDEO_EXTENSIONS and ext not in self.IMAGE_EXTENSIONS:
                continue
            
            # Try to extract emotion from filename or parent directory
            emotion_name = self._extract_emotion_from_path(file_path, split_dir)
            if emotion_name:
                self.emotions.add(emotion_name)
                self.samples.append((str(file_path), emotion_name))
    
    def _extract_emotion_from_path(self, file_path: Path, split_dir: Path) -> Optional[str]:
        """Extract emotion name from file path."""
        # Strategy 1: Parent directory name (if it's not split_dir)
        parent = file_path.parent
        if parent != split_dir:
            parent_name = parent.name.lower().replace('_', ' ').strip()
            # Skip common non-emotion directory names
            skip_names = {'train', 'test', 'val', 'train', 'testing', 'validation', 'dev',
                          'face', 'voice', 'body', 'audio', 'video', 'images'}
            if parent_name not in skip_names:
                return parent_name
        
        # Strategy 2: Filename prefix (e.g., "happy_video1.mp4")
        filename = file_path.stem.lower()
        # Try common separators
        for sep in ['_', '-', ' ']:
            if sep in filename:
                parts = filename.split(sep)
                if len(parts) > 0:
                    potential_emotion = parts[0].strip()
                    if len(potential_emotion) > 2:  # Reasonable emotion name length
                        return potential_emotion
        
        # Strategy 3: Use parent directory if it's one level deep
        if parent.parent == split_dir:
            return parent.name.lower().replace('_', ' ').strip()
        
        return None
    
    def _load_video_frames(self, video_path: Path) -> List[Image.Image]:
        """Load frames from video file."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            raise ValueError(f"Video has no frames: {video_path}")
        
        # Sample frames evenly
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames extracted from video: {video_path}")
        
        return frames
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, emotion = self.samples[idx]
        file_path = Path(file_path)
        
        # Load image or video frame
        if file_path.suffix.lower() in self.VIDEO_EXTENSIONS:
            # Load video and extract middle frame (or average)
            frames = self._load_video_frames(file_path)
            # Use middle frame as representative
            image = frames[len(frames) // 2]
        else:
            # Load static image
            image = Image.open(file_path).convert('RGB')
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        return image, emotion

