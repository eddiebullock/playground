"""
FER2013 Dataset loader for fine-tuning CLIP on emotion recognition.

FER2013 (Facial Expression Recognition 2013) is a standard emotion recognition
dataset with 7 basic emotions: angry, disgust, fear, happy, neutral, sad, surprise.

This is an external dataset (no overlap with CAM), making it ideal for
rigorous fine-tuning that shows general emotion recognition ability.
"""

import os
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class FER2013Dataset(Dataset):
    """
    Dataset loader for FER2013.
    
    FER2013 structure:
    fer2013/
    ├── train/
    │   ├── angry/
    │   ├── disgust/
    │   ├── fear/
    │   ├── happy/
    │   ├── neutral/
    │   ├── sad/
    │   └── surprise/
    ├── test/
    └── val/ (optional)
    """
    
    def __init__(self, fer2013_dir: str, split: str = "train", transform=None):
        """
        Initialize FER2013 dataset.
        
        Args:
            fer2013_dir: Root directory of FER2013 dataset
            split: "train", "test", or "val"
            transform: Optional image transforms
        """
        self.fer2013_dir = Path(fer2013_dir)
        self.split = split
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        # Load image paths and labels
        self.samples = []
        split_dir = self.fer2013_dir / split
        
        if not split_dir.exists():
            raise ValueError(f"FER2013 {split} directory not found: {split_dir}")
        
        # FER2013 has 7 emotion classes
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.emotion_to_idx = {emotion: i for i, emotion in enumerate(self.emotions)}
        
        # Load all images
        for emotion in self.emotions:
            emotion_dir = split_dir / emotion
            if emotion_dir.exists():
                for img_file in emotion_dir.glob("*.jpg"):
                    self.samples.append((str(img_file), emotion))
        
        print(f"Loaded {len(self.samples)} images from FER2013 {split} split")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, emotion = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, emotion





