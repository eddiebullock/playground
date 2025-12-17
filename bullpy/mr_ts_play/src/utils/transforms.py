"""
Data transforms for video frames.
"""

import torch
import torchvision.transforms as transforms
from typing import Tuple
import numpy as np


class VideoToTensor:
    """Convert numpy array of frames to torch tensor."""
    
    def __call__(self, frames: np.ndarray) -> torch.Tensor:
        """
        Args:
            frames: (T, H, W, C) numpy array
        Returns:
            tensor: (T, C, H, W) torch tensor
        """
        # Convert to (T, C, H, W)
        frames = frames.transpose(0, 3, 1, 2)
        # Convert to float and normalize to [0, 1]
        frames = frames.astype(np.float32) / 255.0
        return torch.from_numpy(frames)


class VideoNormalize:
    """Normalize video frames using ImageNet stats."""
    
    def __init__(self, mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                 std: Tuple[float, float, float] = (0.229, 0.224, 0.225)):
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
    
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (T, C, H, W) tensor
        Returns:
            normalized: (T, C, H, W) tensor
        """
        T = frames.shape[0]
        mean = self.mean.expand(T, -1, -1, -1)
        std = self.std.expand(T, -1, -1, -1)
        return (frames - mean) / std


class VideoResize:
    """Resize video frames."""
    
    def __init__(self, size: Tuple[int, int] = (224, 224)):
        self.size = size
    
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        """
        Args:
            frames: (T, H, W, C) numpy array
        Returns:
            resized: (T, H, W, C) numpy array
        """
        import cv2
        resized = []
        for frame in frames:
            frame_resized = cv2.resize(frame, self.size)
            resized.append(frame_resized)
        return np.array(resized)


class VideoRandomCrop:
    """Random crop for data augmentation."""
    
    def __init__(self, size: Tuple[int, int] = (224, 224)):
        self.size = size
    
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        """
        Args:
            frames: (T, H, W, C) numpy array
        Returns:
            cropped: (T, H, W, C) numpy array
        """
        import cv2
        T, H, W, C = frames.shape
        target_h, target_w = self.size
        
        # Random crop
        if H > target_h and W > target_w:
            top = np.random.randint(0, H - target_h)
            left = np.random.randint(0, W - target_w)
            cropped = frames[:, top:top+target_h, left:left+target_w, :]
        else:
            # If image is smaller, resize first
            cropped = []
            for frame in frames:
                frame_resized = cv2.resize(frame, (target_w, target_h))
                cropped.append(frame_resized)
            cropped = np.array(cropped)
        
        return cropped


class VideoCenterCrop:
    """Center crop (for validation/test)."""
    
    def __init__(self, size: Tuple[int, int] = (224, 224)):
        self.size = size
    
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        """
        Args:
            frames: (T, H, W, C) numpy array
        Returns:
            cropped: (T, H, W, C) numpy array
        """
        import cv2
        T, H, W, C = frames.shape
        target_h, target_w = self.size
        
        # Center crop
        if H > target_h and W > target_w:
            top = (H - target_h) // 2
            left = (W - target_w) // 2
            cropped = frames[:, top:top+target_h, left:left+target_w, :]
        else:
            # If image is smaller, resize
            cropped = []
            for frame in frames:
                frame_resized = cv2.resize(frame, (target_w, target_h))
                cropped.append(frame_resized)
            cropped = np.array(cropped)
        
        return cropped


class VideoColorJitter:
    """Color jitter augmentation (applied per frame)."""
    
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        """
        Args:
            frames: (T, H, W, C) numpy array in [0, 255]
        Returns:
            jittered: (T, H, W, C) numpy array
        """
        import cv2
        jittered = []
        for frame in frames:
            # Convert to HSV for easier manipulation
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # Brightness
            if self.brightness > 0:
                hsv[:, :, 2] *= (1.0 + np.random.uniform(-self.brightness, self.brightness))
                hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
            
            # Saturation
            if self.saturation > 0:
                hsv[:, :, 1] *= (1.0 + np.random.uniform(-self.saturation, self.saturation))
                hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            
            # Hue
            if self.hue > 0:
                hsv[:, :, 0] += np.random.uniform(-self.hue * 180, self.hue * 180)
                hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 180)
            
            # Convert back to RGB
            frame_jittered = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            jittered.append(frame_jittered)
        
        return np.array(jittered)


def get_default_transform(size: Tuple[int, int] = (224, 224), augment: bool = False) -> transforms.Compose:
    """
    Get default transform pipeline.
    
    Args:
        size: Target size (H, W)
        augment: Whether to apply data augmentation
    """
    if augment:
        return transforms.Compose([
            VideoRandomCrop(size),
            VideoColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            VideoToTensor(),
            VideoNormalize(),
        ])
    else:
        return transforms.Compose([
            VideoCenterCrop(size),
            VideoToTensor(),
            VideoNormalize(),
        ])



