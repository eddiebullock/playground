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


def get_default_transform(size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """Get default transform pipeline."""
    return transforms.Compose([
        VideoResize(size),
        VideoToTensor(),
        VideoNormalize(),
    ])

