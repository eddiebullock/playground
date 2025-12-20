#!/usr/bin/env python3
"""Quick analysis of V video frames."""

from PIL import Image
import numpy as np
from pathlib import Path

frames_dir = Path("video_inspection")

print("Analyzing V video frames...")
print("=" * 60)

for i in range(1, 4):
    frame_path = frames_dir / f"V_frame_{i}.png"
    if frame_path.exists():
        img = Image.open(frame_path)
        img_array = np.array(img)
        
        print(f"\nFrame {i}:")
        print(f"  Size: {img.size[0]}x{img.size[1]}")
        print(f"  Mode: {img.mode}")
        print(f"  Mean pixel values (RGB): {img_array.mean(axis=(0,1))}")
        print(f"  Unique colors: {len(np.unique(img_array.reshape(-1, 3), axis=0))}")
        
        # Check if it's mostly one color (might indicate text on solid background)
        std = img_array.std(axis=(0,1))
        if std.mean() < 20:
            print(f"  Note: Low variance - might be mostly solid color/text")
        else:
            print(f"  Note: High variance - appears to be natural image/video")

print("\n" + "=" * 60)
print("Summary: V videos appear to be standard video content.")
print("Check the PNG files visually to see the actual content.")




