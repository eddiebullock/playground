#!/usr/bin/env python3
"""
Test if OpenCV can open specific CAM video files.
"""

import cv2
from pathlib import Path
import json
import sys

def test_video_opening(trial_definitions_file: str, data_root: str):
    """Test opening videos from trial definitions."""
    
    print("=" * 60)
    print("Testing OpenCV Video Opening")
    print("=" * 60)
    print(f"Trial definitions: {trial_definitions_file}")
    print(f"Data root: {data_root}")
    print()
    
    # Load trial definitions
    with open(trial_definitions_file, 'r') as f:
        trial_data = json.load(f)
    
    print(f"Total trials: {len(trial_data['trials'])}")
    print()
    
    # Test first 10 trials
    opened = 0
    failed = 0
    failed_files = []
    
    for trial in trial_data['trials'][:10]:
        trial_id = trial['trial_id']
        stimulus_path = trial['stimulus_path']
        
        # Resolve path
        if not Path(stimulus_path).is_absolute():
            video_path = Path(data_root) / stimulus_path
        else:
            video_path = Path(stimulus_path)
        
        print(f"Testing {trial_id}: {video_path.name}")
        
        # Check if file exists
        if not video_path.exists():
            print(f"  ❌ File does not exist")
            failed += 1
            failed_files.append((trial_id, str(video_path), "file_not_found"))
            continue
        
        # Check file size
        file_size = video_path.stat().st_size
        print(f"  File size: {file_size:,} bytes")
        
        # Try to open with OpenCV
        try:
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                # Try to read first frame
                ret, frame = cap.read()
                if ret:
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"  ✅ Opened successfully")
                    print(f"     Frames: {total_frames}, FPS: {fps:.2f}, Size: {width}x{height}")
                    opened += 1
                else:
                    print(f"  ❌ Opened but cannot read frames")
                    failed += 1
                    failed_files.append((trial_id, str(video_path), "cannot_read_frames"))
                cap.release()
            else:
                print(f"  ❌ Cannot open with OpenCV")
                failed += 1
                failed_files.append((trial_id, str(video_path), "opencv_failed"))
        except Exception as e:
            print(f"  ❌ Exception: {e}")
            failed += 1
            failed_files.append((trial_id, str(video_path), str(e)))
        
        print()
    
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Opened successfully: {opened}/10")
    print(f"Failed: {failed}/10")
    print()
    
    if failed_files:
        print("Failed files:")
        for trial_id, path, reason in failed_files:
            print(f"  {trial_id}: {reason}")
            print(f"    {path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test OpenCV video opening")
    parser.add_argument('--trial_definitions', type=str,
                       default='results/cam_test/test_trials.json',
                       help='Path to trial definitions JSON')
    parser.add_argument('--data_root', type=str,
                       default='/home/eb2007/data/CAM',
                       help='Root directory of CAM data')
    
    args = parser.parse_args()
    
    test_video_opening(args.trial_definitions, args.data_root)

