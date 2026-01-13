#!/usr/bin/env python3
"""Diagnose video path issues for video model fine-tuning."""

import json
from pathlib import Path
import sys

def diagnose_paths(trial_file: str, data_root: str, num_samples: int = 10):
    """Check if video paths in trial definitions can be resolved."""
    
    data_root = Path(data_root)
    
    print(f"Data root: {data_root}")
    print(f"Data root exists: {data_root.exists()}")
    print()
    
    # Load trials
    with open(trial_file, 'r') as f:
        trials = json.load(f)
    
    print(f"Total trials: {len(trials)}")
    print()
    
    # Check first few trials
    found = 0
    not_found = 0
    
    for i, trial in enumerate(trials[:num_samples]):
        stimulus_path = trial.get('stimulus_path', '')
        print(f"Trial {i+1}:")
        print(f"  stimulus_path: {stimulus_path}")
        
        # Try to resolve path
        if Path(stimulus_path).is_absolute():
            video_path = Path(stimulus_path)
            print(f"  (absolute path)")
        else:
            video_path = data_root / stimulus_path
            print(f"  (relative path)")
        
        print(f"  Full path: {video_path}")
        print(f"  Exists: {video_path.exists()}")
        
        if video_path.exists():
            found += 1
            print(f"  ✅ Found")
        else:
            not_found += 1
            print(f"  ❌ Not found")
            
            # Try to find by filename
            filename = Path(stimulus_path).name
            found_files = list(data_root.rglob(filename))
            if found_files:
                print(f"  💡 Found by filename: {found_files[0]}")
        
        print()
    
    print(f"Summary (first {num_samples} trials):")
    print(f"  Found: {found}")
    print(f"  Not found: {not_found}")
    
    # Check all trials
    total_found = 0
    total_not_found = 0
    
    for trial in trials:
        stimulus_path = trial.get('stimulus_path', '')
        if Path(stimulus_path).is_absolute():
            video_path = Path(stimulus_path)
        else:
            video_path = data_root / stimulus_path
        
        if video_path.exists():
            total_found += 1
        else:
            total_not_found += 1
    
    print()
    print(f"Summary (all {len(trials)} trials):")
    print(f"  Found: {total_found}")
    print(f"  Not found: {total_not_found}")
    print(f"  Success rate: {100 * total_found / len(trials):.1f}%")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python diagnose_video_paths.py <trial_file> <data_root> [num_samples]")
        sys.exit(1)
    
    trial_file = sys.argv[1]
    data_root = sys.argv[2]
    num_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
    diagnose_paths(trial_file, data_root, num_samples)
