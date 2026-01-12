#!/usr/bin/env python3
"""
Diagnostic script to investigate why CAM trials are being skipped.
Checks path resolution and file existence.
"""

import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.cam_human_like.dataset import CAMDataset

def diagnose_cam_trials(trial_definitions_file: str, data_root: str):
    """Diagnose path issues with CAM trials."""
    
    print("=" * 60)
    print("CAM Trial Path Diagnosis")
    print("=" * 60)
    print(f"Trial definitions: {trial_definitions_file}")
    print(f"Data root: {data_root}")
    print()
    
    # Load trial definitions directly
    with open(trial_definitions_file, 'r') as f:
        trial_data = json.load(f)
    
    print(f"Total trials in file: {len(trial_data['trials'])}")
    print()
    
    # Check first few trials
    print("Checking first 10 trials:")
    print("-" * 60)
    
    missing_count = 0
    found_count = 0
    path_issues = []
    
    for i, trial in enumerate(trial_data['trials'][:10]):
        trial_id = trial['trial_id']
        stimulus_path_rel = trial['stimulus_path']
        
        print(f"\nTrial {trial_id}:")
        print(f"  Original path (from JSON): {stimulus_path_rel}")
        
        # Try different path resolutions
        paths_to_try = []
        
        # 1. As-is (if relative)
        if not Path(stimulus_path_rel).is_absolute():
            paths_to_try.append(Path(data_root) / stimulus_path_rel)
        else:
            paths_to_try.append(Path(stimulus_path_rel))
        
        # 2. Try with data_root prepended (if it looks like it needs it)
        if not stimulus_path_rel.startswith(str(data_root)):
            paths_to_try.append(Path(data_root) / stimulus_path_rel)
        
        # 3. Search by filename
        filename = Path(stimulus_path_rel).name
        found_files = list(Path(data_root).rglob(filename))
        if found_files:
            paths_to_try.append(found_files[0])
        
        # Check which paths exist
        found = False
        for path in paths_to_try:
            if path.exists():
                print(f"  ✅ Found at: {path}")
                found = True
                found_count += 1
                break
            else:
                print(f"  ❌ Not found: {path}")
        
        if not found:
            print(f"  ❌❌ FILE NOT FOUND - This trial will be skipped")
            missing_count += 1
            path_issues.append({
                'trial_id': trial_id,
                'original_path': stimulus_path_rel,
                'tried_paths': [str(p) for p in paths_to_try]
            })
    
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Checked: 10 trials")
    print(f"Found: {found_count}")
    print(f"Missing: {missing_count}")
    print()
    
    if path_issues:
        print("Path issues found:")
        for issue in path_issues[:5]:  # Show first 5
            print(f"  {issue['trial_id']}: {issue['original_path']}")
            print(f"    Tried: {issue['tried_paths'][0]}")
    
    # Now check using CAMDataset
    print()
    print("=" * 60)
    print("Testing with CAMDataset")
    print("=" * 60)
    
    dataset = CAMDataset(
        data_root=data_root,
        trial_definitions_file=trial_definitions_file,
        split_name="test",
        use_actor_filtering=False,
    )
    
    print(f"Loaded {len(dataset.trials)} trials via CAMDataset")
    print()
    
    # Check first 10 trials from dataset
    missing_dataset = 0
    for i, trial in enumerate(dataset.trials[:10]):
        path = Path(trial.stimulus_path)
        exists = path.exists()
        print(f"Trial {trial.trial_id}: {trial.stimulus_path}")
        print(f"  Exists: {exists}")
        if not exists:
            missing_dataset += 1
            # Try to find it
            found = list(Path(data_root).rglob(path.name))
            if found:
                print(f"  But found by filename search: {found[0]}")
    
    print()
    print(f"Missing from dataset trials: {missing_dataset}/10")
    
    # Check actual file structure
    print()
    print("=" * 60)
    print("Checking actual CAM directory structure")
    print("=" * 60)
    
    data_path = Path(data_root)
    if data_path.exists():
        # List first few directories
        dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])[:5]
        print(f"First 5 directories in {data_root}:")
        for d in dirs:
            print(f"  {d.name}/")
            # Check for .mov files
            mov_files = list(d.rglob("*.mov"))
            if mov_files:
                print(f"    Found {len(mov_files)} .mov files")
                print(f"    Example: {mov_files[0].relative_to(data_path)}")
    else:
        print(f"❌ Data root does not exist: {data_root}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose CAM trial path issues")
    parser.add_argument('--trial_definitions', type=str, 
                       default='data/cam_trial_definitions_20concepts.json',
                       help='Path to trial definitions JSON')
    parser.add_argument('--data_root', type=str,
                       default='/home/eb2007/data/CAM',
                       help='Root directory of CAM data')
    
    args = parser.parse_args()
    
    diagnose_cam_trials(args.trial_definitions, args.data_root)





