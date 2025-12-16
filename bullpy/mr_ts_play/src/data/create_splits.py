"""
Create train/val/test splits with actor independence.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import argparse


def parse_filename(filename: str) -> Dict:
    """Parse video filename."""
    import re
    base = filename.replace(".mov", "")
    
    match = re.search(r'([VT])([a-z]+(?:-[a-z]+)*)$', base)
    if not match:
        return None
    
    modality = match.group(1)
    emotion = match.group(2)
    prefix = base[:match.start()]
    
    scenario_match = re.match(r'^(\d{7})', prefix)
    if not scenario_match:
        return None
    
    scenario_id = scenario_match.group(1)
    actor_part = prefix[7:]
    
    actor_match = re.match(r'^([A-Z]+)(\d+)', actor_part)
    if actor_match:
        actor = actor_match.group(1)
    else:
        actor = actor_part[0] if actor_part else "?"
    
    return {
        "scenario_id": scenario_id,
        "actor": actor,
        "modality": modality,
        "emotion": emotion,
    }


def create_splits(
    data_root: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    modality: str = "both",
) -> None:
    """
    Create actor-independent train/val/test splits.
    
    Strategy:
    1. Group videos by actor
    2. Split actors into train/val/test sets
    3. Ensure class balance across splits
    """
    np.random.seed(seed)
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all videos
    print("Collecting videos...")
    videos = []
    for emotion_folder in sorted(data_root.glob("[0-9][0-9]")):
        if not emotion_folder.is_dir():
            continue
        
        for scenario_folder in sorted(emotion_folder.glob("[0-9]*")):
            if not scenario_folder.is_dir():
                continue
            
            for video_file in scenario_folder.glob("*.mov"):
                parsed = parse_filename(video_file.name)
                if parsed:
                    parsed['video_path'] = str(video_file.relative_to(data_root))
                    videos.append(parsed)
    
    df = pd.DataFrame(videos)
    print(f"Total videos: {len(df)}")
    
    # Filter by modality if needed
    if modality != "both":
        df = df[df['modality'] == modality].reset_index(drop=True)
        print(f"After filtering by modality {modality}: {len(df)}")
    
    # Group by actor
    actors = df['actor'].unique()
    print(f"Unique actors: {len(actors)}")
    print(f"Actor distribution:\n{df['actor'].value_counts()}")
    
    # Get emotion distribution per actor
    actor_emotions = defaultdict(set)
    for _, row in df.iterrows():
        actor_emotions[row['actor']].add(row['emotion'])
    
    # Split actors into train/val/test
    # Strategy: Sort actors by number of unique emotions (descending)
    # Then assign to splits to maintain emotion coverage
    actors_sorted = sorted(actors, key=lambda a: len(actor_emotions[a]), reverse=True)
    
    train_actors = []
    val_actors = []
    test_actors = []
    
    # Simple round-robin assignment (can be improved)
    for i, actor in enumerate(actors_sorted):
        if i % 3 == 0:
            train_actors.append(actor)
        elif i % 3 == 1:
            val_actors.append(actor)
        else:
            test_actors.append(actor)
    
    # Adjust to match desired ratios
    total = len(actors_sorted)
    target_train = int(total * train_ratio)
    target_val = int(total * val_ratio)
    
    # Reassign if needed
    if len(train_actors) != target_train:
        # Simple adjustment: move actors between splits
        while len(train_actors) < target_train and val_actors:
            train_actors.append(val_actors.pop(0))
        while len(train_actors) > target_train and len(val_actors) < target_val:
            val_actors.append(train_actors.pop())
        while len(val_actors) < target_val and test_actors:
            val_actors.append(test_actors.pop(0))
        while len(val_actors) > target_val:
            test_actors.append(val_actors.pop())
    
    print(f"\nActor split:")
    print(f"  Train: {len(train_actors)} actors")
    print(f"  Val: {len(val_actors)} actors")
    print(f"  Test: {len(test_actors)} actors")
    
    # Create splits
    train_df = df[df['actor'].isin(train_actors)].copy()
    val_df = df[df['actor'].isin(val_actors)].copy()
    test_df = df[df['actor'].isin(test_actors)].copy()
    
    print(f"\nVideo split:")
    print(f"  Train: {len(train_df)} videos")
    print(f"  Val: {len(val_df)} videos")
    print(f"  Test: {len(test_df)} videos")
    
    # Check class balance
    print(f"\nClass distribution:")
    print(f"  Train: {len(train_df['emotion'].unique())} unique emotions")
    print(f"  Val: {len(val_df['emotion'].unique())} unique emotions")
    print(f"  Test: {len(test_df['emotion'].unique())} unique emotions")
    
    # Save splits
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    
    # Save actor assignments
    with open(output_dir / "actor_splits.txt", "w") as f:
        f.write("Train actors:\n")
        f.write(", ".join(sorted(train_actors)) + "\n\n")
        f.write("Val actors:\n")
        f.write(", ".join(sorted(val_actors)) + "\n\n")
        f.write("Test actors:\n")
        f.write(", ".join(sorted(test_actors)) + "\n")
    
    print(f"\nSplits saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create actor-independent splits")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/mindreading_transporter_files/Mindreading emotions library/Emotions",
        help="Root directory of the dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/splits",
        help="Output directory for split files"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Training set ratio"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Validation set ratio"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Test set ratio"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="both",
        choices=["V", "T", "both"],
        help="Modality to use"
    )
    
    args = parser.parse_args()
    create_splits(
        args.data_root,
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
        args.modality,
    )

