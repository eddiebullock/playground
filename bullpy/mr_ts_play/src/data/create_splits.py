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
    
    # Get emotion and modality distribution per actor
    actor_emotions = defaultdict(set)
    actor_modalities = defaultdict(set)
    for _, row in df.iterrows():
        actor_emotions[row['actor']].add(row['emotion'])
        actor_modalities[row['actor']].add(row['modality'])
    
    # Separate actors by their primary modality
    # Since actors typically specialize in one modality, we split them separately
    # to ensure each split has both V and T videos
    v_actors = [a for a in actors if 'V' in actor_modalities[a] and 'T' not in actor_modalities[a]]
    t_actors = [a for a in actors if 'T' in actor_modalities[a] and 'V' not in actor_modalities[a]]
    both_actors = [a for a in actors if len(actor_modalities[a]) == 2]
    
    print(f"V-only actors: {len(v_actors)}")
    print(f"T-only actors: {len(t_actors)}")
    print(f"Actors with both: {len(both_actors)}")
    
    # Shuffle each group
    np.random.shuffle(v_actors)
    np.random.shuffle(t_actors)
    np.random.shuffle(both_actors)
    
    def split_actors(actor_list, train_ratio, val_ratio, min_per_split=1):
        """Split a list of actors into train/val/test, ensuring minimum per split."""
        total = len(actor_list)
        if total == 0:
            return [], [], []
        
        # Ensure at least min_per_split in each split if possible
        if total >= 3 * min_per_split:
            target_train = max(min_per_split, int(total * train_ratio))
            target_val = max(min_per_split, int(total * val_ratio))
            target_test = max(min_per_split, total - target_train - target_val)
            
            # Adjust if total doesn't match
            if target_train + target_val + target_test > total:
                # Reduce from largest split
                excess = (target_train + target_val + target_test) - total
                if target_train >= excess + min_per_split:
                    target_train -= excess
                elif target_val >= excess + min_per_split:
                    target_val -= excess
                else:
                    target_test -= excess
        else:
            # Distribute evenly if we have fewer than 3*min_per_split
            target_train = max(1, total // 3)
            target_val = max(1, (total - target_train) // 2)
            target_test = total - target_train - target_val
        
        train = actor_list[:target_train]
        val = actor_list[target_train:target_train + target_val]
        test = actor_list[target_train + target_val:]
        
        return train, val, test
    
    # Split V actors - ensure at least 1 in each split
    v_train, v_val, v_test = split_actors(v_actors, train_ratio, val_ratio, min_per_split=1)
    
    # Split T actors
    t_train, t_val, t_test = split_actors(t_actors, train_ratio, val_ratio, min_per_split=1)
    
    # Split actors with both modalities
    both_train, both_val, both_test = split_actors(both_actors, train_ratio, val_ratio, min_per_split=0)
    
    # Combine actors for each split
    train_actors = v_train + t_train + both_train
    val_actors = v_val + t_val + both_val
    test_actors = v_test + t_test + both_test
    
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
    
    # Check modality distribution per split
    print(f"\nModality distribution:")
    print(f"  Train - V: {len(train_df[train_df['modality'] == 'V'])}, T: {len(train_df[train_df['modality'] == 'T'])}")
    print(f"  Val - V: {len(val_df[val_df['modality'] == 'V'])}, T: {len(val_df[val_df['modality'] == 'T'])}")
    print(f"  Test - V: {len(test_df[test_df['modality'] == 'V'])}, T: {len(test_df[test_df['modality'] == 'T'])}")
    
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



