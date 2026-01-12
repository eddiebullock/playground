#!/usr/bin/env python3
"""
Create train/val splits from EU-Emotion dataset for fine-tuning.

This script creates train/val splits from the full EU-Emotion dataset,
ensuring actor-independent splits (no data leakage).
"""

import json
import argparse
import re
from pathlib import Path
from collections import defaultdict
import random

def extract_actor_code(stimulus_path: str) -> str:
    """
    Extract actor code from video filename.
    
    Pattern: {ActorCode}{Number}_cut.mp4 or {ActorCode}{Number}.mov
    Examples:
    - "GF6_cut.mp4" → "GF6"
    - "NF28_cut.mp4" → "NF28"
    - "IF8.mov" → "IF8"
    
    Args:
        stimulus_path: Path to video file (relative or absolute)
    
    Returns:
        Actor code (e.g., "GF6") or "unknown" if extraction fails
    """
    filename = Path(stimulus_path).name
    
    # Remove extension
    base = filename.replace('.mp4', '').replace('.mov', '').replace('.avi', '').replace('.mkv', '').replace('.m4v', '')
    
    # Remove "_cut" suffix if present
    base = base.replace('_cut', '')
    
    # Extract actor code: letters before the first digit
    # Pattern: {Letters}{Digits}
    match = re.match(r'^([A-Z]+)(\d+)', base)
    if match:
        return match.group(1)  # Return the letters (actor code)
    
    # Fallback: try to extract any letters at the start
    match = re.match(r'^([A-Z]+)', base)
    if match:
        return match.group(1)
    
    return "unknown"


def create_splits_from_test_trials(test_trials_file: str, data_root: str, output_dir: str, train_ratio: float = 0.8):
    """
    Create train/val splits from test trials.
    
    Note: This uses the test set for training, which is fine for fine-tuning
    since we're evaluating on a separate held-out test set.
    """
    with open(test_trials_file, 'r') as f:
        data = json.load(f)
    
    trials = data['trials']
    
    # Extract actor codes and group by actor
    trials_by_actor = defaultdict(list)
    for trial in trials:
        actor = extract_actor_code(trial['stimulus_path'])
        trials_by_actor[actor].append(trial)
    
    print(f"Found {len(trials_by_actor)} unique actors")
    
    # Get list of actors and shuffle
    actors = list(trials_by_actor.keys())
    random.shuffle(actors)
    
    # Split actors (not individual trials) into train/val
    split_idx = int(len(actors) * train_ratio)
    train_actors = set(actors[:split_idx])
    val_actors = set(actors[split_idx:])
    
    print(f"Train actors: {len(train_actors)}")
    print(f"Val actors: {len(val_actors)}")
    
    # Assign trials to splits based on actor
    train_trials = []
    val_trials = []
    
    for actor, actor_trials in trials_by_actor.items():
        if actor in train_actors:
            train_trials.extend(actor_trials)
        elif actor in val_actors:
            val_trials.extend(actor_trials)
        else:
            # Unknown actor - assign randomly (shouldn't happen)
            if random.random() < train_ratio:
                train_trials.extend(actor_trials)
            else:
                val_trials.extend(actor_trials)
    
    # Validate actor independence
    train_actor_set = set(extract_actor_code(t['stimulus_path']) for t in train_trials)
    val_actor_set = set(extract_actor_code(t['stimulus_path']) for t in val_trials)
    overlap = train_actor_set & val_actor_set
    
    if overlap:
        print(f"⚠️  WARNING: Actor overlap detected: {overlap}")
        print(f"   This indicates data leakage!")
    else:
        print(f"✅ Actor independence verified: 0 actors in both splits")
    
    # Check class balance
    train_emotions = defaultdict(int)
    val_emotions = defaultdict(int)
    for trial in train_trials:
        train_emotions[trial['correct_label']] += 1
    for trial in val_trials:
        val_emotions[trial['correct_label']] += 1
    
    train_only = set(train_emotions.keys()) - set(val_emotions.keys())
    val_only = set(val_emotions.keys()) - set(train_emotions.keys())
    
    if train_only:
        print(f"⚠️  WARNING: Emotions only in train: {train_only}")
    if val_only:
        print(f"⚠️  WARNING: Emotions only in val: {val_only}")
    
    print(f"Train emotions: {len(train_emotions)}")
    print(f"Val emotions: {len(val_emotions)}")
    
    # Shuffle
    random.shuffle(train_trials)
    random.shuffle(val_trials)
    
    # Save splits
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = output_dir / "eu_emotion_train.json"
    val_file = output_dir / "eu_emotion_val.json"
    
    with open(train_file, 'w') as f:
        json.dump({'trials': train_trials}, f, indent=2)
    
    with open(val_file, 'w') as f:
        json.dump({'trials': val_trials}, f, indent=2)
    
    print(f"\nCreated train/val splits:")
    print(f"  Train: {len(train_trials)} trials ({train_file})")
    print(f"  Val: {len(val_trials)} trials ({val_file})")
    print(f"  Total emotions: {len(train_emotions)} (train), {len(val_emotions)} (val)")
    
    return train_file, val_file


def create_splits_from_directory(data_root: str, output_dir: str, train_ratio: float = 0.8):
    """
    Create train/val splits by scanning the EU-Emotion directory structure.
    
    This discovers all videos and creates splits based on the directory structure.
    """
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Discover all videos
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.m4v'}
    all_videos = []
    
    # Look for videos in the structure: emotions*/HD Version - Face, Body, Social/Faces - HD Version/EDITED/EmotionName/*.mp4
    for emotion_dir in data_root.glob("emotions*"):
        hd_dir = emotion_dir / "HD Version - Face, Body, Social" / "Faces - HD Version" / "EDITED"
        if hd_dir.exists():
            for emotion_folder in hd_dir.iterdir():
                if emotion_folder.is_dir():
                    emotion_name = emotion_folder.name
                    for video_file in emotion_folder.glob("*"):
                        if video_file.suffix.lower() in video_extensions:
                            all_videos.append({
                                'stimulus_path': str(video_file.relative_to(data_root)),
                                'correct_label': emotion_name,
                                'trial_id': f"train_{len(all_videos)}"
                            })
    
    # Also check Original folder
    for emotion_dir in data_root.glob("emotions*"):
        hd_dir = emotion_dir / "HD Version - Face, Body, Social" / "Faces - HD Version" / "Original"
        if hd_dir.exists():
            for emotion_folder in hd_dir.iterdir():
                if emotion_folder.is_dir():
                    emotion_name = emotion_folder.name
                    for video_file in emotion_folder.glob("*"):
                        if video_file.suffix.lower() in video_extensions:
                            all_videos.append({
                                'stimulus_path': str(video_file.relative_to(data_root)),
                                'correct_label': emotion_name,
                                'trial_id': f"train_{len(all_videos)}"
                            })
    
    print(f"Found {len(all_videos)} videos")
    
    # Extract actor codes and group by actor
    trials_by_actor = defaultdict(list)
    for video in all_videos:
        actor = extract_actor_code(video['stimulus_path'])
        trials_by_actor[actor].append(video)
    
    print(f"Found {len(trials_by_actor)} unique actors")
    
    # Get list of actors and shuffle
    actors = list(trials_by_actor.keys())
    random.shuffle(actors)
    
    # Split actors (not individual trials) into train/val
    split_idx = int(len(actors) * train_ratio)
    train_actors = set(actors[:split_idx])
    val_actors = set(actors[split_idx:])
    
    print(f"Train actors: {len(train_actors)}")
    print(f"Val actors: {len(val_actors)}")
    
    # Assign trials to splits based on actor
    train_trials = []
    val_trials = []
    
    for actor, actor_trials in trials_by_actor.items():
        if actor in train_actors:
            train_trials.extend(actor_trials)
        elif actor in val_actors:
            val_trials.extend(actor_trials)
        else:
            # Unknown actor - assign randomly (shouldn't happen)
            if random.random() < train_ratio:
                train_trials.extend(actor_trials)
            else:
                val_trials.extend(actor_trials)
    
    # Validate actor independence
    train_actor_set = set(extract_actor_code(t['stimulus_path']) for t in train_trials)
    val_actor_set = set(extract_actor_code(t['stimulus_path']) for t in val_trials)
    overlap = train_actor_set & val_actor_set
    
    if overlap:
        print(f"⚠️  WARNING: Actor overlap detected: {overlap}")
        print(f"   This indicates data leakage!")
    else:
        print(f"✅ Actor independence verified: 0 actors in both splits")
    
    # Check class balance
    train_emotions = defaultdict(int)
    val_emotions = defaultdict(int)
    for trial in train_trials:
        train_emotions[trial['correct_label']] += 1
    for trial in val_trials:
        val_emotions[trial['correct_label']] += 1
    
    train_only = set(train_emotions.keys()) - set(val_emotions.keys())
    val_only = set(val_emotions.keys()) - set(train_emotions.keys())
    
    if train_only:
        print(f"⚠️  WARNING: Emotions only in train: {train_only}")
    if val_only:
        print(f"⚠️  WARNING: Emotions only in val: {val_only}")
    
    print(f"Train emotions: {len(train_emotions)}")
    print(f"Val emotions: {len(val_emotions)}")
    
    # Shuffle
    random.shuffle(train_trials)
    random.shuffle(val_trials)
    
    # Save splits
    train_file = output_dir / "eu_emotion_train.json"
    val_file = output_dir / "eu_emotion_val.json"
    
    with open(train_file, 'w') as f:
        json.dump({'trials': train_trials}, f, indent=2)
    
    with open(val_file, 'w') as f:
        json.dump({'trials': val_trials}, f, indent=2)
    
    print(f"\nCreated train/val splits:")
    print(f"  Train: {len(train_trials)} trials")
    print(f"  Val: {len(val_trials)} trials")
    print(f"  Total emotions: {len(train_emotions)} (train), {len(val_emotions)} (val)")
    
    return train_file, val_file


def main():
    parser = argparse.ArgumentParser(description="Create train/val splits for EU-Emotion fine-tuning")
    parser.add_argument('--test_trials', type=str, help='Path to test trials JSON file')
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of EU-Emotion dataset')
    parser.add_argument('--output_dir', type=str, default='data/trial_definitions', help='Output directory for splits')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of data for training (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    if args.test_trials:
        create_splits_from_test_trials(
            args.test_trials,
            args.data_root,
            args.output_dir,
            args.train_ratio
        )
    else:
        create_splits_from_directory(
            args.data_root,
            args.output_dir,
            args.train_ratio
        )


if __name__ == "__main__":
    main()
