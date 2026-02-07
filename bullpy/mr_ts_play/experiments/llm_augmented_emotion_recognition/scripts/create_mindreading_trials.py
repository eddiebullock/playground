#!/usr/bin/env python3
"""
Create trial definitions for the MindReading Emotions dataset.

This script:
1. Discovers all videos in the MindReading dataset
2. Groups videos by emotion
3. Generates forced-choice trials (1 correct + 3 foils)
4. Creates train/test splits (80/20)
5. Outputs trial definitions in the same format as EU emotions
"""

import json
import argparse
import sys
import re
import logging
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def discover_mindreading_videos(data_root: str) -> Dict[str, List[Dict]]:
    """
    Discover all videos in the MindReading dataset.
    
    Structure:
    - /Volumes/MindReading/Emotions/01/0100104/0100104M1Vhumiliating.mov
    - /Volumes/MindReading/Emotions/02/0200102/0200102M1Vanguished.mov
    etc.
    
    Returns:
        Dictionary mapping emotion -> list of video info dicts
    """
    data_root_path = Path(data_root)
    emotion_to_videos = defaultdict(list)
    
    # Get all numbered folders (01, 02, 03, etc.)
    video_folders = sorted([d for d in data_root_path.iterdir() 
                           if d.is_dir() and d.name.isdigit()])
    
    logger.info(f"Found {len(video_folders)} video folders")
    
    total_videos = 0
    for vf in video_folders:
        # Get all emotion code subfolders
        emotion_codes = sorted([d for d in vf.iterdir() if d.is_dir()])
        
        for code_dir in emotion_codes:
            code = code_dir.name
            videos = list(code_dir.glob("*.mov"))
            
            if not videos:
                continue
            
            # Extract emotion from first video filename
            # Pattern: {code}{actor}{emotion}.mov
            match = re.match(r'\d{7}[A-Z]\d+[A-Z]([a-z-]+)\.mov', videos[0].name)
            if not match:
                logger.warning(f"Could not parse emotion from {videos[0].name}")
                continue
            
            emotion = match.group(1)
            
            # Add all videos for this emotion code
            for video in videos:
                # Create relative path from data_root
                relative_path = video.relative_to(data_root_path)
                
                emotion_to_videos[emotion].append({
                    'video_path': str(relative_path),
                    'code': code,
                    'video_file': video.name,
                    'folder': vf.name
                })
                total_videos += 1
    
    logger.info(f"Discovered {total_videos} videos across {len(emotion_to_videos)} emotions")
    
    # Log emotion distribution
    emotion_counts = {emotion: len(videos) for emotion, videos in emotion_to_videos.items()}
    logger.info(f"Emotions with most videos: {sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:10]}")
    
    return emotion_to_videos


def select_foils(correct_emotion: str, all_emotions: List[str], num_foils: int = 3, seed: int = None) -> List[str]:
    """Select random foils from other emotions."""
    if seed is not None:
        random.seed(seed)
    
    other_emotions = [e for e in all_emotions if e != correct_emotion]
    foils = random.sample(other_emotions, min(num_foils, len(other_emotions)))
    return foils


def generate_trials_for_emotion(
    emotion: str,
    videos: List[Dict],
    all_emotions: List[str],
    num_trials: int = None,
    seed: int = None
) -> List[Dict]:
    """
    Generate forced-choice trials for an emotion.
    
    If num_trials is None, creates one trial per video.
    Otherwise, randomly samples videos to create num_trials.
    """
    if seed is not None:
        random.seed(seed)
    
    if num_trials is None:
        num_trials = len(videos)
    
    # Sample videos if we have more than num_trials
    selected_videos = random.sample(videos, min(num_trials, len(videos))) if len(videos) > num_trials else videos
    
    trials = []
    for i, video_info in enumerate(selected_videos):
        # Select foils
        foils = select_foils(emotion, all_emotions, num_foils=3, seed=seed + i if seed is not None else None)
        
        # Create candidate labels (correct + foils, shuffled)
        candidate_labels = [emotion] + foils
        random.shuffle(candidate_labels)
        
        # Find correct index
        correct_idx = candidate_labels.index(emotion)
        
        trial = {
            'stimulus_path': video_info['video_path'],
            'correct_label': emotion,
            'candidate_labels': candidate_labels,
            'correct_idx': correct_idx,
            'emotion': emotion,
            'code': video_info['code'],
            'folder': video_info['folder']
        }
        
        trials.append(trial)
    
    return trials


def create_train_test_split(
    all_trials: List[Dict],
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """Create train/test split stratified by emotion."""
    random.seed(seed)
    
    # Group trials by emotion
    emotion_to_trials = defaultdict(list)
    for trial in all_trials:
        emotion_to_trials[trial['emotion']].append(trial)
    
    train_trials = []
    test_trials = []
    
    for emotion, trials in emotion_to_trials.items():
        # Shuffle trials for this emotion
        random.shuffle(trials)
        
        # Split
        split_idx = int(len(trials) * train_ratio)
        train_trials.extend(trials[:split_idx])
        test_trials.extend(trials[split_idx:])
    
    # Shuffle final lists
    random.shuffle(train_trials)
    random.shuffle(test_trials)
    
    return train_trials, test_trials


def main():
    parser = argparse.ArgumentParser(
        description="Create trial definitions for MindReading Emotions dataset"
    )
    parser.add_argument(
        '--data-root',
        type=str,
        required=True,
        help='Root directory of MindReading dataset (/Volumes/MindReading/Emotions)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/trial_definitions',
        help='Output directory for trial definitions'
    )
    parser.add_argument(
        '--trials-per-emotion',
        type=int,
        default=None,
        help='Number of trials per emotion (default: one per video)'
    )
    parser.add_argument(
        '--min-videos-per-emotion',
        type=int,
        default=1,
        help='Minimum number of videos required per emotion (default: 1)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Train/test split ratio (default: 0.8)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Discover videos
    logger.info("Discovering videos...")
    emotion_to_videos = discover_mindreading_videos(args.data_root)
    
    # Filter emotions with insufficient videos
    filtered_emotions = {
        emotion: videos for emotion, videos in emotion_to_videos.items()
        if len(videos) >= args.min_videos_per_emotion
    }
    
    if len(filtered_emotions) < len(emotion_to_videos):
        logger.warning(
            f"Filtered out {len(emotion_to_videos) - len(filtered_emotions)} emotions "
            f"with < {args.min_videos_per_emotion} videos"
        )
    
    logger.info(f"Generating trials for {len(filtered_emotions)} emotions")
    
    # Generate trials
    all_trials = []
    all_emotions = list(filtered_emotions.keys())
    trial_id = 1
    
    for emotion, videos in filtered_emotions.items():
        emotion_trials = generate_trials_for_emotion(
            emotion=emotion,
            videos=videos,
            all_emotions=all_emotions,
            num_trials=args.trials_per_emotion,
            seed=args.seed + trial_id
        )
        
        # Add trial IDs
        for trial in emotion_trials:
            trial['trial_id'] = f"mindreading_trial_{trial_id:05d}"
            trial_id += 1
        
        all_trials.extend(emotion_trials)
    
    logger.info(f"Generated {len(all_trials)} trials")
    
    # Create train/test split
    train_trials, test_trials = create_train_test_split(
        all_trials,
        train_ratio=args.train_ratio,
        seed=args.seed
    )
    
    logger.info(f"Train trials: {len(train_trials)}")
    logger.info(f"Test trials: {len(test_trials)}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save trial definitions
    train_output = output_dir / "mindreading_emotions_train.json"
    test_output = output_dir / "mindreading_emotions_test.json"
    all_output = output_dir / "mindreading_emotions_all.json"
    
    for output_file, trials_list, split_name in [
        (train_output, train_trials, "train"),
        (test_output, test_trials, "test"),
        (all_output, all_trials, "all"),
    ]:
        output_data = {
            'trials': trials_list,
            'metadata': {
                'num_trials': len(trials_list),
                'num_emotions': len(filtered_emotions),
                'trials_per_emotion': args.trials_per_emotion,
                'seed': args.seed,
                'split': split_name,
                'train_ratio': args.train_ratio if split_name != "all" else None,
                'data_root': args.data_root,
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Saved {split_name} trials to {output_file}")
        logger.info(f"  - {len(trials_list)} trials")
        logger.info(f"  - {len(filtered_emotions)} emotions")


if __name__ == "__main__":
    main()
