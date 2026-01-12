#!/usr/bin/env python3
"""
Create EU-Emotion audio forced-choice trial definitions.

This script generates trial definitions for EU-Emotion audio dataset:
1. Discovers all .mp3 files in the audio directory
2. Groups audio files by emotion
3. Generates forced-choice trials (1 correct + 3 foils)
4. Creates train/test splits (80/20)
5. Outputs trial definitions in same format as video trials
"""

import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import random

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logger = None


def discover_audio_files(
    audio_dir: str,
    min_file_size: int = 1000,  # Minimum file size in bytes
) -> Dict[str, List[Dict]]:
    """
    Discover all audio files grouped by emotion.
    
    Args:
        audio_dir: Directory containing emotion subdirectories with audio files
        min_file_size: Minimum file size to consider valid
    
    Returns:
        Dictionary mapping emotion -> list of audio file info dicts
        Format: {
            "emotion_name": [
                {"path": "...", "full_path": "...", "filename": "..."},
                ...
            ]
        }
    """
    audio_path = Path(audio_dir)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    
    emotions = defaultdict(list)
    audio_extensions = {'.mp3', '.wav', '.m4a', '.aac', '.flac'}
    
    # Iterate through emotion directories
    for emotion_dir in sorted(audio_path.iterdir()):
        if not emotion_dir.is_dir():
            continue
        
        # Skip VoiceScripts.docx and other non-emotion directories
        if emotion_dir.name.lower() in ['voicescripts.docx', 'voice_scripts.docx', '.ds_store']:
            continue
        
        # Normalize emotion name
        emotion_name = emotion_dir.name.lower().replace('_', ' ').replace('-', ' ').strip()
        
        # Handle "Low Intensity" variants - collapse to base emotion
        if 'low intensity' in emotion_name:
            emotion_name = emotion_name.replace('low intensity', '').strip()
        
        # Discover audio files in this emotion directory
        for audio_file in emotion_dir.iterdir():
            if not audio_file.is_file():
                continue
            
            if audio_file.suffix.lower() not in audio_extensions:
                continue
            
            # Check file size
            try:
                file_size = audio_file.stat().st_size
                if file_size < min_file_size:
                    continue
            except (OSError, PermissionError):
                continue
            
            # Get relative path from audio directory parent
            # We want path relative to EU_emotions root
            # Audio files are in: EU Emotion - UK Voices/Fixed - amplified volume/Emotion/file.mp3
            try:
                # Find the "EU_emotions" root by going up
                rel_path = audio_file.relative_to(audio_path.parent.parent)
            except ValueError:
                # Fallback: use path relative to audio_dir
                rel_path = audio_file.relative_to(audio_path.parent)
            
            emotions[emotion_name].append({
                'path': str(rel_path),
                'full_path': str(audio_file),
                'filename': audio_file.name,
            })
    
    return dict(emotions)


def select_foils(
    target_emotion: str,
    all_emotions: Set[str],
    num_foils: int = 3,
    seed: Optional[int] = None,
) -> List[str]:
    """
    Select foil emotions that are semantically different from target.
    
    Args:
        target_emotion: Target emotion name
        all_emotions: Set of all available emotions
        num_foils: Number of foils to select
        seed: Random seed
    
    Returns:
        List of foil emotion names
    """
    if seed is not None:
        random.seed(seed)
    
    candidates = [e for e in all_emotions if e != target_emotion]
    
    if len(candidates) < num_foils:
        # Not enough candidates, use what we have and repeat if needed
        foils = candidates.copy()
        while len(foils) < num_foils:
            foils.append(random.choice(candidates))
        return foils[:num_foils]
    
    # Randomly sample foils
    return random.sample(candidates, num_foils)


def generate_trials_for_emotion(
    emotion: str,
    audio_stimuli: List[Dict],
    all_emotions: Set[str],
    num_trials: int = 10,
    seed: Optional[int] = None,
) -> List[Dict]:
    """
    Generate trials for an emotion.
    
    Args:
        emotion: Emotion name (target label)
        audio_stimuli: List of audio stimuli for this emotion
        all_emotions: Set of all available emotions for foil selection
        num_trials: Number of trials per emotion
        seed: Random seed
    
    Returns:
        List of trial dictionaries
    """
    if seed is not None:
        random.seed(seed)
    
    if len(audio_stimuli) == 0:
        return []
    
    # Sample stimuli (with replacement if needed)
    if len(audio_stimuli) >= num_trials:
        sampled_stimuli = random.sample(audio_stimuli, num_trials)
    else:
        # Not enough stimuli, sample with replacement
        sampled_stimuli = [random.choice(audio_stimuli) for _ in range(num_trials)]
    
    trials = []
    
    for i, stimulus in enumerate(sampled_stimuli):
        # Generate foils
        foils = select_foils(emotion, all_emotions, num_foils=3, seed=seed + i)
        
        # Create candidate labels: target + foils
        candidate_labels = [emotion] + foils
        
        # Randomize order
        random.shuffle(candidate_labels)
        
        # Find correct index
        correct_idx = candidate_labels.index(emotion)
        
        trial = {
            'stimulus_path': stimulus['path'],
            'modality': 'voice',
            'correct_label': emotion,
            'candidate_labels': candidate_labels,
            'correct_idx': correct_idx,
            'emotion': emotion,
        }
        
        trials.append(trial)
    
    return trials


def create_train_test_split(
    all_trials: List[Dict],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Create train/test split from all trials.
    
    Args:
        all_trials: List of all trial dictionaries
        train_ratio: Proportion of trials for training
        seed: Random seed
    
    Returns:
        Tuple of (train_trials, test_trials)
    """
    random.seed(seed)
    
    # Shuffle trials
    shuffled = all_trials.copy()
    random.shuffle(shuffled)
    
    # Split
    split_idx = int(len(shuffled) * train_ratio)
    train_trials = shuffled[:split_idx]
    test_trials = shuffled[split_idx:]
    
    return train_trials, test_trials


def main():
    parser = argparse.ArgumentParser(
        description="Create EU-Emotion audio forced-choice trial definitions"
    )
    parser.add_argument(
        '--audio-dir',
        type=str,
        required=True,
        help='Directory containing emotion subdirectories with audio files'
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
        default=10,
        help='Number of trials per emotion (default: 10)'
    )
    parser.add_argument(
        '--min-stimuli-per-emotion',
        type=int,
        default=5,
        help='Minimum number of stimuli required per emotion (default: 5)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Proportion of trials for training (default: 0.8)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    print("=" * 60)
    print("EU-Emotion Audio Trial Definition Generator")
    print("=" * 60)
    print(f"Audio directory: {args.audio_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Trials per emotion: {args.trials_per_emotion}")
    print(f"Min stimuli per emotion: {args.min_stimuli_per_emotion}")
    print(f"Train ratio: {args.train_ratio}")
    print(f"Seed: {args.seed}")
    print("=" * 60)
    print()
    
    # Discover audio files
    print("Discovering audio files...")
    emotions_data = discover_audio_files(args.audio_dir)
    
    print(f"Found {len(emotions_data)} emotions with audio files")
    for emotion, stimuli in sorted(emotions_data.items()):
        print(f"  {emotion}: {len(stimuli)} files")
    print()
    
    # Filter emotions with insufficient stimuli
    all_emotions = set(emotions_data.keys())
    emotions_with_insufficient = []
    
    for emotion, stimuli in emotions_data.items():
        if len(stimuli) < args.min_stimuli_per_emotion:
            emotions_with_insufficient.append(emotion)
    
    if emotions_with_insufficient:
        print(f"Warning: {len(emotions_with_insufficient)} emotions have "
              f"< {args.min_stimuli_per_emotion} stimuli and will be skipped")
        print(f"Skipped emotions: {emotions_with_insufficient}")
        print()
    
    # Generate trials
    print("Generating trials...")
    all_trials = []
    trial_id = 1
    
    for emotion, stimuli in emotions_data.items():
        if emotion in emotions_with_insufficient:
            continue
        
        emotion_trials = generate_trials_for_emotion(
            emotion=emotion,
            audio_stimuli=stimuli,
            all_emotions=all_emotions,
            num_trials=args.trials_per_emotion,
            seed=args.seed + trial_id,
        )
        
        # Add trial IDs
        for trial in emotion_trials:
            trial['trial_id'] = f"eu_audio_trial_{trial_id:03d}"
            trial_id += 1
        
        all_trials.extend(emotion_trials)
    
    print(f"Generated {len(all_trials)} trials across "
          f"{len(emotions_data) - len(emotions_with_insufficient)} emotions")
    print()
    
    # Create train/test split
    train_trials, test_trials = create_train_test_split(
        all_trials,
        train_ratio=args.train_ratio,
        seed=args.seed
    )
    
    print(f"Train trials: {len(train_trials)}")
    print(f"Test trials: {len(test_trials)}")
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save trial definitions
    train_output = output_dir / "eu_emotion_audio_train.json"
    test_output = output_dir / "eu_emotion_audio_test.json"
    all_output = output_dir / "eu_emotion_audio_all.json"
    
    with open(train_output, 'w') as f:
        json.dump({'trials': train_trials}, f, indent=2)
    print(f"Saved train trials to {train_output}")
    
    with open(test_output, 'w') as f:
        json.dump({'trials': test_trials}, f, indent=2)
    print(f"Saved test trials to {test_output}")
    
    with open(all_output, 'w') as f:
        json.dump({'trials': all_trials}, f, indent=2)
    print(f"Saved all trials to {all_output}")
    
    print()
    print("=" * 60)
    print("Trial generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
