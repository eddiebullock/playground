#!/usr/bin/env python3
"""
Create EU-Emotion forced-choice trial definitions following Golan/CAM methodology.

This script generates trial definitions for EU-Emotion dataset:
1. Discovers all EU-Emotion emotions from dataset structure
2. Groups videos by emotion and modality (face/voice)
3. Generates 5 trials per emotion with counterbalanced face/voice distribution
4. Selects foils from different emotion groups (semantically different)
5. Creates train/test splits (80/20)
6. Outputs trial definitions in CAM-compatible format

Golan Methodology:
- 5 trials per emotion concept
- Counterbalanced: 3 face + 2 voice OR 2 face + 3 voice per concept
- 4 options per trial: 1 target + 3 foils
- Foils from different emotion groups
- Randomize order of candidate labels
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

from experiments.cam_human_like.training.eu_emotion_dataset import EUEmotionDataset


def discover_eu_emotion_stimuli(
    eu_emotion_dir: str,
    modality: str = "face"
) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Discover EU-Emotion stimuli grouped by emotion and modality.
    
    Args:
        eu_emotion_dir: Root directory of EU-Emotion dataset
        modality: "face", "voice", or "all"
    
    Returns:
        Dictionary mapping emotion -> modality -> list of video info dicts
        Format: {
            "emotion_name": {
                "face": [{"path": "...", "full_path": "..."}, ...],
                "voice": [{"path": "...", "full_path": "..."}, ...],
            }
        }
    """
    data_path = Path(eu_emotion_dir)
    emotions = defaultdict(lambda: {"face": [], "voice": []})
    
    # Use EUEmotionDataset to discover structure
    # We'll load all samples and group them
    try:
        dataset = EUEmotionDataset(
            eu_emotion_dir=eu_emotion_dir,
            split="train",  # Will create random split, but we want all samples
            modality=modality if modality != "all" else "face",
            num_frames=8,
        )
        
        # Get all samples (file_path, emotion)
        for file_path_str, emotion_name in dataset.samples:
            file_path = Path(file_path_str)
            
            # Determine modality from path or default to face
            # EU-Emotion structure: currently only face files are available
            # If path contains "voice" or "audio", it's voice; otherwise face
            path_lower = str(file_path).lower()
            if "voice" in path_lower or "audio" in path_lower or "vocal" in path_lower:
                mod = "voice"
            else:
                mod = "face"  # Default to face for EU-Emotion
            
            # Only include if matches requested modality
            if modality != "all" and mod != modality:
                continue
            
            # Get relative path from dataset root
            try:
                rel_path = file_path.relative_to(data_path)
            except ValueError:
                # If not relative, use absolute path (will be handled later)
                rel_path = Path(file_path_str)
            
            emotions[emotion_name][mod].append({
                'path': str(rel_path),
                'full_path': str(file_path),
            })
    except Exception as e:
        print(f"Error loading EU-Emotion dataset: {e}")
        print("Trying direct file discovery...")
        
        # Fallback: direct file discovery
        emotions_dirs = sorted(data_path.glob("emotions*"))
        video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.flv', '.wmv'}
        audio_extensions = {'.mp3', '.wav', '.m4a', '.aac', '.flac'}
        
        # Discover face files
        for emotions_dir in emotions_dirs:
            if not emotions_dir.is_dir():
                continue
            
            faces_dir = emotions_dir / "HD Version - Face, Body, Social" / "Faces - HD Version"
            if not faces_dir.exists():
                continue
            
            for subdir_name in ["EDITED", "Original"]:
                subdir = faces_dir / subdir_name
                if not subdir.exists():
                    continue
                
                for emotion_dir in subdir.iterdir():
                    if not emotion_dir.is_dir():
                        continue
                    
                    emotion_name = emotion_dir.name.lower().replace('_', ' ').strip()
                    
                    for video_file in emotion_dir.iterdir():
                        if video_file.is_file() and video_file.suffix.lower() in video_extensions:
                            # Check if file exists locally
                            try:
                                if video_file.exists() and video_file.stat().st_size > 0:
                                    rel_path = video_file.relative_to(data_path)
                                    emotions[emotion_name]["face"].append({
                                        'path': str(rel_path),
                                        'full_path': str(video_file),
                                    })
                            except (OSError, PermissionError):
                                continue
        
        # Discover voice files (audio files in "EU Emotion - UK Voices" directory)
        voices_dir = data_path / "EU Emotion - UK Voices" / "Original"
        if voices_dir.exists():
            for emotion_dir in voices_dir.iterdir():
                if not emotion_dir.is_dir():
                    continue
                
                # Normalize emotion name (handle variations like "Afraid-Low Intensity" vs "afraid low intensity")
                emotion_name = emotion_dir.name.lower().replace('_', ' ').replace('-', ' ').strip()
                
                for audio_file in emotion_dir.iterdir():
                    if audio_file.is_file() and audio_file.suffix.lower() in audio_extensions:
                        # Check if file exists locally
                        try:
                            if audio_file.exists() and audio_file.stat().st_size > 0:
                                rel_path = audio_file.relative_to(data_path)
                                emotions[emotion_name]["voice"].append({
                                    'path': str(rel_path),
                                    'full_path': str(audio_file),
                                })
                        except (OSError, PermissionError):
                            continue
    
    return dict(emotions)


def select_foils(
    target_emotion: str,
    all_emotions: Set[str],
    num_foils: int = 3,
    seed: Optional[int] = None
) -> List[str]:
    """
    Select foil emotions for a target emotion.
    
    Tries to select foils that are semantically different from target.
    For EU-Emotion, we use simple heuristics:
    - Avoid emotions with similar names (e.g., "afraid" and "afraid low intensity")
    - Prefer emotions from different valence groups if possible
    
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
    
    # Remove target from candidates
    candidates = [e for e in all_emotions if e != target_emotion]
    
    # Simple heuristic: avoid emotions with similar base names
    # (e.g., "afraid" and "afraid low intensity" should not be foils)
    target_base = target_emotion.split()[0].lower()
    filtered_candidates = [
        e for e in candidates
        if e.split()[0].lower() != target_base
    ]
    
    # If we don't have enough after filtering, use all candidates
    if len(filtered_candidates) < num_foils:
        filtered_candidates = candidates
    
    # Randomly sample foils
    if len(filtered_candidates) >= num_foils:
        foils = random.sample(filtered_candidates, num_foils)
    else:
        # Not enough candidates, use what we have and repeat if needed
        foils = filtered_candidates.copy()
        while len(foils) < num_foils:
            foils.append(random.choice(candidates))
        foils = foils[:num_foils]
    
    return foils


def generate_trials_for_emotion(
    emotion: str,
    face_stimuli: List[Dict],
    voice_stimuli: List[Dict],
    all_emotions: Set[str],
    num_trials: int = 5,
    seed: Optional[int] = None,
) -> List[Dict]:
    """
    Generate 5 trials for an emotion with counterbalanced face/voice distribution.
    
    Args:
        emotion: Emotion name (target label)
        face_stimuli: List of face stimuli for this emotion
        voice_stimuli: List of voice stimuli for this emotion
        all_emotions: Set of all available emotions for foil selection
        num_trials: Number of trials per emotion (default 5)
        seed: Random seed for reproducibility
    
    Returns:
        List of trial dictionaries
    """
    if seed is not None:
        random.seed(seed)
    
    # Determine face/voice distribution (counterbalanced if both available)
    # If only one modality available, use all trials from that modality
    if not voice_stimuli or len(voice_stimuli) == 0:
        # Only face available: use all trials from face
        num_face = num_trials
        num_voice = 0
    elif not face_stimuli or len(face_stimuli) == 0:
        # Only voice available: use all trials from voice
        num_face = 0
        num_voice = num_trials
    else:
        # Both available: counterbalanced (3+2 or 2+3)
        num_face = 3 if random.random() < 0.5 else 2
        num_voice = num_trials - num_face
    
    # Sample stimuli
    sampled_face = random.sample(face_stimuli, min(num_face, len(face_stimuli))) if face_stimuli and num_face > 0 else []
    sampled_voice = random.sample(voice_stimuli, min(num_voice, len(voice_stimuli))) if voice_stimuli and num_voice > 0 else []
    
    # If we don't have enough stimuli, adjust distribution
    if len(sampled_face) < num_face and len(voice_stimuli) > 0:
        num_voice = num_trials - len(sampled_face)
        sampled_voice = random.sample(voice_stimuli, min(num_voice, len(voice_stimuli))) if voice_stimuli else []
    
    if len(sampled_voice) < num_voice and len(face_stimuli) > 0:
        num_face = num_trials - len(sampled_voice)
        sampled_face = random.sample(face_stimuli, min(num_face, len(face_stimuli))) if face_stimuli else []
    
    # Generate trials
    trials = []
    
    for stimulus in sampled_face + sampled_voice:
        # Determine modality
        mod = "face" if stimulus in sampled_face else "voice"
        
        # Generate foils
        foils = select_foils(emotion, all_emotions, num_foils=3, seed=seed + len(trials))
        
        # Create candidate labels: target + foils
        candidate_labels = [emotion] + foils
        
        # Randomize order (Golan methodology: options presented in random order)
        random.shuffle(candidate_labels)
        
        # Find correct index
        correct_idx = candidate_labels.index(emotion)
        
        trial = {
            'stimulus_path': stimulus['path'],
            'modality': mod,
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
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Create train/test split from all trials.
    
    Args:
        all_trials: List of all trial dictionaries
        train_ratio: Proportion of trials for training (default 0.8)
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
        description="Create EU-Emotion forced-choice trial definitions following Golan methodology"
    )
    parser.add_argument(
        '--eu-emotion-dir',
        type=str,
        required=True,
        help='Root directory of EU-Emotion dataset'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Output directory for trial definitions (default: data)'
    )
    parser.add_argument(
        '--modality',
        type=str,
        default='face',
        choices=['face', 'voice', 'all'],
        help='Modality to use (default: face)'
    )
    parser.add_argument(
        '--trials-per-emotion',
        type=int,
        default=10,
        help='Number of trials per emotion (default: 10, can increase if more files available)'
    )
    parser.add_argument(
        '--min-stimuli-per-emotion',
        type=int,
        default=3,
        help='Minimum stimuli required per emotion (default: 3, lowered since we have enough files)'
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
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print(f"Discovering EU-Emotion stimuli from {args.eu_emotion_dir}...")
    print(f"Modality: {args.modality}")
    
    emotions_data = discover_eu_emotion_stimuli(
        args.eu_emotion_dir,
        modality=args.modality
    )
    
    print(f"Found {len(emotions_data)} emotions")
    
    # Get all unique emotions for foil generation
    all_emotions = set(emotions_data.keys())
    print(f"Total unique emotions: {len(all_emotions)}")
    print(f"Emotions: {sorted(all_emotions)}")
    
    # Generate trials for each emotion
    all_trials = []
    trial_id = 1
    
    emotions_with_insufficient_stimuli = []
    
    for emotion, modalities in emotions_data.items():
        face_stimuli = modalities['face']
        voice_stimuli = modalities['voice']
        total_stimuli = len(face_stimuli) + len(voice_stimuli)
        
        if total_stimuli < args.min_stimuli_per_emotion:
            emotions_with_insufficient_stimuli.append(emotion)
            continue
        
        # Generate trials for this emotion
        emotion_trials = generate_trials_for_emotion(
            emotion=emotion,
            face_stimuli=face_stimuli,
            voice_stimuli=voice_stimuli,
            all_emotions=all_emotions,
            num_trials=args.trials_per_emotion,
            seed=args.seed + trial_id,
        )
        
        # Add trial IDs
        for trial in emotion_trials:
            trial['trial_id'] = f"eu_trial_{trial_id:03d}"
            trial_id += 1
        
        all_trials.extend(emotion_trials)
    
    if emotions_with_insufficient_stimuli:
        print(f"\nWarning: {len(emotions_with_insufficient_stimuli)} emotions have "
              f"< {args.min_stimuli_per_emotion} stimuli and were skipped")
        print(f"Skipped emotions: {emotions_with_insufficient_stimuli}")
    
    print(f"\nGenerated {len(all_trials)} trials across {len(emotions_data) - len(emotions_with_insufficient_stimuli)} emotions")
    
    # Create train/test split
    train_trials, test_trials = create_train_test_split(
        all_trials,
        train_ratio=args.train_ratio,
        seed=args.seed
    )
    
    print(f"Train trials: {len(train_trials)}")
    print(f"Test trials: {len(test_trials)}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save trial definitions
    train_output = output_dir / "eu_emotion_trial_definitions_train.json"
    test_output = output_dir / "eu_emotion_trial_definitions_test.json"
    all_output = output_dir / "eu_emotion_trial_definitions_all.json"
    
    for output_file, trials_list, split_name in [
        (train_output, train_trials, "train"),
        (test_output, test_trials, "test"),
        (all_output, all_trials, "all"),
    ]:
        output_data = {
            'trials': trials_list,
            'metadata': {
                'num_trials': len(trials_list),
                'num_emotions': len(emotions_data) - len(emotions_with_insufficient_stimuli),
                'trials_per_emotion': args.trials_per_emotion,
                'seed': args.seed,
                'modality': args.modality,
                'split': split_name,
                'train_ratio': args.train_ratio if split_name != "all" else None,
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nSaved {split_name} trials to {output_file}")
        print(f"  - {len(trials_list)} trials")
        print(f"  - {len(emotions_data) - len(emotions_with_insufficient_stimuli)} emotions")


if __name__ == "__main__":
    main()

