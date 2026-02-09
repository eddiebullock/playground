#!/usr/bin/env python3
"""
Run multimodal (video + audio) or video-only emotion recognition on MindReading dataset.

Multimodal: video frames + audio. Video-only: same video frames, no audio (--video-only).

Reproducibility (see REPRODUCIBILITY.md):
  - Model: Google Gemini 2.5 Flash (gemini-2.5-flash).
  - Prompt: Default emotion 4AFC with EMOTION:/REASONING: format; full text in llm_wrapper._classify_google_gemini (prompt_version: default_with_reasoning).
  - Seed: 42 for trial generation and candidate label generation.
  - Preprocessing: 4 frames per video (uniform sampling), JPEG for API; audio sent as provided (single-word utterances).
  - Methods note: "Audio consisted of single-word utterances of the emotion label."
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict
from PIL import Image
import cv2

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.llm_augmented_emotion_recognition.models.llm_wrapper import LLMWrapper
from experiments.llm_augmented_emotion_recognition.models.mindreading_audio_matcher import find_audio_files_for_trials

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_trial_definitions(trial_file: str) -> List[Dict]:
    """Load trial definitions from JSON file."""
    with open(trial_file, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'trials' in data:
        return data['trials']
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unexpected format in {trial_file}")


def resolve_video_path(stimulus_path: str, data_root: str) -> Path:
    """Resolve video path from stimulus path and data root."""
    data_root_path = Path(data_root)
    video_path = data_root_path / stimulus_path
    if not video_path.exists():
        # Try alternative paths
        alt_paths = [
            data_root_path / stimulus_path.lstrip('/'),
            Path(stimulus_path) if Path(stimulus_path).exists() else None
        ]
        for alt_path in alt_paths:
            if alt_path and alt_path.exists():
                return alt_path
        raise FileNotFoundError(f"Video not found: {video_path}")
    return video_path


def load_video_frames(video_path: Path, num_frames: int = 4) -> List[Image.Image]:
    """Load frames from video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        logger.warning(f"Video has 0 frames: {video_path}")
        cap.release()
        return frames
    
    # Sample frames uniformly
    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        else:
            break
    
    cap.release()
    return frames


def main():
    parser = argparse.ArgumentParser(
        description="Run multimodal (video + audio) emotion recognition experiment on MindReading dataset"
    )
    parser.add_argument(
        '--trial-definitions',
        type=str,
        required=True,
        help='Path to trial definitions JSON file'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        required=True,
        help='Root directory for video files (/Volumes/MindReading/Emotions)'
    )
    parser.add_argument(
        '--audio-base-dir',
        type=str,
        required=True,
        help='Base directory for audio files (/Volumes/MindReading/Emotions/Audio)'
    )
    parser.add_argument(
        '--audio-folder',
        type=str,
        default='1',
        choices=['1', '2', '3'],
        help='Which audio folder to use (1, 2, or 3, default: 1)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/mindreading_multimodal',
        help='Output directory for results'
    )
    parser.add_argument(
        '--provider',
        type=str,
        default='google',
        choices=['google', 'openai', 'anthropic'],
        help='LLM provider'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gemini-2.5-flash',
        help='Model name'
    )
    parser.add_argument(
        '--num-frames',
        type=int,
        default=4,
        help='Number of frames to extract per video'
    )
    parser.add_argument(
        '--use-audio',
        action='store_true',
        help='Include audio files in multimodal input'
    )
    parser.add_argument(
        '--video-only',
        action='store_true',
        help='Video-only baseline (no audio). Overrides --use-audio. Use same trials to compare with multimodal.'
    )
    parser.add_argument(
        '--skip-failed',
        action='store_true',
        help='Skip trials where video or audio files are missing'
    )
    
    args = parser.parse_args()
    if args.video_only:
        args.use_audio = False
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load trial definitions
    logger.info(f"Loading trial definitions from {args.trial_definitions}...")
    trials = load_trial_definitions(args.trial_definitions)
    logger.info(f"Loaded {len(trials)} trials")
    
    # Find audio files for trials
    audio_mapping = {}
    if args.use_audio:
        logger.info(f"Finding audio files in {args.audio_base_dir} (folder {args.audio_folder})...")
        audio_base_dir = Path(args.audio_base_dir)
        video_data_root = Path(args.data_root)
        audio_mapping = find_audio_files_for_trials(
            trials,
            video_data_root,
            audio_base_dir,
            audio_folder=args.audio_folder
        )
        audio_found = sum(1 for v in audio_mapping.values() if v is not None)
        logger.info(f"Found audio files for {audio_found}/{len(trials)} trials")
    else:
        logger.info("Audio disabled - using video only")
    
    # Initialize LLM wrapper (separate cache for video-only vs multimodal)
    use_audio = getattr(args, 'use_audio', False)
    cache_version = "mindreading_video_only_1.0" if not use_audio else "mindreading_multimodal_1.0"
    logger.info(f"Initializing LLM wrapper: {args.provider}/{args.model} (cache: {cache_version})...")
    llm = LLMWrapper(
        provider=args.provider,
        model=args.model,
        vision_model=args.model,
        cache_dir=str(output_dir / "cache"),
        use_cache=True,
        cache_version=cache_version
    )
    
    # Get all unique emotions for generating candidate labels if missing
    all_emotions = set()
    for trial in trials:
        emotion = trial.get('correct_label') or trial.get('emotion', '')
        if emotion:
            all_emotions.add(emotion)
    all_emotions = sorted(list(all_emotions))
    
    # Set seed for reproducible candidate label generation
    import random
    random.seed(42)
    
    # Process trials
    predictions = []
    failed_trials = []
    
    logger.info("Processing trials...")
    for i, trial in enumerate(trials):
        trial_id = trial.get('trial_id', f'trial_{i}')
        stimulus_path = trial.get('stimulus_path', '')
        candidate_labels = trial.get('candidate_labels', [])
        correct_label = trial.get('correct_label') or trial.get('emotion', '')
        
        # Generate candidate_labels if missing (like vision training script does)
        if not candidate_labels:
            if not correct_label:
                logger.warning(f"Trial {trial_id} has no correct_label or candidate_labels, skipping")
                failed_trials.append(trial_id)
                continue
            
            # Generate 4 candidate labels: correct + 3 random foils
            # Use trial_id as seed for reproducible per-trial randomization
            trial_seed = hash(trial_id) % (2**31)
            random.seed(trial_seed)
            other_emotions = [e for e in all_emotions if e != correct_label]
            if len(other_emotions) < 3:
                logger.warning(f"Trial {trial_id}: Not enough emotions for foils, skipping")
                failed_trials.append(trial_id)
                continue
            
            foils = random.sample(other_emotions, min(3, len(other_emotions)))
            candidate_labels = [correct_label] + foils
            random.shuffle(candidate_labels)
        
        try:
            # Resolve video path
            video_path = resolve_video_path(stimulus_path, args.data_root)
            
            # Load video frames
            frames = load_video_frames(video_path, num_frames=args.num_frames)
            if not frames:
                raise ValueError(f"No frames extracted from {video_path}")
            
            # Get audio path if available
            audio_path = audio_mapping.get(trial_id) if args.use_audio else None
            
            # Classify emotion
            logger.info(f"Processing trial {i+1}/{len(trials)}: {trial_id} (audio: {'yes' if audio_path else 'no'})")
            result = llm.classify_emotion_directly(
                frames=frames,
                candidate_labels=candidate_labels,
                video_path=str(video_path),
                audio_path=audio_path,
                include_reasoning=True
            )
            
            # Get predicted label
            predicted_label = result.get('predicted_label')
            scores = result.get('scores', {})
            
            # Find predicted index
            predicted_idx = None
            if predicted_label:
                try:
                    predicted_idx = candidate_labels.index(predicted_label)
                except ValueError:
                    # Try case-insensitive match
                    for idx, label in enumerate(candidate_labels):
                        if label.lower() == predicted_label.lower():
                            predicted_idx = idx
                            break
            
            # Determine correctness - find correct index in candidate_labels
            correct_idx = trial.get('correct_idx')
            if correct_idx is None:
                # Find correct index in generated candidate_labels
                try:
                    correct_idx = candidate_labels.index(correct_label)
                except ValueError:
                    # Fallback: try case-insensitive
                    for idx, label in enumerate(candidate_labels):
                        if label.lower() == correct_label.lower():
                            correct_idx = idx
                            break
                    if correct_idx is None:
                        logger.warning(f"Trial {trial_id}: Correct label '{correct_label}' not in candidate_labels")
                        correct_idx = None
            
            is_correct = (predicted_idx == correct_idx) if predicted_idx is not None and correct_idx is not None else None
            
            predictions.append({
                'trial_id': trial_id,
                'stimulus_path': stimulus_path,
                'video_path': str(video_path),
                'audio_path': audio_path,
                'candidate_labels': candidate_labels,
                'correct_label': correct_label,
                'correct_idx': correct_idx,
                'predicted_label': predicted_label,
                'predicted_idx': predicted_idx,
                'is_correct': is_correct,
                'scores': scores,
                'reasoning': result.get('reasoning')
            })
            
        except Exception as e:
            logger.error(f"Error processing trial {trial_id}: {e}")
            if args.skip_failed:
                failed_trials.append(trial_id)
                continue
            else:
                raise
    
    # Save predictions
    predictions_file = output_dir / "predictions.json"
    with open(predictions_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    logger.info(f"Saved predictions to {predictions_file}")
    
    # Calculate metrics
    correct_predictions = [p for p in predictions if p.get('is_correct') is True]
    total_valid = len([p for p in predictions if p.get('is_correct') is not None])
    accuracy = len(correct_predictions) / total_valid if total_valid > 0 else 0.0
    
    # Per-emotion accuracy (valid trials only)
    from collections import defaultdict
    by_emotion = defaultdict(lambda: {"count": 0, "correct": 0})
    for p in predictions:
        correct_label = p.get("correct_label") or ""
        if not correct_label:
            continue
        is_correct = p.get("is_correct")
        if is_correct is None:
            continue
        by_emotion[correct_label]["count"] += 1
        if is_correct is True:
            by_emotion[correct_label]["correct"] += 1
    per_emotion = {}
    for emotion in sorted(by_emotion.keys()):
        d = by_emotion[emotion]
        n, c = d["count"], d["correct"]
        per_emotion[emotion] = {"count": n, "correct": c, "accuracy": round(c / n, 4) if n > 0 else 0.0}
    
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total trials: {len(trials)}")
    logger.info(f"Processed: {len(predictions)}")
    logger.info(f"Failed: {len(failed_trials)}")
    logger.info(f"Valid predictions: {total_valid}")
    logger.info(f"Correct: {len(correct_predictions)}")
    logger.info(f"Accuracy: {accuracy:.2%}")
    logger.info("=" * 60)
    
    # Save summary (include per_emotion)
    summary = {
        'total_trials': len(trials),
        'processed': len(predictions),
        'failed': len(failed_trials),
        'valid_predictions': total_valid,
        'correct': len(correct_predictions),
        'accuracy': accuracy,
        'per_emotion': per_emotion,
        'failed_trials': failed_trials,
        'use_audio': args.use_audio,
        'video_only': getattr(args, 'video_only', False),
        'audio_found': sum(1 for v in audio_mapping.values() if v is not None) if args.use_audio else 0,
        'audio_folder': args.audio_folder,
    }
    
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_file}")
    
    # Also save per_emotion.json for easy use
    per_emotion_file = output_dir / "per_emotion.json"
    with open(per_emotion_file, 'w') as f:
        json.dump(per_emotion, f, indent=2)
    logger.info(f"Saved per-emotion scores to {per_emotion_file}")
    
    return accuracy


if __name__ == "__main__":
    main()
