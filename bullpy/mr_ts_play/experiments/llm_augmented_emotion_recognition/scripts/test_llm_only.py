#!/usr/bin/env python3
"""
Test LLM-only emotion recognition (no CLIP, no fusion).

This script evaluates LLMs directly on emotion recognition tasks.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_trial_definitions(trial_file: str) -> List[Dict]:
    """Load trial definitions from JSON file."""
    with open(trial_file, 'r') as f:
        data = json.load(f)
    
    # Handle both formats: {"trials": [...]} and [...]
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


def load_video_frames(video_path: Path, num_frames: int = 4) -> List:
    """Load frames from video file."""
    try:
        import cv2
        import numpy as np
        from PIL import Image
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning(f"Could not open video: {video_path}")
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return []
        
        # Sample frames uniformly
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
        
        cap.release()
        return frames[:num_frames]
    except Exception as e:
        logger.warning(f"Error loading frames from {video_path}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Test LLM-only emotion recognition"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='experiments/llm_augmented_emotion_recognition/configs/llm_config.yaml',
        help='Path to LLM config YAML file'
    )
    parser.add_argument(
        '--test_trials',
        type=str,
        required=True,
        help='Path to test trial definitions JSON file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/llm_only_eu_emotion',
        help='Output directory for results'
    )
    parser.add_argument(
        '--num_frames',
        type=int,
        default=4,
        help='Number of frames per video'
    )
    
    args = parser.parse_args()
    
    # Load config
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    llm_config = config['llm']
    data_config = config['data']
    
    # Determine data root
    dataset_type = data_config.get('dataset_type', 'eu_emotion')
    if dataset_type == 'eu_emotion':
        data_root = data_config['eu_emotion_data_root']
    else:
        data_root = data_config.get('cam_data_root', '')
    
    # Load LLM wrapper
    from experiments.llm_augmented_emotion_recognition.models.llm_wrapper import LLMWrapper
    
    provider = llm_config.get('provider', 'openai')
    model = llm_config.get(provider, {}).get('model', 'gpt-4o-mini')
    vision_model = llm_config.get(provider, {}).get('vision_model', model)
    cache_dir = llm_config.get('cache_dir', 'data/llm_cache')
    use_cache = llm_config.get('use_cache', True)
    cache_version = llm_config.get('cache_version', '1.2')
    vision_detail = llm_config.get('vision_detail', 'low')
    max_frames = llm_config.get('max_frames_per_video', 4)
    
    llm_wrapper = LLMWrapper(
        provider=provider,
        model=model,
        embedding_model=llm_config.get(provider, {}).get('embedding_model', None),
        vision_model=vision_model,
        cache_dir=cache_dir,
        use_cache=use_cache,
        cache_version=cache_version,
        vision_detail=vision_detail
    )
    
    # Load trial definitions
    logger.info(f"Loading trial definitions from: {args.test_trials}")
    trials = load_trial_definitions(args.test_trials)
    logger.info(f"Loaded {len(trials)} trials")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate LLM on each trial
    logger.info("="*60)
    logger.info("Running LLM-only evaluation...")
    logger.info("="*60)
    
    predictions = []
    correct = 0
    total = 0
    
    # Get all unique emotions for auto-generating candidate labels
    import random
    all_emotions = set()
    for t in trials:
        if 'correct_label' in t:
            all_emotions.add(t['correct_label'])
        elif 'emotion' in t:
            all_emotions.add(t['emotion'])
    all_emotions = sorted(list(all_emotions))
    logger.info(f"Found {len(all_emotions)} unique emotions for foil generation")
    
    from tqdm import tqdm
    
    for trial in tqdm(trials, desc="Evaluating"):
        trial_id = trial.get('trial_id', f"trial_{total}")
        stimulus_path = trial.get('stimulus_path', '')
        correct_label = trial.get('correct_label', trial.get('emotion', 'unknown'))
        candidate_labels = trial.get('candidate_labels', [])
        
        # Auto-generate candidate labels if missing (like training scripts do)
        if not candidate_labels:
            # Generate 4 candidate labels: correct + 3 random foils
            other_emotions = [e for e in all_emotions if e != correct_label]
            foils = random.sample(other_emotions, min(3, len(other_emotions)))
            candidate_labels = [correct_label] + foils
            random.shuffle(candidate_labels)
            logger.debug(f"Auto-generated candidate_labels for {trial_id}: {candidate_labels}")
        
        try:
            # Load video frames
            video_path = resolve_video_path(stimulus_path, data_root)
            frames = load_video_frames(video_path, num_frames=min(max_frames, args.num_frames))
            
            if not frames:
                logger.warning(f"No frames loaded for {trial_id}, skipping")
                continue
            
            # Classify emotion using LLM (with reasoning)
            result = llm_wrapper.classify_emotion_directly(
                frames=frames,
                candidate_labels=candidate_labels,
                video_path=str(video_path),
                include_reasoning=True
            )
            
            # Extract scores and predicted label from result
            scores = result.get('scores', {})
            predicted_label = result.get('predicted_label') or max(scores.items(), key=lambda x: x[1])[0]
            reasoning = result.get('reasoning')
            
            is_correct = (predicted_label.lower().strip() == correct_label.lower().strip())
            
            if is_correct:
                correct += 1
            total += 1
            
            prediction_entry = {
                'trial_id': trial_id,
                'correct_label': correct_label,
                'predicted_label': predicted_label,
                'candidate_labels': candidate_labels,
                'scores': scores,
                'is_correct': is_correct,
                'video_path': str(video_path)
            }
            
            # Add reasoning if available
            if reasoning:
                prediction_entry['reasoning'] = reasoning
            
            predictions.append(prediction_entry)
            
        except Exception as e:
            logger.warning(f"Error processing {trial_id}: {e}")
            continue
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0.0
    
    # Save results
    results = {
        'predictions': predictions,
        'metrics': {
            'overall_accuracy': accuracy,
            'num_correct': correct,
            'num_total': total
        }
    }
    
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("="*60)
    logger.info("Evaluation complete!")
    logger.info("="*60)
    logger.info(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
    logger.info(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
