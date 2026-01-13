#!/usr/bin/env python3
"""
Simple script to test LLM models only (no CLIP comparison).

Tests emotion recognition using LLM vision models (OpenAI, Anthropic, Google).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.llm_augmented_emotion_recognition.models.llm_wrapper import LLMWrapper
from experiments.llm_augmented_emotion_recognition.evaluation.three_way_comparison import (
    resolve_video_path,
    load_video_frames,
)
from experiments.llm_augmented_emotion_recognition.evaluation.metrics import compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Test LLM models only (no CLIP comparison)"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='experiments/llm_augmented_emotion_recognition/configs/llm_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['cam', 'eu_emotion'],
        default='eu_emotion',
        help='Dataset type'
    )
    parser.add_argument(
        '--num_frames',
        type=int,
        default=4,
        help='Number of frames per video'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.is_absolute():
        project_root = Path.cwd()
        config_path = project_root / args.config
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    
    # Get provider from config
    provider = config['llm']['provider']
    logger.info(f"Testing provider: {provider}")
    
    # Determine dataset-specific paths
    dataset_type = args.dataset
    
    if dataset_type == 'cam':
        data_root = config['data']['cam_data_root']
        trial_definitions = config['data']['cam_test_trials']
    elif dataset_type == 'eu_emotion':
        data_root = config['data']['eu_emotion_data_root']
        trial_definitions = config['data']['eu_emotion_test_trials']
    else:
        logger.error(f"Unknown dataset type: {dataset_type}")
        sys.exit(1)
    
    # Resolve paths relative to project root
    project_root = Path(config['data']['project_root'])
    trial_definitions_path = project_root / trial_definitions
    
    # Create output directory
    output_dir = project_root / "results" / f"llm_only_{dataset_type}_{provider}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("LLM-Only Emotion Recognition Test")
    logger.info("=" * 60)
    logger.info(f"Provider: {provider}")
    logger.info(f"Dataset: {dataset_type}")
    logger.info(f"Data root: {data_root}")
    logger.info(f"Trial definitions: {trial_definitions_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Frames per video: {args.num_frames}")
    logger.info("=" * 60)
    
    # Load trials
    logger.info(f"Loading trial definitions from: {trial_definitions_path}")
    with open(trial_definitions_path, 'r') as f:
        trial_data = json.load(f)
    trials = trial_data['trials']
    logger.info(f"Loaded {len(trials)} trials")
    
    # Initialize LLM wrapper
    llm_config = config['llm']
    
    # Extract provider-specific config (supports both old and new config formats)
    if provider in llm_config:
        provider_config = llm_config[provider]
        model = provider_config.get('model', llm_config.get('model', 'text-embedding-3-small'))
        embedding_model = provider_config.get('embedding_model', llm_config.get('embedding_model'))
        vision_model = provider_config.get('vision_model', llm_config.get('vision_model', model))
    else:
        # Old config format: flat structure
        model = llm_config.get('model', 'text-embedding-3-small')
        embedding_model = llm_config.get('embedding_model')
        vision_model = llm_config.get('vision_model', model)
    
    llm_wrapper = LLMWrapper(
        provider=provider,
        model=model,
        embedding_model=embedding_model,
        vision_model=vision_model,
        cache_dir=llm_config['cache_dir'],
        use_cache=llm_config['use_cache'],
        cache_version=llm_config.get('cache_version', '1.0'),
        vision_detail=llm_config.get('vision_detail', 'low'),
    )
    # Set max_frames from config
    llm_wrapper.max_frames = llm_config.get('max_frames_per_video', 4)
    
    # Run LLM-only evaluation
    predictions = []
    
    for i, trial in enumerate(trials, 1):
        logger.info(f"[LLM-only] Trial {i}/{len(trials)}: {trial.get('trial_id', 'unknown')}")
        
        try:
            # Resolve video path
            video_path = resolve_video_path(
                trial['stimulus_path'],
                data_root,
                dataset_type,
            )
            
            # Load video frames
            video_frames = load_video_frames(video_path, args.num_frames)
            
            # Direct emotion classification
            candidate_labels = trial['candidate_labels']
            scores = llm_wrapper.classify_emotion_directly(
                video_frames,
                candidate_labels,
                video_path=video_path,
                detail=llm_config.get('vision_detail', 'low'),
                max_frames=llm_config.get('max_frames_per_video', 4),
            )
            
            # Get predicted label
            predicted_label = max(scores.items(), key=lambda x: x[1])[0]
            correct_label = trial['correct_label']
            
            predictions.append({
                'trial_id': trial.get('trial_id', f'trial_{i}'),
                'correct_label': correct_label,
                'predicted_label': predicted_label,
                'candidate_labels': candidate_labels,
                'scores': scores,
                'is_correct': predicted_label == correct_label,
            })
        except Exception as e:
            logger.error(f"Error processing trial {trial.get('trial_id', 'unknown')}: {e}")
            predictions.append({
                'trial_id': trial.get('trial_id', f'trial_{i}'),
                'correct_label': trial['correct_label'],
                'predicted_label': 'ERROR',
                'candidate_labels': trial['candidate_labels'],
                'scores': {},
                'is_correct': False,
                'error': str(e),
            })
    
    # Calculate metrics
    metrics = compute_metrics(predictions)
    
    # Save results
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'predictions': predictions,
            'metrics': metrics,
            'config': {
                'provider': provider,
                'dataset': dataset_type,
                'num_frames': args.num_frames,
            }
        }, f, indent=2)
    
    # Print summary
    correct = sum(1 for p in predictions if p.get('is_correct', False))
    incorrect = len(predictions) - correct
    
    logger.info("=" * 60)
    logger.info("Results Summary")
    logger.info("=" * 60)
    logger.info(f"Accuracy: {metrics['overall_accuracy']:.2%}")
    logger.info(f"Total trials: {len(predictions)}")
    logger.info(f"Correct: {correct}")
    logger.info(f"Incorrect: {incorrect}")
    logger.info(f"Results saved to: {results_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
