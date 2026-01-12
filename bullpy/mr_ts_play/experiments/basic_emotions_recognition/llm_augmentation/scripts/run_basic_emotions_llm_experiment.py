#!/usr/bin/env python3
"""
Main experiment runner for LLM-augmented basic emotions recognition.

Runs three-way comparison: CLIP-only, LLM-only, LLM-augmented.
Uses 7-way classification (model selects from all 7 basic emotions).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import from existing LLM augmentation experiment
from experiments.llm_augmented_emotion_recognition.evaluation.three_way_comparison import (
    run_three_way_comparison
)

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
        description="Run LLM-augmented basic emotions recognition experiment"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='llm_augmentation/configs/basic_emotions_llm_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['cam', 'eu_emotion'],
        help='Dataset type (overrides config)'
    )
    parser.add_argument(
        '--fusion_method',
        type=str,
        choices=['weighted_average', 'attention'],
        help='Fusion method (overrides config)'
    )
    parser.add_argument(
        '--clip_weight',
        type=float,
        help='CLIP weight for fusion (overrides config)'
    )
    parser.add_argument(
        '--use_cache',
        action='store_true',
        help='Use cached LLM responses'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device to run on (cpu, cuda)'
    )
    parser.add_argument(
        '--num_frames',
        type=int,
        default=8,
        help='Number of frames per video'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # Try relative to script directory, then project root
        script_config_path = Path(__file__).parent.parent / args.config
        if script_config_path.exists():
            config_path = script_config_path
        else:
            project_root = Path(__file__).parent.parent.parent.parent
            project_config_path = project_root / args.config
            if project_config_path.exists():
                config_path = project_config_path
            else:
                logger.error(f"Configuration file not found: {args.config}")
                sys.exit(1)
    
    config = load_config(config_path)
    logger.info(f"Loaded configuration from: {config_path}")
    
    # Override config with command-line arguments
    if args.dataset:
        config['data']['dataset_type'] = args.dataset
    if args.fusion_method:
        config['fusion']['method'] = args.fusion_method
    if args.clip_weight is not None:
        config['fusion']['clip_weight'] = args.clip_weight
        config['fusion']['llm_weight'] = 1.0 - args.clip_weight
    if args.use_cache:
        config['llm']['use_cache'] = True
    
    # Determine dataset type
    dataset_type = config['data']['dataset_type']
    logger.info(f"Dataset type: {dataset_type}")
    
    # Load trial definitions
    project_root = Path(config['data'].get('project_root', Path(__file__).parent.parent.parent.parent))
    
    if dataset_type == 'cam':
        trial_file = project_root / config['data'].get('cam_test_trials', 'data/trial_definitions/cam_test.json')
        model_path = config['models']['cam_basic_emotions_model']
        data_root = config['data']['cam_data_root']
    else:  # eu_emotion
        trial_file = project_root / config['data'].get('eu_emotion_test_trials', 'data/trial_definitions/eu_emotion_test.json')
        model_path = config['models']['eu_emotion_basic_emotions_model']
        data_root = config['data']['eu_emotion_data_root']
    
    # For basic emotions, we need to use the basic emotion trial definitions
    # Update paths to use basic emotion trials
    if dataset_type == 'cam':
        trial_file = project_root / "experiments/basic_emotions_recognition/data/trial_definitions/cam_basic_emotions_test.json"
    else:
        trial_file = project_root / "experiments/basic_emotions_recognition/data/trial_definitions/eu_emotion_basic_emotions_test.json"
    
    if not trial_file.exists():
        logger.error(f"Trial definitions not found: {trial_file}")
        logger.error("Please generate basic emotion trials first using create_basic_emotion_trials.py")
        sys.exit(1)
    
    logger.info(f"Loading trial definitions from: {trial_file}")
    with open(trial_file, 'r') as f:
        trial_data = json.load(f)
    
    trials = trial_data.get('trials', [])
    logger.info(f"Loaded {len(trials)} trials")
    
    # Verify all trials have 7 candidate labels (basic emotions)
    basic_emotions = ["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"]
    for trial in trials[:5]:  # Check first 5
        candidate_labels = trial.get('all_candidate_labels', trial.get('candidate_labels', []))
        if len(candidate_labels) != 7:
            logger.warning(f"Trial {trial.get('trial_id')} has {len(candidate_labels)} candidate labels (expected 7)")
    
    # Update trials to use 'candidate_labels' key (expected by three_way_comparison)
    for trial in trials:
        if 'all_candidate_labels' in trial:
            trial['candidate_labels'] = trial['all_candidate_labels']
        if 'basic_emotion' in trial:
            trial['correct_label'] = trial['basic_emotion']
    
    # Run three-way comparison
    logger.info("Running three-way comparison...")
    results = run_three_way_comparison(
        clip_model_path=str(model_path),
        llm_config=config['llm'],
        fusion_config=config['fusion'],
        trials=trials,
        data_root=str(data_root),
        dataset_type=dataset_type,
        device=args.device,
        num_frames=args.num_frames,
    )
    
    # Save results
    output_dir = Path(__file__).parent.parent.parent / "results" / f"basic_emotions_{dataset_type}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results for each condition
    for condition, condition_results in results.items():
        output_file = output_dir / f"{condition}_results.json"
        with open(output_file, 'w') as f:
            json.dump(condition_results, f, indent=2)
        logger.info(f"Saved {condition} results to: {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("BASIC EMOTIONS LLM AUGMENTATION RESULTS")
    print("="*60)
    for condition, condition_results in results.items():
        if 'metrics' in condition_results:
            metrics = condition_results['metrics']
            print(f"\n{condition.upper()}:")
            print(f"  Accuracy: {metrics.get('accuracy', 0):.2%}")
            if 'per_emotion_accuracy' in metrics:
                print("  Per-Emotion Accuracy:")
                for emotion, acc in metrics['per_emotion_accuracy'].items():
                    print(f"    {emotion:12s}: {acc:.2%}")
    print("\n" + "="*60)
    
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()


