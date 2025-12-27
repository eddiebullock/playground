#!/usr/bin/env python3
"""
Main experiment runner for LLM-augmented emotion recognition.

Runs three-way comparison: CLIP-only, LLM-only, LLM-augmented.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.llm_augmented_emotion_recognition.evaluation.three_way_comparison import run_three_way_comparison

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
        description="Run LLM-augmented emotion recognition experiment"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/llm_config.yaml',
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
    # Resolve config path: if relative, try relative to script dir, then project root
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # First try relative to script directory
        script_config_path = Path(__file__).parent.parent / args.config
        if script_config_path.exists():
            config_path = script_config_path
        else:
            # Try relative to project root (where script is run from)
            project_root = Path.cwd()
            project_config_path = project_root / args.config
            if project_config_path.exists():
                config_path = project_config_path
            else:
                logger.error(f"Configuration file not found: {args.config}")
                logger.error(f"Tried: {script_config_path}")
                logger.error(f"Tried: {project_config_path}")
                sys.exit(1)
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    
    # Override with command-line arguments
    if args.dataset:
        config['data']['dataset_type'] = args.dataset
    if args.fusion_method:
        config['fusion']['method'] = args.fusion_method
    if args.clip_weight is not None:
        config['fusion']['clip_weight'] = args.clip_weight
        config['fusion']['llm_weight'] = 1.0 - args.clip_weight
    if args.use_cache:
        config['llm']['use_cache'] = True
    
    # Determine dataset-specific paths
    dataset_type = config['data'].get('dataset_type', 'cam')
    
    if dataset_type == 'cam':
        data_root = config['data']['cam_data_root']
        trial_definitions = config['data']['cam_test_trials']
        clip_model_path = config['models']['cam_clip_model_path']
    elif dataset_type == 'eu_emotion':
        data_root = config['data']['eu_emotion_data_root']
        trial_definitions = config['data']['eu_emotion_test_trials']
        clip_model_path = config['models']['eu_emotion_clip_model_path']
    else:
        logger.error(f"Unknown dataset type: {dataset_type}")
        sys.exit(1)
    
    # Resolve paths relative to project root
    project_root = Path(config['data']['project_root'])
    trial_definitions = project_root / trial_definitions
    
    # Create output directory
    output_dir = project_root / "results" / f"llm_augmented_{dataset_type}_{config['fusion']['method']}"
    
    logger.info("=" * 60)
    logger.info("LLM-Augmented Emotion Recognition Experiment")
    logger.info("=" * 60)
    logger.info(f"Dataset: {dataset_type}")
    logger.info(f"Fusion method: {config['fusion']['method']}")
    logger.info(f"CLIP weight: {config['fusion']['clip_weight']}")
    logger.info(f"LLM weight: {config['fusion']['llm_weight']}")
    logger.info(f"Data root: {data_root}")
    logger.info(f"Trial definitions: {trial_definitions}")
    logger.info(f"CLIP model: {clip_model_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)
    
    # Run experiment
    try:
        results = run_three_way_comparison(
            clip_model_path=str(clip_model_path),
            llm_config=config['llm'],
            fusion_config=config['fusion'],
            trial_definitions_file=str(trial_definitions),
            data_root=str(data_root),
            dataset_type=dataset_type,
            output_dir=str(output_dir),
            num_frames=args.num_frames,
            device=args.device,
        )
        
        logger.info("Experiment completed successfully!")
        logger.info(f"Results saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

