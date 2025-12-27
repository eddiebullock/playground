#!/usr/bin/env python3
"""
Pre-generate LLM cache for all emotions in trial definitions.

This script ensures all emotion embeddings are cached before running experiments,
making experiments fully reproducible without API calls.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.llm_augmented_emotion_recognition.models.llm_wrapper import LLMWrapper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_emotions_from_trials(trial_definitions_file: str) -> set:
    """Extract all unique emotion labels from trial definitions."""
    with open(trial_definitions_file, 'r') as f:
        trial_data = json.load(f)
    
    emotions = set()
    for trial in trial_data['trials']:
        emotions.add(trial['correct_label'])
        emotions.update(trial['candidate_labels'])
    
    return emotions


def main():
    parser = argparse.ArgumentParser(
        description="Pre-generate LLM cache for all emotions"
    )
    parser.add_argument(
        '--trial_definitions',
        type=str,
        nargs='+',
        required=True,
        help='Path(s) to trial definitions JSON file(s)'
    )
    parser.add_argument(
        '--provider',
        type=str,
        default='openai',
        help='LLM provider (openai, anthropic, google)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='text-embedding-3-small',
        help='LLM model name'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='data/llm_cache',
        help='Cache directory'
    )
    parser.add_argument(
        '--cache_version',
        type=str,
        default='1.0',
        help='Cache version'
    )
    
    args = parser.parse_args()
    
    # Extract all emotions from trial definitions
    all_emotions = set()
    for trial_file in args.trial_definitions:
        logger.info(f"Loading emotions from: {trial_file}")
        emotions = extract_emotions_from_trials(trial_file)
        all_emotions.update(emotions)
        logger.info(f"  Found {len(emotions)} unique emotions")
    
    logger.info(f"Total unique emotions: {len(all_emotions)}")
    
    # Resolve cache directory path
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        # Relative to experiment directory
        experiment_dir = Path(__file__).parent.parent
        cache_dir = experiment_dir / args.cache_dir
    
    # Initialize LLM wrapper
    llm_wrapper = LLMWrapper(
        provider=args.provider,
        model=args.model,
        cache_dir=str(cache_dir),
        use_cache=True,
        cache_version=args.cache_version,
    )
    
    # Cache all emotions
    llm_wrapper.cache_all_emotions(
        list(all_emotions),
        use_cache=True,
        verbose=True,
    )
    
    logger.info("Cache generation complete!")


if __name__ == "__main__":
    main()

