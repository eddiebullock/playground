#!/usr/bin/env python3
"""
Pre-generate LLM cache for 7 basic emotions.

This script generates LLM embeddings for the 7 basic emotions only:
- happy, sad, angry, fear, surprise, disgust, neutral

Much smaller cache than fine-grained (7 emotions vs 20-405).
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

# Basic emotion categories
BASIC_EMOTIONS = ["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"]


def main():
    parser = argparse.ArgumentParser(
        description="Pre-generate LLM cache for 7 basic emotions"
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
        help='LLM embedding model name'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='data/llm_cache',
        help='Cache directory (relative to experiment directory)'
    )
    parser.add_argument(
        '--cache_version',
        type=str,
        default='1.0',
        help='Cache version'
    )
    
    args = parser.parse_args()
    
    logger.info(f"Generating LLM cache for {len(BASIC_EMOTIONS)} basic emotions")
    logger.info(f"Emotions: {', '.join(BASIC_EMOTIONS)}")
    
    # Resolve cache directory path
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        # Relative to experiment directory
        experiment_dir = Path(__file__).parent.parent.parent
        cache_dir = experiment_dir / args.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Cache directory: {cache_dir}")
    
    # Initialize LLM wrapper
    llm_wrapper = LLMWrapper(
        provider=args.provider,
        embedding_model=args.model,
        cache_dir=str(cache_dir),
        use_cache=True,
        cache_version=args.cache_version,
    )
    
    # Generate embeddings for all basic emotions
    logger.info("Generating embeddings...")
    embeddings = {}
    
    for emotion in BASIC_EMOTIONS:
        logger.info(f"Processing: {emotion}")
        try:
            embedding = llm_wrapper.get_emotion_embedding(emotion)
            embeddings[emotion] = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
            logger.info(f"  ✓ Cached embedding for '{emotion}'")
        except Exception as e:
            logger.error(f"  ✗ Error processing '{emotion}': {e}")
            continue
    
    # Save embeddings to JSON file
    output_file = cache_dir / "basic_emotions_embeddings.json"
    with open(output_file, 'w') as f:
        json.dump(embeddings, f, indent=2)
    
    logger.info(f"\n✓ LLM cache generation complete!")
    logger.info(f"  Cached {len(embeddings)} emotion embeddings")
    logger.info(f"  Saved to: {output_file}")


if __name__ == "__main__":
    main()

