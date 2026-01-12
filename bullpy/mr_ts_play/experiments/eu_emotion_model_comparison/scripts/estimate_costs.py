#!/usr/bin/env python3
"""
Cost estimation tool for LLM models.

Estimates the cost of running LLM evaluations before actually running them.
"""

import argparse
import json
import yaml
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI pricing
PRICING = {
    "gpt-4o-mini": {
        "input": 0.15 / 1_000_000,
        "output": 0.60 / 1_000_000,
    },
    "gpt-4o": {
        "input": 2.50 / 1_000_000,
        "output": 10.00 / 1_000_000,
    },
    "gpt-4-turbo": {
        "input": 10.00 / 1_000_000,
        "output": 30.00 / 1_000_000,
    },
}


def estimate_llm_cost(
    num_trials: int,
    model_name: str,
    frames_per_video: int = 4,
    vision_detail: str = "low",
) -> dict:
    """
    Estimate cost for LLM evaluation.
    
    Args:
        num_trials: Number of trials to evaluate
        model_name: LLM model name
        frames_per_video: Number of frames per video
        vision_detail: Image detail level ("low" or "high")
    
    Returns:
        Dictionary with cost estimates
    """
    if model_name not in PRICING:
        logger.warning(f"Pricing not available for {model_name}")
        return {}
    
    pricing = PRICING[model_name]
    
    # Token estimates:
    # - Low detail: ~85 tokens per image
    # - High detail: ~765 tokens per image
    # - Prompt: ~100 tokens
    # - Output: ~50-75 tokens per response
    
    tokens_per_image = 85 if vision_detail == "low" else 765
    num_images = num_trials * frames_per_video
    
    input_tokens = (num_images * tokens_per_image) + (num_trials * 100)  # Images + prompts
    output_tokens = num_trials * 75  # Average response length
    
    input_cost = input_tokens * pricing["input"]
    output_cost = output_tokens * pricing["output"]
    total_cost = input_cost + output_cost
    
    return {
        "model": model_name,
        "num_trials": num_trials,
        "frames_per_video": frames_per_video,
        "num_images": num_images,
        "vision_detail": vision_detail,
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": output_tokens,
        "estimated_input_cost_usd": input_cost,
        "estimated_output_cost_usd": output_cost,
        "estimated_total_cost_usd": total_cost,
        "estimated_total_cost_gbp": total_cost * 0.79,  # Approximate GBP conversion
    }


def main():
    parser = argparse.ArgumentParser(
        description="Estimate costs for LLM model evaluation"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='experiments/eu_emotion_model_comparison/configs/comparison_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--trial-definitions',
        type=str,
        help='Path to trial definitions JSON (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent.parent.parent.parent / args.config
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get trial definitions
    if args.trial_definitions:
        trial_file = Path(args.trial_definitions)
    else:
        project_root = Path(__file__).parent.parent.parent.parent
        trial_file = project_root / config['dataset']['trial_definitions']
    
    # Count trials
    with open(trial_file, 'r') as f:
        data = json.load(f)
        trials = data.get('trials', data)
        num_trials = len(trials)
    
    logger.info(f"Found {num_trials} trials in {trial_file}")
    
    # Get LLM models from config
    llm_models = config['models']['llm_models']
    llm_settings = config.get('llm_settings', {})
    frames_per_video = llm_settings.get('max_frames_per_video', 4)
    vision_detail = llm_settings.get('vision_detail', 'low')
    
    # Estimate costs
    estimates = []
    for model_name in llm_models:
        estimate = estimate_llm_cost(
            num_trials=num_trials,
            model_name=model_name,
            frames_per_video=frames_per_video,
            vision_detail=vision_detail,
        )
        estimates.append(estimate)
    
    # Print summary
    print("\n" + "=" * 60)
    print("LLM Cost Estimates")
    print("=" * 60)
    print(f"Number of trials: {num_trials}")
    print(f"Frames per video: {frames_per_video}")
    print(f"Vision detail: {vision_detail}")
    print("\n")
    
    total_cost_usd = 0.0
    for est in estimates:
        print(f"Model: {est['model']}")
        print(f"  Estimated input tokens: {est['estimated_input_tokens']:,}")
        print(f"  Estimated output tokens: {est['estimated_output_tokens']:,}")
        print(f"  Estimated cost: ${est['estimated_total_cost_usd']:.4f} USD / £{est['estimated_total_cost_gbp']:.4f} GBP")
        print()
        total_cost_usd += est['estimated_total_cost_usd']
    
    print("=" * 60)
    print(f"Total estimated cost: ${total_cost_usd:.4f} USD / £{total_cost_usd * 0.79:.4f} GBP")
    print("=" * 60)
    
    # Check against budget
    cost_limit_gbp = llm_settings.get('cost_limit_gbp', 10.0)
    if total_cost_usd * 0.79 > cost_limit_gbp:
        print(f"\n⚠️  WARNING: Estimated cost exceeds budget of £{cost_limit_gbp:.2f}")
    else:
        print(f"\n✅ Estimated cost is within budget of £{cost_limit_gbp:.2f}")


if __name__ == "__main__":
    main()
