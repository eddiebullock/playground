#!/usr/bin/env python3
"""
Main experiment runner for EU-Emotion model comparison.

Evaluates multiple vision and LLM models on the EU-Emotion dataset.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.eu_emotion_model_comparison.models.model_factory import create_model
from experiments.eu_emotion_model_comparison.evaluation.evaluator import ModelEvaluator
from experiments.eu_emotion_model_comparison.evaluation.metrics import compute_confusion_matrix

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


def create_comparison_report(
    results: List[Dict],
    output_dir: Path,
    cost_summary: Dict = None,
):
    """Create comprehensive comparison report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Overall results table
    overall_rows = []
    for result in results:
        metrics = result['metrics']
        overall_rows.append({
            'model': result['model_name'],
            'accuracy': metrics['overall_accuracy'],
            'num_trials': metrics['num_trials'],
        })
    
    import pandas as pd
    overall_df = pd.DataFrame(overall_rows)
    overall_df = overall_df.sort_values('accuracy', ascending=False)
    overall_df.to_csv(output_dir / "overall_results.csv", index=False)
    
    # Create markdown report
    report_file = output_dir / "comparison_report.md"
    with open(report_file, 'w') as f:
        f.write("# EU-Emotion Model Comparison Report\n\n")
        f.write("## Overall Results\n\n")
        f.write("| Model | Accuracy | Trials |\n")
        f.write("|-------|----------|--------|\n")
        for _, row in overall_df.iterrows():
            f.write(f"| {row['model']} | {row['accuracy']:.4f} | {row['num_trials']} |\n")
        f.write("\n")
        
        # Per-emotion summary
        f.write("## Per-Emotion Performance\n\n")
        f.write("See individual model directories for detailed per-emotion results.\n\n")
        
        # Cost summary if available
        if cost_summary:
            f.write("## LLM Cost Summary\n\n")
            f.write(f"Total cost: ${cost_summary.get('total_cost_usd', 0):.4f}\n")
            f.write(f"Total tokens: {cost_summary.get('total_tokens', 0):,}\n")
            f.write(f"API calls: {cost_summary.get('api_calls', 0)}\n")
            f.write(f"Remaining budget: ${cost_summary.get('remaining_budget_usd', 0):.4f}\n")
    
    logger.info(f"Comparison report saved to {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run EU-Emotion model comparison experiment"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='experiments/eu_emotion_model_comparison/configs/comparison_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help='Device to run on'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        help='Specific models to evaluate (overrides config)'
    )
    parser.add_argument(
        '--skip-failed',
        action='store_true',
        help='Skip models that fail to load instead of exiting'
    )
    parser.add_argument(
        '--trial-definitions',
        type=str,
        default=None,
        help='Override trial definitions file path (relative to project root)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # Try relative to script, then project root
        script_config = Path(__file__).parent.parent / args.config
        project_config = Path(__file__).parent.parent.parent.parent / args.config
        if script_config.exists():
            config_path = script_config
        elif project_config.exists():
            config_path = project_config
        else:
            logger.error(f"Configuration file not found: {args.config}")
            sys.exit(1)
    
    config = load_config(config_path)
    project_root = Path(__file__).parent.parent.parent.parent
    
    # Determine dataset paths
    data_root = config['dataset']['data_root']
    # Use override if provided, otherwise use config
    if args.trial_definitions:
        trial_definitions = project_root / args.trial_definitions
    else:
        trial_definitions = project_root / config['dataset']['trial_definitions']
    output_dir = project_root / config['output']['results_dir']
    
    # Determine which models to evaluate
    if args.models:
        model_types = args.models
    else:
        model_types = config['models']['vision_models'] + config['models']['llm_models']
    
    logger.info("=" * 60)
    logger.info("EU-Emotion Model Comparison Experiment")
    logger.info("=" * 60)
    logger.info(f"Data root: {data_root}")
    logger.info(f"Trial definitions: {trial_definitions}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Models to evaluate: {', '.join(model_types)}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 60)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        trial_definitions_file=str(trial_definitions),
        data_root=data_root,
        output_dir=str(output_dir),
    )
    
    # Evaluate each model
    results = []
    cost_summaries = {}
    
    for model_type in model_types:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {model_type}")
        logger.info(f"{'='*60}")
        
        # Get model config
        model_configs = config.get('models', {}).get('model_configs', {})
        model_config = model_configs.get(model_type, {})
        
        # Create model
        try:
            model = create_model(
                model_type=model_type,
                model_config=model_config,
                device=args.device,
            )
            
            if model is None:
                if args.skip_failed:
                    logger.warning(f"Skipping {model_type} (failed to create)")
                    continue
                else:
                    logger.error(f"Failed to create model: {model_type}")
                    sys.exit(1)
            
            # Evaluate model
            result = evaluator.evaluate_model(
                model=model,
                model_name=model_type,
                save_results=config['output'].get('save_predictions', True),
                verbose=True,
            )
            results.append(result)
            
            # Get cost summary if LLM model
            if hasattr(model, 'get_cost_summary'):
                cost_summaries[model_type] = model.get_cost_summary()
                logger.info(f"Cost summary for {model_type}:")
                for key, value in cost_summaries[model_type].items():
                    logger.info(f"  {key}: {value}")
        
        except Exception as e:
            error_str = str(e).lower()
            # Check for quota errors - always skip these gracefully
            if 'quota' in error_str or 'insufficient_quota' in error_str or 'billing' in error_str:
                logger.error(
                    f"OpenAI API quota exceeded for {model_type}. "
                    f"Skipping this model. Please check your OpenAI account billing."
                )
                continue
            
            if args.skip_failed:
                logger.error(f"Error evaluating {model_type}: {e}", exc_info=True)
                continue
            else:
                logger.error(f"Error evaluating {model_type}: {e}", exc_info=True)
                raise
    
    # Create comparison report
    logger.info("\n" + "=" * 60)
    logger.info("Creating comparison report...")
    logger.info("=" * 60)
    
    # Aggregate cost summaries
    total_cost = sum(s.get('total_cost_usd', 0) for s in cost_summaries.values())
    total_tokens = sum(s.get('total_tokens', 0) for s in cost_summaries.values())
    total_calls = sum(s.get('api_calls', 0) for s in cost_summaries.values())
    
    cost_summary = {
        'total_cost_usd': total_cost,
        'total_tokens': total_tokens,
        'api_calls': total_calls,
        'per_model': cost_summaries,
    }
    
    # Save cost breakdown
    cost_file = output_dir / "cost_breakdown.json"
    with open(cost_file, 'w') as f:
        json.dump(cost_summary, f, indent=2)
    logger.info(f"Cost breakdown saved to {cost_file}")
    
    create_comparison_report(results, output_dir, cost_summary)
    
    logger.info("\n" + "=" * 60)
    logger.info("Experiment completed successfully!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
