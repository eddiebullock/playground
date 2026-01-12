#!/usr/bin/env python3
"""
Analyze confusion matrices to identify common error patterns across models.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from collections import defaultdict, Counter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_confusion_matrix(results_dir: Path, model_name: str) -> pd.DataFrame:
    """Load confusion matrix for a model."""
    model_dir = results_dir / model_name.lower().replace('-', '_')
    confusion_file = model_dir / "confusion_matrix.csv"
    
    if not confusion_file.exists():
        logger.warning(f"Confusion matrix not found for {model_name}")
        return pd.DataFrame()
    
    return pd.read_csv(confusion_file, index_col=0)


def analyze_confusion_matrix(
    confusion_matrix: pd.DataFrame,
    model_name: str,
) -> Dict:
    """Analyze a single confusion matrix."""
    if confusion_matrix.empty:
        return {}
    
    analysis = {
        'model': model_name,
        'total_errors': 0,
        'most_confused_pairs': [],
        'emotions_with_most_errors': [],
        'emotions_with_least_errors': [],
    }
    
    # Count total errors (off-diagonal)
    total = 0
    errors = 0
    emotion_errors = defaultdict(int)
    confusion_pairs = Counter()
    
    for true_emotion in confusion_matrix.index:
        for pred_emotion in confusion_matrix.columns:
            count = confusion_matrix.loc[true_emotion, pred_emotion]
            total += count
            
            if true_emotion != pred_emotion:
                errors += count
                emotion_errors[true_emotion] += count
                confusion_pairs[(true_emotion, pred_emotion)] += count
    
    analysis['total_errors'] = errors
    analysis['total_predictions'] = total
    analysis['error_rate'] = errors / total if total > 0 else 0
    
    # Most confused pairs
    analysis['most_confused_pairs'] = [
        {'true': true, 'predicted': pred, 'count': count}
        for (true, pred), count in confusion_pairs.most_common(10)
    ]
    
    # Emotions with most/least errors
    sorted_errors = sorted(emotion_errors.items(), key=lambda x: x[1], reverse=True)
    analysis['emotions_with_most_errors'] = [
        {'emotion': emotion, 'errors': count}
        for emotion, count in sorted_errors[:10]
    ]
    
    analysis['emotions_with_least_errors'] = [
        {'emotion': emotion, 'errors': count}
        for emotion, count in sorted_errors[-10:] if count > 0
    ]
    
    return analysis


def find_common_errors(
    all_analyses: Dict[str, Dict],
) -> Dict:
    """Find errors common across multiple models."""
    common_errors = {
        'common_confused_pairs': [],
        'consistently_difficult_emotions': [],
        'consistently_easy_emotions': [],
    }
    
    # Count how many models confuse each pair
    pair_counts = Counter()
    emotion_error_counts = defaultdict(int)
    emotion_correct_counts = defaultdict(int)
    
    for model_name, analysis in all_analyses.items():
        if not analysis:
            continue
        
        # Count confused pairs
        for pair in analysis['most_confused_pairs']:
            pair_key = (pair['true'], pair['predicted'])
            pair_counts[pair_key] += 1
        
        # Count emotion errors
        for emotion_info in analysis['emotions_with_most_errors']:
            emotion_error_counts[emotion_info['emotion']] += 1
        
        for emotion_info in analysis['emotions_with_least_errors']:
            emotion_correct_counts[emotion_info['emotion']] += 1
    
    # Find pairs confused by multiple models
    num_models = len(all_analyses)
    threshold = max(2, num_models // 2)  # At least half of models
    
    common_errors['common_confused_pairs'] = [
        {'true': true, 'predicted': pred, 'models': count}
        for (true, pred), count in pair_counts.most_common(20)
        if count >= threshold
    ]
    
    # Find consistently difficult emotions
    common_errors['consistently_difficult_emotions'] = [
        {'emotion': emotion, 'models_with_errors': count}
        for emotion, count in sorted(emotion_error_counts.items(), key=lambda x: x[1], reverse=True)
        if count >= threshold
    ][:10]
    
    # Find consistently easy emotions
    common_errors['consistently_easy_emotions'] = [
        {'emotion': emotion, 'models_correct': count}
        for emotion, count in sorted(emotion_correct_counts.items(), key=lambda x: x[1], reverse=True)
        if count >= threshold
    ][:10]
    
    return common_errors


def main():
    parser = argparse.ArgumentParser(
        description="Analyze confusion matrices to identify error patterns"
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='results/eu_emotion_model_comparison',
        help='Results directory containing model results'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        help='Model names to analyze (default: all models in results_dir)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/eu_emotion_model_comparison/confusion_analysis.json',
        help='Output file for analysis results'
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    # Get models to analyze
    if args.models:
        model_names = args.models
    else:
        # Find all models with confusion matrices
        model_names = []
        for model_dir in results_dir.iterdir():
            if model_dir.is_dir() and (model_dir / "confusion_matrix.csv").exists():
                model_names.append(model_dir.name)
    
    logger.info(f"Analyzing {len(model_names)} models: {', '.join(model_names)}")
    
    # Load and analyze each model
    all_analyses = {}
    for model_name in model_names:
        confusion_matrix = load_confusion_matrix(results_dir, model_name)
        if not confusion_matrix.empty:
            analysis = analyze_confusion_matrix(confusion_matrix, model_name)
            all_analyses[model_name] = analysis
            logger.info(f"{model_name}: {analysis.get('error_rate', 0):.2%} error rate")
    
    # Find common errors
    common_errors = find_common_errors(all_analyses)
    
    # Create summary report
    report = {
        'models_analyzed': list(all_analyses.keys()),
        'individual_analyses': all_analyses,
        'common_errors': common_errors,
    }
    
    # Save report
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"✅ Analysis saved to {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("CONFUSION MATRIX ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\nModels analyzed: {len(all_analyses)}")
    for model_name, analysis in all_analyses.items():
        print(f"  {model_name}: {analysis.get('error_rate', 0):.2%} error rate")
    
    print(f"\nCommon confused pairs (across {len(all_analyses)} models):")
    for pair in common_errors['common_confused_pairs'][:10]:
        print(f"  {pair['true']} → {pair['predicted']} ({pair['models']} models)")
    
    print(f"\nConsistently difficult emotions:")
    for emotion_info in common_errors['consistently_difficult_emotions'][:5]:
        print(f"  {emotion_info['emotion']} ({emotion_info['models_with_errors']} models)")
    
    print(f"\nConsistently easy emotions:")
    for emotion_info in common_errors['consistently_easy_emotions'][:5]:
        print(f"  {emotion_info['emotion']} ({emotion_info['models_correct']} models)")


if __name__ == '__main__':
    main()
