#!/usr/bin/env python3
"""
Comprehensive analysis of all model results (LLMs and vision models).
"""
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

def load_llm_results(result_path):
    """Load LLM results and calculate metrics."""
    with open(result_path, 'r') as f:
        data = json.load(f)
    
    predictions = data['predictions']
    metrics = data.get('metrics', {})
    
    # Calculate per-emotion metrics
    emotion_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for pred in predictions:
        emotion = pred['correct_label']
        emotion_stats[emotion]['total'] += 1
        if pred.get('is_correct', False):
            emotion_stats[emotion]['correct'] += 1
    
    per_emotion = {}
    for emotion, stats in emotion_stats.items():
        per_emotion[emotion] = {
            'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0,
            'correct': stats['correct'],
            'total': stats['total']
        }
    
    return {
        'overall_accuracy': metrics.get('overall_accuracy', 0.0),
        'per_emotion': per_emotion,
        'total_trials': len(predictions)
    }

def load_vision_results(result_dir):
    """Load vision model results."""
    metrics_file = result_dir / 'metrics.json'
    per_emotion_file = result_dir / 'per_emotion_results.csv'
    
    if not metrics_file.exists():
        return None
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    per_emotion = {}
    
    # Try to get from per_emotion_metrics in metrics.json first
    if 'per_emotion_metrics' in metrics:
        for emotion, emotion_metrics in metrics['per_emotion_metrics'].items():
            support = emotion_metrics.get('support', 0)
            accuracy = emotion_metrics.get('accuracy', 0.0)
            correct = int(accuracy * support) if support > 0 else 0
            per_emotion[emotion] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': support
            }
    elif per_emotion_file.exists():
        # Fallback to CSV
        df = pd.read_csv(per_emotion_file)
        for _, row in df.iterrows():
            emotion = row['emotion']
            support = int(row.get('support', 0))
            accuracy = row.get('accuracy', 0.0)
            correct = int(accuracy * support) if support > 0 else 0
            per_emotion[emotion] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': support
            }
    
    return {
        'overall_accuracy': metrics.get('overall_accuracy', 0.0),
        'per_emotion': per_emotion,
        'total_trials': metrics.get('num_trials', metrics.get('total_trials', 0))
    }

def main():
    results_dir = Path('results')
    
    # Load LLM results
    llm_results = {}
    for provider in ['google', 'anthropic', 'openai']:
        result_file = results_dir / f'llm_only_eu_emotion_{provider}' / 'results.json'
        if result_file.exists():
            llm_results[provider] = load_llm_results(result_file)
            print(f"✅ Loaded {provider} results")
        else:
            print(f"⚠️  {provider} results not found")
    
    # Load vision model results
    vision_results = {}
    vision_dir = results_dir / 'eu_emotion_model_comparison'
    for model in ['clip_finetuned', 'resnet50', 'vit_base', 'efficientnet_b0']:
        model_dir = vision_dir / model
        if model_dir.exists():
            result = load_vision_results(model_dir)
            if result:
                vision_results[model] = result
                print(f"✅ Loaded {model} results")
        else:
            print(f"⚠️  {model} results not found")
    
    # Get all emotions
    all_emotions = set()
    for results in list(llm_results.values()) + list(vision_results.values()):
        all_emotions.update(results['per_emotion'].keys())
    all_emotions = sorted(all_emotions)
    
    # Create comprehensive comparison
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL COMPARISON - EU-EMOTION DATASET")
    print("="*80)
    
    # Overall accuracy comparison
    print("\n## OVERALL ACCURACY")
    print("-"*80)
    print(f"{'Model':<30} {'Accuracy':<15} {'Trials':<10}")
    print("-"*80)
    
    # Vision models
    for model_name, results in sorted(vision_results.items(), key=lambda x: x[1]['overall_accuracy'], reverse=True):
        print(f"{model_name:<30} {results['overall_accuracy']:>6.2%}      {results['total_trials']:<10}")
    
    # LLM models
    for provider, results in sorted(llm_results.items(), key=lambda x: x[1]['overall_accuracy'], reverse=True):
        provider_name = f"LLM ({provider.upper()})"
        print(f"{provider_name:<30} {results['overall_accuracy']:>6.2%}      {results['total_trials']:<10}")
    
    # Per-emotion comparison
    print("\n## PER-EMOTION ACCURACY")
    print("-"*80)
    
    # Create DataFrame for easier comparison
    comparison_data = []
    for emotion in all_emotions:
        row = {'emotion': emotion}
        
        # Vision models
        for model_name, results in vision_results.items():
            if emotion in results['per_emotion']:
                row[model_name] = results['per_emotion'][emotion]['accuracy']
                row[f"{model_name}_n"] = results['per_emotion'][emotion]['total']
            else:
                row[model_name] = None
                row[f"{model_name}_n"] = 0
        
        # LLM models
        for provider, results in llm_results.items():
            if emotion in results['per_emotion']:
                row[f"llm_{provider}"] = results['per_emotion'][emotion]['accuracy']
                row[f"llm_{provider}_n"] = results['per_emotion'][emotion]['total']
            else:
                row[f"llm_{provider}"] = None
                row[f"llm_{provider}_n"] = 0
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Print per-emotion table
    print(f"\n{'Emotion':<25}", end="")
    for model_name in sorted(vision_results.keys()):
        print(f"{model_name[:15]:>15}", end="")
    for provider in sorted(llm_results.keys()):
        print(f"LLM-{provider[:8]:>15}", end="")
    print()
    print("-"*80)
    
    for _, row in df.iterrows():
        print(f"{row['emotion']:<25}", end="")
        for model_name in sorted(vision_results.keys()):
            if row[model_name] is not None:
                print(f"{row[model_name]:>6.1%} ({row[f'{model_name}_n']:>2})", end="")
            else:
                print(f"{'N/A':>15}", end="")
        for provider in sorted(llm_results.keys()):
            if row[f"llm_{provider}"] is not None:
                print(f"{row[f'llm_{provider}']:>6.1%} ({row[f'llm_{provider}_n']:>2})", end="")
            else:
                print(f"{'N/A':>15}", end="")
        print()
    
    # Save to file
    output_file = results_dir / 'COMPREHENSIVE_MODEL_COMPARISON.md'
    with open(output_file, 'w') as f:
        f.write("# Comprehensive Model Comparison - EU-Emotion Dataset\n\n")
        f.write("## Overall Accuracy\n\n")
        f.write("| Model | Accuracy | Trials |\n")
        f.write("|-------|----------|--------|\n")
        
        # Vision models
        for model_name, results in sorted(vision_results.items(), key=lambda x: x[1]['overall_accuracy'], reverse=True):
            f.write(f"| {model_name} | {results['overall_accuracy']:.2%} | {results['total_trials']} |\n")
        
        # LLM models
        for provider, results in sorted(llm_results.items(), key=lambda x: x[1]['overall_accuracy'], reverse=True):
            f.write(f"| LLM ({provider.upper()}) | {results['overall_accuracy']:.2%} | {results['total_trials']} |\n")
        
        f.write("\n## Per-Emotion Accuracy\n\n")
        f.write("| Emotion | " + " | ".join([m[:15] for m in sorted(vision_results.keys())] + [f"LLM-{p[:8]}" for p in sorted(llm_results.keys())]) + " |\n")
        f.write("|---------|" + "|".join(["---" for _ in range(len(vision_results) + len(llm_results))]) + "|\n")
        
        for _, row in df.iterrows():
            values = []
            for model_name in sorted(vision_results.keys()):
                if row[model_name] is not None:
                    values.append(f"{row[model_name]:.1%} (n={row[f'{model_name}_n']})")
                else:
                    values.append("N/A")
            for provider in sorted(llm_results.keys()):
                if row[f"llm_{provider}"] is not None:
                    values.append(f"{row[f'llm_{provider}']:.1%} (n={row[f'llm_{provider}_n']})")
                else:
                    values.append("N/A")
            f.write(f"| {row['emotion']} | " + " | ".join(values) + " |\n")
    
    print(f"\n✅ Comprehensive comparison saved to: {output_file}")
    
    # Save CSV
    csv_file = results_dir / 'comprehensive_comparison.csv'
    df.to_csv(csv_file, index=False)
    print(f"✅ CSV comparison saved to: {csv_file}")

if __name__ == "__main__":
    main()
