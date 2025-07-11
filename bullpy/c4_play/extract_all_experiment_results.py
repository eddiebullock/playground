#!/usr/bin/env python3
"""
Extract and summarize results from all experiment output folders.
Updated to work with new experiment structure where each experiment saves to its own directory.
"""
import os
import re
import glob
from pathlib import Path

def extract_metrics_from_yaml(yaml_path):
    """Extract readable metrics from a YAML file as text."""
    with open(yaml_path, 'r') as f:
        content = f.read()
    # Try to find test_metrics section
    test_metrics_match = re.search(r'test_metrics:\s*\n(.*?)(?=\n\w+:|$)', content, re.DOTALL)
    if not test_metrics_match:
        return None
    test_metrics_text = test_metrics_match.group(1)
    # Extract model results
    model_sections = re.findall(r'(\w+):\s*\n(.*?)(?=\n\w+:|$)', test_metrics_text, re.DOTALL)
    results = []
    for model_name, metrics_text in model_sections:
        metrics = {}
        for line in metrics_text.split('\n'):
            line = line.strip()
            if ':' in line and not line.startswith('-') and not '!!python' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                try:
                    float_value = float(value)
                    metrics[key] = float_value
                except ValueError:
                    continue
        if metrics:
            results.append((model_name, metrics))
    return results if results else None

def main():
    base_dir = 'experiments/outputs/'
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("ALL EXPERIMENT RESULTS SUMMARY")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    found_any = False
    
    # Track all results for comparison
    all_results = []
    
    for exp_folder in sorted(os.listdir(base_dir)):
        exp_path = os.path.join(base_dir, exp_folder)
        if not os.path.isdir(exp_path):
            continue
            
        # Look for model_results.yaml in the experiment directory
        yaml_files = [
            os.path.join(exp_path, f) for f in ["model_results.yaml"]
            if os.path.exists(os.path.join(exp_path, f))
        ]
        
        if not yaml_files:
            continue
            
        # Use the first found YAML file
        yaml_path = yaml_files[0]
        metrics = extract_metrics_from_yaml(yaml_path)
        if not metrics:
            continue
            
        found_any = True
        summary_lines.append(f"EXPERIMENT: {exp_folder}")
        summary_lines.append("-" * 60)
        
        for model_name, model_metrics in metrics:
            summary_lines.append(f"MODEL: {model_name}")
            for k, v in model_metrics.items():
                summary_lines.append(f"  {k}: {v:.4f}")
            summary_lines.append("")
            
            # Store for comparison
            all_results.append({
                'experiment': exp_folder,
                'model': model_name,
                'metrics': model_metrics
            })
        
        summary_lines.append("")
    
    if not found_any:
        summary_lines.append("No experiment results found.")
        summary_lines.append("Check that experiments have completed successfully.")
    else:
        # Add comparison section
        summary_lines.append("=" * 80)
        summary_lines.append("PERFORMANCE COMPARISON (Sorted by F1 Score)")
        summary_lines.append("=" * 80)
        summary_lines.append("")
        
        # Sort by F1 score
        all_results.sort(key=lambda x: x['metrics'].get('f1_score', 0), reverse=True)
        
        # Header
        header = f"{'Experiment':<30} {'Model':<20} {'F1-Score':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'ROC-AUC':<10}"
        summary_lines.append(header)
        summary_lines.append("-" * len(header))
        
        # Add rows
        for result in all_results:
            metrics = result['metrics']
            row = f"{result['experiment']:<30} {result['model']:<20} {metrics.get('f1_score', 0):<10.3f} {metrics.get('accuracy', 0):<10.3f} {metrics.get('precision', 0):<10.3f} {metrics.get('recall', 0):<10.3f} {metrics.get('roc_auc', 0):<10.3f}"
            summary_lines.append(row)
        
        summary_lines.append("")
        summary_lines.append("")
        
        # Best performing experiments
        summary_lines.append("TOP 5 PERFORMING EXPERIMENTS (by F1-Score)")
        summary_lines.append("-" * 50)
        summary_lines.append("")
        
        for i, result in enumerate(all_results[:5], 1):
            metrics = result['metrics']
            summary_lines.append(f"{i}. {result['experiment']} - {result['model']}")
            summary_lines.append(f"   F1-Score: {metrics.get('f1_score', 0):.3f}")
            summary_lines.append(f"   Accuracy: {metrics.get('accuracy', 0):.3f}")
            summary_lines.append(f"   Precision: {metrics.get('precision', 0):.3f}")
            summary_lines.append(f"   Recall: {metrics.get('recall', 0):.3f}")
            summary_lines.append(f"   ROC-AUC: {metrics.get('roc_auc', 0):.3f}")
            summary_lines.append("")
    
    # Save summary
    with open('all_experiment_results_summary.txt', 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print('\n'.join(summary_lines))
    print("\nResults also saved to: all_experiment_results_summary.txt")

if __name__ == "__main__":
    main() 