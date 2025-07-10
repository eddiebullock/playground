#!/usr/bin/env python3
"""
Analyze all experiment results and create clear, readable text reports.
This script reads all experiment outputs and creates summary text files.
"""

import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_yaml_results(yaml_file):
    """Parse YAML results file and extract metrics."""
    try:
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        
        if not data:
            return None
            
        # Extract test metrics
        test_metrics = data.get('test_metrics', {})
        if not test_metrics:
            return None
            
        results = {}
        for model_name, metrics in test_metrics.items():
            results[model_name] = {
                'accuracy': metrics.get('accuracy', 0.0),
                'precision': metrics.get('precision', 0.0),
                'recall': metrics.get('recall', 0.0),
                'f1_score': metrics.get('f1_score', 0.0),
                'roc_auc': metrics.get('roc_auc', 0.0),
                'balanced_accuracy': metrics.get('balanced_accuracy', 0.0),
                'sensitivity': metrics.get('sensitivity', 0.0),
                'specificity': metrics.get('specificity', 0.0)
            }
        
        return results
    except Exception as e:
        logger.warning(f"Could not parse {yaml_file}: {e}")
        return None

def analyze_experiment_outputs(outputs_dir):
    """Analyze all experiment outputs and create summary."""
    outputs_path = Path(outputs_dir)
    if not outputs_path.exists():
        logger.error(f"Outputs directory {outputs_dir} does not exist")
        return
    
    all_results = {}
    experiment_summaries = {}
    
    # Find all experiment directories
    for exp_dir in outputs_path.iterdir():
        if not exp_dir.is_dir():
            continue
            
        exp_name = exp_dir.name
        logger.info(f"Analyzing experiment: {exp_name}")
        
        # Look for temp_config.yaml files
        config_file = exp_dir / 'temp_config.yaml'
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                experiment_summaries[exp_name] = {
                    'config': config,
                    'type': 'model_training'
                }
            except Exception as e:
                logger.warning(f"Could not read config for {exp_name}: {e}")
        
        # Look for model_results.yaml in logs
        logs_dir = Path('experiments/logs')
        model_results_file = logs_dir / 'model_results.yaml'
        if model_results_file.exists():
            results = parse_yaml_results(model_results_file)
            if results:
                all_results[exp_name] = results
    
    return all_results, experiment_summaries

def create_performance_summary(all_results, output_file):
    """Create a clear performance summary text file."""
    
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("AUTISM DIAGNOSIS PREDICTION - EXPERIMENT RESULTS SUMMARY")
    summary_lines.append("=" * 80)
    summary_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append("")
    
    if not all_results:
        summary_lines.append("No results found in experiment outputs.")
        summary_lines.append("Check that experiments have completed successfully.")
    else:
        # Create comparison table
        summary_lines.append("PERFORMANCE COMPARISON BY EXPERIMENT")
        summary_lines.append("-" * 80)
        summary_lines.append("")
        
        # Header
        header = f"{'Experiment':<25} {'Model':<15} {'F1-Score':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'ROC-AUC':<10}"
        summary_lines.append(header)
        summary_lines.append("-" * len(header))
        
        # Sort experiments by best F1 score
        all_experiments = []
        for exp_name, results in all_results.items():
            for model_name, metrics in results.items():
                all_experiments.append({
                    'experiment': exp_name,
                    'model': model_name,
                    'f1': metrics['f1_score'],
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'roc_auc': metrics['roc_auc']
                })
        
        # Sort by F1 score (descending)
        all_experiments.sort(key=lambda x: x['f1'], reverse=True)
        
        # Add rows
        for exp in all_experiments:
            row = f"{exp['experiment']:<25} {exp['model']:<15} {exp['f1']:<10.3f} {exp['accuracy']:<10.3f} {exp['precision']:<10.3f} {exp['recall']:<10.3f} {exp['roc_auc']:<10.3f}"
            summary_lines.append(row)
        
        summary_lines.append("")
        summary_lines.append("")
        
        # Best performing experiments
        summary_lines.append("TOP PERFORMING EXPERIMENTS (by F1-Score)")
        summary_lines.append("-" * 50)
        summary_lines.append("")
        
        for i, exp in enumerate(all_experiments[:5], 1):
            summary_lines.append(f"{i}. {exp['experiment']} - {exp['model']}")
            summary_lines.append(f"   F1-Score: {exp['f1']:.3f}")
            summary_lines.append(f"   Accuracy: {exp['accuracy']:.3f}")
            summary_lines.append(f"   Precision: {exp['precision']:.3f}")
            summary_lines.append(f"   Recall: {exp['recall']:.3f}")
            summary_lines.append(f"   ROC-AUC: {exp['roc_auc']:.3f}")
            summary_lines.append("")
        
        # F1 Score improvements analysis
        summary_lines.append("F1-SCORE IMPROVEMENT ANALYSIS")
        summary_lines.append("-" * 40)
        summary_lines.append("")
        
        if len(all_experiments) > 1:
            best_f1 = all_experiments[0]['f1']
            worst_f1 = all_experiments[-1]['f1']
            improvement = best_f1 - worst_f1
            improvement_pct = (improvement / worst_f1) * 100 if worst_f1 > 0 else 0
            
            summary_lines.append(f"Best F1-Score: {best_f1:.3f} ({all_experiments[0]['experiment']} - {all_experiments[0]['model']})")
            summary_lines.append(f"Worst F1-Score: {worst_f1:.3f} ({all_experiments[-1]['experiment']} - {all_experiments[-1]['model']})")
            summary_lines.append(f"Absolute Improvement: {improvement:.3f}")
            summary_lines.append(f"Percentage Improvement: {improvement_pct:.1f}%")
            summary_lines.append("")
    
    # Write summary file
    with open(output_file, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    logger.info(f"Performance summary saved to: {output_file}")

def create_detailed_reports(all_results, output_dir):
    """Create detailed reports for each experiment."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for exp_name, results in all_results.items():
        report_file = output_path / f"{exp_name}_detailed_report.txt"
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append(f"DETAILED REPORT: {exp_name.upper()}")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        for model_name, metrics in results.items():
            report_lines.append(f"MODEL: {model_name}")
            report_lines.append("-" * 30)
            report_lines.append(f"F1-Score:        {metrics['f1_score']:.4f}")
            report_lines.append(f"Accuracy:        {metrics['accuracy']:.4f}")
            report_lines.append(f"Precision:       {metrics['precision']:.4f}")
            report_lines.append(f"Recall:          {metrics['recall']:.4f}")
            report_lines.append(f"ROC-AUC:         {metrics['roc_auc']:.4f}")
            report_lines.append(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
            report_lines.append(f"Sensitivity:     {metrics['sensitivity']:.4f}")
            report_lines.append(f"Specificity:     {metrics['specificity']:.4f}")
            report_lines.append("")
        
        # Write detailed report
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Detailed report saved: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze experiment results and create text reports')
    parser.add_argument('--outputs-dir', type=str, default='experiments/outputs',
                       help='Directory containing experiment outputs')
    parser.add_argument('--output-file', type=str, default='experiment_results_summary.txt',
                       help='Output file for performance summary')
    parser.add_argument('--detailed-dir', type=str, default='experiment_reports',
                       help='Directory for detailed reports')
    
    args = parser.parse_args()
    
    logger.info("Analyzing experiment results...")
    
    # Analyze all experiment outputs
    all_results, experiment_summaries = analyze_experiment_outputs(args.outputs_dir)
    
    if not all_results:
        logger.warning("No results found. Check that experiments completed successfully.")
        return
    
    # Create performance summary
    create_performance_summary(all_results, args.output_file)
    
    # Create detailed reports
    create_detailed_reports(all_results, args.detailed_dir)
    
    logger.info("Analysis complete!")
    logger.info(f"Summary saved to: {args.output_file}")
    logger.info(f"Detailed reports saved to: {args.detailed_dir}")

if __name__ == "__main__":
    main() 