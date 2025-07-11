#!/usr/bin/env python3
"""
Extract and display current experiment results from model_results.yaml
"""

import yaml
import sys
from pathlib import Path
import numpy as np

def safe_yaml_load(file_path):
    """Safely load YAML file with numpy objects."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Try to extract just the test_metrics section
        lines = content.split('\n')
        test_metrics_start = None
        test_metrics_end = None
        
        for i, line in enumerate(lines):
            if line.strip() == 'test_metrics:':
                test_metrics_start = i
            elif test_metrics_start and line.startswith('  ') and not line.startswith('    '):
                test_metrics_end = i
                break
        
        if test_metrics_start is not None:
            if test_metrics_end is None:
                test_metrics_end = len(lines)
            
            # Extract just the test_metrics section
            metrics_lines = lines[test_metrics_start:test_metrics_end]
            metrics_content = '\n'.join(metrics_lines)
            
            # Parse this section
            metrics_data = yaml.safe_load(metrics_content)
            return {'test_metrics': metrics_data}
        
        return None
    except Exception as e:
        print(f"Could not parse YAML file: {e}")
        return None

def extract_current_results():
    """Extract results from the current model_results.yaml file."""
    
    # Look for model_results.yaml in experiments/logs
    results_file = Path('experiments/logs/model_results.yaml')
    
    if not results_file.exists():
        print("ERROR: model_results.yaml not found in experiments/logs/")
        print("Available files in experiments/logs/:")
        logs_dir = Path('experiments/logs')
        if logs_dir.exists():
            for file in logs_dir.iterdir():
                if file.is_file():
                    print(f"  - {file.name}")
        return
    
    # Try to parse the file
    data = safe_yaml_load(results_file)
    
    if not data:
        print("ERROR: Could not parse model_results.yaml")
        print("This might be due to numpy objects in the file.")
        print("Let me try to read the file manually...")
        
        # Try to read the file manually and extract basic info
        try:
            with open(results_file, 'r') as f:
                content = f.read()
            
            print("\nFile content preview:")
            print("-" * 50)
            lines = content.split('\n')[:20]
            for line in lines:
                print(line)
            print("...")
            
            # Look for model names and metrics
            print("\nAttempting to extract model information...")
            model_sections = []
            current_model = None
            
            for line in content.split('\n'):
                line = line.strip()
                if line and not line.startswith('  ') and ':' in line and not line.startswith('#'):
                    if ':' in line and not line.startswith('  '):
                        current_model = line.split(':')[0]
                        model_sections.append(current_model)
            
            if model_sections:
                print(f"Found potential model sections: {model_sections}")
            else:
                print("No clear model sections found")
            
        except Exception as e:
            print(f"Could not read file: {e}")
        return
    
    print("=" * 80)
    print("CURRENT EXPERIMENT RESULTS")
    print("=" * 80)
    print()
    
    # Extract test metrics
    test_metrics = data.get('test_metrics', {})
    if not test_metrics:
        print("No test metrics found in results file")
        return
    
    print("MODEL PERFORMANCE COMPARISON")
    print("-" * 50)
    print()
    
    # Create comparison table
    print(f"{'Model':<20} {'F1-Score':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'ROC-AUC':<10}")
    print("-" * 80)
    
    for model_name, metrics in test_metrics.items():
        if metrics is None:
            print(f"{model_name:<20} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
            continue
            
        f1 = metrics.get('f1_score', 0.0)
        acc = metrics.get('accuracy', 0.0)
        prec = metrics.get('precision', 0.0)
        rec = metrics.get('recall', 0.0)
        roc = metrics.get('roc_auc', 0.0)
        
        print(f"{model_name:<20} {f1:<10.3f} {acc:<10.3f} {prec:<10.3f} {rec:<10.3f} {roc:<10.3f}")
    
    print()
    print()
    
    # Find best model by F1 score
    valid_models = [(name, metrics) for name, metrics in test_metrics.items() if metrics is not None]
    if valid_models:
        best_model = max(valid_models, key=lambda x: x[1].get('f1_score', 0.0))
        best_f1 = best_model[1].get('f1_score', 0.0)
    else:
        best_model = None
        best_f1 = 0.0
    
    print("BEST PERFORMING MODEL")
    print("-" * 30)
    if best_model:
        print(f"Model: {best_model[0]}")
        print(f"F1-Score: {best_f1:.3f}")
    else:
        print("No valid models found")
    print()
    
    # Save to text file
    output_file = "current_results_summary.txt"
    with open(output_file, 'w') as f:
        f.write("CURRENT EXPERIMENT RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("MODEL PERFORMANCE COMPARISON\n")
        f.write("-" * 50 + "\n\n")
        
        for model_name, metrics in test_metrics.items():
            if metrics is None:
                f.write(f"MODEL: {model_name}\n")
                f.write("  No metrics available\n\n")
                continue
                
            f.write(f"MODEL: {model_name}\n")
            f.write(f"  F1-Score:        {metrics.get('f1_score', 0.0):.4f}\n")
            f.write(f"  Accuracy:        {metrics.get('accuracy', 0.0):.4f}\n")
            f.write(f"  Precision:       {metrics.get('precision', 0.0):.4f}\n")
            f.write(f"  Recall:          {metrics.get('recall', 0.0):.4f}\n")
            f.write(f"  ROC-AUC:         {metrics.get('roc_auc', 0.0):.4f}\n")
            f.write(f"  Balanced Accuracy: {metrics.get('balanced_accuracy', 0.0):.4f}\n")
            f.write(f"  Sensitivity:     {metrics.get('sensitivity', 0.0):.4f}\n")
            f.write(f"  Specificity:     {metrics.get('specificity', 0.0):.4f}\n")
            f.write("\n")
    
    print(f"Results also saved to: {output_file}")

if __name__ == "__main__":
    extract_current_results() 