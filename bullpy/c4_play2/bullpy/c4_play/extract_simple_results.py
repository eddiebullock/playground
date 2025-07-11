#!/usr/bin/env python3
"""
Extract simple results from the YAML file, ignoring numpy objects.
"""

import yaml
import re

def extract_readable_metrics(yaml_file):
    """Extract readable metrics from YAML file, ignoring numpy objects."""
    
    with open(yaml_file, 'r') as f:
        content = f.read()
    
    print("=" * 80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)
    print()
    
    # Extract test_metrics section
    test_metrics_match = re.search(r'test_metrics:\s*\n(.*?)(?=\n\w+:|$)', content, re.DOTALL)
    if test_metrics_match:
        test_metrics_text = test_metrics_match.group(1)
        print("TEST METRICS")
        print("-" * 30)
        
        # Extract model results from test_metrics
        model_sections = re.findall(r'(\w+):\s*\n(.*?)(?=\n\w+:|$)', test_metrics_text, re.DOTALL)
        
        for model_name, metrics_text in model_sections:
            print(f"MODEL: {model_name}")
            print("-" * 20)
            
            # Extract readable metrics
            for line in metrics_text.split('\n'):
                line = line.strip()
                if ':' in line and not line.startswith('-') and not '!!python' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    try:
                        # Try to convert to float
                        float_value = float(value)
                        print(f"  {key}: {float_value:.4f}")
                    except ValueError:
                        # Skip if not a number
                        continue
            
            print()
    
    # Extract model_comparison section
    model_comparison_match = re.search(r'model_comparison:\s*\n(.*?)(?=\n\w+:|$)', content, re.DOTALL)
    if model_comparison_match:
        model_comparison_text = model_comparison_match.group(1)
        print("MODEL COMPARISON")
        print("-" * 30)
        
        # Extract individual model results
        model_sections = re.findall(r'- model: (\w+)\s*\n(.*?)(?=\n- model:|$)', model_comparison_text, re.DOTALL)
        
        for model_name, metrics_text in model_sections:
            print(f"MODEL: {model_name}")
            print("-" * 20)
            
            # Extract readable metrics
            for line in metrics_text.split('\n'):
                line = line.strip()
                if ':' in line and not line.startswith('-') and not '!!python' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    try:
                        # Try to convert to float
                        float_value = float(value)
                        print(f"  {key}: {float_value:.4f}")
                    except ValueError:
                        # Skip if not a number
                        continue
            
            print()
    
    # Save to file
    with open('simple_results_summary.txt', 'w') as f:
        f.write("EXPERIMENT RESULTS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        if test_metrics_match:
            f.write("TEST METRICS\n")
            f.write("-" * 30 + "\n\n")
            
            for model_name, metrics_text in model_sections:
                f.write(f"MODEL: {model_name}\n")
                f.write("-" * 20 + "\n")
                
                for line in metrics_text.split('\n'):
                    line = line.strip()
                    if ':' in line and not line.startswith('-') and not '!!python' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        try:
                            float_value = float(value)
                            f.write(f"  {key}: {float_value:.4f}\n")
                        except ValueError:
                            continue
                
                f.write("\n")
        
        if model_comparison_match:
            f.write("MODEL COMPARISON\n")
            f.write("-" * 30 + "\n\n")
            
            for model_name, metrics_text in model_sections:
                f.write(f"MODEL: {model_name}\n")
                f.write("-" * 20 + "\n")
                
                for line in metrics_text.split('\n'):
                    line = line.strip()
                    if ':' in line and not line.startswith('-') and not '!!python' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        try:
                            float_value = float(value)
                            f.write(f"  {key}: {float_value:.4f}\n")
                        except ValueError:
                            continue
                
                f.write("\n")
    
    print(f"Results also saved to: simple_results_summary.txt")

if __name__ == "__main__":
    extract_readable_metrics("experiments/logs/model_results.yaml") 