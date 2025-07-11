#!/usr/bin/env python3
"""
Extract and display experiment results from YAML files.
Handles numpy objects and provides readable output.
"""

import yaml
import re
from pathlib import Path

def extract_numeric_values(yaml_content):
    """Extract numeric values from YAML content using regex."""
    # Pattern to match key-value pairs with numeric values
    pattern = r'(\w+):\s*([0-9]+\.?[0-9]*)'
    matches = re.findall(pattern, yaml_content)
    return dict(matches)

def display_results():
    """Display results from model_results.yaml."""
    results_file = Path("experiments/logs/model_results.yaml")
    
    if not results_file.exists():
        print("❌ Results file not found")
        return
    
    print("📊 MODEL RESULTS")
    print("="*50)
    
    with open(results_file, 'r') as f:
        content = f.read()
    
    # Extract numeric values
    results = extract_numeric_values(content)
    
    if results:
        print("\n📈 Performance Metrics:")
        for metric, value in results.items():
            try:
                float_val = float(value)
                print(f"  • {metric}: {float_val:.4f}")
            except ValueError:
                print(f"  • {metric}: {value}")
    else:
        print("⚠️  No numeric results found")
        print("💡 Check the file content manually:")
        print(content[:500] + "..." if len(content) > 500 else content)

def display_experiment_configs():
    """Display experiment configurations."""
    outputs_dir = Path("experiments/outputs")
    
    if not outputs_dir.exists():
        print("❌ No experiment outputs found")
        return
    
    print("\n🔬 EXPERIMENT CONFIGURATIONS")
    print("="*50)
    
    for exp_dir in outputs_dir.iterdir():
        if exp_dir.is_dir():
            config_file = exp_dir / "temp_config.yaml"
            if config_file.exists():
                print(f"\n📂 {exp_dir.name}:")
                try:
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    # Extract key settings
                    imb_method = config.get('imbalance_handling', {}).get('method', 'none')
                    models = list(config.get('models', {}).keys())
                    tuning_method = config.get('hyperparameter_tuning', {}).get('method', 'none')
                    
                    print(f"  • Imbalance handling: {imb_method}")
                    print(f"  • Models: {', '.join(models)}")
                    print(f"  • Tuning: {tuning_method}")
                    
                except Exception as e:
                    print(f"  ⚠️  Could not parse config: {e}")

def main():
    display_results()
    display_experiment_configs()
    
    print("\n" + "="*50)
    print("NEXT STEPS")
    print("="*50)
    print("1. Compare F1 scores across experiments")
    print("2. Look for the highest performing model")
    print("3. Check experiments/models/ for plots and saved models")
    print("4. Use the best settings in your main config files")

if __name__ == "__main__":
    main() 