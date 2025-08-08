#!/usr/bin/env python3
"""
HPC Results Analysis Script
This script analyzes and compares the results from HPC optimization runs.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob

class HPCResultsAnalyzer:
    """Analyze HPC optimization results"""
    
    def __init__(self, results_dir="results"):
        """Initialize the analyzer"""
        self.results_dir = results_dir
        self.results = self.load_all_results()
        
    def load_all_results(self):
        """Load all result files"""
        results = {}
        
        # Find all JSON result files
        json_files = glob.glob(os.path.join(self.results_dir, "*_results_*.json"))
        
        for file_path in json_files:
            with open(file_path, 'r') as f:
                result = json.load(f)
                model_name = result.get('model_name', result.get('ensemble_name', 'unknown'))
                results[model_name] = result
        
        return results
    
    def create_summary_table(self):
        """Create a summary table of all results"""
        summary_data = []
        
        for model_name, result in self.results.items():
            summary_data.append({
                'Model': model_name,
                'Test F1': result.get('test_f1', 0),
                'Optimized F1': result.get('optimized_f1', 0),
                'Test AUC': result.get('test_auc', 0),
                'CV Score': result.get('cv_score', 0),
                'Best Threshold': result.get('best_threshold', 0),
                'Timestamp': result.get('timestamp', '')
            })
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('Optimized F1', ascending=False)
        
        return df
    
    def plot_performance_comparison(self, save_path="plots/performance_comparison.png"):
        """Plot performance comparison"""
        df = self.create_summary_table()
        
        plt.figure(figsize=(12, 8))
        
        # Create bar plot
        x = np.arange(len(df))
        width = 0.35
        
        plt.bar(x - width/2, df['Test F1'], width, label='Test F1', alpha=0.8)
        plt.bar(x + width/2, df['Optimized F1'], width, label='Optimized F1', alpha=0.8)
        
        plt.xlabel('Models')
        plt.ylabel('F1 Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x, df['Model'], rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return save_path
    
    def plot_auc_comparison(self, save_path="plots/auc_comparison.png"):
        """Plot AUC comparison"""
        df = self.create_summary_table()
        
        plt.figure(figsize=(10, 6))
        
        plt.bar(df['Model'], df['Test AUC'], alpha=0.8, color='skyblue')
        plt.xlabel('Models')
        plt.ylabel('ROC-AUC Score')
        plt.title('Model AUC Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return save_path
    
    def analyze_feature_importance(self, model_name, save_path=None):
        """Analyze feature importance for a specific model"""
        # This would require loading the actual model files
        # For now, we'll create a placeholder
        print(f"Feature importance analysis for {model_name}")
        print("(This requires loading the actual model files)")
        
        return None
    
    def generate_report(self, output_file="hpc_optimization_report.md"):
        """Generate a comprehensive report"""
        df = self.create_summary_table()
        
        report = f"""# HPC Optimization Report

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary

Total models optimized: {len(self.results)}

## Performance Summary

{df.to_markdown(index=False)}

## Best Performing Models

### Top 3 by Optimized F1 Score:
"""
        
        top_3 = df.head(3)
        for i, row in top_3.iterrows():
            report += f"""
{i+1}. **{row['Model']}**
   - Optimized F1: {row['Optimized F1']:.4f}
   - Test F1: {row['Test F1']:.4f}
   - Test AUC: {row['Test AUC']:.4f}
   - Best Threshold: {row['Best Threshold']:.3f}
"""
        
        report += f"""
## Performance Improvements

### F1 Score Improvements:
"""
        
        for i, row in df.iterrows():
            improvement = row['Optimized F1'] - row['Test F1']
            report += f"- {row['Model']}: +{improvement:.4f}\n"
        
        report += f"""
## Recommendations

1. **Best Overall Model**: {df.iloc[0]['Model']} with F1={df.iloc[0]['Optimized F1']:.4f}
2. **Best AUC Model**: {df.loc[df['Test AUC'].idxmax(), 'Model']} with AUC={df['Test AUC'].max():.4f}
3. **Most Improved**: {df.loc[(df['Optimized F1'] - df['Test F1']).idxmax(), 'Model']}

## Next Steps

1. Deploy the best performing model
2. Conduct additional validation
3. Consider ensemble methods combining top models
4. Perform feature importance analysis
"""
        
        # Save report
        with open(output_file, 'w') as f:
            f.write(report)
        
        print(f"Report saved to: {output_file}")
        return output_file
    
    def compare_with_baseline(self, baseline_f1=0.68, baseline_auc=0.75):
        """Compare results with baseline performance"""
        df = self.create_summary_table()
        
        print("=== COMPARISON WITH BASELINE ===")
        print(f"Baseline F1: {baseline_f1:.4f}")
        print(f"Baseline AUC: {baseline_auc:.4f}")
        print()
        
        improvements = []
        for i, row in df.iterrows():
            f1_improvement = row['Optimized F1'] - baseline_f1
            auc_improvement = row['Test AUC'] - baseline_auc
            
            improvements.append({
                'Model': row['Model'],
                'F1_Improvement': f1_improvement,
                'AUC_Improvement': auc_improvement,
                'F1_Improvement_Percent': (f1_improvement / baseline_f1) * 100,
                'AUC_Improvement_Percent': (auc_improvement / baseline_auc) * 100
            })
        
        improvements_df = pd.DataFrame(improvements)
        improvements_df = improvements_df.sort_values('F1_Improvement', ascending=False)
        
        print("Improvements over baseline:")
        print(improvements_df.to_string(index=False))
        
        return improvements_df

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze HPC optimization results')
    parser.add_argument('--results-dir', default='results', help='Results directory')
    parser.add_argument('--baseline-f1', type=float, default=0.68, help='Baseline F1 score')
    parser.add_argument('--baseline-auc', type=float, default=0.75, help='Baseline AUC score')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = HPCResultsAnalyzer(args.results_dir)
    
    # Generate summary table
    df = analyzer.create_summary_table()
    print("=== HPC OPTIMIZATION RESULTS ===")
    print(df.to_string(index=False))
    
    # Create plots
    analyzer.plot_performance_comparison()
    analyzer.plot_auc_comparison()
    
    # Generate report
    analyzer.generate_report()
    
    # Compare with baseline
    analyzer.compare_with_baseline(args.baseline_f1, args.baseline_auc)
    
    print("\nAnalysis completed!")

if __name__ == "__main__":
    main() 