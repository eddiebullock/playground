#!/usr/bin/env python3
"""
Generate comparison report from existing results.

Useful when some models failed or were skipped.
"""

import json
import pandas as pd
from pathlib import Path
import sys
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_existing_results(results_dir: Path) -> list:
    """Load results from existing model directories."""
    results = []
    
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        metrics_file = model_dir / "metrics.json"
        if not metrics_file.exists():
            continue
        
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            results.append({
                'model_name': model_dir.name,
                'accuracy': metrics.get('overall_accuracy', 0.0),
                'num_trials': metrics.get('num_trials', 0),
                'top_2_accuracy': metrics.get('top_2_accuracy', None),
            })
            logger.info(f"Loaded results for {model_dir.name}: {metrics.get('overall_accuracy', 0.0):.4f}")
        except Exception as e:
            logger.warning(f"Error loading {model_dir.name}: {e}")
    
    return results


def create_report(results: list, output_dir: Path):
    """Create comparison report from results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Overall results table
    overall_rows = []
    for result in results:
        row = {
            'model': result['model_name'],
            'accuracy': result['accuracy'],
            'num_trials': result['num_trials'],
        }
        if result.get('top_2_accuracy') is not None:
            row['top_2_accuracy'] = result['top_2_accuracy']
        overall_rows.append(row)
    
    df = pd.DataFrame(overall_rows)
    df = df.sort_values('accuracy', ascending=False)
    df.to_csv(output_dir / "overall_results.csv", index=False)
    
    # Create markdown report
    report_file = output_dir / "comparison_report.md"
    with open(report_file, 'w') as f:
        f.write("# EU-Emotion Model Comparison Report\n\n")
        f.write("## Overall Results\n\n")
        f.write("| Model | Accuracy | Top-2 Accuracy | Trials |\n")
        f.write("|-------|----------|----------------|--------|\n")
        for _, row in df.iterrows():
            top2 = f"{row.get('top_2_accuracy', 0):.4f}" if 'top_2_accuracy' in row and pd.notna(row.get('top_2_accuracy')) else "N/A"
            f.write(f"| {row['model']} | {row['accuracy']:.4f} | {top2} | {row['num_trials']} |\n")
        f.write("\n")
        
        f.write("## Key Findings\n\n")
        best_model = df.iloc[0]
        f.write(f"- **Best Model**: {best_model['model']} ({best_model['accuracy']:.1%} accuracy)\n")
        f.write(f"- **Random Baseline**: ~25% (4-choice forced-choice)\n")
        f.write(f"- **Improvement over Random**: {best_model['accuracy'] / 0.25:.2f}x\n\n")
        
        f.write("## Per-Emotion Analysis\n\n")
        f.write("See individual model directories for detailed per-emotion results:\n")
        for _, row in df.iterrows():
            f.write(f"- `{row['model']}/per_emotion_results.csv`\n")
        f.write("\n")
        
        f.write("## Next Steps\n\n")
        f.write("1. Review per-emotion performance to identify strengths/weaknesses\n")
        f.write("2. Fine-tune models on emotions with low accuracy\n")
        f.write("3. Consider ensemble methods combining top models\n")
        f.write("4. Run LLM models when API quota is available\n")
    
    logger.info(f"Comparison report saved to {report_file}")
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    print(f"{'='*60}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate comparison report from existing results")
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results/eu_emotion_model_comparison',
        help='Results directory'
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        sys.exit(1)
    
    results = load_existing_results(results_dir)
    
    if not results:
        logger.error("No results found!")
        sys.exit(1)
    
    create_report(results, results_dir)


if __name__ == "__main__":
    main()
