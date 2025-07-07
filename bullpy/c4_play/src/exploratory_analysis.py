"""
Exploratory Data Analysis (EDA) module for autism diagnosis prediction project.

This module handles:
- Dataset summarization
- Distribution visualization
- Missing value analysis
- Bias detection
- Feature correlation analysis
- Target variable analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any
import yaml
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ExploratoryAnalyzer:
    """
    Handles comprehensive exploratory data analysis for the autism diagnosis dataset.
    """
    
    def __init__(self, output_dir: str = "results/figures"):
        """
        Initialize the exploratory analyzer.
        
        Args:
            output_dir: Directory to save plots and analysis results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_results = {}
        
    def analyze_dataset_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Provide a comprehensive overview of the dataset.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with dataset overview information
        """
        logger.info("Analyzing dataset overview...")
        
        overview = {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum(),
            'dtypes': df.dtypes.value_counts().to_dict(),
            'missing_values': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
            'duplicates': df.duplicated().sum(),
            'unique_values_per_column': df.nunique().to_dict()
        }
        
        # Categorize columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        overview['numeric_columns'] = numeric_cols
        overview['categorical_columns'] = categorical_cols
        
        # Basic statistics for numeric columns
        if numeric_cols:
            overview['numeric_stats'] = df[numeric_cols].describe().to_dict()
        
        logger.info(f"Dataset shape: {overview['shape']}")
        logger.info(f"Memory usage: {overview['memory_usage'] / 1024**2:.2f} MB")
        logger.info(f"Missing values: {overview['missing_values']} ({overview['missing_percentage']:.2f}%)")
        
        self.analysis_results['overview'] = overview
        return overview
    
    def analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detailed analysis of missing values.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with missing value analysis
        """
        logger.info("Analyzing missing values...")
        
        missing_analysis = {}
        
        # Overall missing values
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        missing_df = pd.DataFrame({
            'missing_count': missing_counts,
            'missing_percentage': missing_percentages
        }).sort_values('missing_count', ascending=False)
        
        missing_analysis['missing_summary'] = missing_df.to_dict()
        
        # Missing patterns
        missing_analysis['columns_with_missing'] = missing_df[missing_df['missing_count'] > 0].index.tolist()
        missing_analysis['total_missing'] = missing_counts.sum()
        missing_analysis['total_missing_percentage'] = (missing_counts.sum() / (df.shape[0] * df.shape[1])) * 100
        
        # Create missing value heatmap
        self._plot_missing_heatmap(df)
        
        logger.info(f"Columns with missing values: {len(missing_analysis['columns_with_missing'])}")
        logger.info(f"Total missing values: {missing_analysis['total_missing']}")
        
        self.analysis_results['missing_analysis'] = missing_analysis
        return missing_analysis
    
    def analyze_target_variable(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """
        Analyze the target variable distribution and characteristics.
        
        Args:
            df: DataFrame containing the target
            target_col: Name of the target column
            
        Returns:
            Dictionary with target analysis
        """
        logger.info(f"Analyzing target variable: {target_col}")
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        target_analysis = {}
        y = df[target_col]
        
        # Basic statistics
        target_analysis['value_counts'] = y.value_counts().to_dict()
        target_analysis['value_counts_normalized'] = y.value_counts(normalize=True).to_dict()
        target_analysis['total_samples'] = len(y)
        target_analysis['unique_values'] = y.nunique()
        
        # Class imbalance analysis
        class_counts = y.value_counts()
        target_analysis['class_imbalance_ratio'] = class_counts.max() / class_counts.min()
        target_analysis['minority_class'] = class_counts.idxmin()
        target_analysis['majority_class'] = class_counts.idxmax()
        
        # Create target distribution plot
        self._plot_target_distribution(y, target_col)
        
        logger.info(f"Target distribution: {target_analysis['value_counts']}")
        logger.info(f"Class imbalance ratio: {target_analysis['class_imbalance_ratio']:.2f}")
        
        self.analysis_results['target_analysis'] = target_analysis
        return target_analysis
    
    def analyze_feature_distributions(self, df: pd.DataFrame, target_col: str = None) -> Dict[str, Any]:
        """
        Analyze distributions of features, optionally stratified by target.
        
        Args:
            df: DataFrame to analyze
            target_col: Optional target column for stratified analysis
            
        Returns:
            Dictionary with feature distribution analysis
        """
        logger.info("Analyzing feature distributions...")
        
        distribution_analysis = {}
        
        # Numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col and target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        if numeric_cols:
            distribution_analysis['numeric_features'] = self._analyze_numeric_distributions(
                df, numeric_cols, target_col
            )
        
        # Categorical features
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            distribution_analysis['categorical_features'] = self._analyze_categorical_distributions(
                df, categorical_cols, target_col
            )
        
        self.analysis_results['distribution_analysis'] = distribution_analysis
        return distribution_analysis
    
    def analyze_correlations(self, df: pd.DataFrame, target_col: str = None) -> Dict[str, Any]:
        """
        Analyze correlations between features and with target.
        
        Args:
            df: DataFrame to analyze
            target_col: Optional target column
            
        Returns:
            Dictionary with correlation analysis
        """
        logger.info("Analyzing correlations...")
        
        correlation_analysis = {}
        
        # Numeric columns for correlation
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col and target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        if len(numeric_cols) > 1:
            # Feature correlations
            corr_matrix = df[numeric_cols].corr()
            correlation_analysis['feature_correlations'] = corr_matrix.to_dict()
            
            # Create correlation heatmap
            self._plot_correlation_heatmap(corr_matrix)
            
            # High correlations
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.8:
                        high_corr_pairs.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': corr_val
                        })
            
            correlation_analysis['high_correlations'] = high_corr_pairs
        
        # Target correlations
        if target_col and target_col in df.columns:
            target_correlations = df[numeric_cols].corrwith(df[target_col]).sort_values(key=abs, ascending=False)
            correlation_analysis['target_correlations'] = target_correlations.to_dict()
            
            # Create target correlation plot
            self._plot_target_correlations(target_correlations, target_col)
        
        self.analysis_results['correlation_analysis'] = correlation_analysis
        return correlation_analysis
    
    def analyze_questionnaire_blocks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Specialized analysis for questionnaire blocks (SPQ, EQ, SQR, AQ).
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with questionnaire analysis
        """
        logger.info("Analyzing questionnaire blocks...")
        
        questionnaire_analysis = {}
        
        # Identify questionnaire columns
        questionnaire_blocks = {
            'SPQ': [col for col in df.columns if 'spq' in col.lower()],
            'EQ': [col for col in df.columns if 'eq' in col.lower()],
            'SQR': [col for col in df.columns if 'sqr' in col.lower()],
            'AQ': [col for col in df.columns if 'aq' in col.lower()]
        }
        
        questionnaire_analysis['blocks'] = questionnaire_blocks
        
        # Analyze each block
        for block_name, block_cols in questionnaire_blocks.items():
            if block_cols:
                block_df = df[block_cols]
                questionnaire_analysis[block_name] = {
                    'num_questions': len(block_cols),
                    'missing_values': block_df.isnull().sum().sum(),
                    'mean_score': block_df.mean(axis=1).mean(),
                    'std_score': block_df.mean(axis=1).std(),
                    'score_range': [block_df.mean(axis=1).min(), block_df.mean(axis=1).max()]
                }
        
        # Create questionnaire analysis plots
        self._plot_questionnaire_analysis(df, questionnaire_blocks)
        
        self.analysis_results['questionnaire_analysis'] = questionnaire_analysis
        return questionnaire_analysis
    
    def analyze_demographics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze demographic variables.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with demographic analysis
        """
        logger.info("Analyzing demographic variables...")
        
        demographic_analysis = {}
        
        # Common demographic columns
        demo_cols = ['age', 'sex', 'handedness', 'education', 'occupation', 'country_region']
        available_demo_cols = [col for col in demo_cols if col in df.columns]
        
        demographic_analysis['available_demographics'] = available_demo_cols
        
        for col in available_demo_cols:
            if col in df.columns:
                if df[col].dtype in ['object', 'category']:
                    # Categorical demographic
                    demographic_analysis[col] = {
                        'type': 'categorical',
                        'value_counts': df[col].value_counts().to_dict(),
                        'missing_count': df[col].isnull().sum()
                    }
                else:
                    # Numeric demographic (e.g., age)
                    demographic_analysis[col] = {
                        'type': 'numeric',
                        'mean': df[col].mean(),
                        'std': df[col].std(),
                        'min': df[col].min(),
                        'max': df[col].max(),
                        'missing_count': df[col].isnull().sum()
                    }
        
        # Create demographic plots
        self._plot_demographics(df, available_demo_cols)
        
        self.analysis_results['demographic_analysis'] = demographic_analysis
        return demographic_analysis
    
    def _analyze_numeric_distributions(self, df: pd.DataFrame, numeric_cols: List[str], 
                                     target_col: str = None) -> Dict[str, Any]:
        """Analyze distributions of numeric features."""
        numeric_analysis = {}
        
        for col in numeric_cols[:10]:  # Limit to first 10 for performance
            numeric_analysis[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'skewness': df[col].skew(),
                'missing_count': df[col].isnull().sum()
            }
            
            # Create distribution plot
            self._plot_numeric_distribution(df, col, target_col)
        
        return numeric_analysis
    
    def _analyze_categorical_distributions(self, df: pd.DataFrame, categorical_cols: List[str],
                                        target_col: str = None) -> Dict[str, Any]:
        """Analyze distributions of categorical features."""
        categorical_analysis = {}
        
        for col in categorical_cols[:10]:  # Limit to first 10 for performance
            categorical_analysis[col] = {
                'unique_values': df[col].nunique(),
                'value_counts': df[col].value_counts().head(10).to_dict(),
                'missing_count': df[col].isnull().sum()
            }
            
            # Create distribution plot
            self._plot_categorical_distribution(df, col, target_col)
        
        return categorical_analysis
    
    def _plot_missing_heatmap(self, df: pd.DataFrame) -> None:
        """Create missing value heatmap."""
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
        plt.title('Missing Values Heatmap')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'missing_values_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_target_distribution(self, y: pd.Series, target_col: str) -> None:
        """Create target distribution plot."""
        plt.figure(figsize=(10, 6))
        
        # Bar plot
        plt.subplot(1, 2, 1)
        y.value_counts().plot(kind='bar')
        plt.title(f'{target_col} Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        
        # Pie chart
        plt.subplot(1, 2, 2)
        y.value_counts().plot(kind='pie', autopct='%1.1f%%')
        plt.title(f'{target_col} Distribution (%)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{target_col}_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_heatmap(self, corr_matrix: pd.DataFrame) -> None:
        """Create correlation heatmap."""
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_target_correlations(self, target_correlations: pd.Series, target_col: str) -> None:
        """Create target correlation plot."""
        plt.figure(figsize=(10, 6))
        target_correlations.plot(kind='bar')
        plt.title(f'Feature Correlations with {target_col}')
        plt.xlabel('Features')
        plt.ylabel('Correlation')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{target_col}_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_questionnaire_analysis(self, df: pd.DataFrame, questionnaire_blocks: Dict[str, List[str]]) -> None:
        """Create questionnaire analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, (block_name, block_cols) in enumerate(questionnaire_blocks.items()):
            if block_cols and i < 4:
                block_scores = df[block_cols].mean(axis=1)
                axes[i].hist(block_scores, bins=30, alpha=0.7)
                axes[i].set_title(f'{block_name} Score Distribution')
                axes[i].set_xlabel('Mean Score')
                axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'questionnaire_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_demographics(self, df: pd.DataFrame, demo_cols: List[str]) -> None:
        """Create demographic analysis plots."""
        n_cols = len(demo_cols)
        if n_cols == 0:
            return
        
        fig, axes = plt.subplots(2, min(3, n_cols), figsize=(15, 10))
        if n_cols == 1:
            axes = [axes]
        else:
            axes = axes.ravel()
        
        for i, col in enumerate(demo_cols[:6]):  # Limit to 6 plots
            if col in df.columns:
                if df[col].dtype in ['object', 'category']:
                    # Categorical plot
                    df[col].value_counts().head(10).plot(kind='bar', ax=axes[i])
                    axes[i].set_title(f'{col} Distribution')
                    axes[i].tick_params(axis='x', rotation=45)
                else:
                    # Numeric plot
                    axes[i].hist(df[col].dropna(), bins=30, alpha=0.7)
                    axes[i].set_title(f'{col} Distribution')
                    axes[i].set_xlabel(col)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'demographics_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_numeric_distribution(self, df: pd.DataFrame, col: str, target_col: str = None) -> None:
        """Create distribution plot for numeric feature."""
        plt.figure(figsize=(10, 6))
        
        if target_col and target_col in df.columns:
            # Stratified by target
            for target_val in df[target_col].unique():
                subset = df[df[target_col] == target_val][col]
                plt.hist(subset.dropna(), alpha=0.7, label=f'{target_col}={target_val}')
            plt.legend()
        else:
            plt.hist(df[col].dropna(), bins=30, alpha=0.7)
        
        plt.title(f'{col} Distribution')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{col}_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_categorical_distribution(self, df: pd.DataFrame, col: str, target_col: str = None) -> None:
        """Create distribution plot for categorical feature."""
        plt.figure(figsize=(10, 6))
        
        if target_col and target_col in df.columns:
            # Crosstab with target
            crosstab = pd.crosstab(df[col], df[target_col], normalize='index')
            crosstab.plot(kind='bar', stacked=True)
            plt.title(f'{col} Distribution by {target_col}')
        else:
            df[col].value_counts().head(10).plot(kind='bar')
            plt.title(f'{col} Distribution')
        
        plt.xlabel(col)
        plt.ylabel('Proportion' if target_col else 'Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{col}_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_analysis_results(self, output_file: str = None) -> None:
        """Save analysis results to YAML file."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"experiments/logs/eda_results_{timestamp}.yaml"
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to native Python types for YAML serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            return obj
        
        # Recursively convert numpy types
        def recursive_convert(obj):
            if isinstance(obj, dict):
                return {k: recursive_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_convert(item) for item in obj]
            else:
                return convert_numpy_types(obj)
        
        converted_results = recursive_convert(self.analysis_results)
        
        with open(output_path, 'w') as f:
            yaml.dump(converted_results, f, default_flow_style=False)
        
        logger.info(f"Analysis results saved to {output_path}")

    def save_summary(self, target_col: str = None, output_file: str = None) -> None:
        """Save a concise summary of the EDA results in multiple formats."""
        summary = self._create_summary_dict(target_col)
        
        # Save as clean YAML
        yaml_file = output_file.replace('.html', '.yaml') if output_file else None
        if yaml_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            yaml_file = f"experiments/logs/eda_summary_{timestamp}.yaml"
        
        self._save_clean_yaml(summary, yaml_file)
        
        # Save as HTML report
        html_file = output_file if output_file and output_file.endswith('.html') else None
        if html_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_file = f"experiments/logs/eda_report_{timestamp}.html"
        
        self._save_html_report(summary, html_file)
        
        # Save as markdown summary
        md_file = html_file.replace('.html', '.md')
        self._save_markdown_summary(summary, md_file)
        
        logger.info(f"EDA results saved in multiple formats:")
        logger.info(f"  - YAML: {yaml_file}")
        logger.info(f"  - HTML: {html_file}")
        logger.info(f"  - Markdown: {md_file}")

    def _create_summary_dict(self, target_col: str = None) -> Dict[str, Any]:
        """Create a clean summary dictionary."""
        overview = self.analysis_results.get('overview', {})
        missing = self.analysis_results.get('missing_analysis', {})
        target = self.analysis_results.get('target_analysis', {})
        correlations = self.analysis_results.get('correlation_analysis', {})
        
        summary = {
            'dataset_info': {
                'shape': overview.get('shape', (0, 0)),
                'memory_mb': round(overview.get('memory_usage', 0) / (1024**2), 2),
                'missing_percentage': round(overview.get('missing_percentage', 0), 2),
                'duplicates': overview.get('duplicates', 0),
                'numeric_features': len(overview.get('numeric_columns', [])),
                'categorical_features': len(overview.get('categorical_columns', []))
            },
            'missing_data': {
                'total_missing': missing.get('total_missing', 0),
                'total_missing_percentage': round(missing.get('total_missing_percentage', 0), 2),
                'columns_with_missing': len(missing.get('columns_with_missing', [])),
                'top_missing_columns': []
            },
            'target_analysis': {},
            'feature_insights': {
                'top_correlated_features': [],
                'data_quality_score': 0
            }
        }
        
        # Add top missing columns
        missing_summary = missing.get('missing_summary', {})
        if 'missing_count' in missing_summary:
            missing_counts = missing_summary['missing_count']
            top_missing = sorted(missing_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            summary['missing_data']['top_missing_columns'] = [
                {'column': k, 'missing_count': int(v), 'missing_percentage': round((v / overview.get('shape', (1, 1))[0]) * 100, 2)}
                for k, v in top_missing if v > 0
            ]
        
        # Add target analysis
        if target:
            summary['target_analysis'] = {
                'total_samples': target.get('total_samples', 0),
                'unique_values': target.get('unique_values', 0),
                'class_imbalance_ratio': round(target.get('class_imbalance_ratio', 1), 2),
                'distribution': target.get('value_counts', {}),
                'distribution_percentage': target.get('value_counts_normalized', {})
            }
        
        # Add correlation insights
        if correlations and 'target_correlations' in correlations:
            corrs = correlations['target_correlations']
            top_corrs = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
            summary['feature_insights']['top_correlated_features'] = [
                {'feature': k, 'correlation': round(v, 3)}
                for k, v in top_corrs
            ]
        
        # Calculate data quality score
        missing_pct = summary['dataset_info']['missing_percentage']
        duplicate_pct = (summary['dataset_info']['duplicates'] / summary['dataset_info']['shape'][0]) * 100 if summary['dataset_info']['shape'][0] > 0 else 0
        quality_score = max(0, 100 - missing_pct - duplicate_pct)
        summary['feature_insights']['data_quality_score'] = round(quality_score, 1)
        
        return summary

    def _save_clean_yaml(self, summary: Dict[str, Any], output_file: str) -> None:
        """Save summary as clean YAML."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Clean YAML summary saved to {output_path}")

    def _save_html_report(self, summary: Dict[str, Any], output_file: str) -> None:
        """Save comprehensive HTML report."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        html_content = self._generate_html_report(summary)
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {output_path}")

    def _save_markdown_summary(self, summary: Dict[str, Any], output_file: str) -> None:
        """Save markdown summary."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        md_content = self._generate_markdown_summary(summary)
        
        with open(output_path, 'w') as f:
            f.write(md_content)
        
        logger.info(f"Markdown summary saved to {output_path}")

    def _generate_html_report(self, summary: Dict[str, Any]) -> str:
        """Generate comprehensive HTML report."""
        dataset_info = summary['dataset_info']
        missing_data = summary['missing_data']
        target_analysis = summary['target_analysis']
        feature_insights = summary['feature_insights']
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EDA Report - Autism Diagnosis Prediction</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .metric-card {{ background: #ecf0f1; padding: 20px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #3498db; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-item {{ background: white; padding: 15px; border-radius: 6px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
        .metric-label {{ color: #7f8c8d; font-size: 14px; text-transform: uppercase; }}
        .quality-score {{ font-size: 32px; font-weight: bold; text-align: center; padding: 20px; }}
        .quality-excellent {{ color: #27ae60; }}
        .quality-good {{ color: #f39c12; }}
        .quality-poor {{ color: #e74c3c; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f8f9fa; font-weight: bold; }}
        .warning {{ background-color: #fff3cd; border: 1px solid #ffeaa7; color: #856404; padding: 15px; border-radius: 6px; margin: 20px 0; }}
        .success {{ background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; padding: 15px; border-radius: 6px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Exploratory Data Analysis Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <h2>📈 Dataset Overview</h2>
        <div class="metric-grid">
            <div class="metric-item">
                <div class="metric-value">{dataset_info['shape'][0]:,}</div>
                <div class="metric-label">Total Samples</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">{dataset_info['shape'][1]}</div>
                <div class="metric-label">Features</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">{dataset_info['memory_mb']:.1f} MB</div>
                <div class="metric-label">Memory Usage</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">{dataset_info['missing_percentage']:.1f}%</div>
                <div class="metric-label">Missing Data</div>
            </div>
        </div>
        
        <div class="metric-grid">
            <div class="metric-item">
                <div class="metric-value">{dataset_info['numeric_features']}</div>
                <div class="metric-label">Numeric Features</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">{dataset_info['categorical_features']}</div>
                <div class="metric-label">Categorical Features</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">{dataset_info['duplicates']}</div>
                <div class="metric-label">Duplicate Rows</div>
            </div>
        </div>
        
        <h2>🎯 Data Quality Assessment</h2>
        <div class="quality-score {'quality-excellent' if feature_insights['data_quality_score'] >= 80 else 'quality-good' if feature_insights['data_quality_score'] >= 60 else 'quality-poor'}">
            {feature_insights['data_quality_score']}/100
        </div>
        <p style="text-align: center; color: #7f8c8d;">Data Quality Score</p>
        
        {self._generate_quality_assessment_html(summary)}
        
        <h2>❌ Missing Data Analysis</h2>
        <div class="metric-card">
            <p><strong>Total Missing Values:</strong> {missing_data['total_missing']:,} ({missing_data['total_missing_percentage']:.2f}%)</p>
            <p><strong>Columns with Missing Data:</strong> {missing_data['columns_with_missing']}</p>
        </div>
        
        {self._generate_missing_table_html(missing_data)}
        
        <h2>🎯 Target Variable Analysis</h2>
        {self._generate_target_analysis_html(target_analysis)}
        
        <h2>🔗 Feature Correlations</h2>
        {self._generate_correlation_table_html(feature_insights)}
        
        <h2>📊 Generated Visualizations</h2>
        <p>The following plots have been generated in the <code>results/figures/</code> directory:</p>
        <ul>
            <li>Missing value heatmap</li>
            <li>Target variable distribution</li>
            <li>Correlation heatmap</li>
            <li>Feature distribution plots</li>
            <li>Questionnaire analysis plots</li>
            <li>Demographics analysis plots</li>
        </ul>
        
        <div class="warning">
            <strong>⚠️ Recommendations:</strong>
            <ul>
                <li>Review columns with high missing percentages for potential removal or imputation strategies</li>
                <li>Consider class imbalance in target variable for model selection</li>
                <li>Examine highly correlated features for potential multicollinearity</li>
                <li>Validate data quality before proceeding with model training</li>
            </ul>
        </div>
    </div>
</body>
</html>
        """
        return html

    def _generate_quality_assessment_html(self, summary: Dict[str, Any]) -> str:
        """Generate quality assessment section."""
        missing_pct = summary['dataset_info']['missing_percentage']
        duplicate_pct = (summary['dataset_info']['duplicates'] / summary['dataset_info']['shape'][0]) * 100 if summary['dataset_info']['shape'][0] > 0 else 0
        
        if missing_pct < 5 and duplicate_pct < 1:
            return '<div class="success"><strong>✅ Excellent Data Quality:</strong> Low missing data and duplicates.</div>'
        elif missing_pct < 15 and duplicate_pct < 5:
            return '<div class="warning"><strong>⚠️ Good Data Quality:</strong> Moderate missing data, consider imputation strategies.</div>'
        else:
            return '<div class="warning"><strong>⚠️ Poor Data Quality:</strong> High missing data or duplicates detected.</div>'

    def _generate_missing_table_html(self, missing_data: Dict[str, Any]) -> str:
        """Generate missing data table."""
        if not missing_data['top_missing_columns']:
            return '<p>No missing data detected.</p>'
        
        html = '<table><thead><tr><th>Column</th><th>Missing Count</th><th>Missing %</th></tr></thead><tbody>'
        for item in missing_data['top_missing_columns'][:10]:
            html += f'<tr><td>{item["column"]}</td><td>{item["missing_count"]:,}</td><td>{item["missing_percentage"]:.1f}%</td></tr>'
        html += '</tbody></table>'
        return html

    def _generate_target_analysis_html(self, target_analysis: Dict[str, Any]) -> str:
        """Generate target analysis section."""
        if not target_analysis:
            return '<p>No target variable analysis available.</p>'
        
        html = f"""
        <div class="metric-card">
            <p><strong>Total Samples:</strong> {target_analysis['total_samples']:,}</p>
            <p><strong>Unique Values:</strong> {target_analysis['unique_values']}</p>
            <p><strong>Class Imbalance Ratio:</strong> {target_analysis['class_imbalance_ratio']:.2f}</p>
        </div>
        """
        
        if target_analysis['distribution']:
            html += '<h3>Target Distribution</h3><table><thead><tr><th>Class</th><th>Count</th><th>Percentage</th></tr></thead><tbody>'
            for class_name, count in target_analysis['distribution'].items():
                pct = target_analysis['distribution_percentage'].get(class_name, 0) * 100
                html += f'<tr><td>{class_name}</td><td>{count:,}</td><td>{pct:.1f}%</td></tr>'
            html += '</tbody></table>'
        
        return html

    def _generate_correlation_table_html(self, feature_insights: Dict[str, Any]) -> str:
        """Generate correlation table."""
        if not feature_insights['top_correlated_features']:
            return '<p>No correlation analysis available.</p>'
        
        html = '<table><thead><tr><th>Feature</th><th>Correlation</th><th>Strength</th></tr></thead><tbody>'
        for item in feature_insights['top_correlated_features'][:10]:
            corr = item['correlation']
            strength = 'Strong' if abs(corr) > 0.7 else 'Moderate' if abs(corr) > 0.3 else 'Weak'
            html += f'<tr><td>{item["feature"]}</td><td>{corr:.3f}</td><td>{strength}</td></tr>'
        html += '</tbody></table>'
        return html

    def _generate_markdown_summary(self, summary: Dict[str, Any]) -> str:
        """Generate markdown summary."""
        dataset_info = summary['dataset_info']
        missing_data = summary['missing_data']
        target_analysis = summary['target_analysis']
        feature_insights = summary['feature_insights']
        
        md = f"""# EDA Summary Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Dataset Overview
- **Shape:** {dataset_info['shape'][0]:,} samples × {dataset_info['shape'][1]} features
- **Memory Usage:** {dataset_info['memory_mb']:.1f} MB
- **Missing Data:** {dataset_info['missing_percentage']:.1f}%
- **Numeric Features:** {dataset_info['numeric_features']}
- **Categorical Features:** {dataset_info['categorical_features']}
- **Duplicates:** {dataset_info['duplicates']}

## Data Quality Score
**{feature_insights['data_quality_score']}/100**

## Missing Data Analysis
- **Total Missing:** {missing_data['total_missing']:,} ({missing_data['total_missing_percentage']:.2f}%)
- **Columns with Missing Data:** {missing_data['columns_with_missing']}

### Top Missing Columns
"""
        
        for item in missing_data['top_missing_columns'][:5]:
            md += f"- **{item['column']}:** {item['missing_count']:,} ({item['missing_percentage']:.1f}%)\n"
        
        if target_analysis:
            md += f"""
## Target Variable Analysis
- **Total Samples:** {target_analysis['total_samples']:,}
- **Unique Values:** {target_analysis['unique_values']}
- **Class Imbalance Ratio:** {target_analysis['class_imbalance_ratio']:.2f}

### Target Distribution
"""
            for class_name, count in target_analysis['distribution'].items():
                pct = target_analysis['distribution_percentage'].get(class_name, 0) * 100
                md += f"- **{class_name}:** {count:,} ({pct:.1f}%)\n"
        
        if feature_insights['top_correlated_features']:
            md += """
## Top Correlated Features
"""
            for item in feature_insights['top_correlated_features'][:10]:
                md += f"- **{item['feature']}:** {item['correlation']:.3f}\n"
        
        md += """
## Recommendations
- Review columns with high missing percentages
- Consider class imbalance in model selection
- Examine highly correlated features for multicollinearity
- Validate data quality before model training

## Generated Files
- HTML Report: `eda_report_*.html`
- YAML Summary: `eda_summary_*.yaml`
- Visualizations: `results/figures/`
"""
        
        return md

def main():
    """Main function to run exploratory analysis pipeline."""
    
    # Initialize analyzer
    analyzer = ExploratoryAnalyzer()
    
    # Load processed data (assuming it exists)
    try:
        df = pd.read_csv("data/processed/features_full.csv")
        logger.info(f"Loaded processed data: {df.shape}")
    except FileNotFoundError:
        logger.error("Processed data not found. Run data preprocessing first.")
        return
    
    # Run comprehensive analysis
    analyzer.analyze_dataset_overview(df)
    analyzer.analyze_missing_values(df)
    
    # Analyze target variable (assuming 'has_autism' exists)
    target_col = 'has_autism' if 'has_autism' in df.columns else None
    if target_col:
        analyzer.analyze_target_variable(df, target_col)
    
    # Analyze feature distributions
    analyzer.analyze_feature_distributions(df, target_col)
    
    # Analyze correlations
    analyzer.analyze_correlations(df, target_col)
    
    # Specialized analyses
    analyzer.analyze_questionnaire_blocks(df)
    analyzer.analyze_demographics(df)
    
    # Save results
    analyzer.save_analysis_results()
    # Save concise summary
    analyzer.save_summary(target_col)
    
    logger.info("Exploratory analysis completed successfully!")

if __name__ == "__main__":
    main() 