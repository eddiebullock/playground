"""
Statistical analysis module for autism screening assessments.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class AssessmentAnalyzer:
    """Analyzes statistical properties of assessment data."""
    
    def __init__(self):
        """Initialize the assessment analyzer."""
        self.subscales = {
            'AQ': ['Social_Skills', 'Attention_Switching', 'Attention_to_Detail', 
                   'Communication', 'Imagination'],
            'SQ': ['Pattern_Recognition', 'Systemizing', 'Technical_Interest'],
            'EQ': ['Cognitive_Empathy', 'Emotional_Reactivity', 'Social_Skills']
        }
    
    def analyze_distributions(self, df: pd.DataFrame, assessment_type: str) -> Dict:
        """
        Analyze the distributions of assessment scores.
        
        Args:
            df: DataFrame containing assessment data
            assessment_type: Type of assessment
            
        Returns:
            Dictionary containing distribution statistics
        """
        results = {
            'total_score': self._analyze_score_distribution(df[f'{assessment_type}_Total']),
            'subscales': {}
        }
        
        # Analyze subscale distributions if available
        for subscale in self.subscales.get(assessment_type, []):
            if f'{assessment_type}_{subscale}' in df.columns:
                results['subscales'][subscale] = self._analyze_score_distribution(
                    df[f'{assessment_type}_{subscale}']
                )
        
        return results
    
    def _analyze_score_distribution(self, series: pd.Series) -> Dict:
        """
        Analyze the distribution of a score series.
        
        Args:
            series: Series containing scores
            
        Returns:
            Dictionary containing distribution statistics
        """
        # Basic statistics
        stats_dict = {
            'mean': series.mean(),
            'median': series.median(),
            'std': series.std(),
            'min': series.min(),
            'max': series.max(),
            'skewness': series.skew(),
            'kurtosis': series.kurtosis()
        }
        
        # Normality test
        normality_test = stats.normaltest(series.dropna())
        stats_dict['normality_test'] = {
            'statistic': normality_test[0],
            'p_value': normality_test[1]
        }
        
        return stats_dict
    
    def plot_distributions(self, df: pd.DataFrame, assessment_type: str, 
                         save_path: str = None) -> None:
        """
        Create distribution plots for assessment scores.
        
        Args:
            df: DataFrame containing assessment data
            assessment_type: Type of assessment
            save_path: Optional path to save the plots
        """
        # Set up the plotting style
        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.2)
        
        # Create figure with subplots
        n_subscales = len(self.subscales.get(assessment_type, []))
        fig, axes = plt.subplots(1 + n_subscales, 1, figsize=(10, 5 * (1 + n_subscales)))
        
        # Plot total score distribution
        sns.histplot(data=df, x=f'{assessment_type}_Total', ax=axes[0])
        axes[0].set_title(f'{assessment_type} Total Score Distribution')
        
        # Plot subscale distributions
        for idx, subscale in enumerate(self.subscales.get(assessment_type, []), 1):
            if f'{assessment_type}_{subscale}' in df.columns:
                sns.histplot(data=df, x=f'{assessment_type}_{subscale}', ax=axes[idx])
                axes[idx].set_title(f'{assessment_type} {subscale} Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def compare_groups(self, df: pd.DataFrame, assessment_type: str, 
                      group_column: str) -> Dict:
        """
        Compare assessment scores between different groups.
        
        Args:
            df: DataFrame containing assessment data
            assessment_type: Type of assessment
            group_column: Column name containing group labels
            
        Returns:
            Dictionary containing comparison statistics
        """
        results = {}
        
        # Compare total scores
        groups = df[group_column].unique()
        if len(groups) == 2:
            # T-test for two groups
            group1_scores = df[df[group_column] == groups[0]][f'{assessment_type}_Total']
            group2_scores = df[df[group_column] == groups[1]][f'{assessment_type}_Total']
            
            t_stat, p_val = stats.ttest_ind(group1_scores, group2_scores)
            results['total_score'] = {
                't_statistic': t_stat,
                'p_value': p_val,
                'effect_size': self._calculate_cohens_d(group1_scores, group2_scores)
            }
        else:
            # ANOVA for more than two groups
            f_stat, p_val = stats.f_oneway(
                *[df[df[group_column] == group][f'{assessment_type}_Total'] 
                  for group in groups]
            )
            results['total_score'] = {
                'f_statistic': f_stat,
                'p_value': p_val
            }
        
        return results
    
    def _calculate_cohens_d(self, group1: pd.Series, group2: pd.Series) -> float:
        """
        Calculate Cohen's d effect size.
        
        Args:
            group1: Scores for first group
            group2: Scores for second group
            
        Returns:
            Cohen's d effect size
        """
        n1, n2 = len(group1), len(group2)
        var1, var2 = group1.var(), group2.var()
        
        # Calculate pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        # Calculate Cohen's d
        return (group1.mean() - group2.mean()) / pooled_std 