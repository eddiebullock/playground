#!/usr/bin/env python3
"""
Comprehensive analysis script for autism diagnosis prediction.
Addresses class imbalance and provides strategies for improving F1 score.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_raw_data(data_path: str) -> pd.DataFrame:
    """Load the raw data and perform initial exploration."""
    print(f"Loading data from {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"Raw data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return df

def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """Create the target variable based on diagnosis and autism_diagnosis columns."""
    print("Creating target variable...")
    
    # Get diagnosis columns
    diagnosis_cols = [col for col in df.columns if col.startswith('diagnosis_')]
    autism_diag_cols = [col for col in df.columns if col.startswith('autism_diagnosis_')]
    
    print(f"Found {len(diagnosis_cols)} diagnosis columns: {diagnosis_cols}")
    print(f"Found {len(autism_diag_cols)} autism diagnosis columns: {autism_diag_cols}")
    
    # Create target variable
    df['autism_any'] = 0  # Default to 0
    
    # Check autism diagnosis columns first
    if autism_diag_cols:
        for col in autism_diag_cols:
            # Check if any value indicates autism (1, 2, 3)
            mask = df[col].isin([1.0, 2.0, 3.0])
            df.loc[mask, 'autism_any'] = 1
    
    # Also check diagnosis columns for autism-related codes
    if diagnosis_cols:
        # Assuming autism might be coded as specific values in diagnosis columns
        # This is a heuristic - you may need to adjust based on your coding scheme
        for col in diagnosis_cols:
            # Check for autism-related codes (this is speculative)
            mask = df[col].isin([1.0, 2.0, 3.0])  # Adjust based on your coding
            df.loc[mask, 'autism_any'] = 1
    
    print(f"Target variable 'autism_any' created:")
    print(df['autism_any'].value_counts())
    print(f"Class balance: {df['autism_any'].value_counts(normalize=True)}")
    
    return df

def analyze_class_imbalance(df: pd.DataFrame) -> dict:
    """Analyze class imbalance and provide recommendations."""
    print("\n" + "="*50)
    print("CLASS IMBALANCE ANALYSIS")
    print("="*50)
    
    target_counts = df['autism_any'].value_counts()
    target_balance = df['autism_any'].value_counts(normalize=True)
    
    print(f"Class distribution:")
    print(f"  Class 0 (No Autism): {target_counts[0]} ({target_balance[0]:.1%})")
    print(f"  Class 1 (Autism): {target_counts[1]} ({target_balance[1]:.1%})")
    
    imbalance_ratio = target_balance[1] / target_balance[0]
    print(f"Imbalance ratio (minority/majority): {imbalance_ratio:.3f}")
    
    # Recommendations
    recommendations = []
    
    if imbalance_ratio < 0.1:  # Very imbalanced
        recommendations.append("SEVERE IMBALANCE DETECTED")
        recommendations.append("Recommended approaches:")
        recommendations.append("  1. SMOTE or ADASYN for synthetic minority oversampling")
        recommendations.append("  2. Class weights in model training")
        recommendations.append("  3. Focal Loss for deep learning models")
        recommendations.append("  4. Ensemble methods with balanced subsamples")
        recommendations.append("  5. Consider stratified sampling for validation")
    elif imbalance_ratio < 0.3:  # Moderately imbalanced
        recommendations.append("MODERATE IMBALANCE DETECTED")
        recommendations.append("Recommended approaches:")
        recommendations.append("  1. SMOTE for balanced dataset")
        recommendations.append("  2. Class weights in model training")
        recommendations.append("  3. Focus on precision/recall trade-off")
    else:
        recommendations.append("MILD IMBALANCE - standard approaches should work")
    
    print("\n".join(recommendations))
    
    return {
        'class_counts': target_counts.to_dict(),
        'class_balance': target_balance.to_dict(),
        'imbalance_ratio': imbalance_ratio,
        'recommendations': recommendations
    }

def analyze_questionnaire_data(df: pd.DataFrame) -> dict:
    """Analyze questionnaire data (AQ, EQ, SQ, SPQ) distributions."""
    print("\n" + "="*50)
    print("QUESTIONNAIRE DATA ANALYSIS")
    print("="*50)
    
    # Get questionnaire columns
    spq_cols = [col for col in df.columns if col.startswith('spq_')]
    eq_cols = [col for col in df.columns if col.startswith('eq_')]
    sqr_cols = [col for col in df.columns if col.startswith('sqr_')]
    aq_cols = [col for col in df.columns if col.startswith('aq_')]
    
    questionnaire_info = {}
    
    for name, cols in [('SPQ', spq_cols), ('EQ', eq_cols), ('SQR', sqr_cols), ('AQ', aq_cols)]:
        if cols:
            print(f"\n{name} Questionnaire ({len(cols)} items):")
            
            # Calculate scores
            df[f'{name.lower()}_total'] = df[cols].sum(axis=1)
            df[f'{name.lower()}_mean'] = df[cols].mean(axis=1)
            
            # Analyze by target
            autism_scores = df[df['autism_any'] == 1][f'{name.lower()}_total']
            control_scores = df[df['autism_any'] == 0][f'{name.lower()}_total']
            
            print(f"  Total score range: {df[f'{name.lower()}_total'].min():.1f} - {df[f'{name.lower()}_total'].max():.1f}")
            print(f"  Autism group mean: {autism_scores.mean():.1f} ± {autism_scores.std():.1f}")
            print(f"  Control group mean: {control_scores.mean():.1f} ± {control_scores.std():.1f}")
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((autism_scores.var() * (len(autism_scores) - 1)) + 
                                 (control_scores.var() * (len(control_scores) - 1))) / 
                                (len(autism_scores) + len(control_scores) - 2))
            cohens_d = (autism_scores.mean() - control_scores.mean()) / pooled_std
            print(f"  Effect size (Cohen's d): {cohens_d:.3f}")
            
            questionnaire_info[name] = {
                'n_items': len(cols),
                'autism_mean': autism_scores.mean(),
                'control_mean': control_scores.mean(),
                'effect_size': cohens_d
            }
    
    return questionnaire_info

def create_visualizations(df: pd.DataFrame, output_dir: str = 'analysis_output'):
    """Create comprehensive visualizations."""
    print(f"\nCreating visualizations in {output_dir}...")
    Path(output_dir).mkdir(exist_ok=True)
    
    # 1. Target distribution
    plt.figure(figsize=(10, 6))
    target_counts = df['autism_any'].value_counts()
    plt.pie(target_counts.values, labels=['No Autism', 'Autism'], autopct='%1.1f%%')
    plt.title('Target Variable Distribution')
    plt.savefig(f'{output_dir}/target_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Questionnaire scores by target
    questionnaire_cols = ['spq_total', 'eq_total', 'sqr_total', 'aq_total']
    available_cols = [col for col in questionnaire_cols if col in df.columns]
    
    if available_cols:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, col in enumerate(available_cols):
            if i < 4:  # Only plot first 4
                autism_scores = df[df['autism_any'] == 1][col]
                control_scores = df[df['autism_any'] == 0][col]
                
                axes[i].hist(autism_scores, alpha=0.7, label='Autism', bins=30)
                axes[i].hist(control_scores, alpha=0.7, label='Control', bins=30)
                axes[i].set_title(f'{col.upper()} Distribution')
                axes[i].set_xlabel('Score')
                axes[i].set_ylabel('Frequency')
                axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/questionnaire_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Correlation heatmap
    if len(available_cols) > 1:
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[available_cols + ['autism_any']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Questionnaire Score Correlations')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

def recommend_modeling_strategies(imbalance_info: dict, questionnaire_info: dict) -> dict:
    """Provide specific modeling recommendations based on data characteristics."""
    print("\n" + "="*50)
    print("MODELING STRATEGY RECOMMENDATIONS")
    print("="*50)
    
    recommendations = {
        'class_imbalance_strategy': [],
        'feature_engineering': [],
        'model_selection': [],
        'evaluation_metrics': [],
        'hyperparameter_tuning': []
    }
    
    # Class imbalance strategies
    if imbalance_info['imbalance_ratio'] < 0.1:
        recommendations['class_imbalance_strategy'].extend([
            "Use SMOTE or ADASYN for synthetic minority oversampling",
            "Implement class weights in all models",
            "Consider Focal Loss for deep learning",
            "Use stratified k-fold cross-validation",
            "Focus on precision and recall rather than accuracy"
        ])
    else:
        recommendations['class_imbalance_strategy'].extend([
            "Use class weights in model training",
            "Consider SMOTE for balanced training",
            "Monitor precision/recall trade-off"
        ])
    
    # Feature engineering based on questionnaire analysis
    best_questionnaire = max(questionnaire_info.items(), key=lambda x: abs(x[1]['effect_size']))
    recommendations['feature_engineering'].extend([
        f"Focus on {best_questionnaire[0]} scores (highest effect size: {best_questionnaire[1]['effect_size']:.3f})",
        "Create interaction features between questionnaire scores",
        "Consider polynomial features for non-linear relationships",
        "Create age-stratified features if age is available",
        "Engineer ratio features (e.g., AQ/EQ ratio)"
    ])
    
    # Model selection
    recommendations['model_selection'].extend([
        "Start with XGBoost or LightGBM (handles imbalance well)",
        "Try Random Forest with class weights",
        "Consider SVM with balanced class weights",
        "For deep learning: use Focal Loss or weighted cross-entropy",
        "Ensemble multiple models with different sampling strategies"
    ])
    
    # Evaluation metrics
    recommendations['evaluation_metrics'].extend([
        "Primary: F1-Score (harmonic mean of precision and recall)",
        "Secondary: Precision, Recall, AUROC",
        "Use stratified cross-validation",
        "Report confusion matrix for interpretability",
        "Consider precision-recall curves"
    ])
    
    # Hyperparameter tuning
    recommendations['hyperparameter_tuning'].extend([
        "Use Bayesian optimization for efficiency",
        "Focus on class_weight parameter",
        "Tune sampling ratio for SMOTE",
        "Optimize for F1-score, not accuracy",
        "Use early stopping to prevent overfitting"
    ])
    
    # Print recommendations
    for category, recs in recommendations.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for rec in recs:
            print(f"  • {rec}")
    
    return recommendations

def main():
    """Main analysis pipeline."""
    data_path = "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/data_c4_raw.csv"
    
    print("="*60)
    print("COMPREHENSIVE AUTISM DIAGNOSIS PREDICTION ANALYSIS")
    print("="*60)
    
    # Load and explore data
    df = load_raw_data(data_path)
    
    # Create target variable
    df = create_target_variable(df)
    
    # Analyze class imbalance
    imbalance_info = analyze_class_imbalance(df)
    
    # Analyze questionnaire data
    questionnaire_info = analyze_questionnaire_data(df)
    
    # Create visualizations
    create_visualizations(df)
    
    # Get modeling recommendations
    recommendations = recommend_modeling_strategies(imbalance_info, questionnaire_info)
    
    # Save processed data
    output_path = 'processed_data_with_target.csv'
    df.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to {output_path}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Next steps:")
    print(f"1. Review the visualizations in analysis_output/")
    print(f"2. Use the processed data: {output_path}")
    print(f"3. Implement the recommended modeling strategies")
    print(f"4. Focus on F1-score optimization for class 1")

if __name__ == "__main__":
    main() 