"""
Unified dashboard for both exploratory data analysis and model evaluation.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import json
import os

# Add the project root to the Python path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

# Now we can import from src
from src.data.processing.assessment_processor import AssessmentProcessor
from src.data.analysis.statistical_analyzer import AssessmentAnalyzer
from src.visualization.model_dashboard import ModelDashboard

def load_data(assessment_type: str) -> pd.DataFrame:
    """Load processed assessment data and calculate total score."""
    processor = AssessmentProcessor()
    data_path = project_root / 'data' / 'processed' / f'{assessment_type.lower()}_processed.csv'
    df = processor.load_data(str(data_path), assessment_type)
    
    # Calculate total score from individual question responses
    question_cols = [col for col in df.columns if col.startswith(f'{assessment_type}_')]
    df['total_score'] = df[question_cols].sum(axis=1)
    
    return df

def create_distribution_plot(df: pd.DataFrame, assessment_type: str) -> go.Figure:
    """Create an interactive distribution plot."""
    fig = px.histogram(
        df,
        x='total_score',
        nbins=30,
        title=f'{assessment_type} Score Distribution',
        labels={'total_score': 'Total Score', 'count': 'Frequency'},
        marginal='box'
    )
    fig.update_layout(
        showlegend=False,
        template='plotly_white'
    )
    return fig

def create_subscale_plot(df: pd.DataFrame, assessment_type: str) -> go.Figure:
    """Create an interactive subscale comparison plot."""
    subscale_cols = [col for col in df.columns if col.startswith('subscale_')]
    if not subscale_cols:
        return None
    
    subscale_data = df[subscale_cols].melt()
    fig = px.box(
        subscale_data,
        x='variable',
        y='value',
        title=f'{assessment_type} Subscale Scores',
        labels={'variable': 'Subscale', 'value': 'Score'}
    )
    fig.update_layout(
        template='plotly_white',
        xaxis_tickangle=-45
    )
    return fig

def display_eda_tab(df: pd.DataFrame, assessment_type: str):
    """Display the Exploratory Data Analysis tab content."""
    st.header("Basic Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Mean Score", f"{df['total_score'].mean():.2f}")
    with col2:
        st.metric("Median Score", f"{df['total_score'].median():.2f}")
    with col3:
        st.metric("Standard Deviation", f"{df['total_score'].std():.2f}")
    
    st.header("Score Distribution")
    fig_dist = create_distribution_plot(df, assessment_type)
    st.plotly_chart(fig_dist, use_container_width=True)
    
    st.header("Subscale Analysis")
    fig_subscale = create_subscale_plot(df, assessment_type)
    if fig_subscale:
        st.plotly_chart(fig_subscale, use_container_width=True)
    else:
        st.info("No subscale data available for this assessment type")
    
    st.header("Data Preview")
    st.dataframe(df.head())
    
    st.download_button(
        label="Download Processed Data",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name=f"{assessment_type.lower()}_processed_data.csv",
        mime='text/csv'
    )

def display_model_evaluation_tab(assessment_type: str):
    """Display the Model Evaluation tab content."""
    models_dir = project_root / 'models' / assessment_type.lower()
    if not models_dir.exists():
        st.warning("No model evaluation data available. Please run the model training pipeline first.")
        return
    
    dashboard = ModelDashboard(str(models_dir))
    
    st.header("Model Performance Metrics")
    metrics_fig = dashboard.create_metrics_comparison()
    st.plotly_chart(metrics_fig, use_container_width=True)
    
    st.header("ROC Curves")
    roc_fig = dashboard.create_roc_curves()
    st.plotly_chart(roc_fig, use_container_width=True)
    
    st.header("Confusion Matrices")
    cm_fig = dashboard.create_confusion_matrices()
    st.plotly_chart(cm_fig, use_container_width=True)
    
    # Display model parameters
    st.header("Model Parameters")
    for model_file in models_dir.glob('*_metadata.json'):
        model_name = model_file.stem.replace('_metadata', '')
        with open(model_file, 'r') as f:
            metadata = json.load(f)
            st.subheader(model_name)
            st.json(metadata.get('best_params', {}))

def main():
    st.set_page_config(
        page_title="Autism Assessment Analysis Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("Autism Assessment Analysis Dashboard")
    st.write("Interactive analysis of assessment data and model performance")
    
    # Assessment type selector
    assessment_type = st.sidebar.selectbox(
        "Select Assessment Type",
        ["AQ", "SQ", "EQ"]
    )
    
    # Create tabs
    tab1, tab2 = st.tabs(["Exploratory Data Analysis", "Model Evaluation"])
    
    try:
        df = load_data(assessment_type)
        
        with tab1:
            display_eda_tab(df, assessment_type)
        
        with tab2:
            display_model_evaluation_tab(assessment_type)
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please ensure the processed data files exist in the data/processed directory")

if __name__ == "__main__":
    main() 