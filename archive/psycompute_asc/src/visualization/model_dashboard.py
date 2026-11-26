"""
Model evaluation dashboard creation.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import numpy as np

class ModelDashboard:
    """Create interactive dashboards for model evaluation."""
    
    def __init__(self, models_dir: str):
        """
        Initialize the dashboard creator.
        
        Args:
            models_dir: Directory containing model files
        """
        self.models_dir = Path(models_dir)
        self.metrics = {}
        self.load_metrics()
    
    def load_metrics(self):
        """Load metrics from all model files."""
        for model_file in self.models_dir.glob('*_metadata.json'):
            model_name = model_file.stem.replace('_metadata', '')
            with open(model_file, 'r') as f:
                self.metrics[model_name] = json.load(f)['metrics']
    
    def create_metrics_comparison(self) -> go.Figure:
        """Create a comparison of metrics across models."""
        # Prepare data for plotting
        metrics_data = []
        for model_name, metrics in self.metrics.items():
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)) and metric_name not in ['confusion_matrix', 'roc_curve', 'class_distribution']:
                    metrics_data.append({
                        'Model': model_name,
                        'Metric': metric_name,
                        'Value': value
                    })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Create bar plot
        fig = px.bar(
            metrics_df,
            x='Metric',
            y='Value',
            color='Model',
            barmode='group',
            title='Model Performance Comparison'
        )
        
        return fig
    
    def create_roc_curves(self) -> go.Figure:
        """Create ROC curves for all models."""
        fig = go.Figure()
        
        for model_name, metrics in self.metrics.items():
            roc_data = metrics['roc_curve']
            fig.add_trace(
                go.Scatter(
                    x=roc_data['fpr'],
                    y=roc_data['tpr'],
                    name=model_name,
                    mode='lines'
                )
            )
        
        # Add diagonal line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                name='Random',
                mode='lines',
                line=dict(dash='dash')
            )
        )
        
        fig.update_layout(
            title='ROC Curves',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            showlegend=True
        )
        
        return fig
    
    def create_confusion_matrices(self) -> go.Figure:
        """Create confusion matrices for all models."""
        n_models = len(self.metrics)
        fig = go.Figure()
        
        for i, (model_name, metrics) in enumerate(self.metrics.items()):
            cm = np.array(metrics['confusion_matrix'])
            
            # Create heatmap
            fig.add_trace(
                go.Heatmap(
                    z=cm,
                    x=['Predicted 0', 'Predicted 1'],
                    y=['Actual 0', 'Actual 1'],
                    name=model_name,
                    showscale=False,
                    text=cm,
                    texttemplate='%{text}',
                    textfont={"size": 16},
                    visible=(i == 0)  # Only show first model initially
                )
            )
        
        # Add buttons for model selection
        buttons = []
        for i, model_name in enumerate(self.metrics.keys()):
            button = dict(
                method='update',
                args=[{'visible': [j == i for j in range(n_models)]},
                      {'title': f'Confusion Matrix - {model_name}'}],
                label=model_name
            )
            buttons.append(button)
        
        fig.update_layout(
            title='Confusion Matrix',
            updatemenus=[{
                'buttons': buttons,
                'direction': 'down',
                'showactive': True,
                'x': 0.1,
                'y': 1.1
            }]
        )
        
        return fig
    
    def create_dashboard(self, output_dir: str):
        """Create and save the complete dashboard."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create individual plots
        metrics_fig = self.create_metrics_comparison()
        roc_fig = self.create_roc_curves()
        cm_fig = self.create_confusion_matrices()
        
        # Save plots
        metrics_fig.write_html(output_path / 'metrics_comparison.html')
        roc_fig.write_html(output_path / 'roc_curves.html')
        cm_fig.write_html(output_path / 'confusion_matrices.html')
        
        # Create index page
        index_html = f"""
        <html>
        <head>
            <title>Model Evaluation Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                .plot-container {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Model Evaluation Dashboard</h1>
            <div class="plot-container">
                <h2>Metrics Comparison</h2>
                <iframe src="metrics_comparison.html" width="100%" height="600px" frameborder="0"></iframe>
            </div>
            <div class="plot-container">
                <h2>ROC Curves</h2>
                <iframe src="roc_curves.html" width="100%" height="600px" frameborder="0"></iframe>
            </div>
            <div class="plot-container">
                <h2>Confusion Matrices</h2>
                <iframe src="confusion_matrices.html" width="100%" height="600px" frameborder="0"></iframe>
            </div>
        </body>
        </html>
        """
        
        with open(output_path / 'index.html', 'w') as f:
            f.write(index_html) 