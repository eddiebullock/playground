"""
Logging utilities for experiments.
"""

import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import torch


class ExperimentLogger:
    """Simple experiment logger."""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create experiment directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = self.log_dir / f"exp_{timestamp}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = {}
        self.metrics = []
    
    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        self.config = config
        with open(self.exp_dir / "config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def log_metric(self, step: int, metric_name: str, value: float):
        """Log a metric value."""
        self.metrics.append({
            "step": step,
            "metric": metric_name,
            "value": value,
        })
    
    def log_metrics(self, step: int, metrics: Dict[str, float]):
        """Log multiple metrics at once."""
        for name, value in metrics.items():
            self.log_metric(step, name, value)
    
    def save_checkpoint(self, model: torch.nn.Module, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
        }
        torch.save(checkpoint, self.exp_dir / f"checkpoint_epoch_{epoch}.pth")
    
    def save_summary(self, summary: Dict[str, Any]):
        """Save experiment summary."""
        with open(self.exp_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
    
    def get_exp_dir(self) -> Path:
        """Get experiment directory."""
        return self.exp_dir



