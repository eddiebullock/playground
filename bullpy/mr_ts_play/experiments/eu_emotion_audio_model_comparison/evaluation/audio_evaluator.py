"""
Main evaluation pipeline for audio-based model comparison.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

from .metrics import (
    compute_metrics,
    save_per_emotion_results,
    save_confusion_matrix,
    compute_confusion_matrix,
)

logger = logging.getLogger(__name__)


class AudioModelEvaluator:
    """Evaluator for audio-based emotion recognition models."""
    
    def __init__(
        self,
        trial_definitions_file: str,
        data_root: str,
        output_dir: str,
    ):
        """
        Initialize evaluator.
        
        Args:
            trial_definitions_file: Path to trial definitions JSON
            data_root: Root directory for dataset
            output_dir: Output directory for results
        """
        self.trial_definitions_file = Path(trial_definitions_file)
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        
        # Load trials
        with open(self.trial_definitions_file, 'r') as f:
            data = json.load(f)
            self.trials = data.get('trials', data)  # Handle both formats
        
        logger.info(f"Loaded {len(self.trials)} trials from {trial_definitions_file}")
    
    def evaluate_model(
        self,
        model,
        model_name: str,
        save_results: bool = True,
        verbose: bool = True,
    ) -> Dict:
        """
        Evaluate a model on all trials.
        
        Args:
            model: Model instance (must have predict_emotion method)
            model_name: Name of model (for output files)
            save_results: Whether to save results to files
            verbose: Whether to show progress
        
        Returns:
            Dictionary with predictions and metrics
        """
        predictions = []
        
        # Evaluate each trial
        iterator = tqdm(self.trials, desc=f"Evaluating {model_name}") if verbose else self.trials
        
        # Create output directory early for incremental saving
        model_output_dir = self.output_dir / model_name.lower().replace('-', '_')
        model_output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = model_output_dir / "predictions_checkpoint.json"
        
        # Load existing checkpoint if resuming
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r') as f:
                    existing_predictions = json.load(f)
                completed_trial_ids = {p['trial_id'] for p in existing_predictions}
                logger.info(f"Resuming evaluation: {len(existing_predictions)} trials already completed")
            except:
                existing_predictions = []
                completed_trial_ids = set()
        else:
            existing_predictions = []
            completed_trial_ids = set()
        
        for trial in iterator:
            trial_id = trial.get('trial_id', 'unknown')
            
            # Skip if already completed
            if trial_id in completed_trial_ids:
                continue
            
            try:
                audio_path = trial['stimulus_path']  # Changed from video_path
                candidate_labels = trial['candidate_labels']
                correct_label = trial['correct_label']
                
                # Get prediction
                scores = model.predict_emotion(
                    audio_path=audio_path,  # Changed from video_path
                    candidate_labels=candidate_labels,
                    data_root=str(self.data_root),
                )
                
                # Find predicted label (highest score)
                predicted_label = max(scores.items(), key=lambda x: x[1])[0]
                
                prediction = {
                    'trial_id': trial_id,
                    'audio_path': audio_path,  # Changed from video_path
                    'correct_label': correct_label,
                    'predicted_label': predicted_label,
                    'candidate_labels': candidate_labels,
                    'scores': scores,
                }
                
                predictions.append(prediction)
                existing_predictions.append(prediction)
                
                # Save checkpoint after each trial (for expensive models)
                if save_results and hasattr(model, 'get_cost_summary'):
                    with open(checkpoint_file, 'w') as f:
                        json.dump(existing_predictions, f, indent=2)
                
            except Exception as e:
                logger.error(f"Error evaluating trial {trial_id}: {e}")
                # Still save what we have
                if save_results and hasattr(model, 'get_cost_summary'):
                    with open(checkpoint_file, 'w') as f:
                        json.dump(existing_predictions, f, indent=2)
                continue
        
        # Use checkpoint if we have more complete data
        if len(existing_predictions) > len(predictions):
            predictions = existing_predictions
        
        # Compute metrics
        metrics = compute_metrics(predictions)
        
        # Save results if requested
        if save_results:
            self._save_model_results(model_name, predictions, metrics)
        
        return {
            'model_name': model_name,
            'predictions': predictions,
            'metrics': metrics,
        }
    
    def _save_model_results(
        self,
        model_name: str,
        predictions: List[Dict],
        metrics: Dict,
    ):
        """Save results for a single model."""
        # Create output directory
        model_output_dir = self.output_dir / model_name.lower().replace('-', '_')
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save predictions
        predictions_file = model_output_dir / "predictions.json"
        with open(predictions_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        # Remove checkpoint file if it exists (evaluation complete)
        checkpoint_file = model_output_dir / "predictions_checkpoint.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()
        
        # Save metrics
        metrics_file = model_output_dir / "metrics.json"
        # Convert numpy types for JSON
        metrics_json = json.loads(json.dumps(metrics, default=str))
        with open(metrics_file, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        
        # Save per-emotion results
        per_emotion_file = model_output_dir / "per_emotion_results.csv"
        save_per_emotion_results(
            metrics['per_emotion_metrics'],
            per_emotion_file,
            model_name,
        )
        
        # Save confusion matrix
        from .metrics import compute_confusion_matrix
        confusion_matrix = compute_confusion_matrix(predictions, normalize=True)
        confusion_file = model_output_dir / "confusion_matrix.png"
        save_confusion_matrix(confusion_matrix, confusion_file, model_name)
        
        # Also save as CSV
        confusion_matrix.to_csv(model_output_dir / "confusion_matrix.csv")
        
        logger.info(f"Saved results for {model_name} to {model_output_dir}")
