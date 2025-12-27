"""
Temperature scaling calibration for CAM experiments.

Implements temperature scaling (Guo et al., 2017) to calibrate model predictions
using a validation set. This improves model confidence calibration without
changing the model's predictions (only adjusts temperature parameter).
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict
from scipy.optimize import minimize_scalar

from .trials.forced_choice import ForcedChoiceTrial, run_forced_choice_trial
from .dataset import CAMTrial


class TemperatureScaler:
    """
    Temperature scaling for model calibration.
    
    Learns a single temperature parameter T that scales logits:
    scaled_logits = logits / T
    
    T is optimized on a validation set to maximize likelihood of correct predictions.
    """
    
    def __init__(self):
        self.temperature = 1.0  # Default: no scaling
    
    def fit(
        self,
        calibration_trials: List[ForcedChoiceTrial],
    ) -> float:
        """
        Fit temperature parameter on calibration trials.
        
        Args:
            calibration_trials: List of trial results with model scores
        
        Returns:
            Optimal temperature value
        """
        # Extract logits and correct indices
        logits_list = []
        correct_indices = []
        
        for trial_result in calibration_trials:
            # Get scores in order of candidate_labels
            scores = [
                trial_result.model_scores[label]
                for label in trial_result.trial.candidate_labels
            ]
            logits_list.append(scores)
            correct_indices.append(trial_result.trial.correct_idx)
        
        logits_array = np.array(logits_list)  # (n_trials, 4)
        correct_array = np.array(correct_indices)  # (n_trials,)
        
        # Optimize temperature using negative log-likelihood
        def nll(temperature):
            """Negative log-likelihood for temperature T."""
            scaled_logits = logits_array / temperature
            # Numerical stability: subtract max
            scaled_logits = scaled_logits - scaled_logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(scaled_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            
            # Get probability of correct label
            correct_probs = probs[np.arange(len(probs)), correct_array]
            
            # Negative log-likelihood
            nll_value = -np.log(correct_probs + 1e-10).mean()
            return nll_value
        
        # Optimize temperature (typically between 0.1 and 10)
        result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
        self.temperature = result.x
        
        return self.temperature
    
    def get_temperature(self) -> float:
        """Get the fitted temperature parameter."""
        return self.temperature


def calibrate_model(
    model,
    calibration_trials: List[CAMTrial],
    device: str = "cpu",
) -> float:
    """
    Calibrate model using temperature scaling on validation trials.
    
    Args:
        model: Model wrapper
        calibration_trials: List of CAM trials for calibration
        device: Device to run on
    
    Returns:
        Optimal temperature value
    """
    print(f"\nCalibrating model on {len(calibration_trials)} validation trials...")
    
    # Run trials to get model scores
    calibration_results = []
    for i, trial in enumerate(calibration_trials):
        if (i + 1) % 10 == 0:
            print(f"  Processing calibration trial {i + 1}/{len(calibration_trials)}...")
        
        result = run_forced_choice_trial(trial, model, temperature=1.0)
        calibration_results.append(result)
    
    # Fit temperature scaler
    scaler = TemperatureScaler()
    optimal_temp = scaler.fit(calibration_results)
    
    print(f"  Optimal temperature: {optimal_temp:.3f}")
    
    return optimal_temp








