#!/usr/bin/env python3
"""
Main experiment script for CAM Face-Voice Battery replication.

This script runs the computational replication of the Cambridge Mindreading (CAM)
Face-Voice Battery (Golan et al., 2006), evaluating pretrained models as
task-performing agents.

Experimental structure:
1. Load CAM trial definitions
2. Load pretrained model wrapper
3. Run forced-choice trials (model processes stimulus, scores labels, selects highest)
4. Evaluate results using CAM metrics
5. Save results in structured format (JSON/CSV)

Usage:
    python run_experiment.py --config configs/cam_config.yaml
"""

import argparse
import json
import sys
import yaml
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.cam_human_like.dataset import CAMDataset
from experiments.cam_human_like.models import (
    ModelWrapper, CLIPWrapper, AudioWrapper, MultimodalWrapper,
    EmotionModelWrapper, HybridEmotionWrapper
)
from experiments.cam_human_like.trials.forced_choice import run_batch_trials, ForcedChoiceTrial
from experiments.cam_human_like.evaluation.metrics import evaluate_trials, EvaluationResults
from experiments.cam_human_like.calibration import calibrate_model


def load_config(config_path: str) -> Dict:
    """Load experiment configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(config: Dict) -> ModelWrapper:
    """
    Load model wrapper based on configuration.
    
    Supported model types:
    - "clip": CLIP-style vision-language model
    - "audio": Audio-only model (Wav2Vec2, etc.)
    - "multimodal": Combined vision and audio
    - "emotion": Pre-trained emotion recognition model
    - "hybrid": Emotion model + CLIP (recommended for best results)
    """
    model_type = config['model']['type'].lower()
    model_name = config['model'].get('name', None)
    device = config.get('device', 'cpu')
    
    if model_type == "clip":
        return CLIPWrapper(
            model_name=model_name or "openai/clip-vit-base-patch32",
            device=device,
            num_frames=config['model'].get('num_frames', 8),
            aggregation=config['model'].get('aggregation', 'mean'),
        )
    elif model_type == "emotion":
        return EmotionModelWrapper(
            model_name=model_name or "trpakov/vit-face-expression",
            device=device,
            num_frames=config['model'].get('num_frames', 8),
            aggregation=config['model'].get('aggregation', 'mean'),
            emotion_mapping=config['model'].get('emotion_mapping'),
        )
    elif model_type == "hybrid":
        return HybridEmotionWrapper(
            emotion_model_name=config['model'].get('emotion_model', "trpakov/vit-face-expression"),
            clip_model_name=config['model'].get('clip_model', "openai/clip-vit-base-patch32"),
            device=device,
            num_frames=config['model'].get('num_frames', 8),
            aggregation=config['model'].get('aggregation', 'mean'),
            emotion_weight=config['model'].get('emotion_weight', 0.5),
        )
    elif model_type == "audio":
        return AudioWrapper(
            model_name=model_name or "facebook/wav2vec2-base",
            device=device,
            sample_rate=config['model'].get('sample_rate', 16000),
        )
    elif model_type == "multimodal":
        return MultimodalWrapper(
            vision_model_name=config['model'].get('vision_model', "openai/clip-vit-base-patch32"),
            audio_model_name=config['model'].get('audio_model', "facebook/wav2vec2-base"),
            fusion_method=config['model'].get('fusion_method', 'weighted_average'),
            device=device,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def save_results(
    results: EvaluationResults,
    trial_results: List[ForcedChoiceTrial],
    output_dir: Path,
    config: Dict,
) -> None:
    """
    Save experiment results in structured format.
    
    Saves:
    - summary.json: Overall metrics
    - trial_results.csv: Per-trial predictions and scores
    - confusion_matrix.csv: Confusion matrix
    - per_emotion_accuracy.csv: Per-emotion breakdown
    - per_concept_accuracy.csv: Per-concept breakdown
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save summary
    summary = {
        'timestamp': timestamp,
        'overall_accuracy': results.overall_accuracy,
        'face_accuracy': results.face_accuracy,
        'voice_accuracy': results.voice_accuracy,
        'concept_recognition_rate': results.concept_recognition_rate,
        'num_trials': results.num_trials,
        'num_face_trials': results.num_face_trials,
        'num_voice_trials': results.num_voice_trials,
        'num_concepts': results.num_concepts,
        'config': config,
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save trial-level results
    trial_data = []
    for trial_result in trial_results:
        trial_data.append({
            'trial_id': trial_result.trial.trial_id,
            'stimulus_path': trial_result.trial.stimulus_path,
            'modality': trial_result.trial.modality,
            'concept': trial_result.trial.concept,
            'correct_label': trial_result.trial.correct_label,
            'predicted_label': trial_result.predicted_label,
            'predicted_idx': trial_result.predicted_idx,
            'correct_idx': trial_result.trial.correct_idx,
            'is_correct': trial_result.is_correct,
            'confidence': trial_result.confidence,
            'score_correct': trial_result.model_scores[trial_result.trial.correct_label],
            'score_predicted': trial_result.model_scores[trial_result.predicted_label],
            'actor': trial_result.trial.actor,
            'scenario_id': trial_result.trial.scenario_id,
        })
    
    trial_df = pd.DataFrame(trial_data)
    trial_df.to_csv(output_dir / 'trial_results.csv', index=False)
    
    # Save confusion matrix
    results.confusion_matrix.to_csv(output_dir / 'confusion_matrix.csv')
    
    # Save per-emotion accuracy
    emotion_df = pd.DataFrame([
        {'emotion': emotion, 'accuracy': acc}
        for emotion, acc in results.per_emotion_accuracy.items()
    ])
    emotion_df.to_csv(output_dir / 'per_emotion_accuracy.csv', index=False)
    
    # Save per-concept accuracy
    concept_df = pd.DataFrame([
        {'concept': concept, 'accuracy': acc}
        for concept, acc in results.per_concept_accuracy.items()
    ])
    concept_df.to_csv(output_dir / 'per_concept_accuracy.csv', index=False)
    
    # Save concept recognition (passed/failed)
    concept_passed = results.metadata.get('concept_passed', {})
    concept_recognition_df = pd.DataFrame([
        {
            'concept': concept,
            'passed': passed,
            'accuracy': results.per_concept_accuracy.get(concept, 0.0),
        }
        for concept, passed in concept_passed.items()
    ])
    concept_recognition_df.to_csv(output_dir / 'concept_recognition.csv', index=False)
    
    print(f"\nResults saved to {output_dir}")
    print(f"  - summary.json")
    print(f"  - trial_results.csv ({len(trial_data)} trials)")
    print(f"  - confusion_matrix.csv")
    print(f"  - per_emotion_accuracy.csv")
    print(f"  - per_concept_accuracy.csv")
    print(f"  - concept_recognition.csv")


def main():
    """Main experiment loop."""
    parser = argparse.ArgumentParser(
        description="Run CAM Face-Voice Battery experiment"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to experiment configuration YAML file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (default: results/cam_human_like/{model_type}_{timestamp})'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test', 'all'],
        help='Which split to evaluate (default: test). Use "all" to load all trials without actor filtering (matches original CAM)'
    )
    parser.add_argument(
        '--no-actor-filtering',
        action='store_true',
        help='Disable actor filtering (load all trials regardless of split)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")
    
    # Load dataset
    use_actor_filtering = not args.no_actor_filtering and args.split != "all"
    split_display = args.split if use_actor_filtering else "all trials (no actor filtering)"
    print(f"\nLoading CAM dataset ({split_display})...")
    dataset = CAMDataset(
        data_root=config['data']['root'],
        trial_definitions_file=config['data'].get('trial_definitions_file'),
        splits_dir=config['data'].get('splits_dir') if use_actor_filtering else None,
        split_name=args.split,
        seed=config.get('seed', 42),
        use_actor_filtering=use_actor_filtering,
    )
    
    # Load model
    print(f"\nLoading model: {config['model']['type']}...")
    model = load_model(config)
    print(f"Model loaded: {model.model_name}")
    
    # Calibration (if enabled and validation set available)
    temperature = config.get('evaluation', {}).get('temperature', 1.0)
    calibration_config = config.get('evaluation', {}).get('calibration', {})
    
    if calibration_config.get('enabled', False):
        # For "all" split, use a subset of trials for calibration
        if args.split == 'all':
            # Split 100 trials: 20 for calibration, 80 for testing
            all_trials = dataset.get_all_trials()
            np.random.seed(config.get('seed', 42))
            indices = np.random.permutation(len(all_trials))
            calib_indices = indices[:20]
            test_indices = indices[20:]
            
            calibration_trials = [all_trials[i] for i in calib_indices]
            dataset.trials = [all_trials[i] for i in test_indices]  # Update dataset
            
            print(f"Using {len(calibration_trials)} trials for calibration, {len(dataset)} for testing")
        else:
            # Load validation set for calibration
            val_dataset = CAMDataset(
                data_root=config['data']['root'],
                trial_definitions_file=config['data'].get('trial_definitions_file'),
                splits_dir=config['data'].get('splits_dir'),
                split_name='val',
                seed=config.get('seed', 42),
                use_actor_filtering=True,
            )
            calibration_trials = val_dataset.get_all_trials() if len(val_dataset) > 0 else []
        
        if len(calibration_trials) > 0:
            optimal_temp = calibrate_model(model, calibration_trials, config.get('device', 'cpu'))
            temperature = optimal_temp
            print(f"Using calibrated temperature: {temperature:.3f}")
        else:
            print("Warning: No calibration trials available, using default temperature")
    
    # Run trials
    print(f"\nRunning {len(dataset)} forced-choice trials...")
    
    trial_results = run_batch_trials(
        trials=dataset.get_all_trials(),
        model=model,
        temperature=temperature,
        verbose=True,
    )
    
    # Evaluate results
    print("\nEvaluating results...")
    results = evaluate_trials(trial_results)
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT RESULTS")
    print("="*60)
    print(f"Overall Accuracy: {results.overall_accuracy:.3f}")
    print(f"Face Accuracy: {results.face_accuracy:.3f} ({results.num_face_trials} trials)")
    print(f"Voice Accuracy: {results.voice_accuracy:.3f} ({results.num_voice_trials} trials)")
    print(f"Concept Recognition Rate: {results.concept_recognition_rate:.3f} ({results.num_concepts} concepts)")
    print("="*60)
    
    # Save results
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        model_type = config['model']['type']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/cam_human_like/{model_type}_{timestamp}")
    
    save_results(results, trial_results, output_dir, config)
    
    print("\nExperiment complete!")


if __name__ == "__main__":
    main()

