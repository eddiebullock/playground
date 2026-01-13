"""
Three-way comparison evaluation: CLIP-only, LLM-only, LLM-augmented.

Runs three experimental conditions and compares results.
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
import logging

from ..models.clip_model_loader import CLIPModelLoader
from ..models.llm_wrapper import LLMWrapper
from ..models.llm_augmented_wrapper import LLMAugmentedWrapper
from .metrics import compute_metrics, save_results

logger = logging.getLogger(__name__)


def load_video_frames(video_path: str, num_frames: int = 8) -> List[Image.Image]:
    """
    Extract frames from video.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
    
    Returns:
        List of PIL Images
    """
    try:
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file does not exist: {video_path}")
        
        file_size = Path(video_path).stat().st_size
        if file_size < 50 * 1024:  # 50KB threshold
            raise ValueError(f"Video file too small (likely corrupted): {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            raise ValueError(f"Video has no frames: {video_path}")
        
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
            else:
                if len(frames) > 0:
                    frames.append(frames[-1])
                else:
                    frames.append(Image.new('RGB', (224, 224), (0, 0, 0)))
        
        cap.release()
        
        while len(frames) < num_frames:
            frames.append(frames[-1] if frames else Image.new('RGB', (224, 224), (0, 0, 0)))
        
        return frames[:num_frames]
    except Exception as e:
        raise ValueError(f"Error loading video {video_path}: {e}")


def resolve_video_path(stimulus_path: str, data_root: str, dataset_type: str) -> str:
    """
    Resolve video path based on dataset type.
    
    Args:
        stimulus_path: Relative path from trial definition
        data_root: Root directory for dataset
        dataset_type: "cam" or "eu_emotion"
    
    Returns:
        Full path to video file
    """
    data_root = Path(data_root)
    
    if dataset_type == "cam":
        # CAM: stimulus_path is relative to data_root
        video_path = data_root / stimulus_path
    elif dataset_type == "eu_emotion":
        # EU-Emotion: stimulus_path includes emotions* folder structure
        video_path = data_root / stimulus_path
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    if not video_path.exists():
        # Try alternative paths for EU-Emotion
        if dataset_type == "eu_emotion":
            # Try without "emotions " prefix
            alt_path = data_root / stimulus_path.replace("emotions ", "emotions", 1)
            if alt_path.exists():
                return str(alt_path)
        
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    return str(video_path)


def run_clip_only(
    clip_loader: CLIPModelLoader,
    trials: List[Dict],
    data_root: str,
    dataset_type: str,
    num_frames: int = 8,
) -> List[Dict]:
    """
    Run CLIP-only evaluation.
    
    Args:
        clip_loader: CLIP model loader
        trials: List of trial dictionaries
        data_root: Root directory for dataset
        dataset_type: "cam" or "eu_emotion"
        num_frames: Number of frames per video
    
    Returns:
        List of prediction dictionaries
    """
    predictions = []
    
    for i, trial in enumerate(trials, 1):
        logger.info(f"[CLIP-only] Trial {i}/{len(trials)}: {trial.get('trial_id', 'unknown')}")
        
        try:
            # Resolve video path
            video_path = resolve_video_path(
                trial['stimulus_path'],
                data_root,
                dataset_type,
            )
            
            # Load video frames
            video_frames = load_video_frames(video_path, num_frames)
            
            # Score labels
            candidate_labels = trial['candidate_labels']
            scores = clip_loader.score_labels(video_frames, candidate_labels)
            
            # Get predicted label
            predicted_label = max(scores.items(), key=lambda x: x[1])[0]
            correct_label = trial['correct_label']
            
            predictions.append({
                'trial_id': trial.get('trial_id', f'trial_{i}'),
                'correct_label': correct_label,
                'predicted_label': predicted_label,
                'candidate_labels': candidate_labels,
                'scores': scores,
                'is_correct': predicted_label == correct_label,
            })
        except Exception as e:
            logger.error(f"Error processing trial {trial.get('trial_id', 'unknown')}: {e}")
            predictions.append({
                'trial_id': trial.get('trial_id', f'trial_{i}'),
                'correct_label': trial['correct_label'],
                'predicted_label': 'ERROR',
                'candidate_labels': trial['candidate_labels'],
                'scores': {},
                'is_correct': False,
                'error': str(e),
            })
    
    return predictions


def run_llm_only(
    llm_wrapper: LLMWrapper,
    trials: List[Dict],
    data_root: str,
    dataset_type: str,
    num_frames: int = 8,
) -> List[Dict]:
    """
    Run LLM-only evaluation using proper video description.
    
    Now uses vision model to describe videos, then compares descriptions to emotion labels.
    
    Args:
        llm_wrapper: LLM wrapper (must use vision model like gpt-4o)
        trials: List of trial dictionaries
        data_root: Root directory for dataset
        dataset_type: "cam" or "eu_emotion"
        num_frames: Number of frames per video
    
    Returns:
        List of prediction dictionaries
    """
    predictions = []
    
    for i, trial in enumerate(trials, 1):
        logger.info(f"[LLM-only] Trial {i}/{len(trials)}: {trial.get('trial_id', 'unknown')}")
        
        try:
            # Resolve video path
            video_path = resolve_video_path(
                trial['stimulus_path'],
                data_root,
                dataset_type,
            )
            
            # Load video frames
            video_frames = load_video_frames(video_path, num_frames)
            
            # Direct emotion classification (like ChatGPT web interface)
            # This avoids information loss from description → embedding → similarity
            candidate_labels = trial['candidate_labels']
            scores = llm_wrapper.classify_emotion_directly(
                video_frames,
                candidate_labels,
                video_path=video_path,
                detail="low",  # Use "low" for cost efficiency
                max_frames=llm_wrapper.max_frames,  # Use multiple frames for better quality
            )
            
            # Get video description for logging (optional, for analysis)
            video_description = None  # Not needed for direct classification, but can be retrieved if needed
            
            # Get predicted label
            predicted_label = max(scores.items(), key=lambda x: x[1])[0]
            correct_label = trial['correct_label']
            
            predictions.append({
                'trial_id': trial.get('trial_id', f'trial_{i}'),
                'correct_label': correct_label,
                'predicted_label': predicted_label,
                'candidate_labels': candidate_labels,
                'scores': scores,
                'is_correct': predicted_label == correct_label,
                'method': 'direct_classification',  # Indicate we used direct classification
            })
        except Exception as e:
            logger.error(f"Error processing trial {trial.get('trial_id', 'unknown')}: {e}")
            predictions.append({
                'trial_id': trial.get('trial_id', f'trial_{i}'),
                'correct_label': trial['correct_label'],
                'predicted_label': 'ERROR',
                'candidate_labels': trial['candidate_labels'],
                'scores': {},
                'is_correct': False,
                'error': str(e),
            })
    
    return predictions


def run_llm_augmented(
    augmented_wrapper: LLMAugmentedWrapper,
    trials: List[Dict],
    data_root: str,
    dataset_type: str,
    num_frames: int = 8,
) -> List[Dict]:
    """
    Run LLM-augmented evaluation.
    
    Args:
        augmented_wrapper: LLM-augmented wrapper
        trials: List of trial dictionaries
        data_root: Root directory for dataset
        dataset_type: "cam" or "eu_emotion"
        num_frames: Number of frames per video
    
    Returns:
        List of prediction dictionaries
    """
    predictions = []
    
    for i, trial in enumerate(trials, 1):
        logger.info(f"[LLM-augmented] Trial {i}/{len(trials)}: {trial.get('trial_id', 'unknown')}")
        
        try:
            # Resolve video path
            video_path = resolve_video_path(
                trial['stimulus_path'],
                data_root,
                dataset_type,
            )
            
            # Load video frames
            video_frames = load_video_frames(video_path, num_frames)
            
            # Score labels
            candidate_labels = trial['candidate_labels']
            scores = augmented_wrapper.score_labels(
                video_frames,
                candidate_labels,
                video_path=video_path,
            )
            
            # Get predicted label
            predicted_label = max(scores.items(), key=lambda x: x[1])[0]
            correct_label = trial['correct_label']
            
            predictions.append({
                'trial_id': trial.get('trial_id', f'trial_{i}'),
                'correct_label': correct_label,
                'predicted_label': predicted_label,
                'candidate_labels': candidate_labels,
                'scores': scores,
                'is_correct': predicted_label == correct_label,
            })
        except Exception as e:
            logger.error(f"Error processing trial {trial.get('trial_id', 'unknown')}: {e}")
            predictions.append({
                'trial_id': trial.get('trial_id', f'trial_{i}'),
                'correct_label': trial['correct_label'],
                'predicted_label': 'ERROR',
                'candidate_labels': trial['candidate_labels'],
                'scores': {},
                'is_correct': False,
                'error': str(e),
            })
    
    return predictions


def run_three_way_comparison(
    clip_model_path: str,
    llm_config: Dict,
    fusion_config: Dict,
    trial_definitions_file: str,
    data_root: str,
    dataset_type: str,
    output_dir: str,
    num_frames: int = 8,
    device: str = "cpu",
) -> Dict:
    """
    Run three-way comparison: CLIP-only, LLM-only, LLM-augmented.
    
    Args:
        clip_model_path: Path to fine-tuned CLIP model
        llm_config: LLM configuration dictionary
        fusion_config: Fusion configuration dictionary
        trial_definitions_file: Path to trial definitions JSON
        data_root: Root directory for dataset
        dataset_type: "cam" or "eu_emotion"
        output_dir: Output directory for results
        num_frames: Number of frames per video
        device: Device to run on
    
    Returns:
        Dictionary with comparison results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load trial definitions
    logger.info(f"Loading trial definitions from: {trial_definitions_file}")
    with open(trial_definitions_file, 'r') as f:
        trial_data = json.load(f)
    trials = trial_data['trials']
    logger.info(f"Loaded {len(trials)} trials")
    
    # Initialize models
    logger.info("Initializing models...")
    clip_loader = CLIPModelLoader(clip_model_path, device=device)
    
    # Extract provider-specific config (supports both old and new config formats)
    provider = llm_config['provider']
    
    # New config format: provider-specific sections
    if provider in llm_config:
        provider_config = llm_config[provider]
        model = provider_config.get('model', llm_config.get('model', 'text-embedding-3-small'))
        embedding_model = provider_config.get('embedding_model', llm_config.get('embedding_model'))
        vision_model = provider_config.get('vision_model', llm_config.get('vision_model', model))
    else:
        # Old config format: flat structure
        model = llm_config.get('model', 'text-embedding-3-small')
        embedding_model = llm_config.get('embedding_model')
        vision_model = llm_config.get('vision_model', model)
    
    llm_wrapper = LLMWrapper(
        provider=provider,
        model=model,
        embedding_model=embedding_model,
        vision_model=vision_model,
        cache_dir=llm_config['cache_dir'],
        use_cache=llm_config['use_cache'],
        cache_version=llm_config.get('cache_version', '1.0'),
        vision_detail=llm_config.get('vision_detail', 'low'),
    )
    # Set max_frames from config
    llm_wrapper.max_frames = llm_config.get('max_frames_per_video', 4)
    augmented_wrapper = LLMAugmentedWrapper(
        clip_loader=clip_loader,
        llm_wrapper=llm_wrapper,
        fusion_method=fusion_config['method'],
        clip_weight=fusion_config['clip_weight'],
        llm_weight=fusion_config.get('llm_weight'),
        attention_dim=fusion_config.get('attention_dim', 128),
    )
    
    # Run evaluations
    results = {}
    
    # CLIP-only
    logger.info("Running CLIP-only evaluation...")
    clip_predictions = run_clip_only(
        clip_loader, trials, data_root, dataset_type, num_frames
    )
    clip_metrics = compute_metrics(clip_predictions)
    save_results(clip_predictions, clip_metrics, output_dir / "clip_only", "CLIP-only")
    results['clip_only'] = {
        'accuracy': clip_metrics['overall_accuracy'],
        'predictions': clip_predictions,
        'metrics': clip_metrics,
    }
    
    # LLM-only
    logger.info("Running LLM-only evaluation...")
    llm_predictions = run_llm_only(
        llm_wrapper, trials, data_root, dataset_type, num_frames
    )
    llm_metrics = compute_metrics(llm_predictions)
    save_results(llm_predictions, llm_metrics, output_dir / "llm_only", "LLM-only")
    results['llm_only'] = {
        'accuracy': llm_metrics['overall_accuracy'],
        'predictions': llm_predictions,
        'metrics': llm_metrics,
    }
    
    # LLM-augmented
    logger.info("Running LLM-augmented evaluation...")
    augmented_predictions = run_llm_augmented(
        augmented_wrapper, trials, data_root, dataset_type, num_frames
    )
    augmented_metrics = compute_metrics(augmented_predictions)
    fusion_method_name = fusion_config['method']
    save_results(
        augmented_predictions,
        augmented_metrics,
        output_dir / f"llm_augmented_{fusion_method_name}",
        f"LLM-augmented ({fusion_method_name})",
    )
    results['llm_augmented'] = {
        'accuracy': augmented_metrics['overall_accuracy'],
        'predictions': augmented_predictions,
        'metrics': augmented_metrics,
        'fusion_method': fusion_method_name,
    }
    
    # Generate comparison report
    comparison_report = {
        'clip_only_accuracy': results['clip_only']['accuracy'],
        'llm_only_accuracy': results['llm_only']['accuracy'],
        'llm_augmented_accuracy': results['llm_augmented']['accuracy'],
        'improvement_over_clip': results['llm_augmented']['accuracy'] - results['clip_only']['accuracy'],
        'improvement_over_llm': results['llm_augmented']['accuracy'] - results['llm_only']['accuracy'],
        'fusion_method': fusion_method_name,
    }
    
    # Save comparison report
    comparison_file = output_dir / "comparison_report.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison_report, f, indent=2)
    
    # Generate markdown report
    markdown_report = generate_markdown_report(comparison_report, results)
    markdown_file = output_dir / "comparison_report.md"
    with open(markdown_file, 'w') as f:
        f.write(markdown_report)
    
    logger.info("Three-way comparison complete!")
    logger.info(f"CLIP-only accuracy: {results['clip_only']['accuracy']:.4f}")
    logger.info(f"LLM-only accuracy: {results['llm_only']['accuracy']:.4f}")
    logger.info(f"LLM-augmented accuracy: {results['llm_augmented']['accuracy']:.4f}")
    
    return results


def generate_markdown_report(comparison_report: Dict, results: Dict) -> str:
    """Generate human-readable markdown report."""
    report = "# Three-Way Comparison Report\n\n"
    report += "## Summary\n\n"
    report += f"- **CLIP-only accuracy**: {comparison_report['clip_only_accuracy']:.4f}\n"
    report += f"- **LLM-only accuracy**: {comparison_report['llm_only_accuracy']:.4f}\n"
    report += f"- **LLM-augmented accuracy**: {comparison_report['llm_augmented_accuracy']:.4f}\n"
    report += f"- **Fusion method**: {comparison_report['fusion_method']}\n\n"
    report += f"- **Improvement over CLIP-only**: {comparison_report['improvement_over_clip']:+.4f}\n"
    report += f"- **Improvement over LLM-only**: {comparison_report['improvement_over_llm']:+.4f}\n\n"
    
    report += "## Detailed Results\n\n"
    report += "See individual result directories for detailed metrics, confusion matrices, and per-emotion accuracy.\n"
    
    return report

