#!/usr/bin/env python3
"""
Evaluate a fine-tuned CLIP model on the CAM test set.

This script loads a fine-tuned model and evaluates it on CAM,
comparing performance to the zero-shot baseline (37%).
"""

import argparse
import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# CAMDataset is not needed - we use TaskSpecificTrialDataset instead
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image


def compute_metrics(predictions, trials):
    """
    Compute evaluation metrics from predictions and trials.
    
    Args:
        predictions: List of prediction dicts with keys:
            - 'trial_id': str
            - 'is_correct': bool
            - 'predicted_label': str
            - 'correct_label': str
        trials: List of trial objects with 'modality' and 'trial_id' attributes
    
    Returns:
        Dictionary with metrics: accuracy, face_accuracy, voice_accuracy
    """
    if not predictions:
        return {'accuracy': 0.0, 'face_accuracy': 0.0, 'voice_accuracy': 0.0}
    
    # Overall accuracy
    correct = sum(1 for p in predictions if p['is_correct'])
    accuracy = correct / len(predictions)
    
    # Create trial lookup by trial_id (fixes bug where zip assumes index alignment)
    trial_dict = {trial.trial_id: trial for trial in trials}
    
    # Split by modality - match predictions to trials by trial_id
    face_predictions = []
    voice_predictions = []
    
    for pred in predictions:
        trial_id = pred['trial_id']
        if trial_id in trial_dict:
            trial = trial_dict[trial_id]
            if trial.modality == 'face':
                face_predictions.append(pred)
            elif trial.modality == 'voice':
                voice_predictions.append(pred)
    
    face_accuracy = sum(1 for p in face_predictions if p['is_correct']) / len(face_predictions) if face_predictions else 0.0
    voice_accuracy = sum(1 for p in voice_predictions if p['is_correct']) / len(voice_predictions) if voice_predictions else 0.0
    
    return {
        'accuracy': accuracy,
        'face_accuracy': face_accuracy,
        'voice_accuracy': voice_accuracy,
    }


def evaluate_finetuned_model(
    model_path: str,
    trial_definitions_file: str,
    data_root: str,
    dataset_type: str = "cam",
    splits_dir: str = None,
    split_name: str = "test",
    device: str = "cpu",
    num_frames: int = 8,
    use_multiframe: bool = True,
):
    """
    Evaluate a fine-tuned CLIP model on CAM or EU-Emotion test set.
    
    Args:
        model_path: Path to fine-tuned model (directory containing config.json, pytorch_model.bin)
        trial_definitions_file: Path to trial definitions JSON (CAM or EU-Emotion)
        data_root: Root directory of video stimuli
        dataset_type: "cam" or "eu_emotion"
        splits_dir: Directory containing train/val/test splits (CAM only, optional)
        split_name: Which split to evaluate on ("test" or "val")
        device: Device to run on
        num_frames: Number of frames to extract per video
        use_multiframe: Whether to use multiple frames (average features)
    """
    print("=" * 60)
    if dataset_type == 'cam':
        print("Evaluating Fine-Tuned Model on CAM Test Set")
    else:
        print("Evaluating Fine-Tuned Model on EU-Emotion Test Set")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_type.upper()}")
    print(f"Split: {split_name}")
    print(f"Device: {device}")
    print(f"Multi-frame: {use_multiframe}")
    print()
    
    # Load dataset
    if dataset_type == 'cam':
        print("Loading CAM test dataset...")
        from experiments.cam_human_like.training.task_specific_dataset import TaskSpecificTrialDataset
        
        # Load trial definitions to create wrapper
        with open(trial_definitions_file, 'r') as f:
            trial_defs = json.load(f)
        
        # Create a simple trial class for compatibility
        class SimpleTrial:
            def __init__(self, trial_data, data_root):
                self.trial_id = trial_data.get('trial_id', '')
                self.stimulus_path = str(Path(data_root) / trial_data['stimulus_path']) if not Path(trial_data['stimulus_path']).is_absolute() else trial_data['stimulus_path']
                self.modality = trial_data.get('modality', 'face')
                self.correct_label = trial_data.get('correct_label', '')
                self.candidate_labels = trial_data.get('candidate_labels', [])
                self.correct_idx = trial_data.get('correct_idx', 0)
                self.actor = trial_data.get('actor', '')
                self.scenario_id = trial_data.get('scenario_id', '')
                self.concept = trial_data.get('concept', '')
        
        # Create wrapper that provides .trials attribute
        class CAMDatasetWrapper:
            def __init__(self, task_dataset, trial_defs, data_root):
                self.task_dataset = task_dataset
                self.trials = [SimpleTrial(t, data_root) for t in trial_defs.get('trials', [])]
        
        task_dataset = TaskSpecificTrialDataset(
            data_root=data_root,
            trial_definitions_file=trial_definitions_file,
            num_frames=num_frames,
        )
        dataset = CAMDatasetWrapper(task_dataset, trial_defs, data_root)
    else:
        # EU-Emotion: load directly from trial definitions
        print("Loading EU-Emotion test dataset...")
        from experiments.cam_human_like.training.task_specific_dataset import TaskSpecificTrialDataset
        dataset = TaskSpecificTrialDataset(
            data_root=data_root,
            trial_definitions_file=trial_definitions_file,
            num_frames=num_frames,
        )
        # Convert to CAMDataset-like interface for compatibility
        # Need to get actual stimulus paths from trial definitions
        with open(trial_definitions_file, 'r') as f:
            trial_defs = json.load(f)
        
        # Create mapping from trial_id to stimulus_path
        trial_id_to_path = {t['trial_id']: t['stimulus_path'] for t in trial_defs['trials']}
        
        class EUEmotionDatasetWrapper:
            def __init__(self, task_dataset, trial_id_to_path, data_root):
                self.trials = []
                from pathlib import Path
                for i in range(len(task_dataset)):
                    item = task_dataset[i]
                    trial_id = item['trial_id']
                    
                    # Get actual stimulus path from trial definitions
                    stimulus_path_rel = trial_id_to_path.get(trial_id, '')
                    if stimulus_path_rel:
                        # Resolve to absolute path
                        if Path(stimulus_path_rel).is_absolute():
                            stimulus_path = stimulus_path_rel
                        else:
                            stimulus_path = str(Path(data_root) / stimulus_path_rel)
                    else:
                        stimulus_path = ''
                    
                    # Create simple trial object for EU-Emotion
                    class SimpleTrial:
                        def __init__(self, trial_data, data_root):
                            self.trial_id = trial_data.get('trial_id', '')
                            self.stimulus_path = str(Path(data_root) / trial_data['stimulus_path']) if not Path(trial_data['stimulus_path']).is_absolute() else trial_data['stimulus_path']
                            self.modality = trial_data.get('modality', 'face')
                            self.correct_label = trial_data.get('correct_label', '')
                            self.candidate_labels = trial_data.get('candidate_labels', [])
                            self.correct_idx = trial_data.get('correct_idx', 0)
                            self.actor = trial_data.get('actor', 'unknown')
                            self.scenario_id = trial_data.get('scenario_id', '')
                            self.concept = trial_data.get('concept', '')
                    
                    trial_data = {
                        'trial_id': trial_id,
                        'stimulus_path': stimulus_path,
                        'modality': 'face',
                        'correct_label': item['candidate_labels'][item['correct_idx']],
                        'candidate_labels': item['candidate_labels'],
                        'correct_idx': item['correct_idx'],
                        'actor': 'unknown',
                        'scenario_id': '',
                        'concept': item['candidate_labels'][item['correct_idx']],
                    }
                    trial = SimpleTrial(trial_data, data_root)
                    self.trials.append(trial)
        
        dataset_wrapper = EUEmotionDatasetWrapper(dataset, trial_id_to_path, data_root)
        dataset = dataset_wrapper
    
    print(f"Loaded {len(dataset.trials)} trials")
    print()
    
    # Load fine-tuned model
    print(f"Loading fine-tuned model from: {model_path}")
    model = CLIPModel.from_pretrained(model_path)
    processor = CLIPProcessor.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    print("✅ Model loaded successfully")
    print()
    
    def load_video_frames(video_path, num_frames=8):
        """Extract frames from video."""
        try:
            # Check if file exists first
            if not Path(video_path).exists():
                raise FileNotFoundError(f"Video file does not exist: {video_path}")
            
            # Check file permissions
            if not Path(video_path).is_file():
                raise ValueError(f"Path is not a file: {video_path}")
            
            # Check file size - videos smaller than 50KB are likely corrupted/incomplete
            import os
            file_size = os.path.getsize(video_path)
            if file_size < 50 * 1024:  # 50KB threshold
                raise ValueError(f"Video file too small (likely corrupted): {video_path} (size: {file_size:,} bytes, expected >50KB)")
            
            # Try to open with OpenCV
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                # Provide more diagnostic info
                raise ValueError(f"Could not open video: {video_path} (file exists: True, size: {file_size:,} bytes)")
            
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
    
    def score_labels(model, processor, video_path, candidate_labels, device, num_frames=8, use_multiframe=True):
        """Score candidate labels for a video."""
        # Load frames
        video_frames = load_video_frames(video_path, num_frames if use_multiframe else 1)
        
        # Process images
        image_inputs = processor(images=video_frames, return_tensors="pt").to(device)
        
        # Use prompt templates
        prompted_labels = [f"a photo of a person feeling {label}" for label in candidate_labels]
        text_inputs = processor(
            text=prompted_labels,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        
        # Get embeddings
        with torch.no_grad():
            image_features = model.get_image_features(**image_inputs)
            text_features = model.get_text_features(**text_inputs)
            
            # Aggregate video features (mean pooling if multi-frame)
            if use_multiframe and len(video_frames) > 1:
                video_features = image_features.mean(dim=0, keepdim=True)
            else:
                video_features = image_features[0:1]  # Use first frame
            
            # Normalize
            video_features = F.normalize(video_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            
            # Compute similarity
            logits = video_features @ text_features.t()  # (1, num_labels)
            
            # Convert to scores dictionary
            scores = {label: logits[0][i].item() for i, label in enumerate(candidate_labels)}
        
        return scores
    
    # Run evaluation
    print("Running evaluation...")
    predictions = []
    correct_labels = []
    skipped_trials = []
    
    for trial in tqdm(dataset.trials, desc="Evaluating"):
        try:
            # Resolve video path - handle multiple path formats
            video_path = None
            stimulus_path = trial.stimulus_path
            
            # Strategy 1: Use path as-is if it exists
            if Path(stimulus_path).exists():
                video_path = Path(stimulus_path)
            # Strategy 2: If relative, try joining with data_root
            elif not Path(stimulus_path).is_absolute():
                candidate = Path(data_root) / stimulus_path
                if candidate.exists():
                    video_path = candidate
            # Strategy 3: If absolute but doesn't exist, try relative to data_root
            elif Path(stimulus_path).is_absolute():
                # Extract relative part and try with data_root
                try:
                    # If path contains data_root, try extracting relative part
                    if str(data_root) in stimulus_path:
                        rel_part = stimulus_path.split(str(data_root))[-1].lstrip('/')
                        candidate = Path(data_root) / rel_part
                        if candidate.exists():
                            video_path = candidate
                except:
                    pass
            
            # Strategy 4: Search by filename as last resort
            if video_path is None or not video_path.exists():
                filename = Path(stimulus_path).name
                found_files = list(Path(data_root).rglob(filename))
                if found_files:
                    video_path = found_files[0]
                else:
                    raise FileNotFoundError(f"Video not found: {trial.stimulus_path} (tried: {stimulus_path})")
            
            # Score labels
            scores = score_labels(
                model=model,
                processor=processor,
                video_path=video_path,
                candidate_labels=trial.candidate_labels,
                device=device,
                num_frames=num_frames,
                use_multiframe=use_multiframe,
            )
            
            # Get prediction (highest scoring label)
            predicted_label = max(scores.items(), key=lambda x: x[1])[0]
            predicted_idx = trial.candidate_labels.index(predicted_label)
            
            predictions.append({
                'trial_id': trial.trial_id,
                'predicted_label': predicted_label,
                'predicted_idx': predicted_idx,
                'correct_label': trial.correct_label,
                'correct_idx': trial.correct_idx,
                'is_correct': predicted_idx == trial.correct_idx,
                'scores': scores,
            })
            correct_labels.append(trial.correct_label)
        except (ValueError, FileNotFoundError, OSError) as e:
            # Skip trials with missing or corrupted video files
            print(f"Warning: Skipping trial {trial.trial_id}: {e}")
            skipped_trials.append(trial.trial_id)
            continue
    
    if skipped_trials:
        print(f"\nSkipped {len(skipped_trials)} trials due to missing/corrupted files")
        print(f"Evaluating on {len(predictions)} valid trials")
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(predictions, dataset.trials)
    
    # Print results
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Overall Accuracy: {metrics['accuracy']:.2%}")
    print(f"Face Accuracy: {metrics.get('face_accuracy', 'N/A')}")
    print(f"Voice Accuracy: {metrics.get('voice_accuracy', 'N/A')}")
    print()
    if dataset_type == 'cam':
        print(f"Baseline (zero-shot CLIP on CAM): 37.0%")
        print(f"Improvement: {metrics['accuracy'] - 0.37:.2%} ({((metrics['accuracy'] / 0.37) - 1) * 100:.1f}% relative)")
    else:
        print(f"Random baseline (4-option forced-choice): 25.0%")
        print(f"Improvement over random: {metrics['accuracy'] - 0.25:.2%} ({((metrics['accuracy'] / 0.25) - 1) * 100:.1f}% relative)")
    print()
    
    # Save results
    if dataset_type == 'eu_emotion':
        results_file = Path(model_path).parent / f"eu_emotion_evaluation_{split_name}.json"
    else:
        results_file = Path(model_path).parent / f"cam_evaluation_{split_name}.json"
    results = {
        'model_path': model_path,
        'split': split_name,
        'num_trials': len(dataset.trials),
        'num_valid_trials': len(predictions),
        'num_skipped_trials': len(skipped_trials),
        'skipped_trial_ids': skipped_trials,
        'metrics': metrics,
        'predictions': predictions,
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    print()
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned CLIP model on CAM or EU-Emotion test set")
    parser.add_argument('--model_path', type=str, required=True, help='Path to fine-tuned model directory')
    parser.add_argument('--trial_definitions', type=str, default='data/cam_trial_definitions_20concepts.json', help='Path to trial definitions JSON (CAM or EU-Emotion)')
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of video stimuli')
    parser.add_argument('--dataset_type', type=str, choices=['cam', 'eu_emotion'], default='cam', help='Dataset type (default: cam)')
    parser.add_argument('--splits_dir', type=str, help='Directory containing train/val/test splits (CAM only, optional)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], help='Which split to evaluate on')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu, cuda, mps)')
    parser.add_argument('--num_frames', type=int, default=8, help='Number of frames per video')
    parser.add_argument('--use_multiframe', action='store_true', default=True, help='Use multiple frames (average features)')
    parser.add_argument('--single_frame', action='store_true', help='Use only middle frame')
    
    args = parser.parse_args()
    
    use_multiframe = args.use_multiframe and not args.single_frame
    
    metrics = evaluate_finetuned_model(
        model_path=args.model_path,
        trial_definitions_file=args.trial_definitions,
        data_root=args.data_root,
        dataset_type=args.dataset_type,
        splits_dir=args.splits_dir,
        split_name=args.split,
        device=args.device,
        num_frames=args.num_frames,
        use_multiframe=use_multiframe,
    )
    
    print("=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()


