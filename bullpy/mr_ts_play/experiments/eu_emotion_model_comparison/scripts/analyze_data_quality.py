#!/usr/bin/env python3
"""
Comprehensive data quality and preprocessing analysis.

Checks for:
1. Video quality issues (corrupted files, missing files)
2. Frame extraction quality
3. Class imbalance
4. Data split quality (actor independence, balance)
5. Preprocessing consistency
6. Video metadata (resolution, fps, duration)
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
import cv2
import numpy as np
from PIL import Image
import pandas as pd

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.eu_emotion_model_comparison.models.video_utils import extract_frames, get_video_info


def extract_actor_code(stimulus_path: str) -> str:
    """
    Extract actor code from video filename.
    
    Pattern: {ActorCode}{Number}_cut.mp4 or {ActorCode}{Number}.mov
    Examples:
    - "GF6_cut.mp4" → "GF6"
    - "NF28_cut.mp4" → "NF28"
    - "IF8.mov" → "IF8"
    """
    filename = Path(stimulus_path).name
    
    # Remove extension
    base = filename.replace('.mp4', '').replace('.mov', '').replace('.avi', '').replace('.mkv', '').replace('.m4v', '')
    
    # Remove "_cut" suffix if present
    base = base.replace('_cut', '')
    
    # Extract actor code: letters before the first digit
    match = re.match(r'^([A-Z]+)(\d+)', base)
    if match:
        return match.group(1)  # Return the letters (actor code)
    
    # Fallback: try to extract any letters at the start
    match = re.match(r'^([A-Z]+)', base)
    if match:
        return match.group(1)
    
    return "unknown"

def check_video_files(trials: List[Dict], data_root: Path) -> Dict:
    """Check for missing, corrupted, or problematic video files."""
    issues = {
        'missing': [],
        'corrupted': [],
        'too_small': [],
        'zero_frames': [],
        'metadata': [],
    }
    
    for trial in trials:
        video_path = data_root / trial['stimulus_path']
        
        # Check if file exists
        if not video_path.exists():
            issues['missing'].append({
                'trial_id': trial.get('trial_id', 'unknown'),
                'path': str(video_path),
            })
            continue
        
        # Check file size
        file_size = video_path.stat().st_size
        if file_size < 50 * 1024:  # 50KB threshold
            issues['too_small'].append({
                'trial_id': trial.get('trial_id', 'unknown'),
                'path': str(video_path),
                'size': file_size,
            })
        
        # Check video metadata
        try:
            info = get_video_info(str(video_path))
            if info.get('frame_count', 0) == 0:
                issues['zero_frames'].append({
                    'trial_id': trial.get('trial_id', 'unknown'),
                    'path': str(video_path),
                })
            else:
                issues['metadata'].append({
                    'trial_id': trial.get('trial_id', 'unknown'),
                    'path': str(video_path),
                    **info,
                })
        except Exception as e:
            issues['corrupted'].append({
                'trial_id': trial.get('trial_id', 'unknown'),
                'path': str(video_path),
                'error': str(e),
            })
    
    return issues


def check_frame_extraction(trials: List[Dict], data_root: Path, num_frames: int = 8) -> Dict:
    """Check frame extraction quality."""
    issues = {
        'extraction_failures': [],
        'too_few_frames': [],
        'frame_quality': [],
    }
    
    for trial in trials[:10]:  # Sample first 10 for speed
        video_path = data_root / trial['stimulus_path']
        
        if not video_path.exists():
            continue
        
        try:
            frames = extract_frames(str(video_path), num_frames=num_frames)
            
            if len(frames) < num_frames:
                issues['too_few_frames'].append({
                    'trial_id': trial.get('trial_id', 'unknown'),
                    'extracted': len(frames),
                    'expected': num_frames,
                })
            
            # Check frame quality (basic checks)
            for i, frame in enumerate(frames):
                if frame.size[0] < 100 or frame.size[1] < 100:
                    issues['frame_quality'].append({
                        'trial_id': trial.get('trial_id', 'unknown'),
                        'frame': i,
                        'size': frame.size,
                    })
                
                # Check if frame is mostly black
                img_array = np.array(frame)
                if img_array.mean() < 10:  # Very dark
                    issues['frame_quality'].append({
                        'trial_id': trial.get('trial_id', 'unknown'),
                        'frame': i,
                        'issue': 'mostly_black',
                    })
        
        except Exception as e:
            issues['extraction_failures'].append({
                'trial_id': trial.get('trial_id', 'unknown'),
                'error': str(e),
            })
    
    return issues


def analyze_class_balance(trials: List[Dict]) -> Dict:
    """Analyze class distribution and balance."""
    emotion_counts = Counter(t['correct_label'] for t in trials)
    
    total = len(trials)
    num_classes = len(emotion_counts)
    
    # Calculate imbalance metrics
    counts = list(emotion_counts.values())
    max_count = max(counts)
    min_count = min(counts)
    mean_count = np.mean(counts)
    std_count = np.std(counts)
    
    # Imbalance ratio
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    # Classes with very few examples
    rare_classes = [emotion for emotion, count in emotion_counts.items() if count <= 2]
    
    return {
        'total_trials': total,
        'num_classes': num_classes,
        'emotion_counts': dict(emotion_counts),
        'max_count': max_count,
        'min_count': min_count,
        'mean_count': mean_count,
        'std_count': std_count,
        'imbalance_ratio': imbalance_ratio,
        'rare_classes': rare_classes,
        'distribution': {
            emotion: count / total for emotion, count in emotion_counts.items()
        },
    }


def check_data_splits(train_file: str, val_file: str) -> Dict:
    """Check train/val split quality."""
    with open(train_file, 'r') as f:
        train_data = json.load(f)
    with open(val_file, 'r') as f:
        val_data = json.load(f)
    
    train_trials = train_data.get('trials', train_data)
    val_trials = val_data.get('trials', val_data)
    
    # Check actor independence
    train_actors = set()
    val_actors = set()
    
    for trial in train_trials:
        # Extract actor from stimulus_path if not in trial dict
        actor = trial.get('actor', None)
        if actor is None:
            actor = extract_actor_code(trial.get('stimulus_path', ''))
        train_actors.add(actor)
    
    for trial in val_trials:
        # Extract actor from stimulus_path if not in trial dict
        actor = trial.get('actor', None)
        if actor is None:
            actor = extract_actor_code(trial.get('stimulus_path', ''))
        val_actors.add(actor)
    
    overlap = train_actors & val_actors
    
    # Check class balance across splits
    train_emotions = Counter(t['correct_label'] for t in train_trials)
    val_emotions = Counter(t['correct_label'] for t in val_trials)
    
    # Check if all classes present in both splits
    train_only = set(train_emotions.keys()) - set(val_emotions.keys())
    val_only = set(val_emotions.keys()) - set(train_emotions.keys())
    
    return {
        'train_size': len(train_trials),
        'val_size': len(val_trials),
        'train_actors': len(train_actors),
        'val_actors': len(val_actors),
        'actor_overlap': len(overlap),
        'actor_independent': len(overlap) == 0,
        'train_emotions': len(train_emotions),
        'val_emotions': len(val_emotions),
        'train_only_emotions': list(train_only),
        'val_only_emotions': list(val_only),
        'train_emotion_counts': dict(train_emotions),
        'val_emotion_counts': dict(val_emotions),
    }


def analyze_video_metadata(metadata: List[Dict]) -> Dict:
    """Analyze video metadata statistics."""
    if not metadata:
        return {}
    
    df = pd.DataFrame(metadata)
    
    return {
        'fps': {
            'mean': df['fps'].mean() if 'fps' in df else None,
            'std': df['fps'].std() if 'fps' in df else None,
            'min': df['fps'].min() if 'fps' in df else None,
            'max': df['fps'].max() if 'fps' in df else None,
        },
        'frame_count': {
            'mean': df['frame_count'].mean() if 'frame_count' in df else None,
            'std': df['frame_count'].std() if 'frame_count' in df else None,
            'min': df['frame_count'].min() if 'frame_count' in df else None,
            'max': df['frame_count'].max() if 'frame_count' in df else None,
        },
        'duration': {
            'mean': df['duration'].mean() if 'duration' in df else None,
            'std': df['duration'].std() if 'duration' in df else None,
            'min': df['duration'].min() if 'duration' in df else None,
            'max': df['duration'].max() if 'duration' in df else None,
        },
        'resolution': {
            'widths': df['width'].unique().tolist() if 'width' in df else [],
            'heights': df['height'].unique().tolist() if 'height' in df else [],
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze data quality and preprocessing issues"
    )
    parser.add_argument(
        '--trial-definitions',
        type=str,
        required=True,
        help='Trial definitions JSON file'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        required=True,
        help='Data root directory'
    )
    parser.add_argument(
        '--train-file',
        type=str,
        help='Train split file (for split analysis)'
    )
    parser.add_argument(
        '--val-file',
        type=str,
        help='Val split file (for split analysis)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/eu_emotion_model_comparison/data_quality_report.json',
        help='Output file for report'
    )
    parser.add_argument(
        '--num-frames',
        type=int,
        default=8,
        help='Number of frames to extract (for testing)'
    )
    
    args = parser.parse_args()
    
    # Load trials
    with open(args.trial_definitions, 'r') as f:
        data = json.load(f)
    
    trials = data.get('trials', data)
    data_root = Path(args.data_root)
    
    print("="*60)
    print("DATA QUALITY ANALYSIS")
    print("="*60)
    print(f"Trials: {len(trials)}")
    print(f"Data root: {data_root}")
    print()
    
    # 1. Check video files
    print("1. Checking video files...")
    video_issues = check_video_files(trials, data_root)
    print(f"   Missing: {len(video_issues['missing'])}")
    print(f"   Corrupted: {len(video_issues['corrupted'])}")
    print(f"   Too small: {len(video_issues['too_small'])}")
    print(f"   Zero frames: {len(video_issues['zero_frames'])}")
    print()
    
    # 2. Check frame extraction
    print("2. Checking frame extraction (sampling 10 videos)...")
    frame_issues = check_frame_extraction(trials, data_root, args.num_frames)
    print(f"   Extraction failures: {len(frame_issues['extraction_failures'])}")
    print(f"   Too few frames: {len(frame_issues['too_few_frames'])}")
    print(f"   Frame quality issues: {len(frame_issues['frame_quality'])}")
    print()
    
    # 3. Analyze class balance
    print("3. Analyzing class balance...")
    class_balance = analyze_class_balance(trials)
    print(f"   Total trials: {class_balance['total_trials']}")
    print(f"   Number of classes: {class_balance['num_classes']}")
    print(f"   Imbalance ratio: {class_balance['imbalance_ratio']:.2f}")
    print(f"   Rare classes (≤2 examples): {len(class_balance['rare_classes'])}")
    if class_balance['rare_classes']:
        print(f"     {class_balance['rare_classes']}")
    print()
    
    # 4. Check data splits
    split_analysis = None
    if args.train_file and args.val_file:
        print("4. Checking data splits...")
        split_analysis = check_data_splits(args.train_file, args.val_file)
        print(f"   Train size: {split_analysis['train_size']}")
        print(f"   Val size: {split_analysis['val_size']}")
        print(f"   Actor independent: {split_analysis['actor_independent']}")
        print(f"   Actor overlap: {split_analysis['actor_overlap']}")
        print(f"   Train-only emotions: {split_analysis['train_only_emotions']}")
        print(f"   Val-only emotions: {split_analysis['val_only_emotions']}")
        print()
    
    # 5. Analyze video metadata
    print("5. Analyzing video metadata...")
    metadata_stats = analyze_video_metadata(video_issues['metadata'])
    if metadata_stats:
        if metadata_stats.get('fps', {}).get('mean'):
            print(f"   FPS: {metadata_stats['fps']['mean']:.2f} ± {metadata_stats['fps']['std']:.2f}")
        if metadata_stats.get('duration', {}).get('mean'):
            print(f"   Duration: {metadata_stats['duration']['mean']:.2f}s ± {metadata_stats['duration']['std']:.2f}s")
        if metadata_stats.get('frame_count', {}).get('mean'):
            print(f"   Frame count: {metadata_stats['frame_count']['mean']:.0f} ± {metadata_stats['frame_count']['std']:.0f}")
    print()
    
    # Compile report
    report = {
        'video_issues': {
            'missing': len(video_issues['missing']),
            'corrupted': len(video_issues['corrupted']),
            'too_small': len(video_issues['too_small']),
            'zero_frames': len(video_issues['zero_frames']),
            'details': video_issues,
        },
        'frame_issues': {
            'extraction_failures': len(frame_issues['extraction_failures']),
            'too_few_frames': len(frame_issues['too_few_frames']),
            'frame_quality': len(frame_issues['frame_quality']),
            'details': frame_issues,
        },
        'class_balance': class_balance,
        'split_analysis': split_analysis,
        'metadata_stats': metadata_stats,
    }
    
    # Save report
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print("="*60)
    print("SUMMARY")
    print("="*60)
    
    # Critical issues
    critical = []
    if len(video_issues['missing']) > 0:
        critical.append(f"⚠️  {len(video_issues['missing'])} missing video files")
    if len(video_issues['corrupted']) > 0:
        critical.append(f"⚠️  {len(video_issues['corrupted'])} corrupted video files")
    if len(video_issues['zero_frames']) > 0:
        critical.append(f"⚠️  {len(video_issues['zero_frames'])} videos with 0 frames")
    if class_balance['imbalance_ratio'] > 5:
        critical.append(f"⚠️  High class imbalance (ratio: {class_balance['imbalance_ratio']:.2f})")
    if split_analysis and not split_analysis['actor_independent']:
        critical.append(f"⚠️  Data leakage: {split_analysis['actor_overlap']} actors in both train/val")
    
    if critical:
        print("CRITICAL ISSUES FOUND:")
        for issue in critical:
            print(f"  {issue}")
    else:
        print("✅ No critical issues found")
    
    print()
    print(f"Full report saved to: {output_file}")

if __name__ == '__main__':
    main()
