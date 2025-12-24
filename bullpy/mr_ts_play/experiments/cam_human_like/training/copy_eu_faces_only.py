#!/usr/bin/env python3
"""
Copy ONLY face files from already-extracted EU-Emotion dataset.

The EU-Emotion dataset is already extracted with structure:
  emotions*/HD Version - Face, Body, Social/Faces - HD Version/EDITED/EmotionName/*.mp4
  emotions*/HD Version - Face, Body, Social/Faces - HD Version/Original/EmotionName/*.mov

This script finds all face video files and copies them to a target directory,
preserving the emotion folder structure for easy dataset loading.

Usage:
    python experiments/cam_human_like/training/copy_eu_faces_only.py \
        --source_dir "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
        --target_dir "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions_faces"
"""

import argparse
import shutil
from pathlib import Path
import sys
from tqdm import tqdm


def find_face_files(source_dir: Path):
    """
    Find all face video files in the EU-Emotion structure.
    
    Looks for files in:
    - emotions*/HD Version - Face, Body, Social/Faces - HD Version/EDITED/EmotionName/*.mp4
    - emotions*/HD Version - Face, Body, Social/Faces - HD Version/Original/EmotionName/*.mov
    """
    face_files = []
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.m4v'}
    
    # Find all emotions* directories
    emotions_dirs = sorted(source_dir.glob("emotions*"))
    
    print(f"Found {len(emotions_dirs)} emotions directories")
    
    for emotions_dir in emotions_dirs:
        if not emotions_dir.is_dir():
            continue
        
        # Look for Faces - HD Version directory
        faces_dir = emotions_dir / "HD Version - Face, Body, Social" / "Faces - HD Version"
        
        if not faces_dir.exists():
            continue
        
        # Look in EDITED and Original subdirectories
        for subdir_name in ["EDITED", "Original"]:
            subdir = faces_dir / subdir_name
            if not subdir.exists():
                continue
            
            # Find all emotion-named subdirectories
            for emotion_dir in subdir.iterdir():
                if not emotion_dir.is_dir():
                    continue
                
                emotion_name = emotion_dir.name
                
                # Find all video files in this emotion directory
                for video_file in emotion_dir.iterdir():
                    if video_file.is_file() and video_file.suffix.lower() in video_extensions:
                        face_files.append({
                            'source': video_file,
                            'emotion': emotion_name,
                            'subdir': subdir_name,
                        })
    
    return face_files


def copy_face_files(face_files, target_dir: Path, preserve_structure: bool = True):
    """
    Copy face files to target directory.
    
    If preserve_structure=True, maintains emotion folder structure:
      target_dir/EmotionName/video.mp4
    
    If preserve_structure=False, flattens to:
      target_dir/video.mp4
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    
    copied_count = 0
    skipped_count = 0
    error_count = 0
    
    for face_info in tqdm(face_files, desc="Copying files"):
        source_file = face_info['source']
        emotion = face_info['emotion']
        
        if preserve_structure:
            # Preserve emotion folder structure
            target_emotion_dir = target_dir / emotion
            target_emotion_dir.mkdir(parents=True, exist_ok=True)
            target_file = target_emotion_dir / source_file.name
        else:
            # Flatten: use emotion_name_filename format
            target_file = target_dir / f"{emotion}_{source_file.name}"
        
        # Skip if already exists
        if target_file.exists():
            skipped_count += 1
            continue
        
        try:
            # Copy file
            shutil.copy2(source_file, target_file)
            copied_count += 1
        except Exception as e:
            error_count += 1
            print(f"  ⚠️  Error copying {source_file.name}: {e}")
    
    return copied_count, skipped_count, error_count


def main():
    parser = argparse.ArgumentParser(description="Copy only face files from extracted EU-Emotion dataset")
    parser.add_argument('--source_dir', type=str, required=True, help='Directory containing extracted EU-Emotion dataset')
    parser.add_argument('--target_dir', type=str, help='Directory to copy face files to (default: source_dir + "_faces")')
    parser.add_argument('--preserve_structure', action='store_true', default=True, help='Preserve emotion folder structure (default: True)')
    parser.add_argument('--flatten', action='store_true', help='Flatten structure (no emotion folders)')
    
    args = parser.parse_args()
    
    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}")
        sys.exit(1)
    
    if args.target_dir:
        target_dir = Path(args.target_dir)
    else:
        target_dir = source_dir.parent / f"{source_dir.name}_faces"
    
    preserve_structure = args.preserve_structure and not args.flatten
    
    print("=" * 60)
    print("EU-Emotion Dataset - Face Files Copy")
    print("=" * 60)
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    print(f"Structure: {'Preserved (emotion folders)' if preserve_structure else 'Flattened'}")
    print()
    
    # Find all face files
    print("Searching for face video files...")
    face_files = find_face_files(source_dir)
    
    if len(face_files) == 0:
        print("❌ No face files found!")
        print(f"   Searched in: {source_dir}/emotions*/HD Version - Face, Body, Social/Faces - HD Version/")
        sys.exit(1)
    
    print(f"Found {len(face_files)} face video files")
    
    # Count unique emotions
    unique_emotions = set(f['emotion'] for f in face_files)
    print(f"Found {len(unique_emotions)} unique emotions: {', '.join(sorted(unique_emotions)[:10])}{'...' if len(unique_emotions) > 10 else ''}")
    print()
    
    # Copy files
    print("Copying face files...")
    copied_count, skipped_count, error_count = copy_face_files(face_files, target_dir, preserve_structure)
    
    print()
    print("=" * 60)
    print("Copy Summary")
    print("=" * 60)
    print(f"Total files found: {len(face_files)}")
    print(f"Copied: {copied_count}")
    print(f"Skipped (already exists): {skipped_count}")
    print(f"Errors: {error_count}")
    print()
    
    # Calculate total size
    total_size = sum(f['source'].stat().st_size for f in face_files)
    total_size_gb = total_size / (1024 ** 3)
    print(f"Total size: {total_size_gb:.2f} GB")
    print()
    
    print(f"Face files location: {target_dir}")
    print()
    print("Next step: Test the dataset loader:")
    print(f"  python experiments/cam_human_like/training/test_eu_emotion.py \\")
    print(f"      --eu_emotion_dir {target_dir} \\")
    print(f"      --eu_emotion_modality face")


if __name__ == "__main__":
    main()






