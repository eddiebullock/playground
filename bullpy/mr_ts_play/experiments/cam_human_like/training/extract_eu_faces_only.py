#!/usr/bin/env python3
"""
Extract ONLY face files from EU-Emotion dataset ZIP files.

This script extracts only the face videos from the EU-Emotion dataset,
significantly reducing the storage requirement from ~213GB to ~20-40GB.

Usage:
    python experiments/cam_human_like/training/extract_eu_faces_only.py \
        --source_dir "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
        --target_dir "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions_faces"
"""

import argparse
import zipfile
from pathlib import Path
import sys
from tqdm import tqdm
import os


def should_extract_file(file_path: str) -> bool:
    """
    Check if a file should be extracted (only face files).
    
    Face files are in paths containing:
    - "Faces" or "faces" or "Face" or "face"
    - Video extensions: .mp4, .mov, .avi, .mkv
    """
    file_path_lower = file_path.lower()
    
    # Must be in a Faces directory
    if "face" not in file_path_lower:
        return False
    
    # Must be a video file
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.m4v'}
    file_ext = Path(file_path).suffix.lower()
    if file_ext not in video_extensions:
        return False
    
    # Exclude body gestures and social scenes
    if "body" in file_path_lower and "face" not in file_path_lower:
        return False
    if "social" in file_path_lower and "face" not in file_path_lower:
        return False
    
    return True


def extract_faces_from_zip(zip_path: Path, target_dir: Path, skip_existing: bool = True):
    """Extract only face files from a ZIP file."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get all file names
            all_files = zip_ref.namelist()
            
            # Filter for face files only
            face_files = [f for f in all_files if should_extract_file(f)]
            
            if len(face_files) == 0:
                return False, "No face files found in this ZIP"
            
            # Check if already extracted
            if skip_existing:
                first_face_file = face_files[0]
                first_file_path = target_dir / first_face_file
                if first_file_path.exists():
                    return False, "Already extracted"
            
            # Extract only face files
            extracted_count = 0
            for file_name in face_files:
                try:
                    # Create target directory structure
                    target_file_path = target_dir / file_name
                    target_file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Extract file
                    with zip_ref.open(file_name) as source:
                        with open(target_file_path, 'wb') as target:
                            target.write(source.read())
                    extracted_count += 1
                except Exception as e:
                    print(f"  Warning: Could not extract {file_name}: {e}")
                    continue
            
            return True, f"Extracted {extracted_count}/{len(face_files)} face files"
            
    except zipfile.BadZipFile:
        return False, "Bad ZIP file (may be part of split archive)"
    except Exception as e:
        return False, f"Error: {e}"


def find_zip_files(source_dir: Path):
    """Find all ZIP files in source directory."""
    zip_files = []
    
    # Find all .zip files
    for zip_file in source_dir.rglob("*.zip"):
        if not zip_file.name.startswith("."):
            zip_files.append(zip_file)
    
    return sorted(zip_files)


def main():
    parser = argparse.ArgumentParser(description="Extract ONLY face files from EU-Emotion dataset")
    parser.add_argument('--source_dir', type=str, required=True, help='Directory containing ZIP files')
    parser.add_argument('--target_dir', type=str, help='Directory to extract face files to (default: source_dir + "_faces")')
    parser.add_argument('--skip_existing', action='store_true', default=True, help='Skip already extracted files')
    
    args = parser.parse_args()
    
    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}")
        sys.exit(1)
    
    if args.target_dir:
        target_dir = Path(args.target_dir)
    else:
        target_dir = source_dir.parent / f"{source_dir.name}_faces"
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("EU-Emotion Dataset - Face Files Extraction")
    print("=" * 60)
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    print("Note: Only extracting face video files (reduces size from ~213GB to ~20-40GB)")
    print()
    
    # Find ZIP files
    zip_files = find_zip_files(source_dir)
    
    print(f"Found {len(zip_files)} ZIP files")
    print()
    
    # Extract face files from ZIP files
    print("Extracting face files from ZIP files...")
    extracted_count = 0
    skipped_count = 0
    error_count = 0
    total_face_files = 0
    
    for zip_file in tqdm(zip_files, desc="Processing ZIPs"):
        success, message = extract_faces_from_zip(zip_file, target_dir, args.skip_existing)
        if success:
            extracted_count += 1
            # Count extracted files
            if "Extracted" in message:
                try:
                    num_files = int(message.split("/")[0].split()[-1])
                    total_face_files += num_files
                except:
                    pass
        elif "Already extracted" in message:
            skipped_count += 1
        else:
            error_count += 1
            if "No face files" not in message:  # Don't warn about ZIPs with no face files
                print(f"  ⚠️  {zip_file.name}: {message}")
    
    print()
    print("=" * 60)
    print("Extraction Summary")
    print("=" * 60)
    print(f"ZIPs processed: {extracted_count}")
    print(f"ZIPs skipped (already extracted): {skipped_count}")
    print(f"ZIPs with errors: {error_count}")
    print(f"Total face files extracted: {total_face_files}")
    print()
    
    print(f"Extracted face files location: {target_dir}")
    print()
    print("Next step: Test the dataset loader:")
    print(f"  python experiments/cam_human_like/training/test_eu_emotion.py \\")
    print(f"      --eu_emotion_dir {target_dir} \\")
    print(f"      --eu_emotion_modality face")

if __name__ == "__main__":
    main()


