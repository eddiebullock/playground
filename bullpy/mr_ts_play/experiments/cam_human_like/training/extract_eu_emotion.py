#!/usr/bin/env python3
"""
Extract EU-Emotion dataset from ZIP files.

The EU-Emotion dataset comes as multiple ZIP files that need to be extracted.
This script will:
1. Find all ZIP files in the dataset directory
2. Extract them to a target directory
3. Handle split archives (multi-part ZIPs)
4. Skip already extracted files

Usage:
    python experiments/cam_human_like/training/extract_eu_emotion.py \
        --source_dir "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
        --target_dir "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions_extracted"
"""

import argparse
import zipfile
from pathlib import Path
import sys
from tqdm import tqdm


def extract_zip(zip_path: Path, target_dir: Path, skip_existing: bool = True):
    """Extract a ZIP file to target directory."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Check if already extracted
            if skip_existing:
                first_file = zip_ref.namelist()[0] if zip_ref.namelist() else None
                if first_file:
                    first_file_path = target_dir / first_file
                    if first_file_path.exists():
                        return False, "Already extracted"
            
            # Extract all files
            zip_ref.extractall(target_dir)
            return True, f"Extracted {len(zip_ref.namelist())} files"
    except zipfile.BadZipFile:
        return False, "Bad ZIP file (may be part of split archive)"
    except Exception as e:
        return False, f"Error: {e}"


def find_zip_files(source_dir: Path):
    """Find all ZIP files in source directory."""
    zip_files = []
    
    # Find all .zip files
    for zip_file in source_dir.glob("*.zip"):
        if not zip_file.name.startswith("."):
            zip_files.append(zip_file)
    
    # Find split archives (e.g., .zip-001.001, .zip-002.002)
    split_archives = {}
    for file in source_dir.glob("*"):
        if file.suffix == ".002" or file.suffix == ".006" or ".zip-" in file.name:
            # This might be a split archive part
            base_name = file.stem.split("-")[0] if "-" in file.name else file.stem
            if base_name not in split_archives:
                split_archives[base_name] = []
            split_archives[base_name].append(file)
    
    return sorted(zip_files), split_archives


def main():
    parser = argparse.ArgumentParser(description="Extract EU-Emotion dataset from ZIP files")
    parser.add_argument('--source_dir', type=str, required=True, help='Directory containing ZIP files')
    parser.add_argument('--target_dir', type=str, help='Directory to extract to (default: source_dir + "_extracted")')
    parser.add_argument('--skip_existing', action='store_true', default=True, help='Skip already extracted files')
    
    args = parser.parse_args()
    
    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}")
        sys.exit(1)
    
    if args.target_dir:
        target_dir = Path(args.target_dir)
    else:
        target_dir = source_dir.parent / f"{source_dir.name}_extracted"
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("EU-Emotion Dataset Extraction")
    print("=" * 60)
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    print()
    
    # Find ZIP files
    zip_files, split_archives = find_zip_files(source_dir)
    
    print(f"Found {len(zip_files)} ZIP files")
    if split_archives:
        print(f"Found {len(split_archives)} potential split archives")
    print()
    
    # Extract ZIP files
    print("Extracting ZIP files...")
    extracted_count = 0
    skipped_count = 0
    error_count = 0
    
    for zip_file in tqdm(zip_files, desc="Extracting"):
        success, message = extract_zip(zip_file, target_dir, args.skip_existing)
        if success:
            extracted_count += 1
        elif "Already extracted" in message:
            skipped_count += 1
        else:
            error_count += 1
            print(f"  ⚠️  {zip_file.name}: {message}")
    
    print()
    print("=" * 60)
    print("Extraction Summary")
    print("=" * 60)
    print(f"Extracted: {extracted_count}")
    print(f"Skipped (already extracted): {skipped_count}")
    print(f"Errors: {error_count}")
    print()
    
    if split_archives:
        print("Note: Split archives detected. You may need to:")
        print("  1. Combine split parts using a tool like 7-Zip or Keka")
        print("  2. Or extract manually")
        print()
    
    print(f"Extracted dataset location: {target_dir}")
    print()
    print("Next step: Test the dataset loader:")
    print(f"  python experiments/cam_human_like/training/test_eu_emotion.py \\")
    print(f"      --eu_emotion_dir {target_dir}")


if __name__ == "__main__":
    main()







