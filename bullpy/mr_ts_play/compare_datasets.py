#!/usr/bin/env python3
"""
Compare the two datasets to see if they're different.
"""

import re
from pathlib import Path
from collections import Counter, defaultdict

def parse_filename(filename: str) -> dict:
    """Parse MindReading filename format."""
    base = filename.replace(".mov", "").replace(".MOV", "")
    match = re.search(r'([VT])([a-z]+(?:-[a-z]+)*)$', base)
    if not match:
        return None
    modality = match.group(1)
    emotion = match.group(2)
    prefix = base[:match.start()]
    scenario_match = re.match(r'^(\d{7})', prefix)
    if not scenario_match:
        return None
    scenario_id = scenario_match.group(1)
    return {
        "scenario_id": scenario_id,
        "modality": modality,
        "emotion": emotion,
        "filename": filename
    }

def analyze_dataset(dataset_path: str, name: str):
    """Analyze a dataset."""
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"\n{name}: PATH DOES NOT EXIST")
        return None
    
    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"{'='*80}")
    print(f"Path: {dataset_path}")
    
    # Find all .mov files
    video_files = list(dataset_path.rglob("*.mov")) + list(dataset_path.rglob("*.MOV"))
    print(f"Total video files: {len(video_files)}")
    
    # Parse files
    parsed_files = []
    emotions = set()
    scenarios = set()
    modalities = Counter()
    filenames = set()
    
    for video_file in video_files:
        parsed = parse_filename(video_file.name)
        if parsed:
            parsed_files.append(parsed)
            emotions.add(parsed["emotion"])
            scenarios.add(parsed["scenario_id"])
            modalities[parsed["modality"]] += 1
            filenames.add(parsed["filename"])
    
    print(f"Successfully parsed: {len(parsed_files)}")
    print(f"Unique emotions: {len(emotions)}")
    print(f"Unique scenarios: {len(scenarios)}")
    print(f"Modalities: {dict(modalities)}")
    
    # Check directory structure
    top_level = [d.name for d in dataset_path.iterdir() if d.is_dir()]
    print(f"Top-level directories: {len(top_level)}")
    print(f"Sample directories: {sorted(top_level)[:10]}")
    
    return {
        "path": str(dataset_path),
        "total_files": len(video_files),
        "parsed_files": len(parsed_files),
        "emotions": emotions,
        "emotion_count": len(emotions),
        "scenarios": scenarios,
        "scenario_count": len(scenarios),
        "modalities": dict(modalities),
        "filenames": filenames,
        "top_level_dirs": top_level
    }

# Compare datasets
cam_path = "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/mindreading_transporter_files/Mindreading emotions library/Emotions"
mindreading_path = "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/MindReading/Emotions"

cam_data = analyze_dataset(cam_path, "CAM DATASET")
mindreading_data = analyze_dataset(mindreading_path, "MINDREADING/EMOTIONS DATASET")

if cam_data and mindreading_data:
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")
    
    print(f"\nFile count comparison:")
    print(f"  CAM: {cam_data['total_files']} files")
    print(f"  MindReading: {mindreading_data['total_files']} files")
    print(f"  Difference: {abs(cam_data['total_files'] - mindreading_data['total_files'])} files")
    
    print(f"\nEmotion count comparison:")
    print(f"  CAM: {cam_data['emotion_count']} emotions")
    print(f"  MindReading: {mindreading_data['emotion_count']} emotions")
    print(f"  Difference: {abs(cam_data['emotion_count'] - mindreading_data['emotion_count'])} emotions")
    
    # Check overlap
    cam_emotions = cam_data['emotions']
    mr_emotions = mindreading_data['emotions']
    overlap = cam_emotions & mr_emotions
    only_cam = cam_emotions - mr_emotions
    only_mr = mr_emotions - cam_emotions
    
    print(f"\nEmotion overlap:")
    print(f"  Overlapping emotions: {len(overlap)}")
    print(f"  Only in CAM: {len(only_cam)}")
    print(f"  Only in MindReading: {len(only_mr)}")
    
    if len(overlap) > 0:
        print(f"\nSample overlapping emotions: {sorted(list(overlap))[:10]}")
    if len(only_cam) > 0:
        print(f"\nSample emotions only in CAM: {sorted(list(only_cam))[:10]}")
    if len(only_mr) > 0:
        print(f"\nSample emotions only in MindReading: {sorted(list(only_mr))[:10]}")
    
    # Check filename overlap
    cam_filenames = cam_data['filenames']
    mr_filenames = mindreading_data['filenames']
    filename_overlap = cam_filenames & mr_filenames
    only_cam_files = cam_filenames - mr_filenames
    only_mr_files = mr_filenames - cam_filenames
    
    print(f"\nFilename overlap:")
    print(f"  Overlapping filenames: {len(filename_overlap)}")
    print(f"  Only in CAM: {len(only_cam_files)}")
    print(f"  Only in MindReading: {len(only_mr_files)}")
    
    # Check directory structure
    print(f"\nDirectory structure comparison:")
    print(f"  CAM top-level dirs: {len(cam_data['top_level_dirs'])}")
    print(f"  MindReading top-level dirs: {len(mindreading_data['top_level_dirs'])}")
    
    cam_dirs = set(cam_data['top_level_dirs'])
    mr_dirs = set(mindreading_data['top_level_dirs'])
    dir_overlap = cam_dirs & mr_dirs
    only_cam_dirs = cam_dirs - mr_dirs
    only_mr_dirs = mr_dirs - cam_dirs
    
    print(f"  Overlapping directories: {len(dir_overlap)}")
    if len(dir_overlap) > 0:
        print(f"    {sorted(list(dir_overlap))[:10]}")
    if len(only_cam_dirs) > 0:
        print(f"  Only in CAM: {sorted(list(only_cam_dirs))[:10]}")
    if len(only_mr_dirs) > 0:
        print(f"  Only in MindReading: {sorted(list(only_mr_dirs))[:10]}")
    
    # Conclusion
    print(f"\n{'='*80}")
    print("CONCLUSION")
    print(f"{'='*80}")
    
    if len(filename_overlap) > 0.9 * min(len(cam_filenames), len(mr_filenames)):
        print("These appear to be THE SAME DATASET (high filename overlap)")
    elif len(filename_overlap) > 0.5 * min(len(cam_filenames), len(mr_filenames)):
        print("These datasets have SIGNIFICANT OVERLAP (many shared files)")
    elif len(overlap) > 0.8 * min(len(cam_emotions), len(mr_emotions)):
        print("These datasets have SIMILAR EMOTIONS but different files")
    else:
        print("These appear to be DIFFERENT DATASETS")
