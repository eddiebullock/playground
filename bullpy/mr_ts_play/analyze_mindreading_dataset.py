#!/usr/bin/env python3
"""
Analyze the MindReading/Emotions dataset structure and compare with EU emotions experiment.
"""

import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set

def parse_filename(filename: str) -> dict:
    """
    Parse MindReading filename format: 0302501Y6Vgrateful.mov
    Format: [scenario_id][actor][instance][modality][emotion].mov
    - scenario_id: 7 digits
    - actor: 1-2 letters
    - instance: 1-2 digits
    - modality: V (visual/face) or T (textual/voice)
    - emotion: emotion name
    """
    base = filename.replace(".mov", "").replace(".MOV", "")
    
    # Extract modality and emotion (last part after V/T)
    match = re.search(r'([VT])([a-z]+(?:-[a-z]+)*)$', base)
    if not match:
        return None
    
    modality = match.group(1)
    emotion = match.group(2)
    prefix = base[:match.start()]
    
    # Extract scenario ID (first 7 digits)
    scenario_match = re.match(r'^(\d{7})', prefix)
    if not scenario_match:
        return None
    
    scenario_id = scenario_match.group(1)
    actor_part = prefix[7:]
    
    # Extract actor and instance number
    actor_match = re.match(r'^([A-Z]+)(\d+)', actor_part)
    if actor_match:
        actor = actor_match.group(1)
        instance_num = actor_match.group(2)
    else:
        actor = actor_part[0] if actor_part else "?"
        instance_num = actor_part[1:] if len(actor_part) > 1 else "?"
    
    return {
        "scenario_id": scenario_id,
        "actor": actor,
        "instance_num": instance_num,
        "modality": modality,
        "emotion": emotion,
        "filename": filename
    }

def analyze_mindreading_dataset(dataset_path: str):
    """Comprehensive analysis of MindReading/Emotions dataset."""
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        return
    
    print("=" * 80)
    print("MINREADING/EMOTIONS DATASET ANALYSIS")
    print("=" * 80)
    print(f"\nDataset path: {dataset_path}\n")
    
    # Find all .mov files
    print("Scanning for video files...")
    video_files = list(dataset_path.rglob("*.mov")) + list(dataset_path.rglob("*.MOV"))
    print(f"Found {len(video_files)} video files\n")
    
    # Parse all files
    parsed_files = []
    emotions = set()
    modalities = Counter()
    scenarios = set()
    actors = set()
    emotion_modality = defaultdict(lambda: {"V": 0, "T": 0})
    
    for video_file in video_files:
        parsed = parse_filename(video_file.name)
        if parsed:
            parsed_files.append(parsed)
            emotions.add(parsed["emotion"])
            modalities[parsed["modality"]] += 1
            scenarios.add(parsed["scenario_id"])
            actors.add(parsed["actor"])
            emotion_modality[parsed["emotion"]][parsed["modality"]] += 1
        else:
            # Check if it's a scenario file
            if "scen" in video_file.name.lower():
                print(f"  Note: Found scenario file: {video_file.name}")
    
    print("=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)
    print(f"Total video files: {len(video_files)}")
    print(f"Successfully parsed: {len(parsed_files)}")
    print(f"Unique emotions: {len(emotions)}")
    print(f"Unique scenarios: {len(scenarios)}")
    print(f"Unique actors: {len(actors)}")
    print(f"\nModality distribution:")
    for mod, count in modalities.items():
        print(f"  {mod} ({'Visual/Face' if mod == 'V' else 'Textual/Voice'}): {count}")
    
    print("\n" + "=" * 80)
    print("EMOTION LIST (sorted)")
    print("=" * 80)
    sorted_emotions = sorted(emotions)
    for i, emotion in enumerate(sorted_emotions, 1):
        v_count = emotion_modality[emotion]["V"]
        t_count = emotion_modality[emotion]["T"]
        print(f"{i:3d}. {emotion:20s} (V: {v_count:4d}, T: {t_count:4d}, Total: {v_count + t_count:4d})")
    
    print("\n" + "=" * 80)
    print("SAMPLE FILES")
    print("=" * 80)
    print("\nFirst 10 parsed files:")
    for i, parsed in enumerate(parsed_files[:10], 1):
        print(f"{i:2d}. {parsed['filename']}")
        print(f"     Emotion: {parsed['emotion']}, Modality: {parsed['modality']}, "
              f"Scenario: {parsed['scenario_id']}, Actor: {parsed['actor']}")
    
    # Check directory structure
    print("\n" + "=" * 80)
    print("DIRECTORY STRUCTURE")
    print("=" * 80)
    top_level = [d for d in dataset_path.iterdir() if d.is_dir()]
    print(f"\nTop-level directories: {len(top_level)}")
    for d in sorted(top_level)[:20]:
        file_count = len(list(d.rglob("*.mov"))) + len(list(d.rglob("*.MOV")))
        print(f"  {d.name}: {file_count} video files")
    if len(top_level) > 20:
        print(f"  ... and {len(top_level) - 20} more directories")
    
    # Check for special directories
    special_dirs = ["Audio", "scenarios", "Stories", "Rewards", "definitions"]
    print("\nSpecial directories:")
    for sd in special_dirs:
        sd_path = dataset_path / sd
        if sd_path.exists():
            if sd_path.is_dir():
                file_count = len(list(sd_path.rglob("*")))
                print(f"  {sd}/: {file_count} files")
            else:
                print(f"  {sd}: exists (not a directory)")
    
    return {
        "total_files": len(video_files),
        "parsed_files": len(parsed_files),
        "emotions": sorted_emotions,
        "emotion_count": len(emotions),
        "modalities": dict(modalities),
        "scenarios": len(scenarios),
        "actors": len(actors),
        "emotion_modality": dict(emotion_modality)
    }

if __name__ == "__main__":
    dataset_path = "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/MindReading/Emotions"
    results = analyze_mindreading_dataset(dataset_path)
    
    print("\n" + "=" * 80)
    print("COMPARISON WITH CURRENT EU EMOTIONS EXPERIMENT")
    print("=" * 80)
    print("\nThis dataset appears to use the SAME naming convention as the CAM/MindReading dataset!")
    print("The filename format matches exactly: [scenario_id][actor][instance][modality][emotion].mov")
    print("\nThis suggests this IS the MindReading/CAM dataset, not a separate EU-Emotion dataset.")
    print("The EU-Emotion experiment in this repo expects a different structure.")
    print("\nKey differences:")
    print("  - Current EU-Emotion experiment expects: emotions*/HD Version - Face, Body, Social/...")
    print("  - This dataset has: numbered directories (01-24) with scenario subdirectories")
    print("  - Both use same filename format, suggesting same source material")
