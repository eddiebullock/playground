#!/usr/bin/env python3
"""
Create and save emotion mapping from fine-grained to basic emotions.
Analyzes the dataset and creates a mapping file.
"""

import sys
from pathlib import Path
import pandas as pd
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.emotion_mapping import create_emotion_mapping, map_emotion_to_basic, BASIC_EMOTIONS


def analyze_dataset_and_create_mapping(data_root: str, output_file: str = "data/basic_emotion_mapping.json"):
    """
    Analyze dataset and create mapping from fine-grained to basic emotions.
    """
    import re
    from collections import Counter
    
    data_root = Path(data_root)
    emotions = set()
    
    print("Analyzing dataset to find all emotions...")
    
    # Collect all unique emotions
    for emotion_folder in sorted(data_root.glob("[0-9][0-9]")):
        if not emotion_folder.is_dir():
            continue
        for scenario_folder in sorted(emotion_folder.glob("[0-9]*")):
            if not scenario_folder.is_dir():
                continue
            for video_file in scenario_folder.glob("*.mov"):
                base = video_file.name.replace(".mov", "")
                match = re.search(r'([VT])([a-z]+(?:-[a-z]+)*)$', base)
                if match:
                    emotion = match.group(2)
                    emotions.add(emotion)
    
    print(f"Found {len(emotions)} unique emotions")
    
    # Create mapping
    keyword_mapping = create_emotion_mapping()
    emotion_to_basic = {}
    emotion_counts = Counter()
    
    for emotion in sorted(emotions):
        basic = map_emotion_to_basic(emotion, keyword_mapping)
        emotion_to_basic[emotion] = basic
        emotion_counts[basic] += 1
    
    # Save mapping
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(emotion_to_basic, f, indent=2)
    
    print(f"\nMapping saved to {output_path}")
    print(f"\nBasic emotion distribution:")
    for emotion, count in emotion_counts.most_common():
        print(f"  {emotion:15s}: {count:4d} fine-grained emotions")
    
    # Show some examples
    print(f"\nExample mappings:")
    for i, (fine, basic) in enumerate(list(emotion_to_basic.items())[:20]):
        print(f"  {fine:20s} -> {basic}")
    
    return emotion_to_basic


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create basic emotion mapping")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/mindreading_transporter_files/Mindreading emotions library/Emotions",
        help="Root directory of the dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/basic_emotion_mapping.json",
        help="Output file for mapping"
    )
    
    args = parser.parse_args()
    analyze_dataset_and_create_mapping(args.data_root, args.output)

