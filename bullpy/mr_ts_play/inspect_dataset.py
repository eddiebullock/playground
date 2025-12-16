#!/usr/bin/env python3
"""
Dataset inspection script for Mindreading/CAM dataset.
Analyzes structure, labels, and prepares validation report.
"""

import os
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set

DATASET_ROOT = Path(
    "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/mindreading_transporter_files/Mindreading emotions library/Emotions"
)


def parse_filename(filename: str) -> Dict[str, str]:
    """
    Parse CAM video filename.
    Format: {scenario_id}{actor}{number}{V|T}{emotion}.mov
    Example: 0100104M1Vhumiliating.mov
    """
    base = filename.replace(".mov", "")
    
    # Extract emotion (last part after V/T, may contain hyphens)
    match = re.search(r'([VT])([a-z]+(?:-[a-z]+)*)$', base)
    if match:
        modality = match.group(1)  # V or T
        emotion = match.group(2)
        prefix = base[:match.start()]
    else:
        return {"error": f"Could not parse: {filename}"}
    
    # Extract scenario ID (first 7 digits)
    scenario_match = re.match(r'^(\d{7})', prefix)
    if scenario_match:
        scenario_id = scenario_match.group(1)
        actor_part = prefix[7:]
    else:
        return {"error": f"Could not extract scenario ID: {filename}"}
    
    # Extract actor and number
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
        "modality": modality,  # V = Visual, T = Textual?
        "emotion": emotion,
        "filename": filename
    }


def inspect_dataset() -> Dict:
    """Comprehensive dataset inspection."""
    results = {
        "total_videos": 0,
        "scenarios": set(),
        "emotions": set(),
        "actors": set(),
        "modalities": set(),
        "scenario_to_emotion": defaultdict(set),
        "emotion_counts": Counter(),
        "actor_counts": Counter(),
        "modality_counts": Counter(),
        "scenario_emotion_map": defaultdict(str),
        "errors": [],
        "sample_files": []
    }
    
    # Walk through emotion folders (01-24)
    for emotion_folder in sorted(DATASET_ROOT.glob("[0-9][0-9]")):
        if not emotion_folder.is_dir():
            continue
        
        emotion_num = emotion_folder.name
        
        # Walk through scenario folders
        for scenario_folder in sorted(emotion_folder.glob("[0-9]*")):
            if not scenario_folder.is_dir():
                continue
            
            scenario_id = scenario_folder.name
            results["scenarios"].add(scenario_id)
            
            # Process video files
            for video_file in scenario_folder.glob("*.mov"):
                results["total_videos"] += 1
                
                parsed = parse_filename(video_file.name)
                
                if "error" in parsed:
                    results["errors"].append(parsed["error"])
                    continue
                
                results["emotions"].add(parsed["emotion"])
                results["actors"].add(parsed["actor"])
                results["modalities"].add(parsed["modality"])
                results["emotion_counts"][parsed["emotion"]] += 1
                results["actor_counts"][parsed["actor"]] += 1
                results["modality_counts"][parsed["modality"]] += 1
                results["scenario_to_emotion"][scenario_id].add(parsed["emotion"])
                
                # Store scenario->emotion mapping (should be 1:1)
                if scenario_id not in results["scenario_emotion_map"]:
                    results["scenario_emotion_map"][scenario_id] = parsed["emotion"]
                
                # Collect samples
                if len(results["sample_files"]) < 10:
                    results["sample_files"].append({
                        "path": str(video_file.relative_to(DATASET_ROOT)),
                        "parsed": parsed
                    })
    
    return results


def generate_report(results: Dict) -> str:
    """Generate human-readable inspection report."""
    report = []
    report.append("=" * 80)
    report.append("MINREADING/CAM DATASET INSPECTION REPORT")
    report.append("=" * 80)
    report.append("")
    
    report.append("DATASET OVERVIEW")
    report.append("-" * 80)
    report.append(f"Total video files: {results['total_videos']}")
    report.append(f"Total scenarios: {len(results['scenarios'])}")
    report.append(f"Unique emotions: {len(results['emotions'])}")
    report.append(f"Unique actors: {len(results['actors'])}")
    report.append(f"Modalities: {', '.join(sorted(results['modalities']))}")
    report.append("")
    
    report.append("EMOTION DISTRIBUTION (Top 20)")
    report.append("-" * 80)
    for emotion, count in results['emotion_counts'].most_common(20):
        report.append(f"  {emotion:30s} {count:4d} videos")
    report.append("")
    
    report.append("ACTOR DISTRIBUTION")
    report.append("-" * 80)
    for actor, count in sorted(results['actor_counts'].items()):
        report.append(f"  {actor:10s} {count:4d} videos")
    report.append("")
    
    report.append("MODALITY DISTRIBUTION")
    report.append("-" * 80)
    for modality, count in sorted(results['modality_counts'].items()):
        mod_name = "Visual" if modality == "V" else "Textual" if modality == "T" else modality
        report.append(f"  {mod_name:10s} {count:4d} videos")
    report.append("")
    
    # Check scenario-emotion alignment
    report.append("SCENARIO-EMOTION ALIGNMENT")
    report.append("-" * 80)
    multi_emotion_scenarios = []
    for scenario_id, emotions in results['scenario_to_emotion'].items():
        if len(emotions) > 1:
            multi_emotion_scenarios.append((scenario_id, emotions))
    
    if multi_emotion_scenarios:
        report.append(f"WARNING: {len(multi_emotion_scenarios)} scenarios have multiple emotions:")
        for scenario_id, emotions in multi_emotion_scenarios[:10]:
            report.append(f"  {scenario_id}: {emotions}")
    else:
        report.append("✓ All scenarios map to single emotion (1:1 mapping)")
    report.append("")
    
    # Class balance
    report.append("CLASS BALANCE ANALYSIS")
    report.append("-" * 80)
    counts = list(results['emotion_counts'].values())
    if counts:
        min_count = min(counts)
        max_count = max(counts)
        avg_count = sum(counts) / len(counts)
        report.append(f"Min videos per emotion: {min_count}")
        report.append(f"Max videos per emotion: {max_count}")
        report.append(f"Avg videos per emotion: {avg_count:.1f}")
        report.append(f"Balance ratio (min/max): {min_count/max_count:.3f}")
    report.append("")
    
    report.append("SAMPLE FILES")
    report.append("-" * 80)
    for sample in results['sample_files'][:5]:
        report.append(f"  {sample['path']}")
        report.append(f"    Scenario: {sample['parsed']['scenario_id']}")
        report.append(f"    Actor: {sample['parsed']['actor']}")
        report.append(f"    Emotion: {sample['parsed']['emotion']}")
        report.append(f"    Modality: {sample['parsed']['modality']}")
    report.append("")
    
    if results['errors']:
        report.append("PARSING ERRORS")
        report.append("-" * 80)
        for error in results['errors'][:10]:
            report.append(f"  {error}")
        report.append("")
    
    report.append("=" * 80)
    return "\n".join(report)


if __name__ == "__main__":
    print("Inspecting dataset...")
    results = inspect_dataset()
    report = generate_report(results)
    print(report)
    
    # Save report
    output_path = Path(__file__).parent / "dataset_inspection_report.txt"
    with open(output_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {output_path}")

