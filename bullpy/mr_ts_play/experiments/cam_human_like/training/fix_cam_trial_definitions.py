#!/usr/bin/env python3
"""
Fix CAM trial definitions by replacing corrupted T files with valid V files.

This script:
1. Identifies voice trials with corrupted T files (< 50KB)
2. Finds valid V files (video files with audio) in the same directory
3. Updates trial definitions to use valid files
4. Preserves actor information where possible
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional


def find_valid_v_file(
    data_root: Path,
    scenario_id: str,
    emotion: str,
    preferred_actor: Optional[str] = None
) -> Optional[str]:
    """
    Find a valid V file (video file) for a given scenario and emotion.
    
    Args:
        data_root: Root directory of CAM dataset
        scenario_id: Scenario ID (e.g., "1202701")
        emotion: Emotion name (e.g., "confronted")
        preferred_actor: Preferred actor code (e.g., "P", "M")
    
    Returns:
        Relative path to valid V file, or None if not found
    """
    # Find scenario directory (format: XX/scenario_id/)
    scenario_dir = None
    for subdir in data_root.iterdir():
        if subdir.is_dir() and (subdir / scenario_id).exists():
            scenario_dir = subdir / scenario_id
            break
    
    if not scenario_dir or not scenario_dir.exists():
        return None
    
    # Find all V files for this emotion
    pattern = f"*V{emotion}.mov"
    v_files = list(scenario_dir.glob(pattern))
    
    if not v_files:
        return None
    
    # Filter by size (> 50KB)
    valid_files = [f for f in v_files if f.stat().st_size > 50 * 1024]
    
    if not valid_files:
        return None
    
    # If preferred actor specified, try to find matching file
    if preferred_actor:
        for f in valid_files:
            # Extract actor from filename: {scenario_id}{actor}V{emotion}.mov
            parts = f.stem.replace(scenario_id, '').split('V')
            if len(parts) > 0 and preferred_actor in parts[0]:
                rel_path = f.relative_to(data_root)
                return str(rel_path)
    
    # Use first valid file
    rel_path = valid_files[0].relative_to(data_root)
    return str(rel_path)


def extract_emotion_from_path(stimulus_path: str) -> tuple:
    """
    Extract scenario_id, actor, and emotion from stimulus path.
    
    Args:
        stimulus_path: Path like "12/1202701/1202701P6Tconfronted.mov"
    
    Returns:
        (scenario_id, actor, emotion)
    """
    parts = stimulus_path.split('/')
    if len(parts) < 3:
        return None, None, None
    
    scenario_id = parts[1]  # e.g., "1202701"
    filename = parts[2]  # e.g., "1202701P6Tconfronted.mov"
    
    # Remove scenario_id and extension
    name_part = filename.replace(scenario_id, '').replace('.mov', '')
    
    # Extract actor and emotion
    # Format: {actor}{number}T{emotion} or {actor}{number}V{emotion}
    if 'T' in name_part:
        actor_part, emotion = name_part.split('T', 1)
        actor = actor_part[0] if actor_part else None
    elif 'V' in name_part:
        actor_part, emotion = name_part.split('V', 1)
        actor = actor_part[0] if actor_part else None
    else:
        actor = None
        emotion = None
    
    return scenario_id, actor, emotion


def fix_trial_definitions(
    trial_definitions_file: str,
    data_root: str,
    output_file: Optional[str] = None
):
    """
    Fix trial definitions by replacing corrupted T files with valid V files.
    
    Args:
        trial_definitions_file: Path to trial definitions JSON
        data_root: Root directory of CAM dataset
        output_file: Output file path (default: overwrite input)
    """
    data_root = Path(data_root)
    
    # Load trial definitions
    with open(trial_definitions_file, 'r') as f:
        data = json.load(f)
    
    fixed_count = 0
    not_found_count = 0
    
    print("Fixing corrupted voice trial files...\n")
    
    for trial in data['trials']:
        # Only process voice trials
        if trial['modality'] != 'voice':
            continue
        
        stimulus_path = trial['stimulus_path']
        file_path = data_root / stimulus_path
        
        # Check if file is corrupted (< 50KB)
        if file_path.exists() and file_path.stat().st_size < 50 * 1024:
            print(f"Found corrupted file: {trial['trial_id']}")
            print(f"  Current: {stimulus_path} ({file_path.stat().st_size / 1024:.1f} KB)")
            
            # Extract scenario and emotion
            scenario_id, actor, emotion = extract_emotion_from_path(stimulus_path)
            
            if not scenario_id or not emotion:
                print(f"  ⚠️  Could not parse path, skipping")
                continue
            
            # Find valid V file
            valid_file = find_valid_v_file(data_root, scenario_id, emotion, actor)
            
            if valid_file:
                old_path = trial['stimulus_path']
                trial['stimulus_path'] = valid_file
                print(f"  ✅ Fixed: {valid_file}")
                
                # Update actor if it changed
                new_scenario_id, new_actor, _ = extract_emotion_from_path(valid_file)
                if new_actor and new_actor != actor:
                    print(f"  ⚠️  Actor changed: {actor} → {new_actor}")
                    trial['actor'] = new_actor
                
                fixed_count += 1
            else:
                print(f"  ❌ No valid V file found for {scenario_id}/{emotion}")
                not_found_count += 1
        
        print()
    
    # Save updated trial definitions
    output_path = output_file or trial_definitions_file
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print("=" * 60)
    print(f"Fixed {fixed_count} trials")
    if not_found_count > 0:
        print(f"Could not fix {not_found_count} trials")
    print(f"Updated trial definitions saved to: {output_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Fix CAM trial definitions by replacing corrupted files'
    )
    parser.add_argument(
        'trial_definitions_file',
        help='Path to trial definitions JSON file'
    )
    parser.add_argument(
        'data_root',
        help='Root directory of CAM dataset'
    )
    parser.add_argument(
        '--output',
        help='Output file path (default: overwrite input)'
    )
    
    args = parser.parse_args()
    
    fix_trial_definitions(
        args.trial_definitions_file,
        args.data_root,
        args.output
    )



