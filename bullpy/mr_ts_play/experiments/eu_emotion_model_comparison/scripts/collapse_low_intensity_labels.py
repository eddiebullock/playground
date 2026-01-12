#!/usr/bin/env python3
"""
Collapse "low intensity" emotion labels to their base emotions.

This script creates a modified version of the trial definitions where:
- "afraid low intensity" → "afraid"
- "angry low intensity" → "angry"
- etc.

This can help test if the "low intensity" modifier is confusing models.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

def collapse_low_intensity(emotion: str) -> str:
    """Remove 'low intensity' suffix from emotion label."""
    if " low intensity" in emotion:
        return emotion.replace(" low intensity", "").strip()
    return emotion

def process_trials(trials: List[Dict]) -> List[Dict]:
    """Process trials to collapse low intensity labels."""
    processed = []
    
    for trial in trials:
        # Collapse correct label
        original_correct = trial.get('correct_label', trial.get('emotion', ''))
        collapsed_correct = collapse_low_intensity(original_correct)
        
        # Collapse candidate labels
        original_candidates = trial.get('candidate_labels', [])
        collapsed_candidates = [collapse_low_intensity(c) for c in original_candidates]
        
        # Find new correct index
        try:
            collapsed_idx = collapsed_candidates.index(collapsed_correct)
        except ValueError:
            # If collapsed label not in candidates, keep original index
            collapsed_idx = trial.get('correct_idx', 0)
        
        # Create new trial
        new_trial = trial.copy()
        new_trial['correct_label'] = collapsed_correct
        new_trial['emotion'] = collapsed_correct
        new_trial['candidate_labels'] = collapsed_candidates
        new_trial['correct_idx'] = collapsed_idx
        
        # Add metadata about original label
        if original_correct != collapsed_correct:
            new_trial['original_label'] = original_correct
            new_trial['was_low_intensity'] = True
        else:
            new_trial['was_low_intensity'] = False
        
        processed.append(new_trial)
    
    return processed

def main():
    parser = argparse.ArgumentParser(
        description="Collapse low intensity emotion labels to base emotions"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input trial definitions JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output trial definitions JSON file'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Print statistics about label changes'
    )
    
    args = parser.parse_args()
    
    # Load input
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    trials = data.get('trials', data)
    
    # Process trials
    processed_trials = process_trials(trials)
    
    # Statistics
    if args.stats:
        low_intensity_count = sum(1 for t in processed_trials if t.get('was_low_intensity', False))
        unique_emotions = set(t['correct_label'] for t in processed_trials)
        
        print(f"Original trials: {len(trials)}")
        print(f"Low intensity labels collapsed: {low_intensity_count}")
        print(f"Unique emotions after collapse: {len(unique_emotions)}")
        print(f"Emotions: {sorted(unique_emotions)}")
    
    # Save output
    output_data = {'trials': processed_trials}
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Processed trials saved to {args.output}")

if __name__ == '__main__':
    main()
