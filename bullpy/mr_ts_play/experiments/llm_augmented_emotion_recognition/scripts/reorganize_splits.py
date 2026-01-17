#!/usr/bin/env python3
"""
Reorganize data splits for scientifically rigorous evaluation.

This script:
1. Renames current test set to validation set (for optimization)
2. Uses existing val set as final test set (for evaluation)
3. Creates backup of original files
4. Documents the reorganization
"""

import json
import shutil
from pathlib import Path
from datetime import datetime


def reorganize_splits():
    """Reorganize splits for proper ML evaluation."""
    
    base_dir = Path("data/trial_definitions")
    
    # Original files
    original_test = base_dir / "eu_emotion_test.json"
    original_val = base_dir / "eu_emotion_val.json"
    
    # Backup directory
    backup_dir = base_dir / "backup_before_reorganization"
    backup_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("="*80)
    print("REORGANIZING DATA SPLITS FOR SCIENTIFIC RIGOR")
    print("="*80)
    print()
    
    # Step 1: Create backups
    print("Step 1: Creating backups...")
    if original_test.exists():
        backup_test = backup_dir / f"eu_emotion_test_original_{timestamp}.json"
        shutil.copy2(original_test, backup_test)
        print(f"  ✓ Backed up test set to: {backup_test}")
    
    if original_val.exists():
        backup_val = backup_dir / f"eu_emotion_val_original_{timestamp}.json"
        shutil.copy2(original_val, backup_val)
        print(f"  ✓ Backed up val set to: {backup_val}")
    
    print()
    
    # Step 2: Load data
    print("Step 2: Loading data...")
    with open(original_test) as f:
        test_data = json.load(f)
        test_trials = test_data.get('trials', [])
    
    with open(original_val) as f:
        val_data = json.load(f)
        val_trials = val_data.get('trials', [])
    
    print(f"  ✓ Loaded {len(test_trials)} trials from test set")
    print(f"  ✓ Loaded {len(val_trials)} trials from val set")
    print()
    
    # Step 3: Reorganize
    print("Step 3: Reorganizing splits...")
    print()
    print("  NEW STRUCTURE:")
    print("  - Validation set (for optimization): Current test set (54 trials)")
    print("  - Test set (for final evaluation): Current val set (118 trials)")
    print()
    
    # Create new validation set (from old test set)
    new_val_file = base_dir / "eu_emotion_val_for_optimization.json"
    val_metadata = {
        "description": "Validation set for prompt optimization and hyperparameter tuning",
        "source": "Original test set (eu_emotion_test.json)",
        "reorganized_date": timestamp,
        "purpose": "Use this set for systematic prompt optimization. Do NOT use for final evaluation.",
        "num_trials": len(test_trials),
        "trials": test_trials
    }
    
    with open(new_val_file, 'w') as f:
        json.dump(val_metadata, f, indent=2)
    
    print(f"  ✓ Created validation set: {new_val_file}")
    print(f"    Purpose: Prompt optimization")
    print(f"    Trials: {len(test_trials)}")
    
    # Create new test set (from old val set)
    new_test_file = base_dir / "eu_emotion_test_final.json"
    test_metadata = {
        "description": "Final test set for one-time evaluation only",
        "source": "Original val set (eu_emotion_val.json)",
        "reorganized_date": timestamp,
        "purpose": "Use this set ONCE for final evaluation after optimization. Do NOT use for optimization.",
        "num_trials": len(val_trials),
        "trials": val_trials
    }
    
    with open(new_test_file, 'w') as f:
        json.dump(test_metadata, f, indent=2)
    
    print(f"  ✓ Created test set: {new_test_file}")
    print(f"    Purpose: Final evaluation (use ONCE)")
    print(f"    Trials: {len(val_trials)}")
    print()
    
    # Step 4: Create documentation
    doc_file = base_dir / "SPLIT_REORGANIZATION.md"
    doc_content = f"""# Data Split Reorganization

## Date: {timestamp}

## Purpose

Reorganized data splits to follow proper machine learning evaluation practices:
- **Validation set**: Used for prompt optimization and hyperparameter tuning
- **Test set**: Used ONCE for final evaluation only

## Changes Made

### Before
- `eu_emotion_test.json`: 54 trials (was used for optimization ❌)
- `eu_emotion_val.json`: 118 trials

### After
- `eu_emotion_val_for_optimization.json`: 54 trials (use for optimization ✅)
- `eu_emotion_test_final.json`: 118 trials (use for final evaluation ✅)

## Usage Protocol

### For Optimization (Validation Set)
```python
# Use this for trying different prompts
validation_trials = "data/trial_definitions/eu_emotion_val_for_optimization.json"

# Try prompt variations:
# - Baseline prompt
# - Enhanced prompts with explicit distinctions
# - Few-shot examples
# - etc.

# Select best prompt based on validation accuracy
```

### For Final Evaluation (Test Set)
```python
# Use this ONCE after optimization is complete
test_trials = "data/trial_definitions/eu_emotion_test_final.json"

# Run best prompt on test set
# Report this as final result
# Do NOT modify based on test results
```

## Scientific Rigor

This reorganization ensures:
1. ✅ No test set optimization (test set held out)
2. ✅ Proper validation-based optimization
3. ✅ Honest performance estimates
4. ✅ Publishable methodology

## Backups

Original files backed up to:
- `backup_before_reorganization/eu_emotion_test_original_{timestamp}.json`
- `backup_before_reorganization/eu_emotion_val_original_{timestamp}.json`
"""
    
    with open(doc_file, 'w') as f:
        f.write(doc_content)
    
    print(f"  ✓ Created documentation: {doc_file}")
    print()
    
    print("="*80)
    print("REORGANIZATION COMPLETE")
    print("="*80)
    print()
    print("Next steps:")
    print("1. Use 'eu_emotion_val_for_optimization.json' for prompt optimization")
    print("2. Use 'eu_emotion_test_final.json' for final evaluation (ONCE)")
    print("3. See SPLIT_REORGANIZATION.md for details")
    print()


if __name__ == "__main__":
    reorganize_splits()
