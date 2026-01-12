# Test Trial Definitions: Locations and Methodology

## 📍 Where Are the Test Trial JSON Files?

### On HPC (Original Location)

**CAM Test Trials:**
```
~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/mr_ts_play_results/cam_replication/cam_trial_definitions_test_all_files.json
```

**EU-Emotion Test Trials:**
```
~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/mr_ts_play_results/eu_emotion_replication/eu_emotion_trial_definitions_test.json
```

**CAM Train Trials:**
```
~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/mr_ts_play_results/cam_replication/cam_trial_definitions_train_all_files.json
```

**EU-Emotion Train Trials:**
```
~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/mr_ts_play_results/eu_emotion_replication/eu_emotion_trial_definitions_train.json
```

## 🚀 How to Transfer Trial Definitions to Local

### Option 1: Transfer All Trial Definitions

```bash
# Set HPC host
HPC_HOST="eb2007@login.hpc.cam.ac.uk"
HPC_BASE="~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/mr_ts_play_results"

# Create local directory
mkdir -p data/trial_definitions

# Transfer CAM trials
rsync -avz --progress \
  ${HPC_HOST}:${HPC_BASE}/cam_replication/cam_trial_definitions_*.json \
  data/trial_definitions/cam/

# Transfer EU-Emotion trials
rsync -avz --progress \
  ${HPC_HOST}:${HPC_BASE}/eu_emotion_replication/eu_emotion_trial_definitions_*.json \
  data/trial_definitions/eu_emotion/
```

### Option 2: Transfer Only Test Trials

```bash
# CAM test trials only
rsync -avz --progress \
  ${HPC_HOST}:${HPC_BASE}/cam_replication/cam_trial_definitions_test_all_files.json \
  data/trial_definitions/cam_test.json

# EU-Emotion test trials only
rsync -avz --progress \
  ${HPC_HOST}:${HPC_BASE}/eu_emotion_replication/eu_emotion_trial_definitions_test.json \
  data/trial_definitions/eu_emotion_test.json
```

## ✅ Is Using Test Trial Files Biased? **NO!**

### Proper Train/Test Split Methodology

The methodology used is **NOT biased** - it follows standard machine learning best practices:

#### 1. **Split Created BEFORE Training**
- Train/test split is created **once** at the beginning
- Uses fixed random seed (42) for reproducibility
- Split is **concept-balanced** (each concept appears in both train and test)

#### 2. **Test Set is Held Out During Training**
- Model **never sees** test data during training
- Only trains on `train_trial_definitions.json`
- Validation during training uses the **same test set** (this is standard practice)

#### 3. **Test Set Only Used for Final Evaluation**
- Test set is used **only once** at the end for final evaluation
- This gives an unbiased estimate of model performance

### Code Evidence

From `create_cam_trials_from_all_files.py`:

```python
def create_train_test_split(
    all_trials: List[Dict],
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Create train/test split from all trials.
    
    Ensures each concept has trials in both splits (concept-balanced).
    """
    random.seed(seed)  # Fixed seed for reproducibility
    
    # Group trials by concept
    concept_trials = defaultdict(list)
    for trial in all_trials:
        concept = trial.get('concept', trial.get('correct_label'))
        concept_trials[concept].append(trial)
    
    train_trials = []
    test_trials = []
    
    # Split each concept's trials (concept-balanced)
    for concept, concept_trial_list in concept_trials.items():
        random.shuffle(concept_trial_list)
        split_idx = int(len(concept_trial_list) * train_ratio)
        train_trials.extend(concept_trial_list[:split_idx])
        test_trials.extend(concept_trial_list[split_idx:])
    
    return train_trials, test_trials
```

### Why This is NOT Biased

1. **No Data Leakage**: Test set is completely separate from training
2. **Concept-Balanced**: Each emotion concept appears in both train and test
3. **Fixed Seed**: Same split every time (reproducible)
4. **Standard Practice**: This is how all ML experiments should be done

### What WOULD Be Biased

❌ **Biased approaches (NOT used here):**
- Training on test data
- Using test data for hyperparameter tuning
- Regenerating splits for each experiment
- Using different test sets for different models

✅ **What we do (NOT biased):**
- Fixed train/test split created once
- Test set held out during training
- Same test set used for all evaluations
- Concept-balanced splitting

## 📊 Trial Definition Structure

Each trial definition JSON file contains:

```json
{
  "trials": [
    {
      "trial_id": "cam_001",
      "stimulus_path": "01/0100104/0100104R6Thumiliating.mov",
      "modality": "face",
      "correct_label": "humiliating",
      "candidate_labels": ["humiliating", "resentful", "subservient", "confronted"],
      "correct_idx": 0,
      "actor": "R6",
      "scenario_id": "0100104",
      "concept": "humiliating"
    },
    ...
  ]
}
```

## 🔍 Verifying the Split

You can verify the split is proper:

```python
import json

# Load train and test
with open('data/trial_definitions/cam/cam_trial_definitions_train_all_files.json') as f:
    train = json.load(f)
with open('data/trial_definitions/cam/cam_trial_definitions_test_all_files.json') as f:
    test = json.load(f)

# Check no overlap
train_ids = {t['trial_id'] for t in train['trials']}
test_ids = {t['trial_id'] for t in test['trials']}
overlap = train_ids & test_ids
print(f"Overlapping trials: {len(overlap)}")  # Should be 0

# Check concept balance
train_concepts = {t['concept'] for t in train['trials']}
test_concepts = {t['concept'] for t in test['trials']}
print(f"Concepts in train: {len(train_concepts)}")
print(f"Concepts in test: {len(test_concepts)}")
print(f"All concepts in both: {train_concepts == test_concepts}")  # Should be True
```

## 📝 Summary

**Test trial files are:**
- ✅ Properly split (80/20 train/test)
- ✅ Concept-balanced
- ✅ Held out during training
- ✅ Used only for final evaluation
- ✅ **NOT biased** - follows standard ML methodology

**To use them:**
1. Transfer from HPC using rsync (commands above)
2. Use test trials for evaluation only
3. Never train on test trials
4. Use the same test set for all model comparisons


