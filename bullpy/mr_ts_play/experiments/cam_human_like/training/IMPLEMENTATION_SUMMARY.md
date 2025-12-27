# Implementation Summary: Dual Replication System

## Overview

This implementation provides a complete system for computationally replicating the Golan/CAM study on both **EU-Emotion** and **CAM** datasets separately, following the exact methodology with 4-option forced-choice trials.

## Components Created

### 1. EU-Emotion Trial Generator
**File**: `create_eu_emotion_trials.py`

- Discovers all EU-Emotion emotions from dataset structure
- Generates forced-choice trials following Golan methodology:
  - 5 trials per emotion
  - Counterbalanced face/voice distribution (3+2 or 2+3)
  - 4 options per trial (1 target + 3 foils)
  - Foils selected from different emotion groups
- Creates train/test splits (80/20)
- Outputs JSON trial definitions compatible with CAM format

**Usage**:
```bash
python experiments/cam_human_like/training/create_eu_emotion_trials.py \
    --eu-emotion-dir /path/to/EU_emotions \
    --output-dir data \
    --modality face \
    --trials-per-emotion 5
```

### 2. CAM Train/Test Split Creator
**File**: `create_cam_splits.py`

- Loads existing CAM trial definitions
- Creates train/test splits with multiple methods:
  - `concept_balanced`: Ensures each concept has trials in both splits
  - `actor_independent`: No actor overlap between splits
  - `random`: Simple random split
- Saves split definitions to JSON files

**Usage**:
```bash
python experiments/cam_human_like/training/create_cam_splits.py \
    --trial-definitions data/cam_trial_definitions_20concepts.json \
    --output-dir data/cam_splits \
    --split-method concept_balanced \
    --train-ratio 0.8
```

### 3. Task-Specific Dataset
**File**: `task_specific_dataset.py`

- Loads trials with 4 candidate labels
- Extracts multiple frames from videos (multi-frame architecture)
- Returns: (frames, candidate_labels, correct_idx)
- Handles both absolute and relative paths

### 4. Enhanced Fine-Tuning Pipeline
**File**: `finetune_clip_emotions.py` (updated)

**New Features**:
- **Task-specific training**: `finetune_clip_task_specific()` function
  - Cross-entropy loss over 4 candidate labels (not just contrastive learning)
  - Multi-frame processing: extracts 8 frames, averages features
  - Supports both CAM and EU-Emotion datasets
  
- **Multi-frame architecture**:
  - Extracts `num_frames` frames uniformly from video
  - Processes all frames through CLIP
  - Averages frame features before computing similarity
  
- **Task-specific loss**:
  - For each video, computes similarity with 4 candidate labels
  - Uses cross-entropy loss over the 4 options
  - Matches the forced-choice evaluation format

**Usage (Task-Specific)**:
```bash
python experiments/cam_human_like/training/finetune_clip_emotions.py \
    --task_specific \
    --dataset_type cam \
    --train_trials data/cam_splits/train_trials.json \
    --val_trials data/cam_splits/test_trials.json \
    --data_root /path/to/cam/stimuli \
    --output_dir models/clip_cam_finetuned \
    --num_epochs 2 \
    --batch_size 8 \
    --num_frames 8
```

### 5. Enhanced Evaluation Pipeline
**File**: `evaluate_on_cam.py` (updated)

- Supports both CAM and EU-Emotion evaluations
- Uses forced-choice format: scores 4 candidate labels, selects highest
- Computes comprehensive metrics:
  - Overall accuracy
  - Face/voice accuracy
  - Per-emotion/concept accuracy
  - Confusion matrices

**Usage**:
```bash
python experiments/cam_human_like/training/evaluate_on_cam.py \
    --model_path models/clip_cam_finetuned/best_model \
    --trial_definitions data/cam_splits/test_trials.json \
    --data_root /path/to/cam/stimuli \
    --dataset_type cam \
    --num_frames 8 \
    --use_multiframe
```

### 6. Orchestration Script
**File**: `run_dual_replication.sh`

- Automates the entire dual replication process
- Runs EU-Emotion replication (generate trials → fine-tune → evaluate)
- Runs CAM replication (create splits → fine-tune → evaluate)
- Generates comparison report

**Usage**:
```bash
./experiments/cam_human_like/training/run_dual_replication.sh
```

## Architecture Details

### Multi-Frame Processing

1. **Frame Extraction**: Extracts `num_frames` (default 8) frames uniformly from video
2. **Feature Extraction**: Each frame is processed through CLIP image encoder
3. **Aggregation**: Frame features are averaged (mean pooling)
4. **Similarity Computation**: Aggregated video features are compared with text features

### Task-Specific Training

1. **Input**: Video + 4 candidate labels (1 correct + 3 foils)
2. **Processing**: 
   - Extract frames from video
   - Encode frames → image features
   - Encode candidate labels → text features
   - Average frame features → video features
3. **Loss**: Cross-entropy over 4 options (task-specific)
4. **Output**: Model learns to select correct label from 4 options

## File Structure

```
experiments/cam_human_like/training/
├── create_eu_emotion_trials.py      # EU-Emotion trial generator
├── create_cam_splits.py             # CAM split creator
├── task_specific_dataset.py         # Task-specific dataset loader
├── finetune_clip_emotions.py        # Enhanced fine-tuning (updated)
├── evaluate_on_cam.py               # Enhanced evaluation (updated)
└── run_dual_replication.sh          # Orchestration script

data/
├── eu_emotion_trial_definitions_train.json  # Generated
├── eu_emotion_trial_definitions_test.json   # Generated
└── cam_splits/
    ├── train_trials.json            # Generated
    └── test_trials.json             # Generated

results/
├── eu_emotion_replication/
│   ├── model_checkpoints/
│   └── eu_emotion_trial_definitions_*.json
├── cam_replication/
│   ├── model_checkpoints/
│   └── train_trials.json, test_trials.json
└── comparison_report.md
```

## Key Features

✅ **Golan Methodology**: Exact replication with 4-option forced-choice
✅ **Multi-Frame Architecture**: Processes multiple frames, averages features
✅ **Task-Specific Training**: Cross-entropy loss over 4 candidate labels
✅ **Dual Replication**: Separate replications for EU-Emotion and CAM
✅ **Comprehensive Evaluation**: Detailed metrics and comparison reports
✅ **Ready for HPC**: Easy to scale epochs and batch size

## Next Steps

1. **Run Local Test**: Execute `run_dual_replication.sh` to test pipeline (2 epochs)
2. **Review Results**: Check comparison report and individual results
3. **Scale to HPC**: Increase epochs to 10-20 for better performance
4. **Analysis**: Analyze per-emotion/concept performance breakdowns

## Notes

- **Device Detection**: Scripts auto-detect MPS/CUDA/CPU
- **Error Handling**: Gracefully handles missing/corrupted videos
- **Reproducibility**: All random operations use seed=42
- **Path Handling**: Supports both absolute and relative paths





