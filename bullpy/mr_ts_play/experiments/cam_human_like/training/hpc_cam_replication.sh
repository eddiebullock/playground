#!/bin/bash
# HPC Script: CAM Replication Study
# Runs CAM fine-tuning and evaluation on HPC

set -e

# HPC Configuration
# Data locations are separate:
# - CAM data: /home/eb2007/data/CAM (already on HPC)
# - EU emotions: RDS (transferring separately)

# RDS path for EU emotions (project 90416)
# Try different RDS path formats
if [ -d "${HOME}/rds/rds-autism-research-ePtR33Nsgi4/data" ]; then
    RDS_DATA_DIR="${HOME}/rds/rds-autism-research-ePtR33Nsgi4/data"
elif [ -d "/rds/user/eb2007/rds-autism-research-ePtR33Nsgi4/data" ]; then
    RDS_DATA_DIR="/rds/user/eb2007/rds-autism-research-ePtR33Nsgi4/data"
elif [ -d "/rds-d7/project/45718/users/eb2007/data" ]; then
    RDS_DATA_DIR="/rds-d7/project/45718/users/eb2007/data"
else
    RDS_DATA_DIR="${HOME}/rds/rds-autism-research-ePtR33Nsgi4/data"  # Default, will create if needed
fi

# CAM data location - FIXED: Always in /home (separate from EU emotions)
CAM_DATA_ROOT="/home/eb2007/data/CAM"
if [ ! -d "$CAM_DATA_ROOT" ]; then
    echo "❌ Error: CAM data not found at $CAM_DATA_ROOT"
    echo "   CAM data should be at: /home/eb2007/data/CAM"
    exit 1
fi
echo "✅ CAM data location: $CAM_DATA_ROOT"

# EU Emotions data location - On RDS (separate from CAM)
if [ -d "${RDS_DATA_DIR}/EU_emotions" ]; then
    EU_EMOTIONS_DATA_ROOT="${RDS_DATA_DIR}/EU_emotions"
    echo "✅ EU Emotions data on RDS: $EU_EMOTIONS_DATA_ROOT"
else
    EU_EMOTIONS_DATA_ROOT="${RDS_DATA_DIR}/EU_emotions"
    echo "⚠️  EU Emotions data location: $EU_EMOTIONS_DATA_ROOT (may still be transferring)"
fi

CAM_TRIAL_DEFINITIONS="data/cam_trial_definitions_20concepts.json"
OUTPUT_BASE="results"
NUM_EPOCHS=10  # More epochs for HPC
BATCH_SIZE=16  # Larger batch size for HPC GPU (optimized for 16GB RAM)
LEARNING_RATE=1e-5
NUM_FRAMES=8
DEVICE="cuda"  # HPC uses CUDA

# Auto-detect GPU availability
if ! python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "Warning: CUDA not available, falling back to CPU"
    DEVICE="cpu"
    BATCH_SIZE=8  # Reduce batch size for CPU
fi

# Project root (adjust if needed)
PROJECT_ROOT="${HOME}/mr_ts_play"
cd "$PROJECT_ROOT" || { echo "Error: Could not cd to $PROJECT_ROOT"; exit 1; }

# Detect Python (use module or conda environment)
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

echo "============================================================"
echo "CAM Replication Study on HPC"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  CAM data root: $CAM_DATA_ROOT"
echo "  CAM trial definitions: $CAM_TRIAL_DEFINITIONS"
echo "  Output directory: $OUTPUT_BASE"
echo "  Device: $DEVICE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Num frames: $NUM_FRAMES"
echo "  Project root: $PROJECT_ROOT"
echo ""

# Create output directories
mkdir -p "$OUTPUT_BASE/cam_replication"

# ============================================================
# Step 1: Create CAM train/test splits
# ============================================================
echo "============================================================"
echo "Step 1: Creating CAM train/test splits..."
echo "============================================================"
echo ""

$PYTHON_CMD experiments/cam_human_like/training/create_cam_splits.py \
    --trial-definitions "$CAM_TRIAL_DEFINITIONS" \
    --output-dir "$OUTPUT_BASE/cam_replication" \
    --split-method concept_balanced \
    --train-ratio 0.8 \
    --seed 42

CAM_TRAIN_TRIALS="$OUTPUT_BASE/cam_replication/train_trials.json"
CAM_TEST_TRIALS="$OUTPUT_BASE/cam_replication/test_trials.json"

if [ ! -f "$CAM_TRAIN_TRIALS" ] || [ ! -f "$CAM_TEST_TRIALS" ]; then
    echo "Error: Failed to create CAM splits"
    exit 1
fi

echo "CAM splits created successfully"
echo "  Train: $CAM_TRAIN_TRIALS"
echo "  Test: $CAM_TEST_TRIALS"
echo ""

# ============================================================
# Step 2: Fine-tune on CAM
# ============================================================
echo "============================================================"
echo "Step 2: Fine-tuning CLIP on CAM"
echo "============================================================"
echo "This will take approximately 1-3 hours for 10 epochs on GPU..."
echo ""

$PYTHON_CMD experiments/cam_human_like/training/finetune_clip_emotions.py \
    --task_specific \
    --dataset_type cam \
    --train_trials "$CAM_TRAIN_TRIALS" \
    --val_trials "$CAM_TEST_TRIALS" \
    --data_root "$CAM_DATA_ROOT" \
    --output_dir "$OUTPUT_BASE/cam_replication/model_checkpoints" \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --device $DEVICE \
    --num_frames $NUM_FRAMES

CAM_MODEL_PATH="$OUTPUT_BASE/cam_replication/model_checkpoints/best_model"

if [ ! -d "$CAM_MODEL_PATH" ]; then
    echo "Error: CAM fine-tuning failed"
    exit 1
fi

echo "CAM fine-tuning complete!"
echo ""

# ============================================================
# Step 3: Evaluate on CAM test set
# ============================================================
echo "============================================================"
echo "Step 3: Evaluating CAM model on test set..."
echo "============================================================"
echo ""

$PYTHON_CMD experiments/cam_human_like/training/evaluate_on_cam.py \
    --model_path "$CAM_MODEL_PATH" \
    --trial_definitions "$CAM_TEST_TRIALS" \
    --data_root "$CAM_DATA_ROOT" \
    --dataset_type cam \
    --split test \
    --device $DEVICE \
    --num_frames $NUM_FRAMES \
    --use_multiframe

echo "CAM evaluation complete!"
echo ""

echo "============================================================"
echo "CAM Replication Complete!"
echo "============================================================"
echo ""
echo "Results saved to: $OUTPUT_BASE/cam_replication/"
echo "  Model: $CAM_MODEL_PATH"
echo "  Evaluation: $OUTPUT_BASE/cam_replication/model_checkpoints/cam_evaluation_test.json"
echo ""

