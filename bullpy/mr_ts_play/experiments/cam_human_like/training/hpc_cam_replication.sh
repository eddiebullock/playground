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

# Note: CAM trials are now generated from all files, not from pre-defined definitions
# The script create_cam_trials_from_all_files.py will discover all valid files
OUTPUT_BASE="results"
NUM_EPOCHS=20  # Optimized for GPU: 20 epochs for better convergence
BATCH_SIZE=16  # Optimized for GPU: larger batch size for stable training
LEARNING_RATE=5e-5  # Optimized: slightly higher LR for faster convergence
WEIGHT_DECAY=0.01  # Regularization
NUM_FRAMES=16  # Optimized: more frames for better temporal coverage
DEVICE="cuda"  # Using GPU nodes (ukaea-amp partition)

# Check if CUDA is available, fallback to CPU if not
if ! python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo "⚠️  Warning: CUDA not available, falling back to CPU"
    DEVICE="cpu"
    BATCH_SIZE=4  # Smaller batch for CPU
fi

echo "Configuration: GPU training enabled (10-20x faster than CPU)"

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
echo "  Output directory: $OUTPUT_BASE"
echo "  Device: $DEVICE"
echo "  Epochs: $NUM_EPOCHS (optimized for GPU)"
echo "  Batch size: $BATCH_SIZE (optimized for GPU)"
echo "  Learning rate: $LEARNING_RATE (optimized)"
echo "  Weight decay: $WEIGHT_DECAY (regularization)"
echo "  Num frames: $NUM_FRAMES (optimized for temporal coverage)"
echo "  Project root: $PROJECT_ROOT"
echo ""

# Create output directories
mkdir -p "$OUTPUT_BASE/cam_replication"

# ============================================================
# Step 1: Generate CAM trials from ALL available files
# ============================================================
echo "============================================================"
echo "Step 1: Generating CAM trials from ALL available files..."
echo "============================================================"
echo ""

# Generate trials from all valid files (unified methodology)
$PYTHON_CMD experiments/cam_human_like/training/create_cam_trials_from_all_files.py \
    --cam-dir "$CAM_DATA_ROOT" \
    --output-dir "$OUTPUT_BASE/cam_replication" \
    --trials-per-concept 10 \
    --min-file-size-kb 50 \
    --train-ratio 0.8 \
    --seed 42

CAM_TRAIN_TRIALS="$OUTPUT_BASE/cam_replication/cam_trial_definitions_train_all_files.json"
CAM_TEST_TRIALS="$OUTPUT_BASE/cam_replication/cam_trial_definitions_test_all_files.json"

if [ ! -f "$CAM_TRAIN_TRIALS" ] || [ ! -f "$CAM_TEST_TRIALS" ]; then
    echo "Error: Failed to generate CAM trials"
    exit 1
fi

echo "CAM trials generated successfully"
echo "  Train: $CAM_TRAIN_TRIALS"
echo "  Test: $CAM_TEST_TRIALS"
echo ""

# ============================================================
# Step 2: Fine-tune on CAM
# ============================================================
echo "============================================================"
echo "Step 2: Fine-tuning CLIP on CAM"
echo "============================================================"
if [ "$DEVICE" = "cuda" ]; then
    echo "Note: Running on GPU - this will take approximately 1-2 hours for 20 epochs..."
    echo "      (10-20x faster than CPU training)"
else
    echo "Note: Running on CPU - this will take approximately 6-10 hours for 20 epochs..."
    echo "      (Slower than GPU, but will complete successfully)"
fi
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
    --weight_decay $WEIGHT_DECAY \
    --device $DEVICE \
    --num_frames $NUM_FRAMES \
    --use_lr_scheduler \
    --warmup_steps 100

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

