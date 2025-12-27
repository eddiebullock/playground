#!/bin/bash
# HPC Script: CAM Basic Emotions Fine-Tuning
# Fine-tunes CLIP on CAM basic emotions (7-way classification)

set -e

# CAM data location
CAM_DATA_ROOT="/home/eb2007/data/CAM"
if [ ! -d "$CAM_DATA_ROOT" ]; then
    echo "❌ Error: CAM data not found at $CAM_DATA_ROOT"
    exit 1
fi
echo "✅ CAM data location: $CAM_DATA_ROOT"

# Output directory: Use RDS to avoid /home quota issues
if [ -d "${HOME}/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007" ]; then
    RDS_USER_DIR="${HOME}/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007"
elif [ -d "/rds/user/eb2007/rds-autism-research-ePtR33Nsgi4/users/eb2007" ]; then
    RDS_USER_DIR="/rds/user/eb2007/rds-autism-research-ePtR33Nsgi4/users/eb2007"
elif [ -d "/rds-d7/project/45718/users/eb2007" ]; then
    RDS_USER_DIR="/rds-d7/project/45718/users/eb2007"
else
    RDS_USER_DIR="${HOME}/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007"
fi

OUTPUT_BASE="${RDS_USER_DIR}/mr_ts_play_results"
mkdir -p "$OUTPUT_BASE"
echo "✅ Using RDS for results: $OUTPUT_BASE"

# Training configuration (CPU-optimized)
NUM_EPOCHS=20
BATCH_SIZE=4
LEARNING_RATE=5e-5
WEIGHT_DECAY=0.01
NUM_FRAMES=16
DEVICE="cpu"

echo "Configuration: CPU training (icelake partition)"
echo "Note: CPU training will take approximately 8-12 hours for 20 epochs..."

# Project root
PROJECT_ROOT="${HOME}/mr_ts_play"
cd "$PROJECT_ROOT" || { echo "Error: Could not cd to $PROJECT_ROOT"; exit 1; }

# Detect Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

echo "============================================================"
echo "CAM Basic Emotions Fine-Tuning on HPC"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  CAM data root: $CAM_DATA_ROOT"
echo "  Output directory: $OUTPUT_BASE"
echo "  Device: $DEVICE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch size: $BATCH_SIZE (CPU-optimized)"
echo "  Learning rate: $LEARNING_RATE"
echo "  Weight decay: $WEIGHT_DECAY"
echo "  Num frames: $NUM_FRAMES"
echo ""

# Create output directories
mkdir -p "$OUTPUT_BASE/basic_emotions_cam"

# ============================================================
# Step 1: Load existing CAM trial definitions and create basic emotion trials
# ============================================================
echo "============================================================"
echo "Step 1: Creating basic emotion trials from CAM data..."
echo "============================================================"
echo ""

# Find existing CAM trial definitions (use test set as source)
CAM_TRIAL_SOURCE="${PROJECT_ROOT}/data/trial_definitions/cam_test.json"
if [ ! -f "$CAM_TRIAL_SOURCE" ]; then
    # Try alternative location
    CAM_TRIAL_SOURCE="${PROJECT_ROOT}/data/cam_trial_definitions_20concepts.json"
fi

if [ ! -f "$CAM_TRIAL_SOURCE" ]; then
    echo "❌ Error: CAM trial definitions not found. Please generate them first."
    exit 1
fi

echo "Using CAM trial source: $CAM_TRIAL_SOURCE"

# Basic emotion mapping file
CAM_MAPPING="${PROJECT_ROOT}/data/basic_emotion_mapping.json"
if [ ! -f "$CAM_MAPPING" ]; then
    echo "❌ Error: CAM basic emotion mapping not found at $CAM_MAPPING"
    exit 1
fi

CAM_TRAIN_TRIALS="$OUTPUT_BASE/basic_emotions_cam/cam_basic_emotions_train.json"
CAM_TEST_TRIALS="$OUTPUT_BASE/basic_emotions_cam/cam_basic_emotions_test.json"

# Generate basic emotion trials
if [ ! -f "$CAM_TRAIN_TRIALS" ] || [ ! -f "$CAM_TEST_TRIALS" ]; then
    echo "Generating basic emotion trials from CAM data..."
    $PYTHON_CMD experiments/basic_emotions_recognition/training/create_basic_emotion_trials.py \
        --dataset_type cam \
        --input_trials "$CAM_TRIAL_SOURCE" \
        --mapping_file "$CAM_MAPPING" \
        --output_dir "$OUTPUT_BASE/basic_emotions_cam" \
        --train_ratio 0.8 \
        --seed 42
    
    if [ ! -f "$CAM_TRAIN_TRIALS" ] || [ ! -f "$CAM_TEST_TRIALS" ]; then
        echo "❌ Error: Failed to generate basic emotion trials"
        exit 1
    fi
else
    echo "Using existing basic emotion trials:"
    echo "  Train: $CAM_TRAIN_TRIALS"
    echo "  Test: $CAM_TEST_TRIALS"
fi

echo "Basic emotion trials generated successfully"
echo ""

# ============================================================
# Step 2: Fine-tune CLIP on CAM basic emotions
# ============================================================
echo "============================================================"
echo "Step 2: Fine-tuning CLIP on CAM basic emotions..."
echo "============================================================"
echo "Note: Running on CPU - this will take approximately 8-12 hours for 20 epochs..."
echo ""

$PYTHON_CMD experiments/basic_emotions_recognition/training/finetune_basic_emotions.py \
    --dataset_type cam \
    --train_trials "$CAM_TRAIN_TRIALS" \
    --val_trials "$CAM_TEST_TRIALS" \
    --data_root "$CAM_DATA_ROOT" \
    --output_dir "$OUTPUT_BASE/basic_emotions_cam/model_checkpoints" \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --device $DEVICE \
    --num_frames $NUM_FRAMES \
    --early_stopping_patience 5 \
    --early_stopping_min_delta 0.001

CAM_MODEL_PATH="$OUTPUT_BASE/basic_emotions_cam/model_checkpoints/best_model"

if [ ! -d "$CAM_MODEL_PATH" ]; then
    echo "❌ Error: CAM fine-tuning failed"
    exit 1
fi

echo "CAM basic emotions fine-tuning complete!"
echo ""

# ============================================================
# Step 3: Evaluate on CAM basic emotion test set
# ============================================================
echo "============================================================"
echo "Step 3: Evaluating CAM model on basic emotion test set..."
echo "============================================================"
echo ""

$PYTHON_CMD experiments/basic_emotions_recognition/training/evaluate_basic_emotions.py \
    --model_path "$CAM_MODEL_PATH" \
    --trial_definitions "$CAM_TEST_TRIALS" \
    --data_root "$CAM_DATA_ROOT" \
    --device $DEVICE \
    --num_frames $NUM_FRAMES \
    --output_file "$OUTPUT_BASE/basic_emotions_cam/evaluation_results.json"

echo "CAM basic emotions evaluation complete!"
echo ""

echo "============================================================"
echo "CAM Basic Emotions Fine-Tuning Complete!"
echo "============================================================"
echo ""
echo "Results saved to: $OUTPUT_BASE/basic_emotions_cam/"
echo "  Model: $CAM_MODEL_PATH"
echo "  Evaluation: $OUTPUT_BASE/basic_emotions_cam/evaluation_results.json"
echo ""

