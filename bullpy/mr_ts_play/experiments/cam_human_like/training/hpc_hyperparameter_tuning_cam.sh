#!/bin/bash
# HPC Script: Hyperparameter Tuning for CAM Only
# Tests different learning rates to find optimal configuration
# Uses early stopping to prevent overfitting

set -e

# RDS path for results
if [ -d "${HOME}/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007" ]; then
    RDS_USER_DIR="${HOME}/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007"
elif [ -d "/rds/user/eb2007/rds-autism-research-ePtR33Nsgi4/users/eb2007" ]; then
    RDS_USER_DIR="/rds/user/eb2007/rds-autism-research-ePtR33Nsgi4/users/eb2007"
else
    RDS_USER_DIR="${HOME}/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007"
fi

OUTPUT_BASE="${RDS_USER_DIR}/mr_ts_play_results"
mkdir -p "$OUTPUT_BASE"

# Configuration
NUM_FRAMES=16  # Optimized: more frames for better temporal coverage
WEIGHT_DECAY=0.0  # Disabled (was causing issues)
DEVICE="cpu"  # Using CPU nodes (icelake partition)
NUM_EPOCHS=40  # Extended epochs with early stopping
EARLY_STOPPING_PATIENCE=5
EARLY_STOPPING_MIN_DELTA=0.001

# Data paths
CAM_DATA_ROOT="/home/eb2007/data/CAM"

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
echo "CAM Hyperparameter Tuning Study"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Device: $DEVICE (CPU)"
echo "  Num frames: $NUM_FRAMES"
echo "  Weight decay: $WEIGHT_DECAY (disabled)"
echo "  Max epochs: $NUM_EPOCHS (with early stopping)"
echo "  Early stopping patience: $EARLY_STOPPING_PATIENCE epochs"
echo "  CAM data: $CAM_DATA_ROOT"
echo "  Output: $OUTPUT_BASE"
echo ""
echo "Testing 5 learning rate configurations:"
echo "  1. Lower LR: lr=5e-6, batch_size=4 (more stable)"
echo "  2. Baseline: lr=1e-5, batch_size=4 (current best)"
echo "  3. Higher LR: lr=2e-5, batch_size=4 (faster convergence)"
echo "  4. Higher LR2: lr=3e-5, batch_size=4 (test upper limit)"
echo "  5. Lower LR2: lr=8e-6, batch_size=4 (between baseline and lower)"
echo ""

# Hyperparameter configurations: learning_rate:batch_size:name
declare -a CONFIGS=(
    "5e-6:4:lr_lower"
    "1e-5:4:lr_baseline"
    "2e-5:4:lr_higher"
    "3e-5:4:lr_higher2"
    "8e-6:4:lr_lower2"
)

# ============================================================
# CAM Hyperparameter Tuning
# ============================================================
echo "============================================================"
echo "CAM Hyperparameter Tuning"
echo "============================================================"
echo ""

# Generate CAM trials once (shared across all runs)
CAM_TRIALS_DIR="$OUTPUT_BASE/cam_replication/hp_tuning"
mkdir -p "$CAM_TRIALS_DIR"

echo "Generating CAM trials (shared across all runs)..."
$PYTHON_CMD experiments/cam_human_like/training/create_cam_trials_from_all_files.py \
    --cam-dir "$CAM_DATA_ROOT" \
    --output-dir "$CAM_TRIALS_DIR" \
    --trials-per-concept 10 \
    --min-file-size-kb 50 \
    --train-ratio 0.8 \
    --seed 42

CAM_TRAIN_TRIALS="$CAM_TRIALS_DIR/cam_trial_definitions_train_all_files.json"
CAM_TEST_TRIALS="$CAM_TRIALS_DIR/cam_trial_definitions_test_all_files.json"

if [ ! -f "$CAM_TRAIN_TRIALS" ] || [ ! -f "$CAM_TEST_TRIALS" ]; then
    echo "Error: Failed to generate CAM trials"
    exit 1
fi

echo "CAM trials generated successfully"
echo ""

# Run each configuration
for config in "${CONFIGS[@]}"; do
    IFS=':' read -r lr batch_size name <<< "$config"
    
    echo "============================================================"
    echo "CAM Run: $name (lr=$lr, batch_size=$batch_size)"
    echo "============================================================"
    
    RUN_OUTPUT_DIR="$OUTPUT_BASE/cam_replication/hp_tuning/run_${name}"
    mkdir -p "$RUN_OUTPUT_DIR"
    
    $PYTHON_CMD experiments/cam_human_like/training/finetune_clip_emotions.py \
        --task_specific \
        --dataset_type cam \
        --train_trials "$CAM_TRAIN_TRIALS" \
        --val_trials "$CAM_TEST_TRIALS" \
        --data_root "$CAM_DATA_ROOT" \
        --output_dir "$RUN_OUTPUT_DIR/model_checkpoints" \
        --num_epochs $NUM_EPOCHS \
        --batch_size $batch_size \
        --learning_rate $lr \
        --weight_decay $WEIGHT_DECAY \
        --device $DEVICE \
        --num_frames $NUM_FRAMES \
        --early_stopping_patience $EARLY_STOPPING_PATIENCE \
        --early_stopping_min_delta $EARLY_STOPPING_MIN_DELTA
    
    # Evaluate on test set
    echo ""
    echo "Evaluating CAM model ($name) on test set..."
    $PYTHON_CMD experiments/cam_human_like/training/evaluate_on_cam.py \
        --model_path "$RUN_OUTPUT_DIR/model_checkpoints/best_model" \
        --trial_definitions "$CAM_TEST_TRIALS" \
        --data_root "$CAM_DATA_ROOT" \
        --dataset_type cam \
        --split test \
        --device $DEVICE \
        --num_frames $NUM_FRAMES \
        --use_multiframe
    
    echo ""
    echo "CAM run $name complete!"
    echo ""
done

echo "============================================================"
echo "CAM Hyperparameter Tuning Complete!"
echo "============================================================"
echo ""
echo "Results saved to: $OUTPUT_BASE/cam_replication/hp_tuning/"
echo "To compare results, check the evaluation JSON files in each run directory."

