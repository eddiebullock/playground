#!/bin/bash
# HPC Script: Hyperparameter Tuning for EU-Emotion Only
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
if [ -d "${HOME}/rds/rds-autism-research-ePtR33Nsgi4/data/EU_emotions" ]; then
    EU_EMOTIONS_DATA_ROOT="${HOME}/rds/rds-autism-research-ePtR33Nsgi4/data/EU_emotions"
elif [ -d "/rds/rds-autism-research-ePtR33Nsgi4/data/EU_emotions" ]; then
    EU_EMOTIONS_DATA_ROOT="/rds/rds-autism-research-ePtR33Nsgi4/data/EU_emotions"
else
    EU_EMOTIONS_DATA_ROOT="${HOME}/rds/rds-autism-research-ePtR33Nsgi4/data/EU_emotions"
fi

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
echo "EU-Emotion Hyperparameter Tuning Study"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Device: $DEVICE (CPU)"
echo "  Num frames: $NUM_FRAMES"
echo "  Weight decay: $WEIGHT_DECAY (disabled)"
echo "  Max epochs: $NUM_EPOCHS (with early stopping)"
echo "  Early stopping patience: $EARLY_STOPPING_PATIENCE epochs"
echo "  EU-Emotion data: $EU_EMOTIONS_DATA_ROOT"
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
# EU-Emotion Hyperparameter Tuning
# ============================================================
echo "============================================================"
echo "EU-Emotion Hyperparameter Tuning"
echo "============================================================"
echo ""

# Generate EU-Emotion trials once (shared across all runs)
EU_TRIALS_DIR="$OUTPUT_BASE/eu_emotion_replication/hp_tuning"
mkdir -p "$EU_TRIALS_DIR"

echo "Generating EU-Emotion trials (shared across all runs)..."
$PYTHON_CMD experiments/cam_human_like/training/create_eu_emotion_trials.py \
    --eu-emotion-dir "$EU_EMOTIONS_DATA_ROOT" \
    --output-dir "$EU_TRIALS_DIR" \
    --modality face \
    --train-ratio 0.8 \
    --seed 42

EU_TRAIN_TRIALS="$EU_TRIALS_DIR/eu_emotion_trial_definitions_train.json"
EU_TEST_TRIALS="$EU_TRIALS_DIR/eu_emotion_trial_definitions_test.json"

if [ ! -f "$EU_TRAIN_TRIALS" ] || [ ! -f "$EU_TEST_TRIALS" ]; then
    echo "Error: Failed to generate EU-Emotion trials"
    exit 1
fi

echo "EU-Emotion trials generated successfully"
echo ""

# Run each configuration
for config in "${CONFIGS[@]}"; do
    IFS=':' read -r lr batch_size name <<< "$config"
    
    echo "============================================================"
    echo "EU-Emotion Run: $name (lr=$lr, batch_size=$batch_size)"
    echo "============================================================"
    
    RUN_OUTPUT_DIR="$OUTPUT_BASE/eu_emotion_replication/hp_tuning/run_${name}"
    mkdir -p "$RUN_OUTPUT_DIR"
    
    $PYTHON_CMD experiments/cam_human_like/training/finetune_clip_emotions.py \
        --task_specific \
        --dataset_type eu_emotion \
        --train_trials "$EU_TRAIN_TRIALS" \
        --val_trials "$EU_TEST_TRIALS" \
        --data_root "$EU_EMOTIONS_DATA_ROOT" \
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
    echo "Evaluating EU-Emotion model ($name) on test set..."
    $PYTHON_CMD experiments/cam_human_like/training/evaluate_on_cam.py \
        --model_path "$RUN_OUTPUT_DIR/model_checkpoints/best_model" \
        --trial_definitions "$EU_TEST_TRIALS" \
        --data_root "$EU_EMOTIONS_DATA_ROOT" \
        --dataset_type eu_emotion \
        --split test \
        --device $DEVICE \
        --num_frames $NUM_FRAMES \
        --use_multiframe
    
    echo ""
    echo "EU-Emotion run $name complete!"
    echo ""
done

echo "============================================================"
echo "EU-Emotion Hyperparameter Tuning Complete!"
echo "============================================================"
echo ""
echo "Results saved to: $OUTPUT_BASE/eu_emotion_replication/hp_tuning/"
echo "To compare results, check the evaluation JSON files in each run directory."

