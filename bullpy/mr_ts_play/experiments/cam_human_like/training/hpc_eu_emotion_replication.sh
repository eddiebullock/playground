#!/bin/bash
# HPC Script: EU-Emotion Replication Study - FULL VERSION
# Full replication: 10 epochs, full trial set

set -e

# EU Emotions data location - On RDS
# Check multiple possible RDS mount points
if [ -d "/rds/rds-autism-research-ePtR33Nsgi4/data/EU_emotions" ]; then
    EU_EMOTIONS_DATA_ROOT="/rds/rds-autism-research-ePtR33Nsgi4/data/EU_emotions"
elif [ -d "${HOME}/rds/rds-autism-research-ePtR33Nsgi4/data/EU_emotions" ]; then
    EU_EMOTIONS_DATA_ROOT="${HOME}/rds/rds-autism-research-ePtR33Nsgi4/data/EU_emotions"
elif [ -d "/rds/user/eb2007/rds-autism-research-ePtR33Nsgi4/data/EU_emotions" ]; then
    EU_EMOTIONS_DATA_ROOT="/rds/user/eb2007/rds-autism-research-ePtR33Nsgi4/data/EU_emotions"
elif [ -d "/rds-d7/project/45718/users/eb2007/data/EU_emotions" ]; then
    EU_EMOTIONS_DATA_ROOT="/rds-d7/project/45718/users/eb2007/data/EU_emotions"
else
    EU_EMOTIONS_DATA_ROOT="/rds/rds-autism-research-ePtR33Nsgi4/data/EU_emotions"
fi

if [ ! -d "$EU_EMOTIONS_DATA_ROOT" ]; then
    echo "❌ Error: EU Emotions data not found at $EU_EMOTIONS_DATA_ROOT"
    exit 1
fi
echo "✅ EU Emotions data location: $EU_EMOTIONS_DATA_ROOT"

# Output directory: Use RDS to avoid /home quota issues
# RDS has 1099.5GB available vs /home's 50GB quota
if [ -d "${HOME}/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007" ]; then
    RDS_USER_DIR="${HOME}/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007"
elif [ -d "/rds/user/eb2007/rds-autism-research-ePtR33Nsgi4/users/eb2007" ]; then
    RDS_USER_DIR="/rds/user/eb2007/rds-autism-research-ePtR33Nsgi4/users/eb2007"
elif [ -d "/rds-d7/project/45718/users/eb2007" ]; then
    RDS_USER_DIR="/rds-d7/project/45718/users/eb2007"
else
    RDS_USER_DIR="${HOME}/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007"  # Default
fi

# Create results directory on RDS
OUTPUT_BASE="${RDS_USER_DIR}/mr_ts_play_results"
mkdir -p "$OUTPUT_BASE"
echo "✅ Using RDS for results: $OUTPUT_BASE"
NUM_EPOCHS=40  # Extended: 40 epochs for maximum convergence (with early stopping)
BATCH_SIZE=4   # CPU-optimized: smaller batch size for CPU training
LEARNING_RATE=1e-5  # Optimized: Best from HP tuning (85.19% validation accuracy vs 83.33% for 5e-6)
WEIGHT_DECAY=0.0  # Fixed: Removed weight decay (was causing issues with small dataset)
NUM_FRAMES=16  # Optimized: more frames for better temporal coverage
DEVICE="cpu"   # Using CPU nodes (icelake partition - user has CPU access only)

# Note: User account only has access to CPU partitions (icelake)
# Scripts will automatically use CPU with appropriate batch size
echo "Configuration: CPU training (GPU not available for this account)"

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
echo "EU-Emotion Replication Study on HPC - FULL VERSION"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  EU Emotions data root: $EU_EMOTIONS_DATA_ROOT"
echo "  Output directory: $OUTPUT_BASE"
echo "  Device: $DEVICE"
echo "  Epochs: $NUM_EPOCHS (extended with early stopping)"
echo "  Batch size: $BATCH_SIZE (CPU-optimized)"
echo "  Learning rate: $LEARNING_RATE (optimized: best from HP tuning - 85.19% validation accuracy)"
echo "  Weight decay: $WEIGHT_DECAY (disabled for small datasets)"
echo "  Num frames: $NUM_FRAMES (optimized for temporal coverage)"
echo ""

# Create output directories
mkdir -p "$OUTPUT_BASE/eu_emotion_replication"

# ============================================================
# Step 1: Generate EU-Emotion trial definitions
# ============================================================
echo "============================================================"
echo "Step 1: Generating EU-Emotion forced-choice trials..."
echo "============================================================"
echo ""

EU_TRAIN_TRIALS="$OUTPUT_BASE/eu_emotion_replication/eu_emotion_trial_definitions_train.json"
EU_TEST_TRIALS="$OUTPUT_BASE/eu_emotion_replication/eu_emotion_trial_definitions_test.json"

# Only generate trials if they don't already exist (to ensure consistency across runs)
if [ ! -f "$EU_TRAIN_TRIALS" ] || [ ! -f "$EU_TEST_TRIALS" ]; then
    echo "Generating EU-Emotion trial definitions (not found, creating new ones)..."
    $PYTHON_CMD experiments/cam_human_like/training/create_eu_emotion_trials.py \
        --eu-emotion-dir "$EU_EMOTIONS_DATA_ROOT" \
        --output-dir "$OUTPUT_BASE/eu_emotion_replication" \
        --modality face \
        --trials-per-emotion 10 \
        --min-stimuli-per-emotion 3 \
        --train-ratio 0.8 \
        --seed 42
    
    if [ ! -f "$EU_TRAIN_TRIALS" ] || [ ! -f "$EU_TEST_TRIALS" ]; then
        echo "Error: Failed to generate EU-Emotion trial definitions"
        exit 1
    fi
else
    echo "Using existing EU-Emotion trial definitions (to ensure consistency):"
    echo "  Train: $EU_TRAIN_TRIALS"
    echo "  Test: $EU_TEST_TRIALS"
fi

echo "EU-Emotion trials generated successfully"
echo "  Train: $EU_TRAIN_TRIALS"
echo "  Test: $EU_TEST_TRIALS"
echo ""

# ============================================================
# Step 2: Fine-tune on EU-Emotion
# ============================================================
echo "============================================================"
echo "Step 2: Fine-tuning CLIP on EU-Emotion"
echo "============================================================"
echo "Note: Running on CPU - this will take approximately 6-10 hours for 20 epochs..."
echo "      (CPU training is slower than GPU, but will complete successfully)"
echo ""

$PYTHON_CMD experiments/cam_human_like/training/finetune_clip_emotions.py \
    --task_specific \
    --dataset_type eu_emotion \
    --train_trials "$EU_TRAIN_TRIALS" \
    --val_trials "$EU_TEST_TRIALS" \
    --data_root "$EU_EMOTIONS_DATA_ROOT" \
    --output_dir "$OUTPUT_BASE/eu_emotion_replication/model_checkpoints" \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --device $DEVICE \
    --num_frames $NUM_FRAMES \
    --early_stopping_patience 5 \
    --early_stopping_min_delta 0.001

EU_MODEL_PATH="$OUTPUT_BASE/eu_emotion_replication/model_checkpoints/best_model"

if [ ! -d "$EU_MODEL_PATH" ]; then
    echo "Error: EU-Emotion fine-tuning failed"
    exit 1
fi

echo "EU-Emotion fine-tuning complete!"
echo ""

# ============================================================
# Step 3: Evaluate on EU-Emotion test set
# ============================================================
echo "============================================================"
echo "Step 3: Evaluating EU-Emotion model on test set..."
echo "============================================================"
echo ""

$PYTHON_CMD experiments/cam_human_like/training/evaluate_on_cam.py \
    --model_path "$EU_MODEL_PATH" \
    --trial_definitions "$EU_TEST_TRIALS" \
    --data_root "$EU_EMOTIONS_DATA_ROOT" \
    --dataset_type eu_emotion \
    --split test \
    --device $DEVICE \
    --num_frames $NUM_FRAMES \
    --use_multiframe

echo "EU-Emotion evaluation complete!"
echo ""

echo "============================================================"
echo "EU-Emotion Replication Complete!"
echo "============================================================"
echo ""
echo "Results saved to: $OUTPUT_BASE/eu_emotion_replication/"
echo "  Model: $EU_MODEL_PATH"
echo "  Evaluation: $OUTPUT_BASE/eu_emotion_replication/model_checkpoints/eu_emotion_evaluation_test.json"
echo ""

