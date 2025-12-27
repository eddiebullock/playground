#!/bin/bash
# HPC Script: EU-Emotion Basic Emotions Fine-Tuning
# Fine-tunes CLIP on EU-Emotion basic emotions (7-way classification)

set -e

# EU Emotions data location - On RDS
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

# Output directory: Use RDS
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
echo "EU-Emotion Basic Emotions Fine-Tuning on HPC"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  EU Emotions data root: $EU_EMOTIONS_DATA_ROOT"
echo "  Output directory: $OUTPUT_BASE"
echo "  Device: $DEVICE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch size: $BATCH_SIZE (CPU-optimized)"
echo "  Learning rate: $LEARNING_RATE"
echo "  Weight decay: $WEIGHT_DECAY"
echo "  Num frames: $NUM_FRAMES"
echo ""

# Create output directories
mkdir -p "$OUTPUT_BASE/basic_emotions_eu_emotion"

# ============================================================
# Step 1: Generate EU-Emotion trial definitions (if needed) and create basic emotion trials
# ============================================================
echo "============================================================"
echo "Step 1: Creating basic emotion trials from EU-Emotion data..."
echo "============================================================"
echo ""

# First, check if EU-Emotion trial definitions exist, if not generate them
EU_TRIAL_SOURCE="${PROJECT_ROOT}/data/trial_definitions/eu_emotion_test.json"
if [ ! -f "$EU_TRIAL_SOURCE" ]; then
    echo "EU-Emotion trial definitions not found. Generating them first..."
    $PYTHON_CMD experiments/cam_human_like/training/create_eu_emotion_trials.py \
        --eu-emotion-dir "$EU_EMOTIONS_DATA_ROOT" \
        --output-dir "${PROJECT_ROOT}/data/trial_definitions" \
        --modality face \
        --trials-per-emotion 10 \
        --min-stimuli-per-emotion 5 \
        --seed 42
    
    # Use the generated all trials file
    EU_TRIAL_SOURCE="${PROJECT_ROOT}/data/trial_definitions/eu_emotion_trial_definitions_all.json"
fi

if [ ! -f "$EU_TRIAL_SOURCE" ]; then
    echo "❌ Error: EU-Emotion trial definitions not found after generation"
    exit 1
fi

echo "Using EU-Emotion trial source: $EU_TRIAL_SOURCE"

# Basic emotion mapping file
EU_MAPPING="${PROJECT_ROOT}/experiments/basic_emotions_recognition/data/basic_emotion_mappings/eu_emotion_basic_mapping.json"
if [ ! -f "$EU_MAPPING" ]; then
    echo "❌ Error: EU-Emotion basic emotion mapping not found at $EU_MAPPING"
    exit 1
fi

EU_TRAIN_TRIALS="$OUTPUT_BASE/basic_emotions_eu_emotion/eu_emotion_basic_emotions_train.json"
EU_TEST_TRIALS="$OUTPUT_BASE/basic_emotions_eu_emotion/eu_emotion_basic_emotions_test.json"

# Generate basic emotion trials
if [ ! -f "$EU_TRAIN_TRIALS" ] || [ ! -f "$EU_TEST_TRIALS" ]; then
    echo "Generating basic emotion trials from EU-Emotion data..."
    $PYTHON_CMD experiments/basic_emotions_recognition/training/create_basic_emotion_trials.py \
        --dataset_type eu_emotion \
        --input_trials "$EU_TRIAL_SOURCE" \
        --mapping_file "$EU_MAPPING" \
        --output_dir "$OUTPUT_BASE/basic_emotions_eu_emotion" \
        --train_ratio 0.8 \
        --seed 42
    
    if [ ! -f "$EU_TRAIN_TRIALS" ] || [ ! -f "$EU_TEST_TRIALS" ]; then
        echo "❌ Error: Failed to generate basic emotion trials"
        exit 1
    fi
else
    echo "Using existing basic emotion trials:"
    echo "  Train: $EU_TRAIN_TRIALS"
    echo "  Test: $EU_TEST_TRIALS"
fi

echo "Basic emotion trials generated successfully"
echo ""

# ============================================================
# Step 2: Fine-tune CLIP on EU-Emotion basic emotions
# ============================================================
echo "============================================================"
echo "Step 2: Fine-tuning CLIP on EU-Emotion basic emotions..."
echo "============================================================"
echo "Note: Running on CPU - this will take approximately 8-12 hours for 20 epochs..."
echo ""

$PYTHON_CMD experiments/basic_emotions_recognition/training/finetune_basic_emotions.py \
    --dataset_type eu_emotion \
    --train_trials "$EU_TRAIN_TRIALS" \
    --val_trials "$EU_TEST_TRIALS" \
    --data_root "$EU_EMOTIONS_DATA_ROOT" \
    --output_dir "$OUTPUT_BASE/basic_emotions_eu_emotion/model_checkpoints" \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --device $DEVICE \
    --num_frames $NUM_FRAMES \
    --early_stopping_patience 5 \
    --early_stopping_min_delta 0.001

EU_MODEL_PATH="$OUTPUT_BASE/basic_emotions_eu_emotion/model_checkpoints/best_model"

if [ ! -d "$EU_MODEL_PATH" ]; then
    echo "❌ Error: EU-Emotion fine-tuning failed"
    exit 1
fi

echo "EU-Emotion basic emotions fine-tuning complete!"
echo ""

# ============================================================
# Step 3: Evaluate on EU-Emotion basic emotion test set
# ============================================================
echo "============================================================"
echo "Step 3: Evaluating EU-Emotion model on basic emotion test set..."
echo "============================================================"
echo ""

$PYTHON_CMD experiments/basic_emotions_recognition/training/evaluate_basic_emotions.py \
    --model_path "$EU_MODEL_PATH" \
    --trial_definitions "$EU_TEST_TRIALS" \
    --data_root "$EU_EMOTIONS_DATA_ROOT" \
    --device $DEVICE \
    --num_frames $NUM_FRAMES \
    --output_file "$OUTPUT_BASE/basic_emotions_eu_emotion/evaluation_results.json"

echo "EU-Emotion basic emotions evaluation complete!"
echo ""

echo "============================================================"
echo "EU-Emotion Basic Emotions Fine-Tuning Complete!"
echo "============================================================"
echo ""
echo "Results saved to: $OUTPUT_BASE/basic_emotions_eu_emotion/"
echo "  Model: $EU_MODEL_PATH"
echo "  Evaluation: $OUTPUT_BASE/basic_emotions_eu_emotion/evaluation_results.json"
echo ""

