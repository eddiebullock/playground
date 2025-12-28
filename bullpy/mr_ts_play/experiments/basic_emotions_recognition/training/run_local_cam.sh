#!/bin/bash
# Local Script: CAM Basic Emotions Fine-Tuning
# Fine-tunes CLIP on CAM basic emotions (7-way classification)

set -e

# Activate virtual environment if it exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
if [ -d "$PROJECT_ROOT/venv" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
    PYTHON_CMD="$PROJECT_ROOT/venv/bin/python"
elif [ -d "$PROJECT_ROOT/.venv" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
    PYTHON_CMD="$PROJECT_ROOT/.venv/bin/python"
else
    PYTHON_CMD="python3"
fi

# CAM data location (adjust to your local path)
CAM_DATA_ROOT="/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/mindreading_transporter_files/Mindreading emotions library/Emotions"
if [ ! -d "$CAM_DATA_ROOT" ]; then
    echo "❌ Error: CAM data not found at $CAM_DATA_ROOT"
    echo "Please update CAM_DATA_ROOT in this script to point to your CAM data location"
    exit 1
fi
echo "✅ CAM data location: $CAM_DATA_ROOT"

# Output directory (local)
OUTPUT_BASE="models/basic_emotions_cam"
mkdir -p "$OUTPUT_BASE"
echo "✅ Output directory: $OUTPUT_BASE"

# Detect device
DEVICE=$($PYTHON_CMD -c "import torch; print('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')" 2>/dev/null || echo "cpu")

# Training configuration (auto-optimized based on device)
if [ "$DEVICE" = "cpu" ]; then
    NUM_EPOCHS=12
    BATCH_SIZE=4
    echo "Configuration: CPU training (will take ~3-5 hours)"
elif [ "$DEVICE" = "mps" ]; then
    NUM_EPOCHS=12
    BATCH_SIZE=8
    echo "Configuration: MPS (Mac GPU) training (will take ~2-4 hours)"
elif [ "$DEVICE" = "cuda" ]; then
    NUM_EPOCHS=12
    BATCH_SIZE=16
    echo "Configuration: CUDA (GPU) training (will take ~30-60 minutes)"
fi

LEARNING_RATE=5e-5
WEIGHT_DECAY=0.01
NUM_FRAMES=16

cd "$PROJECT_ROOT" || { echo "Error: Could not cd to $PROJECT_ROOT"; exit 1; }

echo "============================================================"
echo "CAM Basic Emotions Fine-Tuning (Local)"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  CAM data root: $CAM_DATA_ROOT"
echo "  Output directory: $OUTPUT_BASE"
echo "  Device: $DEVICE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Weight decay: $WEIGHT_DECAY"
echo "  Num frames: $NUM_FRAMES"
echo ""

# ============================================================
# Step 1: Create basic emotion trials
# ============================================================
echo "============================================================"
echo "Step 1: Creating basic emotion trials from CAM data..."
echo "============================================================"
echo ""

CAM_TRIAL_SOURCE="$PROJECT_ROOT/data/cam_trial_definitions_20concepts.json"
if [ ! -f "$CAM_TRIAL_SOURCE" ]; then
    echo "❌ Error: CAM trial definitions not found at $CAM_TRIAL_SOURCE"
    exit 1
fi

CAM_MAPPING="$PROJECT_ROOT/data/basic_emotion_mapping.json"
if [ ! -f "$CAM_MAPPING" ]; then
    echo "❌ Error: CAM emotion mapping not found at $CAM_MAPPING"
    exit 1
fi

CAM_TRAIN_TRIALS="$OUTPUT_BASE/cam_basic_emotions_train.json"
CAM_VAL_TRIALS="$OUTPUT_BASE/cam_basic_emotions_val.json"
CAM_TEST_TRIALS="$OUTPUT_BASE/cam_basic_emotions_test.json"

# Generate basic emotion trials
if [ ! -f "$CAM_TRAIN_TRIALS" ] || [ ! -f "$CAM_VAL_TRIALS" ] || [ ! -f "$CAM_TEST_TRIALS" ]; then
    echo "Generating basic emotion trials from CAM data..."
    $PYTHON_CMD experiments/basic_emotions_recognition/training/create_basic_emotion_trials.py \
        --dataset_type cam \
        --input_trials "$CAM_TRIAL_SOURCE" \
        --mapping_file "$CAM_MAPPING" \
        --output_dir "$OUTPUT_BASE" \
        --train_ratio 0.8 \
        --seed 42
    
    if [ ! -f "$CAM_TRAIN_TRIALS" ] || [ ! -f "$CAM_VAL_TRIALS" ] || [ ! -f "$CAM_TEST_TRIALS" ]; then
        echo "❌ Error: Failed to generate basic emotion trials"
        exit 1
    fi
    echo "✅ Basic emotion trials generated"
else
    echo "✅ Basic emotion trials already exist, skipping generation"
fi

# ============================================================
# Step 2: Fine-tune CLIP
# ============================================================
echo ""
echo "============================================================"
echo "Step 2: Fine-tuning CLIP on CAM basic emotions..."
echo "============================================================"
echo ""

# Check if model already exists
MODEL_PATH="$OUTPUT_BASE/best_model/best_model"
if [ -f "$MODEL_PATH/config.json" ] || [ -f "$MODEL_PATH/model.safetensors" ] || [ -f "$MODEL_PATH/pytorch_model.bin" ]; then
    echo "✅ Model already exists at $MODEL_PATH, skipping training"
    echo "   To re-train, delete the model directory first"
else
    echo "Training new model..."
    $PYTHON_CMD experiments/basic_emotions_recognition/training/finetune_basic_emotions.py \
    --dataset_type cam \
    --train_trials "$CAM_TRAIN_TRIALS" \
    --val_trials "$CAM_VAL_TRIALS" \
    --data_root "$CAM_DATA_ROOT" \
    --output_dir "$OUTPUT_BASE/best_model" \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --device "$DEVICE" \
    --num_frames $NUM_FRAMES
fi

# ============================================================
# Step 3: Evaluate
# ============================================================
echo ""
echo "============================================================"
echo "Step 3: Evaluating fine-tuned model..."
echo "============================================================"
echo ""

mkdir -p "$OUTPUT_BASE/evaluation"

# Find the actual best model path (could be nested)
BEST_MODEL_PATH="$OUTPUT_BASE/best_model/best_model"
if [ ! -f "$BEST_MODEL_PATH/config.json" ] && [ ! -f "$BEST_MODEL_PATH/model.safetensors" ]; then
    # Try the direct path
    BEST_MODEL_PATH="$OUTPUT_BASE/best_model"
    if [ ! -f "$BEST_MODEL_PATH/config.json" ] && [ ! -f "$BEST_MODEL_PATH/model.safetensors" ]; then
        echo "❌ Error: Could not find trained model"
        echo "   Looked in: $OUTPUT_BASE/best_model/best_model"
        echo "   Looked in: $OUTPUT_BASE/best_model"
        exit 1
    fi
fi

echo "Using model from: $BEST_MODEL_PATH"

$PYTHON_CMD experiments/basic_emotions_recognition/training/evaluate_basic_emotions.py \
    --model_path "$BEST_MODEL_PATH" \
    --trial_definitions "$CAM_TEST_TRIALS" \
    --data_root "$CAM_DATA_ROOT" \
    --output_file "$OUTPUT_BASE/evaluation/results.json" \
    --device "$DEVICE" \
    --num_frames $NUM_FRAMES

echo ""
echo "============================================================"
echo "✅ CAM Basic Emotions Training Complete!"
echo "============================================================"
echo ""
echo "Results:"
echo "  Model: $OUTPUT_BASE/best_model"
echo "  Evaluation: $OUTPUT_BASE/evaluation"
echo ""

