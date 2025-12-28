#!/bin/bash
# Local Script: EU-Emotion Basic Emotions Fine-Tuning
# Fine-tunes CLIP on EU-Emotion basic emotions (7-way classification)

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

# EU-Emotion data location (adjust to your local path)
EU_EMOTIONS_DATA_ROOT="/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions"
if [ ! -d "$EU_EMOTIONS_DATA_ROOT" ]; then
    echo "❌ Error: EU-Emotion data not found at $EU_EMOTIONS_DATA_ROOT"
    echo "Please update EU_EMOTIONS_DATA_ROOT in this script to point to your EU-Emotion data location"
    exit 1
fi
echo "✅ EU-Emotion data location: $EU_EMOTIONS_DATA_ROOT"

# Output directory (local)
OUTPUT_BASE="models/basic_emotions_eu_emotion"
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
echo "EU-Emotion Basic Emotions Fine-Tuning (Local)"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  EU-Emotion data root: $EU_EMOTIONS_DATA_ROOT"
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
echo "Step 1: Creating basic emotion trials from EU-Emotion data..."
echo "============================================================"
echo ""

# Find or create EU-Emotion trial definitions
EU_TRIAL_SOURCE="$PROJECT_ROOT/data/trial_definitions/eu_emotion_trial_definitions_all.json"
if [ ! -f "$EU_TRIAL_SOURCE" ]; then
    echo "⚠️  EU-Emotion trial definitions not found at $EU_TRIAL_SOURCE"
    echo "Creating EU-Emotion trial definitions first..."
    
    # Check if create_eu_emotion_trials.py exists
    if [ -f "experiments/cam_human_like/training/create_eu_emotion_trials.py" ]; then
        $PYTHON_CMD experiments/cam_human_like/training/create_eu_emotion_trials.py \
            --eu-emotion-dir "$EU_EMOTIONS_DATA_ROOT" \
            --output-dir "$PROJECT_ROOT/data/trial_definitions" \
            --modality face \
            --trials-per-emotion 5 \
            --train-ratio 0.8 \
            --seed 42
        EU_TRIAL_SOURCE="$PROJECT_ROOT/data/trial_definitions/eu_emotion_trial_definitions_all.json"
    else
        echo "❌ Error: Cannot create EU-Emotion trials - create_eu_emotion_trials.py not found"
        exit 1
    fi
fi

EU_MAPPING="$PROJECT_ROOT/experiments/basic_emotions_recognition/data/basic_emotion_mappings/eu_emotion_basic_mapping.json"
if [ ! -f "$EU_MAPPING" ]; then
    echo "❌ Error: EU-Emotion emotion mapping not found at $EU_MAPPING"
    exit 1
fi

EU_TRAIN_TRIALS="$OUTPUT_BASE/eu_emotion_basic_emotions_train.json"
EU_VAL_TRIALS="$OUTPUT_BASE/eu_emotion_basic_emotions_val.json"
EU_TEST_TRIALS="$OUTPUT_BASE/eu_emotion_basic_emotions_test.json"

# Generate basic emotion trials
if [ ! -f "$EU_TRAIN_TRIALS" ] || [ ! -f "$EU_VAL_TRIALS" ] || [ ! -f "$EU_TEST_TRIALS" ]; then
    echo "Generating basic emotion trials from EU-Emotion data..."
    $PYTHON_CMD experiments/basic_emotions_recognition/training/create_basic_emotion_trials.py \
        --dataset_type eu_emotion \
        --input_trials "$EU_TRIAL_SOURCE" \
        --mapping_file "$EU_MAPPING" \
        --output_dir "$OUTPUT_BASE" \
        --train_ratio 0.8 \
        --seed 42
    
    if [ ! -f "$EU_TRAIN_TRIALS" ] || [ ! -f "$EU_VAL_TRIALS" ] || [ ! -f "$EU_TEST_TRIALS" ]; then
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
echo "Step 2: Fine-tuning CLIP on EU-Emotion basic emotions..."
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
    --dataset_type eu_emotion \
    --train_trials "$EU_TRAIN_TRIALS" \
    --val_trials "$EU_VAL_TRIALS" \
    --data_root "$EU_EMOTIONS_DATA_ROOT" \
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
    --trial_definitions "$EU_TEST_TRIALS" \
    --data_root "$EU_EMOTIONS_DATA_ROOT" \
    --output_file "$OUTPUT_BASE/evaluation/results.json" \
    --device "$DEVICE" \
    --num_frames $NUM_FRAMES

echo ""
echo "============================================================"
echo "✅ EU-Emotion Basic Emotions Training Complete!"
echo "============================================================"
echo ""
echo "Results:"
echo "  Model: $OUTPUT_BASE/best_model"
echo "  Evaluation: $OUTPUT_BASE/evaluation"
echo ""

