#!/bin/bash
# Full pipeline: Fine-tune on EU-Emotion → Evaluate on CAM
# This replicates the Golan study with a fine-tuned model

set -e  # Exit on error

# Activate virtual environment if it exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
if [ -d "$PROJECT_ROOT/venv" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
    echo "Activated virtual environment: $PROJECT_ROOT/venv"
elif [ -d "$PROJECT_ROOT/.venv" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
    echo "Activated virtual environment: $PROJECT_ROOT/.venv"
fi

# Configuration
EU_EMOTION_DIR="/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions"
CAM_DATA_ROOT="/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/mindreading_transporter_files/Mindreading emotions library/Emotions"
CAM_TRIAL_DEFINITIONS="data/cam_trial_definitions_20concepts.json"
OUTPUT_DIR="models/clip_eu_emotion_finetuned"
NUM_EPOCHS=2  # Quick local test - full training will be done on HPC
BATCH_SIZE=8
LEARNING_RATE=1e-5

# Use venv Python if available
if [ -d "$PROJECT_ROOT/venv" ]; then
    PYTHON="$PROJECT_ROOT/venv/bin/python"
elif [ -d "$PROJECT_ROOT/.venv" ]; then
    PYTHON="$PROJECT_ROOT/.venv/bin/python"
else
    PYTHON="python3"
fi

# Detect device
DEVICE=$($PYTHON -c "import torch; print('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')" 2>/dev/null || echo "cpu")

echo "============================================================"
echo "Full Pipeline: EU-Emotion Fine-Tuning → CAM Evaluation"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  EU-Emotion dir: $EU_EMOTION_DIR"
echo "  CAM data root: $CAM_DATA_ROOT"
echo "  CAM trial definitions: $CAM_TRIAL_DEFINITIONS"
echo "  Output directory: $OUTPUT_DIR"
echo "  Device: $DEVICE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo ""

# Step 1: Fine-tune on EU-Emotion
echo "============================================================"
echo "Step 1: Fine-Tuning CLIP on EU-Emotion (Faces Only)"
echo "============================================================"
echo ""

echo "This will fine-tune CLIP on EU-Emotion dataset (faces only)"
echo "This is a QUICK LOCAL TEST with $NUM_EPOCHS epochs"
echo "Full training will be done on HPC later"
echo ""
echo "Expected time:"
if [ "$DEVICE" = "cpu" ]; then
    echo "  CPU: ~1-2 hours ($NUM_EPOCHS epochs)"
elif [ "$DEVICE" = "mps" ]; then
    echo "  MPS (Mac GPU): ~20-40 minutes ($NUM_EPOCHS epochs)"
elif [ "$DEVICE" = "cuda" ]; then
    echo "  CUDA (GPU): ~10-20 minutes ($NUM_EPOCHS epochs)"
fi
echo ""

$PYTHON experiments/cam_human_like/training/finetune_clip_emotions.py \
    --eu_emotion_dir "$EU_EMOTION_DIR" \
    --eu_emotion_modality face \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --device "$DEVICE" \
    --num_frames 8 \
    --use_multiframe

if [ $? -ne 0 ]; then
    echo "❌ Fine-tuning failed!"
    exit 1
fi

echo ""
echo "✅ Fine-tuning complete!"
echo ""

# Step 2: Evaluate on CAM test set
echo "============================================================"
echo "Step 2: Evaluating Fine-Tuned Model on CAM Test Set"
echo "============================================================"
echo ""

MODEL_PATH="$OUTPUT_DIR/best_model"

if [ ! -d "$MODEL_PATH" ]; then
    echo "⚠️  Warning: best_model directory not found, trying epoch_1..."
    MODEL_PATH="$OUTPUT_DIR/epoch_1"
    if [ ! -d "$MODEL_PATH" ]; then
        echo "❌ Error: Could not find fine-tuned model in $OUTPUT_DIR"
        echo "   Available directories:"
        ls -la "$OUTPUT_DIR" 2>/dev/null || echo "   (output directory doesn't exist)"
        exit 1
    fi
fi

echo "Using model: $MODEL_PATH"
echo ""

$PYTHON experiments/cam_human_like/training/evaluate_on_cam.py \
    --model_path "$MODEL_PATH" \
    --trial_definitions "$CAM_TRIAL_DEFINITIONS" \
    --data_root "$CAM_DATA_ROOT" \
    --split test \
    --device "$DEVICE" \
    --num_frames 8 \
    --use_multiframe

if [ $? -ne 0 ]; then
    echo "❌ CAM evaluation failed!"
    exit 1
fi

echo ""
echo "============================================================"
echo "Pipeline Complete!"
echo "============================================================"
echo ""
echo "Results:"
echo "  - Fine-tuned model: $MODEL_PATH"
echo "  - Evaluation results: $OUTPUT_DIR/cam_evaluation_test.json"
echo ""
echo "Compare to baselines:"
echo "  - Zero-shot CLIP: 37.0%"
echo "  - Previous fine-tuned (1 epoch): 45.95%"
echo ""

