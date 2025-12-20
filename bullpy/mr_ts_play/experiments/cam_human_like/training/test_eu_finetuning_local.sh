#!/bin/bash
# Local test script for EU-Emotion fine-tuning pipeline
# Tests: extraction → dataset loading → fine-tuning → CAM evaluation

set -e  # Exit on error

# Configuration
SOURCE_DIR="/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions"
# Use source directory directly (no copying needed)
EXTRACTED_FACES_DIR="$SOURCE_DIR"
CAM_DATA_ROOT="/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/mindreading_transporter_files/Mindreading emotions library/Emotions"
CAM_TRIAL_DEFINITIONS="data/cam_trial_definitions_20concepts.json"
OUTPUT_DIR="models/clip_eu_emotion_local_test"

# Detect device
if command -v python3 &> /dev/null; then
    DEVICE=$(python3 -c "import torch; print('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')" 2>/dev/null || echo "cpu")
else
    DEVICE="cpu"
fi

echo "============================================================"
echo "EU-Emotion Fine-Tuning - Local Test Pipeline"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Source EU-Emotion: $SOURCE_DIR"
echo "  Extracted faces: $EXTRACTED_FACES_DIR"
echo "  CAM data root: $CAM_DATA_ROOT"
echo "  Output directory: $OUTPUT_DIR"
echo "  Device: $DEVICE"
echo ""

# Step 1: Use EU-Emotion dataset directly (no copying needed)
echo "============================================================"
echo "Step 1: Using EU-Emotion Dataset Directly"
echo "============================================================"
echo ""

echo "Using dataset directly from: $SOURCE_DIR"
echo "   No copying needed - dataset loader works with original structure"
echo "   Structure: emotions*/HD Version - Face, Body, Social/Faces - HD Version/"
echo ""

echo ""
echo "============================================================"
echo "Step 2: Test EU-Emotion Dataset Loader"
echo "============================================================"
echo ""

python3 experiments/cam_human_like/training/test_eu_emotion.py \
    --eu_emotion_dir "$SOURCE_DIR" \
    --modality face

if [ $? -ne 0 ]; then
    echo "❌ Dataset loader test failed!"
    exit 1
fi

echo ""
echo "============================================================"
echo "Step 3: Fine-Tune CLIP on EU-Emotion (1-2 epochs, test run)"
echo "============================================================"
echo ""
echo "This will take approximately:"
if [ "$DEVICE" = "cpu" ]; then
    echo "  CPU: ~4-8 hours (2 epochs)"
elif [ "$DEVICE" = "mps" ]; then
    echo "  MPS (Mac GPU): ~20-40 minutes (2 epochs)"
elif [ "$DEVICE" = "cuda" ]; then
    echo "  CUDA (GPU): ~10-20 minutes (2 epochs)"
fi
echo ""

read -p "Continue with fine-tuning? (y/N): " confirm
if [[ ! "$confirm" =~ ^[yY]$ ]]; then
    echo "Skipping fine-tuning. You can run it manually later."
    exit 0
fi

echo ""
echo "Starting fine-tuning..."
echo ""

python3 experiments/cam_human_like/training/finetune_clip_emotions.py \
    --eu_emotion_dir "$SOURCE_DIR" \
    --eu_emotion_modality face \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs 2 \
    --batch_size 8 \
    --learning_rate 1e-5 \
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
echo "============================================================"
echo "Step 4: Evaluate Fine-Tuned Model on CAM Test Set"
echo "============================================================"
echo ""

echo "Next step: Evaluate the fine-tuned model on CAM:"
echo ""
echo "  python experiments/cam_human_like/run_experiment.py \\"
echo "      --config configs/cam_config.yaml \\"
echo "      --model_name $OUTPUT_DIR/best_model \\"
echo "      --output_dir results/eu_emotion_finetuned"
echo ""
echo "This will compare performance to the 37% zero-shot baseline."
echo ""
echo "============================================================"
echo "Local Test Pipeline Complete!"
echo "============================================================"

