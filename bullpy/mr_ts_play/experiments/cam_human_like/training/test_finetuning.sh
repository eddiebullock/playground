#!/bin/bash
# Quick local test of CAM fine-tuning (1-2 epochs)

# Configuration for quick test
DATA_ROOT="/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/mindreading_transporter_files/Mindreading emotions library/Emotions"
TRAIN_DATA="data/splits/train.csv"
VAL_DATA="data/splits/val.csv"
OUTPUT_DIR="models/clip_cam_finetuned_test"

# Detect device
if command -v python3 &> /dev/null; then
    DEVICE=$(python3 -c "import torch; print('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')" 2>/dev/null || echo "cpu")
else
    DEVICE="cpu"
fi

echo "=========================================="
echo "CAM Fine-Tuning - Quick Local Test"
echo "=========================================="
echo "Data root: $DATA_ROOT"
echo "Train data: $TRAIN_DATA"
echo "Val data: $VAL_DATA"
echo "Output dir: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "Epochs: 2 (test run)"
echo "Batch size: 8 (reduced for faster testing)"
echo "Frames: 4 (reduced for faster testing)"
echo "=========================================="
echo ""
echo "Estimated time:"
if [ "$DEVICE" = "cpu" ]; then
    echo "  CPU: ~4-8 hours (2 epochs × 2-4 hours each)"
elif [ "$DEVICE" = "mps" ]; then
    echo "  MPS (Mac GPU): ~20-40 minutes (2 epochs × 10-20 min each)"
elif [ "$DEVICE" = "cuda" ]; then
    echo "  CUDA (GPU): ~10-20 minutes (2 epochs × 5-10 min each)"
fi
echo ""
echo "Starting test run..."
echo ""

# Run fine-tuning with reduced settings for faster testing
python3 experiments/cam_human_like/training/finetune_clip_emotions.py \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs 2 \
    --batch_size 8 \
    --learning_rate 1e-5 \
    --device "$DEVICE" \
    --num_frames 4

echo ""
echo "=========================================="
echo "Test complete!"
echo "=========================================="
echo "If this worked, you can run the full training with:"
echo "  - 10 epochs"
echo "  - batch_size 16"
echo "  - num_frames 8"
echo ""
echo "Or submit to HPC for much faster training!"


