#!/bin/bash
# Script to run CAM fine-tuning

# Configuration
DATA_ROOT="/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/mindreading_transporter_files/Mindreading emotions library/Emotions"
TRAIN_DATA="data/splits/train.csv"
VAL_DATA="data/splits/val.csv"
OUTPUT_DIR="models/clip_cam_finetuned"
NUM_EPOCHS=10
BATCH_SIZE=16
LEARNING_RATE=1e-5

# Detect device
if command -v python3 &> /dev/null; then
    DEVICE=$(python3 -c "import torch; print('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')")
else
    DEVICE="cpu"
fi

echo "=========================================="
echo "CAM Fine-Tuning Setup"
echo "=========================================="
echo "Data root: $DATA_ROOT"
echo "Train data: $TRAIN_DATA"
echo "Val data: $VAL_DATA"
echo "Output dir: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "Epochs: $NUM_EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "=========================================="

# Run fine-tuning
python3 experiments/cam_human_like/training/finetune_clip_emotions.py \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --device "$DEVICE" \
    --num_frames 8

echo ""
echo "Fine-tuning complete!"
echo "To use the fine-tuned model, update configs/cam_config.yaml:"
echo "  model.name: \"$OUTPUT_DIR/best_model\""






