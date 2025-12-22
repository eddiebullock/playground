#!/bin/bash
# Re-run EU-Emotion training with improved settings
# - 267 trials (213 train, 54 test)
# - 5 epochs
# - Prompt templates
# - Multi-frame processing (8 frames)

set -e

cd /Users/eb2007/playground/bullpy/mr_ts_play

# Configuration
EU_EMOTION_DIR="/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions"
EU_TRAIN_TRIALS="results/eu_emotion_replication/eu_emotion_trial_definitions_train.json"
EU_TEST_TRIALS="results/eu_emotion_replication/eu_emotion_trial_definitions_test.json"
OUTPUT_DIR="results/eu_emotion_replication/model_checkpoints_v3"
NUM_EPOCHS=5
BATCH_SIZE=8
LEARNING_RATE=1e-5
NUM_FRAMES=8
DEVICE="mps"

# Detect Python
if [ -f "venv/bin/python3" ]; then
    PYTHON_CMD="venv/bin/python3"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

# Detect device
if [ -f "venv/bin/python3" ]; then
    DEVICE_DETECTED=$($PYTHON_CMD -c "import torch; print('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')" 2>/dev/null || echo "cpu")
    DEVICE=$DEVICE_DETECTED
fi

echo "============================================================"
echo "Re-running EU-Emotion Training (Improved Settings)"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Train trials: $EU_TRAIN_TRIALS"
echo "  Val trials: $EU_TEST_TRIALS"
echo "  Output directory: $OUTPUT_DIR"
echo "  Device: $DEVICE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Num frames: $NUM_FRAMES"
echo "  Python: $PYTHON_CMD"
echo ""

# Check if trial files exist
if [ ! -f "$EU_TRAIN_TRIALS" ] || [ ! -f "$EU_TEST_TRIALS" ]; then
    echo "Error: Trial definition files not found!"
    echo "  Train: $EU_TRAIN_TRIALS"
    echo "  Test: $EU_TEST_TRIALS"
    echo ""
    echo "Generating trials first..."
    $PYTHON_CMD experiments/cam_human_like/training/create_eu_emotion_trials.py \
        --eu-emotion-dir "$EU_EMOTION_DIR" \
        --output-dir results/eu_emotion_replication \
        --modality face \
        --trials-per-emotion 10 \
        --min-stimuli-per-emotion 3 \
        --train-ratio 0.8 \
        --seed 42
    echo ""
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting training..."
echo "This will take approximately 50-100 minutes for 5 epochs on MPS..."
echo ""

# Run training with logging
$PYTHON_CMD experiments/cam_human_like/training/finetune_clip_emotions.py \
    --task_specific \
    --dataset_type eu_emotion \
    --train_trials "$EU_TRAIN_TRIALS" \
    --val_trials "$EU_TEST_TRIALS" \
    --data_root "$EU_EMOTION_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --device $DEVICE \
    --num_frames $NUM_FRAMES \
    2>&1 | tee results/eu_emotion_replication/training_log_v3.txt

echo ""
echo "============================================================"
echo "Training Complete!"
echo "============================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "Log saved to: results/eu_emotion_replication/training_log_v3.txt"
echo ""
echo "Next steps:"
echo "  1. Check training log for validation accuracy"
echo "  2. Evaluate model: python experiments/cam_human_like/training/evaluate_on_cam.py ..."
echo ""

