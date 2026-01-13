#!/bin/bash
# Fine-tune I3D and TimeSformer models on EU-Emotion dataset

set -e

# Configuration
DATA_ROOT="/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions"
TRAIN_TRIALS="data/trial_definitions/eu_emotion_train.json"
VAL_TRIALS="data/trial_definitions/eu_emotion_val.json"
OUTPUT_BASE="models"

# Training parameters (optimized for speed)
NUM_EPOCHS=12
BATCH_SIZE=8
LEARNING_RATE=2e-4
NUM_FRAMES_I3D=16
NUM_FRAMES_TIMESFORMER=8
GRADIENT_ACCUMULATION=1
EARLY_STOPPING_PATIENCE=4
NUM_WORKERS=2

echo "=========================================="
echo "Fine-tuning I3D and TimeSformer models"
echo "=========================================="
echo ""
echo "Data root: $DATA_ROOT"
echo "Train trials: $TRAIN_TRIALS"
echo "Val trials: $VAL_TRIALS"
echo ""

# Check if train/val splits exist
if [ ! -f "$TRAIN_TRIALS" ]; then
    echo "ERROR: Train trials file not found: $TRAIN_TRIALS"
    echo "Please create train/val splits first using:"
    echo "  python experiments/eu_emotion_model_comparison/training/create_train_val_splits.py \\"
    echo "    --test_trials data/trial_definitions/eu_emotion_test.json \\"
    echo "    --data_root $DATA_ROOT \\"
    echo "    --output_dir data/trial_definitions"
    exit 1
fi

if [ ! -f "$VAL_TRIALS" ]; then
    echo "ERROR: Val trials file not found: $VAL_TRIALS"
    exit 1
fi

# Fine-tune I3D
echo "=========================================="
echo "Fine-tuning I3D model..."
echo "=========================================="
python experiments/eu_emotion_model_comparison/training/finetune_video_models_task_specific.py \
    --model i3d \
    --train_trials "$TRAIN_TRIALS" \
    --val_trials "$VAL_TRIALS" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_BASE/i3d_emotion_finetuned_task_specific" \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_frames $NUM_FRAMES_I3D \
    --frame_sampling uniform \
    --device auto \
    --use_mixed_precision \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --num_workers $NUM_WORKERS

echo ""
echo "I3D fine-tuning completed!"
echo ""

# Fine-tune TimeSformer
echo "=========================================="
echo "Fine-tuning TimeSformer model..."
echo "=========================================="
python experiments/eu_emotion_model_comparison/training/finetune_video_models_task_specific.py \
    --model timesformer \
    --train_trials "$TRAIN_TRIALS" \
    --val_trials "$VAL_TRIALS" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_BASE/timesformer_emotion_finetuned_task_specific" \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_frames $NUM_FRAMES_TIMESFORMER \
    --frame_sampling uniform \
    --device auto \
    --use_mixed_precision \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --num_workers $NUM_WORKERS

echo ""
echo "TimeSformer fine-tuning completed!"
echo ""

echo "=========================================="
echo "All fine-tuning completed!"
echo "=========================================="
echo ""
echo "Models saved to:"
echo "  - $OUTPUT_BASE/i3d_emotion_finetuned_task_specific/best_model.pth"
echo "  - $OUTPUT_BASE/timesformer_emotion_finetuned_task_specific/best_model.pth"
echo ""
echo "Next step: Test the models using:"
echo "  python experiments/eu_emotion_model_comparison/training/test_video_models.py \\"
echo "    --model i3d --model_path $OUTPUT_BASE/i3d_emotion_finetuned_task_specific/best_model.pth"
echo "  python experiments/eu_emotion_model_comparison/training/test_video_models.py \\"
echo "    --model timesformer --model_path $OUTPUT_BASE/timesformer_emotion_finetuned_task_specific/best_model.pth"
