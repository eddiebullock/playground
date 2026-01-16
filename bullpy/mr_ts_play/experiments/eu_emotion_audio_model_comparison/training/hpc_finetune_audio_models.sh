#!/bin/bash
# HPC Script: Fine-tune Audio Models for EU-Emotion
# Trains Wav2Vec2, Whisper models on HPC (CPU or GPU)

set -e

# EU Emotions audio data location - On RDS
# Audio files are in: EU Emotion - UK Voices/Fixed - amplified volume
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

# Audio subdirectory
AUDIO_SUBDIR="EU Emotion - UK Voices/Fixed - amplified volume"
AUDIO_DATA_ROOT="${EU_EMOTIONS_DATA_ROOT}/${AUDIO_SUBDIR}"

if [ ! -d "$AUDIO_DATA_ROOT" ]; then
    echo "❌ Error: Audio data not found at $AUDIO_DATA_ROOT"
    echo "   Please transfer audio files to RDS first"
    exit 1
fi
echo "✅ Audio data location: $AUDIO_DATA_ROOT"

# Trial definitions (should be in project directory)
TRAIN_TRIALS="data/trial_definitions/eu_emotion_audio_train.json"
VAL_TRIALS="data/trial_definitions/eu_emotion_audio_val.json"

if [ ! -f "$TRAIN_TRIALS" ]; then
    echo "❌ Error: Train trials file not found: $TRAIN_TRIALS"
    exit 1
fi

if [ ! -f "$VAL_TRIALS" ]; then
    echo "❌ Error: Val trials file not found: $VAL_TRIALS"
    exit 1
fi

# Output directory: Use RDS to avoid /home quota issues
if [ -d "${HOME}/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007" ]; then
    RDS_USER_DIR="${HOME}/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007"
elif [ -d "/rds/user/eb2007/rds-autism-research-ePtR33Nsgi4/users/eb2007" ]; then
    RDS_USER_DIR="/rds/user/eb2007/rds-autism-research-ePtR33Nsgi4/users/eb2007"
elif [ -d "/rds-d7/project/45718/users/eb2007" ]; then
    RDS_USER_DIR="/rds-d7/project/45718/users/eb2007"
else
    RDS_USER_DIR="${HOME}/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007"
fi

OUTPUT_BASE="${RDS_USER_DIR}/mr_ts_play_results/audio_models"
mkdir -p "$OUTPUT_BASE"
echo "✅ Using RDS for model outputs: $OUTPUT_BASE"

# Training parameters (optimized for HPC)
NUM_EPOCHS=20  # Full training (was 5 for quick test)
BATCH_SIZE=8
LEARNING_RATE=1e-4
DEVICE="auto"  # Will use GPU if available, CPU otherwise

echo "=========================================="
echo "Fine-tuning Audio Models on HPC"
echo "=========================================="
echo ""
echo "Data root: $EU_EMOTIONS_DATA_ROOT"
echo "Audio subdir: $AUDIO_SUBDIR"
echo "Train trials: $TRAIN_TRIALS"
echo "Val trials: $VAL_TRIALS"
echo "Epochs: $NUM_EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Device: $DEVICE (auto-detect)"
echo ""

# Function to train an audio model
train_audio_model() {
    local model=$1
    local output_dir=$2
    local batch_size=$3
    local lr=$4
    
    echo "=========================================="
    echo "Training ${model}..."
    echo "=========================================="
    
    python experiments/eu_emotion_audio_model_comparison/training/finetune_audio_models_task_specific.py \
        --model ${model} \
        --train_trials "$TRAIN_TRIALS" \
        --val_trials "$VAL_TRIALS" \
        --data_root "$EU_EMOTIONS_DATA_ROOT" \
        --audio_subdirectory "$AUDIO_SUBDIR" \
        --output_dir "${OUTPUT_BASE}/${model}_emotion_finetuned_task_specific" \
        --num_epochs $NUM_EPOCHS \
        --batch_size $batch_size \
        --learning_rate $lr \
        --device $DEVICE
    
    if [ $? -eq 0 ]; then
        echo "✅ ${model} training completed!"
    else
        echo "❌ ${model} training failed!"
        return 1
    fi
    echo ""
}

# Train all models
train_audio_model "wav2vec2_base" "${OUTPUT_BASE}/wav2vec2_emotion_finetuned_task_specific" 8 1e-4
train_audio_model "whisper_base" "${OUTPUT_BASE}/whisper_base_emotion_finetuned_task_specific" 8 1e-4
train_audio_model "whisper_tiny" "${OUTPUT_BASE}/whisper_tiny_emotion_finetuned_task_specific" 8 1e-4

# Wav2Vec2-large (optional, may fail due to tokenizer issues)
if train_audio_model "wav2vec2_large" "${OUTPUT_BASE}/wav2vec2_large_emotion_finetuned_task_specific" 4 5e-5; then
    echo "✅ Wav2Vec2-large training completed!"
else
    echo "⚠️  Wav2Vec2-large training failed (expected - tokenizer issues)"
fi

echo "=========================================="
echo "All audio model training completed!"
echo "=========================================="
echo ""
echo "Models saved to: $OUTPUT_BASE"
