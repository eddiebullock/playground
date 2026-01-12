#!/bin/bash
# Train ALL audio models with task-specific fine-tuning
#
# This script trains:
# 1. Wav2Vec2-base (task-specific)
# 2. Wav2Vec2-large (task-specific, if available)
# 3. Whisper-base (task-specific)
# 4. Whisper-tiny (task-specific)
#
# All models use:
# - Fixed actor-independent splits (no data leakage)
# - Class-weighted loss (handles imbalance)
# - Data augmentation (better generalization)
# - Task-specific 4-option forced-choice format

set -e  # Exit on error

# Configuration
DATA_ROOT="/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions"
TRAIN_TRIALS="data/trial_definitions/eu_emotion_audio_train.json"
VAL_TRIALS="data/trial_definitions/eu_emotion_audio_val.json"
NUM_EPOCHS=5
DEVICE="auto"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}============================================================${NC}"
echo -e "${YELLOW}TRAINING ALL AUDIO MODELS WITH TASK-SPECIFIC FINE-TUNING${NC}"
echo -e "${YELLOW}============================================================${NC}"
echo ""
echo -e "${BLUE}Improvements applied:${NC}"
echo "  ✅ Actor-independent splits (no data leakage)"
echo "  ✅ Class-weighted loss (handles imbalance)"
echo "  ✅ Data augmentation (better generalization)"
echo "  ✅ Task-specific 4-option forced-choice format"
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo "  Train: ${TRAIN_TRIALS}"
echo "  Val: ${VAL_TRIALS}"
echo "  Epochs: ${NUM_EPOCHS}"
echo "  Device: ${DEVICE}"
echo ""

# Function to train an audio model
train_audio_model() {
    local model=$1
    local output_dir=$2
    local batch_size=$3
    local lr=$4
    local log_file=$5
    
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}Training ${model} (task-specific with improvements)...${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo "Output: ${output_dir}"
    echo "Log: ${log_file}"
    echo "Batch size: ${batch_size}, Learning rate: ${lr}"
    echo ""
    
    python -u experiments/eu_emotion_audio_model_comparison/training/finetune_audio_models_task_specific.py \
        --model ${model} \
        --train_trials ${TRAIN_TRIALS} \
        --val_trials ${VAL_TRIALS} \
        --data_root "${DATA_ROOT}" \
        --output_dir ${output_dir} \
        --num_epochs ${NUM_EPOCHS} \
        --batch_size ${batch_size} \
        --learning_rate ${lr} \
        --device ${DEVICE} > ${log_file} 2>&1
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ ${model} training completed successfully!${NC}"
        echo "Check ${log_file} for details"
        echo ""
    else
        echo -e "${RED}❌ ${model} training failed!${NC}"
        echo "Check ${log_file} for errors"
        echo ""
        return 1
    fi
}

# Step 1: Train Wav2Vec2-base
echo -e "${BLUE}Step 1/4: Training Wav2Vec2-base...${NC}"
train_audio_model \
    "wav2vec2_base" \
    "models/wav2vec2_emotion_finetuned_task_specific" \
    8 \
    1e-4 \
    "wav2vec2_retraining.log"

# Step 2: Train Whisper-base
echo -e "${BLUE}Step 2/4: Training Whisper-base...${NC}"
train_audio_model \
    "whisper_base" \
    "models/whisper_base_emotion_finetuned_task_specific" \
    8 \
    1e-4 \
    "whisper_base_retraining.log"

# Step 3: Train Whisper-tiny
echo -e "${BLUE}Step 3/4: Training Whisper-tiny...${NC}"
train_audio_model \
    "whisper_tiny" \
    "models/whisper_tiny_emotion_finetuned_task_specific" \
    8 \
    1e-4 \
    "whisper_tiny_retraining.log"

# Step 4: Train Wav2Vec2-large (optional, may fail due to tokenizer issues)
echo -e "${BLUE}Step 4/4: Training Wav2Vec2-large (optional)...${NC}"
if train_audio_model \
    "wav2vec2_large" \
    "models/wav2vec2_large_emotion_finetuned_task_specific" \
    4 \
    5e-5 \
    "wav2vec2_large_retraining.log"; then
    echo -e "${GREEN}✅ Wav2Vec2-large training completed!${NC}"
else
    echo -e "${YELLOW}⚠️  Wav2Vec2-large training failed or skipped${NC}"
    echo "This is OK - wav2vec2_large has tokenizer issues"
    echo "Check wav2vec2_large_retraining.log for details"
fi

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}All audio model training complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
