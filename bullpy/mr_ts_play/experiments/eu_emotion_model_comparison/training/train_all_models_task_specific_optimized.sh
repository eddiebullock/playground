#!/bin/bash
# OPTIMIZED: Train all models with TASK-SPECIFIC approach (faster settings)

# Configuration - OPTIMIZED for speed
DATA_ROOT="/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions"
TRAIN_TRIALS="data/trial_definitions/eu_emotion_train.json"
VAL_TRIALS="data/trial_definitions/eu_emotion_val.json"
NUM_EPOCHS=10  # Reduced from 12
NUM_FRAMES=4   # Reduced from 8 (faster frame extraction)
DEVICE="auto"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting OPTIMIZED TASK-SPECIFIC fine-tuning...${NC}"
echo -e "${BLUE}Settings: ${NUM_EPOCHS} epochs, ${NUM_FRAMES} frames/video, smaller batch sizes${NC}"
echo ""

# Function to train a model
train_model() {
    local model=$1
    local output_dir=$2
    local batch_size=$3
    local lr=$4
    local log_file=$5
    
    echo -e "${GREEN}Training ${model} (task-specific, optimized)...${NC}"
    echo "Output: ${output_dir}"
    echo "Log: ${log_file}"
    echo "Batch size: ${batch_size}, Frames: ${NUM_FRAMES}"
    
    python -u experiments/eu_emotion_model_comparison/training/finetune_vision_models_task_specific.py \
        --model ${model} \
        --train_trials ${TRAIN_TRIALS} \
        --val_trials ${VAL_TRIALS} \
        --data_root "${DATA_ROOT}" \
        --output_dir ${output_dir} \
        --num_epochs ${NUM_EPOCHS} \
        --batch_size ${batch_size} \
        --learning_rate ${lr} \
        --num_frames ${NUM_FRAMES} \
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

# Train ResNet50 (optimized: batch 4, 4 frames)
train_model "resnet50" \
    "models/resnet50_emotion_finetuned_task_specific" \
    4 \
    1e-4 \
    "resnet50_task_specific_optimized.log" || exit 1

# Train ViT (optimized: batch 2, 4 frames - ViT is slower)
train_model "vit_base" \
    "models/vit_emotion_finetuned_task_specific" \
    2 \
    5e-5 \
    "vit_task_specific_optimized.log" || exit 1

# Train EfficientNet (optimized: batch 4, 4 frames)
train_model "efficientnet_b0" \
    "models/efficientnet_b0_emotion_finetuned_task_specific" \
    4 \
    1e-4 \
    "efficientnet_task_specific_optimized.log" || exit 1

echo -e "${GREEN}All models trained with optimized task-specific approach!${NC}"
echo ""
echo "Summary:"
echo "- ResNet50: models/resnet50_emotion_finetuned_task_specific/best_model.pth"
echo "- ViT: models/vit_emotion_finetuned_task_specific/best_model.pth"
echo "- EfficientNet: models/efficientnet_b0_emotion_finetuned_task_specific/best_model.pth"
echo ""
echo "Next step: Update config and evaluate on test set"
