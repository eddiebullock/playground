#!/bin/bash
# Train all models sequentially (one after another)

# Configuration
DATA_ROOT="/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions"
TRAIN_TRIALS="data/trial_definitions/eu_emotion_train.json"
VAL_TRIALS="data/trial_definitions/eu_emotion_val.json"
NUM_EPOCHS=10
DEVICE="auto"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting sequential training of all models...${NC}"
echo ""

# Function to train a model
train_model() {
    local model=$1
    local output_dir=$2
    local batch_size=$3
    local lr=$4
    local log_file=$5
    
    echo -e "${GREEN}Training ${model}...${NC}"
    echo "Output: ${output_dir}"
    echo "Log: ${log_file}"
    
    python -u experiments/eu_emotion_model_comparison/training/finetune_vision_models.py \
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

# Train ResNet50
train_model "resnet50" \
    "models/resnet50_emotion_finetuned" \
    16 \
    1e-4 \
    "resnet50_training.log"

# Train ViT
train_model "vit_base" \
    "models/vit_emotion_finetuned" \
    8 \
    5e-5 \
    "vit_training.log"

# Train EfficientNet
train_model "efficientnet_b0" \
    "models/efficientnet_b0_emotion_finetuned" \
    16 \
    1e-4 \
    "efficientnet_training.log"

echo -e "${GREEN}All models trained!${NC}"
echo ""
echo "Summary:"
echo "- ResNet50: models/resnet50_emotion_finetuned/best_model.pth"
echo "- ViT: models/vit_emotion_finetuned/best_model.pth"
echo "- EfficientNet: models/efficientnet_b0_emotion_finetuned/best_model.pth"
echo ""
echo "Next step: Update config and evaluate on test set"
