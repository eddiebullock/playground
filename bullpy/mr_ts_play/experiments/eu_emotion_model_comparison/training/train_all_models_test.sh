#!/bin/bash
# TEST VERSION: Train all models with 1 epoch each (quick test)

# Configuration
DATA_ROOT="/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions"
TRAIN_TRIALS="data/trial_definitions/eu_emotion_train.json"
VAL_TRIALS="data/trial_definitions/eu_emotion_val.json"
NUM_EPOCHS=1  # TEST: Only 1 epoch
DEVICE="auto"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}TEST MODE: Training all models with 1 epoch each (quick test)${NC}"
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
    "models/resnet50_emotion_finetuned_test" \
    16 \
    1e-4 \
    "resnet50_test.log"

# Train ViT
train_model "vit_base" \
    "models/vit_emotion_finetuned_test" \
    8 \
    5e-5 \
    "vit_test.log"

# Train EfficientNet
train_model "efficientnet_b0" \
    "models/efficientnet_b0_emotion_finetuned_test" \
    16 \
    1e-4 \
    "efficientnet_test.log"

echo -e "${GREEN}Test completed!${NC}"
echo ""
echo "If all models trained successfully, run the full training:"
echo "  bash experiments/eu_emotion_model_comparison/training/train_all_models.sh"
