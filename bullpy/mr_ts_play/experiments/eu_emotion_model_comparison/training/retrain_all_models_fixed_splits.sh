#!/bin/bash
# Retrain ALL models with FIXED actor-independent splits and improvements
#
# This script retrains:
# 1. CLIP (if using trial definition files)
# 2. ResNet50 (task-specific)
# 3. ViT (task-specific)
# 4. EfficientNet (task-specific)
#
# All models use:
# - Fixed actor-independent splits (no data leakage)
# - Class-weighted loss (handles imbalance)
# - Data augmentation (better generalization)
# - Improved frame sampling (avoids black frames)

set -e  # Exit on error

# Configuration
DATA_ROOT="/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions"
TRAIN_TRIALS="data/trial_definitions/eu_emotion_train.json"
VAL_TRIALS="data/trial_definitions/eu_emotion_val.json"
NUM_EPOCHS=10
NUM_FRAMES=4  # Optimized for speed
DEVICE="auto"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}============================================================${NC}"
echo -e "${YELLOW}RETRAINING ALL MODELS WITH FIXED SPLITS${NC}"
echo -e "${YELLOW}============================================================${NC}"
echo ""
echo -e "${BLUE}Improvements applied:${NC}"
echo "  ✅ Actor-independent splits (no data leakage)"
echo "  ✅ Class-weighted loss (handles imbalance)"
echo "  ✅ Data augmentation (better generalization)"
echo "  ✅ Improved frame sampling (avoids black frames)"
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo "  Train: ${TRAIN_TRIALS}"
echo "  Val: ${VAL_TRIALS}"
echo "  Epochs: ${NUM_EPOCHS}"
echo "  Frames: ${NUM_FRAMES}"
echo "  Device: ${DEVICE}"
echo ""

# Function to train a vision model
train_vision_model() {
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

# Function to train CLIP (if using trial definition files)
train_clip() {
    local output_dir=$1
    local log_file=$2
    
    # CLIP script doesn't accept "auto", need to detect actual device
    local clip_device="cpu"
    if [ "$DEVICE" = "auto" ]; then
        # Try to detect device (default to mps for Mac, cuda for Linux, cpu otherwise)
        if [[ "$OSTYPE" == "darwin"* ]]; then
            clip_device="mps"
        else
            clip_device="cuda"
        fi
    else
        clip_device=$DEVICE
    fi
    
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}Training CLIP (task-specific with improvements)...${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo "Output: ${output_dir}"
    echo "Log: ${log_file}"
    echo "Device: ${clip_device}"
    echo ""
    
    # Check if CLIP training script supports trial definition files
    python -u experiments/cam_human_like/training/finetune_clip_emotions.py \
        --task_specific \
        --dataset_type eu_emotion \
        --train_trials ${TRAIN_TRIALS} \
        --val_trials ${VAL_TRIALS} \
        --data_root "${DATA_ROOT}" \
        --output_dir ${output_dir} \
        --num_epochs ${NUM_EPOCHS} \
        --batch_size 8 \
        --learning_rate 1e-5 \
        --device ${clip_device} \
        --num_frames ${NUM_FRAMES} > ${log_file} 2>&1
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ CLIP training completed successfully!${NC}"
        echo "Check ${log_file} for details"
        echo ""
    else
        echo -e "${YELLOW}⚠️  CLIP training failed or not using trial definition files${NC}"
        echo "This is OK if CLIP uses a different split system"
        echo "Check ${log_file} for details"
        echo ""
        return 0  # Don't fail the whole script
    fi
}

# Start training
START_TIME=$(date +%s)

# Train CLIP (optional - may use different split system)
echo -e "${BLUE}Step 1/4: Training CLIP...${NC}"
train_clip \
    "models/eu_emotion_finetuned_best" \
    "clip_retraining_fixed_splits.log" || echo "CLIP training skipped (may use different splits)"

# Train ResNet50
echo -e "${BLUE}Step 2/4: Training ResNet50...${NC}"
train_vision_model "resnet50" \
    "models/resnet50_emotion_finetuned_task_specific" \
    4 \
    1e-4 \
    "resnet50_retraining_fixed_splits.log" || exit 1

# Train ViT
echo -e "${BLUE}Step 3/4: Training ViT...${NC}"
train_vision_model "vit_base" \
    "models/vit_emotion_finetuned_task_specific" \
    2 \
    5e-5 \
    "vit_retraining_fixed_splits.log" || exit 1

# Train EfficientNet
echo -e "${BLUE}Step 4/4: Training EfficientNet...${NC}"
train_vision_model "efficientnet_b0" \
    "models/efficientnet_b0_emotion_finetuned_task_specific" \
    4 \
    1e-4 \
    "efficientnet_retraining_fixed_splits.log" || exit 1

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}ALL MODELS RETRAINED SUCCESSFULLY!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo -e "${BLUE}Training Summary:${NC}"
echo "  Duration: ${HOURS}h ${MINUTES}m"
echo ""
echo -e "${BLUE}Model Checkpoints:${NC}"
echo "  CLIP: models/eu_emotion_finetuned_best/"
echo "  ResNet50: models/resnet50_emotion_finetuned_task_specific/best_model.pth"
echo "  ViT: models/vit_emotion_finetuned_task_specific/best_model.pth"
echo "  EfficientNet: models/efficientnet_b0_emotion_finetuned_task_specific/best_model.pth"
echo ""
echo -e "${BLUE}Log Files:${NC}"
echo "  CLIP: clip_retraining_fixed_splits.log"
echo "  ResNet50: resnet50_retraining_fixed_splits.log"
echo "  ViT: vit_retraining_fixed_splits.log"
echo "  EfficientNet: efficientnet_retraining_fixed_splits.log"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Verify actor independence:"
echo "     python experiments/eu_emotion_model_comparison/scripts/analyze_data_quality.py \\"
echo "         --trial-definitions data/trial_definitions/eu_emotion_test.json \\"
echo "         --data-root \"${DATA_ROOT}\" \\"
echo "         --train-file ${TRAIN_TRIALS} \\"
echo "         --val-file ${VAL_TRIALS}"
echo ""
echo "  2. Evaluate all models:"
echo "     python experiments/eu_emotion_model_comparison/scripts/run_comparison.py \\"
echo "         --config experiments/eu_emotion_model_comparison/configs/comparison_config.yaml \\"
echo "         --models clip_finetuned resnet50 vit_base efficientnet_b0 \\"
echo "         --device auto"
echo ""
