#!/bin/bash
# Transfer fine-tuned models from HPC to local machine
# Usage: ./transfer_models_from_hpc.sh [cam|eu|both]

set -e

# Configuration
HPC_HOST="${HPC_HOST:-eb2007@login.hpc.cam.ac.uk}"
HPC_BASE="~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/mr_ts_play_results"
LOCAL_MODELS_DIR="models"
LOCAL_RESULTS_DIR="results"

# Create local directories
mkdir -p "$LOCAL_MODELS_DIR"
mkdir -p "$LOCAL_RESULTS_DIR"

# Determine what to transfer
TRANSFER_WHAT="${1:-both}"

echo "============================================================"
echo "Transferring Models from HPC to Local"
echo "============================================================"
echo "HPC Host: $HPC_HOST"
echo "Transfer: $TRANSFER_WHAT"
echo ""

# Function to transfer a model
transfer_model() {
    local dataset=$1
    local hpc_path="${HPC_BASE}/${dataset}_replication/model_checkpoints/best_model"
    local local_path="${LOCAL_MODELS_DIR}/${dataset}_finetuned_best"
    
    echo "Transferring ${dataset} model..."
    echo "  From: ${HPC_HOST}:${hpc_path}"
    echo "  To: ${local_path}"
    
    rsync -avz --progress \
        "${HPC_HOST}:${hpc_path}/" \
        "${local_path}/"
    
    if [ $? -eq 0 ]; then
        echo "✅ ${dataset} model transferred successfully"
        echo "   Size: $(du -sh ${local_path} | cut -f1)"
    else
        echo "❌ Failed to transfer ${dataset} model"
        return 1
    fi
    echo ""
}

# Function to transfer evaluation results
transfer_evaluation() {
    local dataset=$1
    local hpc_path="${HPC_BASE}/${dataset}_replication/model_checkpoints"
    local local_file="${LOCAL_RESULTS_DIR}/${dataset}_evaluation_test.json"
    
    echo "Transferring ${dataset} evaluation results..."
    rsync -avz --progress \
        "${HPC_HOST}:${hpc_path}/${dataset}_evaluation_test.json" \
        "${local_file}" 2>/dev/null || echo "⚠️  Evaluation file not found (may not exist yet)"
    echo ""
}

# Transfer based on argument
case "$TRANSFER_WHAT" in
    cam)
        transfer_model "cam"
        transfer_evaluation "cam"
        ;;
    eu|eu_emotion)
        transfer_model "eu_emotion"
        transfer_evaluation "eu_emotion"
        ;;
    both|*)
        transfer_model "cam"
        transfer_evaluation "cam"
        transfer_model "eu_emotion"
        transfer_evaluation "eu_emotion"
        ;;
esac

echo "============================================================"
echo "Transfer Complete!"
echo "============================================================"
echo ""
echo "Models are now available at:"
if [ "$TRANSFER_WHAT" = "cam" ] || [ "$TRANSFER_WHAT" = "both" ]; then
    echo "  CAM: ${LOCAL_MODELS_DIR}/cam_finetuned_best/"
fi
if [ "$TRANSFER_WHAT" = "eu" ] || [ "$TRANSFER_WHAT" = "eu_emotion" ] || [ "$TRANSFER_WHAT" = "both" ]; then
    echo "  EU-Emotion: ${LOCAL_MODELS_DIR}/eu_emotion_finetuned_best/"
fi
echo ""
echo "To use the models in Python:"
echo "  from transformers import CLIPModel, CLIPProcessor"
echo "  model = CLIPModel.from_pretrained('${LOCAL_MODELS_DIR}/cam_finetuned_best')"
echo "  processor = CLIPProcessor.from_pretrained('${LOCAL_MODELS_DIR}/cam_finetuned_best')"


