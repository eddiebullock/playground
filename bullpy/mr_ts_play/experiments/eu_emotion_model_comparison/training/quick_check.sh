#!/bin/bash
# Quick pre-flight check for video fine-tuning on HPC
# Run this on HPC before submitting the job

set -e

echo "=========================================="
echo "Pre-Flight Check for Video Fine-Tuning"
echo "=========================================="
echo ""

cd ~/mr_ts_play

# Check 1: Files exist
echo "1. Checking files..."
if [ ! -f "experiments/eu_emotion_model_comparison/training/hpc_finetune_video_models.sh" ]; then
    echo "❌ Missing: hpc_finetune_video_models.sh"
    exit 1
fi
if [ ! -f "experiments/eu_emotion_model_comparison/training/hpc_finetune_video_models.slurm" ]; then
    echo "❌ Missing: hpc_finetune_video_models.slurm"
    exit 1
fi
if [ ! -f "experiments/eu_emotion_model_comparison/training/finetune_video_models_task_specific.py" ]; then
    echo "❌ Missing: finetune_video_models_task_specific.py"
    exit 1
fi
echo "✅ All required files present"

# Check 2: Data path
echo ""
echo "2. Checking data path..."
DATA_ROOT="${HOME}/rds/rds-autism-research-ePtR33Nsgi4/data/EU_emotions"
if [ ! -d "$DATA_ROOT" ]; then
    echo "❌ Data not found at: $DATA_ROOT"
    echo "   Please check the path in hpc_finetune_video_models.sh"
    exit 1
fi
echo "✅ Data found at: $DATA_ROOT"

# Check 3: Trial definitions
echo ""
echo "3. Checking trial definitions..."
if [ ! -f "data/trial_definitions/eu_emotion_train.json" ]; then
    echo "❌ Missing: data/trial_definitions/eu_emotion_train.json"
    exit 1
fi
if [ ! -f "data/trial_definitions/eu_emotion_val.json" ]; then
    echo "❌ Missing: data/trial_definitions/eu_emotion_val.json"
    exit 1
fi
echo "✅ Trial definitions present"

# Check 4: Python environment
echo ""
echo "4. Checking Python environment..."
if [ ! -f "venv/bin/activate" ]; then
    echo "❌ Virtual environment not found"
    exit 1
fi
source venv/bin/activate

if ! python -c "import torch" 2>/dev/null; then
    echo "❌ PyTorch not installed"
    exit 1
fi
echo "✅ PyTorch installed"

# Check 5: Required packages
echo ""
echo "5. Checking required packages..."
if ! python -c "import pytorchvideo" 2>/dev/null; then
    echo "⚠️  pytorchvideo not installed (needed for I3D)"
    echo "   Install with: pip install pytorchvideo"
else
    echo "✅ pytorchvideo installed"
fi

if ! python -c "from transformers import TimesformerForVideoClassification" 2>/dev/null; then
    echo "⚠️  transformers may not support TimeSformer"
    echo "   Check version: pip show transformers"
else
    echo "✅ transformers supports TimeSformer"
fi

# Check 6: Script syntax
echo ""
echo "6. Checking script syntax..."
if ! bash -n experiments/eu_emotion_model_comparison/training/hpc_finetune_video_models.sh; then
    echo "❌ Syntax error in hpc_finetune_video_models.sh"
    exit 1
fi
echo "✅ Script syntax OK"

# Check 7: Data path in script matches
echo ""
echo "7. Verifying data path in script..."
SCRIPT_DATA_ROOT=$(grep "^DATA_ROOT=" experiments/eu_emotion_model_comparison/training/hpc_finetune_video_models.sh | cut -d'"' -f2)
if [ "$SCRIPT_DATA_ROOT" != "$DATA_ROOT" ]; then
    echo "⚠️  Data path in script doesn't match actual location"
    echo "   Script has: $SCRIPT_DATA_ROOT"
    echo "   Actual is:  $DATA_ROOT"
    echo "   Consider updating the script"
else
    echo "✅ Data path matches"
fi

echo ""
echo "=========================================="
echo "✅ All checks passed!"
echo "=========================================="
echo ""
echo "Ready to submit job:"
echo "  sbatch experiments/eu_emotion_model_comparison/training/hpc_finetune_video_models.slurm"
echo ""
