#!/bin/bash
# HPC Test Script: CAM Replication Study - QUICK TEST VERSION
# Quick test: 2 epochs, reduced trials, 1-2 hours runtime
# Use this to verify the pipeline works before running full replication

set -e

# CAM data location
CAM_DATA_ROOT="/home/eb2007/data/CAM"
if [ ! -d "$CAM_DATA_ROOT" ]; then
    echo "❌ Error: CAM data not found at $CAM_DATA_ROOT"
    exit 1
fi
echo "✅ CAM data location: $CAM_DATA_ROOT"

OUTPUT_BASE="results"
NUM_EPOCHS=2  # Quick test: 2 epochs
BATCH_SIZE=4
LEARNING_RATE=1e-5
NUM_FRAMES=8
DEVICE="cpu"

# Project root
PROJECT_ROOT="${HOME}/mr_ts_play"
cd "$PROJECT_ROOT" || { echo "Error: Could not cd to $PROJECT_ROOT"; exit 1; }

# Detect Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

echo "============================================================"
echo "CAM Replication Study - QUICK TEST"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  CAM data root: $CAM_DATA_ROOT"
echo "  Output directory: $OUTPUT_BASE"
echo "  Device: $DEVICE"
echo "  Epochs: $NUM_EPOCHS (QUICK TEST)"
echo "  Batch size: $BATCH_SIZE"
echo ""

# Create output directories
mkdir -p "$OUTPUT_BASE/cam_test"

# ============================================================
# Step 1: Generate CAM trials from ALL available files
# ============================================================
echo "============================================================"
echo "Step 1: Generating CAM trials from ALL available files..."
echo "============================================================"
echo ""

# Generate trials (reduced for quick test: 5 per concept instead of 10)
$PYTHON_CMD experiments/cam_human_like/training/create_cam_trials_from_all_files.py \
    --cam-dir "$CAM_DATA_ROOT" \
    --output-dir "$OUTPUT_BASE/cam_test" \
    --trials-per-concept 5 \
    --min-file-size-kb 50 \
    --train-ratio 0.8 \
    --seed 42

CAM_TRAIN_TRIALS="$OUTPUT_BASE/cam_test/cam_trial_definitions_train_all_files.json"
CAM_TEST_TRIALS="$OUTPUT_BASE/cam_test/cam_trial_definitions_test_all_files.json"

if [ ! -f "$CAM_TRAIN_TRIALS" ] || [ ! -f "$CAM_TEST_TRIALS" ]; then
    echo "Error: Failed to generate CAM trials"
    exit 1
fi

echo "CAM trials generated successfully"
echo "  Train: $CAM_TRAIN_TRIALS"
echo "  Test: $CAM_TEST_TRIALS"
echo ""

# ============================================================
# Step 2: Fine-tune on CAM (QUICK TEST)
# ============================================================
echo "============================================================"
echo "Step 2: Fine-tuning CLIP on CAM (QUICK TEST - 2 epochs)..."
echo "============================================================"
echo ""

$PYTHON_CMD experiments/cam_human_like/training/finetune_clip_emotions.py \
    --task_specific \
    --dataset_type cam \
    --train_trials "$CAM_TRAIN_TRIALS" \
    --val_trials "$CAM_TEST_TRIALS" \
    --data_root "$CAM_DATA_ROOT" \
    --output_dir "$OUTPUT_BASE/cam_test/model_checkpoints" \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --device $DEVICE \
    --num_frames $NUM_FRAMES

CAM_MODEL_PATH="$OUTPUT_BASE/cam_test/model_checkpoints/best_model"

if [ ! -d "$CAM_MODEL_PATH" ]; then
    echo "Error: CAM fine-tuning failed"
    exit 1
fi

echo "CAM fine-tuning complete!"
echo ""

# ============================================================
# Step 3: Evaluate on CAM test set
# ============================================================
echo "============================================================"
echo "Step 3: Evaluating CAM model on test set..."
echo "============================================================"
echo ""

$PYTHON_CMD experiments/cam_human_like/training/evaluate_on_cam.py \
    --model_path "$CAM_MODEL_PATH" \
    --trial_definitions "$CAM_TEST_TRIALS" \
    --data_root "$CAM_DATA_ROOT" \
    --dataset_type cam \
    --split test \
    --device $DEVICE \
    --num_frames $NUM_FRAMES \
    --use_multiframe

echo "CAM evaluation complete!"
echo ""

echo "============================================================"
echo "CAM Test Complete!"
echo "============================================================"
echo ""
echo "Results saved to: $OUTPUT_BASE/cam_test/"
echo "  Model: $CAM_MODEL_PATH"
echo "  Evaluation: $OUTPUT_BASE/cam_test/model_checkpoints/cam_evaluation_test.json"
echo ""
echo "If this test succeeds, you can run the full replication:"
echo "  sbatch experiments/cam_human_like/training/hpc_cam_replication.slurm"
echo ""
