#!/bin/bash
# HPC Script: Hyperparameter Tuning for CAM and EU-Emotion
# Runs 5 different hyperparameter configurations for each dataset
# Tests different learning rates and batch sizes to find optimal configuration

set -e

# Configuration
OUTPUT_BASE="results"
NUM_FRAMES=16  # Optimized: more frames for better temporal coverage
WEIGHT_DECAY=0.01  # Regularization
DEVICE="cuda"  # Using GPU nodes

# Check if CUDA is available, fallback to CPU if not
if ! python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo "⚠️  Warning: CUDA not available, falling back to CPU"
    DEVICE="cpu"
fi

# Project root
PROJECT_ROOT="${HOME}/mr_ts_play"
cd "$PROJECT_ROOT" || { echo "Error: Could not cd to $PROJECT_ROOT"; exit 1; }

# Detect Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

# RDS path detection
if [ -d "/rds/rds-autism-research-ePtR33Nsgi4/data/EU_emotions" ]; then
    EU_EMOTIONS_DATA_ROOT="/rds/rds-autism-research-ePtR33Nsgi4/data/EU_emotions"
elif [ -d "${HOME}/rds/rds-autism-research-ePtR33Nsgi4/data/EU_emotions" ]; then
    EU_EMOTIONS_DATA_ROOT="${HOME}/rds/rds-autism-research-ePtR33Nsgi4/data/EU_emotions"
else
    EU_EMOTIONS_DATA_ROOT="/rds/rds-autism-research-ePtR33Nsgi4/data/EU_emotions"
fi

CAM_DATA_ROOT="/home/eb2007/data/CAM"

echo "============================================================"
echo "Hyperparameter Tuning Study"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Device: $DEVICE"
echo "  Num frames: $NUM_FRAMES"
echo "  Weight decay: $WEIGHT_DECAY"
echo "  CAM data: $CAM_DATA_ROOT"
echo "  EU-Emotion data: $EU_EMOTIONS_DATA_ROOT"
echo ""
echo "Testing 5 hyperparameter configurations:"
echo "  1. Baseline: lr=1e-5, batch_size=16, epochs=20"
echo "  2. Higher LR: lr=5e-5, batch_size=16, epochs=20"
echo "  3. Highest LR: lr=1e-4, batch_size=16, epochs=20"
echo "  4. Larger batch: lr=5e-5, batch_size=32, epochs=20"
echo "  5. Conservative: lr=1e-5, batch_size=16, epochs=25"
echo ""

# Hyperparameter configurations
declare -a CONFIGS=(
    "1e-5:16:20:baseline"
    "5e-5:16:20:higher_lr"
    "1e-4:16:20:highest_lr"
    "5e-5:32:20:larger_batch"
    "1e-5:16:25:conservative"
)

# ============================================================
# CAM Hyperparameter Tuning
# ============================================================
echo "============================================================"
echo "CAM Hyperparameter Tuning"
echo "============================================================"
echo ""

# Generate CAM trials once (shared across all runs)
CAM_TRIALS_DIR="$OUTPUT_BASE/cam_replication/hp_tuning"
mkdir -p "$CAM_TRIALS_DIR"

echo "Generating CAM trials (shared across all runs)..."
$PYTHON_CMD experiments/cam_human_like/training/create_cam_trials_from_all_files.py \
    --cam-dir "$CAM_DATA_ROOT" \
    --output-dir "$CAM_TRIALS_DIR" \
    --trials-per-concept 10 \
    --min-file-size-kb 50 \
    --train-ratio 0.8 \
    --seed 42

CAM_TRAIN_TRIALS="$CAM_TRIALS_DIR/cam_trial_definitions_train_all_files.json"
CAM_TEST_TRIALS="$CAM_TRIALS_DIR/cam_trial_definitions_test_all_files.json"

if [ ! -f "$CAM_TRAIN_TRIALS" ] || [ ! -f "$CAM_TEST_TRIALS" ]; then
    echo "Error: Failed to generate CAM trials"
    exit 1
fi

# Run each configuration
for config in "${CONFIGS[@]}"; do
    IFS=':' read -r lr batch_size epochs config_name <<< "$config"
    
    echo ""
    echo "============================================================"
    echo "CAM Run: $config_name"
    echo "  Learning rate: $lr"
    echo "  Batch size: $batch_size"
    echo "  Epochs: $epochs"
    echo "============================================================"
    
    RUN_DIR="$OUTPUT_BASE/cam_replication/hp_tuning/run_${config_name}"
    mkdir -p "$RUN_DIR/model_checkpoints"
    
    # Fine-tune
    $PYTHON_CMD experiments/cam_human_like/training/finetune_clip_emotions.py \
        --task_specific \
        --dataset_type cam \
        --train_trials "$CAM_TRAIN_TRIALS" \
        --val_trials "$CAM_TEST_TRIALS" \
        --data_root "$CAM_DATA_ROOT" \
        --output_dir "$RUN_DIR/model_checkpoints" \
        --num_epochs $epochs \
        --batch_size $batch_size \
        --learning_rate $lr \
        --weight_decay $WEIGHT_DECAY \
        --device $DEVICE \
        --num_frames $NUM_FRAMES \
        --use_lr_scheduler \
        --warmup_steps 100
    
    # Evaluate
    $PYTHON_CMD experiments/cam_human_like/training/evaluate_on_cam.py \
        --model_path "$RUN_DIR/model_checkpoints/best_model" \
        --trial_definitions "$CAM_TEST_TRIALS" \
        --data_root "$CAM_DATA_ROOT" \
        --dataset_type cam \
        --split test \
        --device $DEVICE \
        --num_frames $NUM_FRAMES \
        --use_multiframe
    
    echo "✅ CAM run $config_name complete!"
done

# ============================================================
# EU-Emotion Hyperparameter Tuning
# ============================================================
echo ""
echo "============================================================"
echo "EU-Emotion Hyperparameter Tuning"
echo "============================================================"
echo ""

# Generate EU-Emotion trials once (shared across all runs)
EU_TRIALS_DIR="$OUTPUT_BASE/eu_emotion_replication/hp_tuning"
mkdir -p "$EU_TRIALS_DIR"

echo "Generating EU-Emotion trials (shared across all runs)..."
$PYTHON_CMD experiments/cam_human_like/training/create_eu_emotion_trials.py \
    --eu-emotion-dir "$EU_EMOTIONS_DATA_ROOT" \
    --output-dir "$EU_TRIALS_DIR" \
    --modality face \
    --trials-per-emotion 10 \
    --min-stimuli-per-emotion 3 \
    --train-ratio 0.8 \
    --seed 42

EU_TRAIN_TRIALS="$EU_TRIALS_DIR/eu_emotion_trial_definitions_train.json"
EU_TEST_TRIALS="$EU_TRIALS_DIR/eu_emotion_trial_definitions_test.json"

if [ ! -f "$EU_TRAIN_TRIALS" ] || [ ! -f "$EU_TEST_TRIALS" ]; then
    echo "Error: Failed to generate EU-Emotion trial definitions"
    exit 1
fi

# Run each configuration
for config in "${CONFIGS[@]}"; do
    IFS=':' read -r lr batch_size epochs config_name <<< "$config"
    
    echo ""
    echo "============================================================"
    echo "EU-Emotion Run: $config_name"
    echo "  Learning rate: $lr"
    echo "  Batch size: $batch_size"
    echo "  Epochs: $epochs"
    echo "============================================================"
    
    RUN_DIR="$OUTPUT_BASE/eu_emotion_replication/hp_tuning/run_${config_name}"
    mkdir -p "$RUN_DIR/model_checkpoints"
    
    # Fine-tune
    $PYTHON_CMD experiments/cam_human_like/training/finetune_clip_emotions.py \
        --task_specific \
        --dataset_type eu_emotion \
        --train_trials "$EU_TRAIN_TRIALS" \
        --val_trials "$EU_TEST_TRIALS" \
        --data_root "$EU_EMOTIONS_DATA_ROOT" \
        --output_dir "$RUN_DIR/model_checkpoints" \
        --num_epochs $epochs \
        --batch_size $batch_size \
        --learning_rate $lr \
        --weight_decay $WEIGHT_DECAY \
        --device $DEVICE \
        --num_frames $NUM_FRAMES \
        --use_lr_scheduler \
        --warmup_steps 100
    
    # Evaluate
    $PYTHON_CMD experiments/cam_human_like/training/evaluate_on_cam.py \
        --model_path "$RUN_DIR/model_checkpoints/best_model" \
        --trial_definitions "$EU_TEST_TRIALS" \
        --data_root "$EU_EMOTIONS_DATA_ROOT" \
        --dataset_type eu_emotion \
        --split test \
        --device $DEVICE \
        --num_frames $NUM_FRAMES \
        --use_multiframe
    
    echo "✅ EU-Emotion run $config_name complete!"
done

# ============================================================
# Generate Summary Report
# ============================================================
echo ""
echo "============================================================"
echo "Generating Hyperparameter Tuning Summary"
echo "============================================================"
echo ""

SUMMARY_FILE="$OUTPUT_BASE/hp_tuning_summary.json"
$PYTHON_CMD << EOF
import json
import glob
from pathlib import Path

results = {
    "cam": {},
    "eu_emotion": {}
}

# Collect CAM results
for eval_file in glob.glob("$OUTPUT_BASE/cam_replication/hp_tuning/run_*/model_checkpoints/cam_evaluation_test.json"):
    run_name = Path(eval_file).parent.parent.name.replace("run_", "")
    with open(eval_file) as f:
        data = json.load(f)
        results["cam"][run_name] = {
            "accuracy": data["metrics"]["accuracy"],
            "face_accuracy": data["metrics"].get("face_accuracy", 0),
            "voice_accuracy": data["metrics"].get("voice_accuracy", 0),
            "num_trials": data["num_valid_trials"]
        }

# Collect EU-Emotion results
for eval_file in glob.glob("$OUTPUT_BASE/eu_emotion_replication/hp_tuning/run_*/model_checkpoints/eu_emotion_evaluation_test.json"):
    run_name = Path(eval_file).parent.parent.name.replace("run_", "")
    with open(eval_file) as f:
        data = json.load(f)
        results["eu_emotion"][run_name] = {
            "accuracy": data["metrics"]["accuracy"],
            "num_trials": data["num_valid_trials"]
        }

# Find best configurations
best_cam = max(results["cam"].items(), key=lambda x: x[1]["accuracy"])
best_eu = max(results["eu_emotion"].items(), key=lambda x: x[1]["accuracy"])

summary = {
    "results": results,
    "best_configurations": {
        "cam": {
            "config": best_cam[0],
            "accuracy": best_cam[1]["accuracy"]
        },
        "eu_emotion": {
            "config": best_eu[0],
            "accuracy": best_eu[1]["accuracy"]
        }
    }
}

with open("$SUMMARY_FILE", "w") as f:
    json.dump(summary, f, indent=2)

print("Summary saved to: $SUMMARY_FILE")
print(f"\nBest CAM configuration: {best_cam[0]} (accuracy: {best_cam[1]['accuracy']:.2%})")
print(f"Best EU-Emotion configuration: {best_eu[0]} (accuracy: {best_eu[1]['accuracy']:.2%})")
EOF

echo ""
echo "============================================================"
echo "Hyperparameter Tuning Complete!"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  CAM: $OUTPUT_BASE/cam_replication/hp_tuning/"
echo "  EU-Emotion: $OUTPUT_BASE/eu_emotion_replication/hp_tuning/"
echo "  Summary: $SUMMARY_FILE"
echo ""

