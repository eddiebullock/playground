#!/bin/bash
# Orchestration script for dual replication: EU-Emotion and CAM
# Replicates Golan/CAM study computationally for both datasets separately

set -e  # Exit on error

# Configuration
EU_EMOTION_DIR="/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions"
CAM_DATA_ROOT="/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/mindreading_transporter_files/Mindreading emotions library/Emotions"
CAM_TRIAL_DEFINITIONS="data/cam_trial_definitions_20concepts.json"
OUTPUT_BASE="results"
NUM_EPOCHS=5  # Increased from 2 to 5 for better training
BATCH_SIZE=8
LEARNING_RATE=1e-5
NUM_FRAMES=8
DEVICE="mps"  # Will auto-detect if not available

# Detect device
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

# Get device
if [ -f "venv/bin/python3" ]; then
    DEVICE_DETECTED=$($PYTHON_CMD -c "import torch; print('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')" 2>/dev/null || echo "cpu")
    DEVICE=$DEVICE_DETECTED
fi

echo "============================================================"
echo "Dual Replication: EU-Emotion and CAM"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  EU-Emotion dir: $EU_EMOTION_DIR"
echo "  CAM data root: $CAM_DATA_ROOT"
echo "  CAM trial definitions: $CAM_TRIAL_DEFINITIONS"
echo "  Output directory: $OUTPUT_BASE"
echo "  Device: $DEVICE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Num frames: $NUM_FRAMES"
echo ""

# Create output directories
mkdir -p "$OUTPUT_BASE/eu_emotion_replication"
mkdir -p "$OUTPUT_BASE/cam_replication"

# ============================================================
# Step 1: EU-Emotion Replication
# ============================================================
echo "============================================================"
echo "Step 1: EU-Emotion Replication"
echo "============================================================"
echo ""

# Generate EU-Emotion trial definitions
echo "Generating EU-Emotion forced-choice trials..."
$PYTHON_CMD experiments/cam_human_like/training/create_eu_emotion_trials.py \
    --eu-emotion-dir "$EU_EMOTION_DIR" \
    --output-dir "$OUTPUT_BASE/eu_emotion_replication" \
    --modality face \
    --trials-per-emotion 10 \
    --min-stimuli-per-emotion 3 \
    --train-ratio 0.8 \
    --seed 42

EU_TRAIN_TRIALS="$OUTPUT_BASE/eu_emotion_replication/eu_emotion_trial_definitions_train.json"
EU_TEST_TRIALS="$OUTPUT_BASE/eu_emotion_replication/eu_emotion_trial_definitions_test.json"

if [ ! -f "$EU_TRAIN_TRIALS" ] || [ ! -f "$EU_TEST_TRIALS" ]; then
    echo "Error: Failed to generate EU-Emotion trial definitions"
    exit 1
fi

echo "EU-Emotion trials generated successfully"
echo ""

# Fine-tune on EU-Emotion
echo "Fine-tuning CLIP on EU-Emotion (task-specific, 4-option forced-choice)..."
echo "This will take approximately 50-100 minutes for 5 epochs on MPS..."

$PYTHON_CMD experiments/cam_human_like/training/finetune_clip_emotions.py \
    --task_specific \
    --dataset_type eu_emotion \
    --train_trials "$EU_TRAIN_TRIALS" \
    --val_trials "$EU_TEST_TRIALS" \
    --data_root "$EU_EMOTION_DIR" \
    --output_dir "$OUTPUT_BASE/eu_emotion_replication/model_checkpoints" \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --device $DEVICE \
    --num_frames $NUM_FRAMES

EU_MODEL_PATH="$OUTPUT_BASE/eu_emotion_replication/model_checkpoints/best_model"

if [ ! -d "$EU_MODEL_PATH" ]; then
    echo "Error: EU-Emotion fine-tuning failed"
    exit 1
fi

echo "EU-Emotion fine-tuning complete!"
echo ""

# Evaluate on EU-Emotion test set
echo "Evaluating EU-Emotion model on test set..."
$PYTHON_CMD experiments/cam_human_like/training/evaluate_on_cam.py \
    --model_path "$EU_MODEL_PATH" \
    --trial_definitions "$EU_TEST_TRIALS" \
    --data_root "$EU_EMOTION_DIR" \
    --dataset_type eu_emotion \
    --split test \
    --device $DEVICE \
    --num_frames $NUM_FRAMES \
    --use_multiframe

echo "EU-Emotion evaluation complete!"
echo ""

# ============================================================
# Step 2: CAM Replication
# ============================================================
echo "============================================================"
echo "Step 2: CAM Replication"
echo "============================================================"
echo ""

# Create CAM train/test splits
echo "Creating CAM train/test splits..."
$PYTHON_CMD experiments/cam_human_like/training/create_cam_splits.py \
    --trial-definitions "$CAM_TRIAL_DEFINITIONS" \
    --output-dir "$OUTPUT_BASE/cam_replication" \
    --split-method concept_balanced \
    --train-ratio 0.8 \
    --seed 42

CAM_TRAIN_TRIALS="$OUTPUT_BASE/cam_replication/train_trials.json"
CAM_TEST_TRIALS="$OUTPUT_BASE/cam_replication/test_trials.json"

if [ ! -f "$CAM_TRAIN_TRIALS" ] || [ ! -f "$CAM_TEST_TRIALS" ]; then
    echo "Error: Failed to create CAM splits"
    exit 1
fi

echo "CAM splits created successfully"
echo ""

# Fine-tune on CAM
echo "Fine-tuning CLIP on CAM (task-specific, 4-option forced-choice)..."
echo "This will take approximately 50-100 minutes for 5 epochs on MPS..."

$PYTHON_CMD experiments/cam_human_like/training/finetune_clip_emotions.py \
    --task_specific \
    --dataset_type cam \
    --train_trials "$CAM_TRAIN_TRIALS" \
    --val_trials "$CAM_TEST_TRIALS" \
    --data_root "$CAM_DATA_ROOT" \
    --output_dir "$OUTPUT_BASE/cam_replication/model_checkpoints" \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --device $DEVICE \
    --num_frames $NUM_FRAMES

CAM_MODEL_PATH="$OUTPUT_BASE/cam_replication/model_checkpoints/best_model"

if [ ! -d "$CAM_MODEL_PATH" ]; then
    echo "Error: CAM fine-tuning failed"
    exit 1
fi

echo "CAM fine-tuning complete!"
echo ""

# Evaluate on CAM test set
echo "Evaluating CAM model on test set..."
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

# ============================================================
# Step 3: Generate Comparison Report
# ============================================================
echo "============================================================"
echo "Step 3: Generating Comparison Report"
echo "============================================================"
echo ""

$PYTHON_CMD << EOF
import json
from pathlib import Path

eu_results_file = Path("$OUTPUT_BASE/eu_emotion_replication/model_checkpoints/eu_emotion_evaluation_test.json")
cam_results_file = Path("$OUTPUT_BASE/cam_replication/model_checkpoints/cam_evaluation_test.json")

# Try to find results files (they might have different names)
if not eu_results_file.exists():
    # Look for any evaluation JSON in the directory
    eu_dir = Path("$OUTPUT_BASE/eu_emotion_replication/model_checkpoints")
    eu_files = list(eu_dir.glob("*evaluation*.json"))
    if eu_files:
        eu_results_file = eu_files[0]

if not cam_results_file.exists():
    cam_dir = Path("$OUTPUT_BASE/cam_replication/model_checkpoints")
    cam_files = list(cam_dir.glob("*evaluation*.json"))
    if cam_files:
        cam_results_file = cam_files[0]

report = []
report.append("=" * 80)
report.append("DUAL REPLICATION COMPARISON REPORT")
report.append("=" * 80)
report.append("")

# Load EU-Emotion results
if eu_results_file.exists():
    with open(eu_results_file, 'r') as f:
        eu_data = json.load(f)
    eu_metrics = eu_data.get('metrics', {})
    eu_acc = eu_metrics.get('accuracy', 0.0)
    eu_face_acc = eu_metrics.get('face_accuracy', 0.0)
    eu_voice_acc = eu_metrics.get('voice_accuracy', 0.0)
    
    report.append("EU-Emotion Replication Results:")
    report.append(f"  Overall Accuracy: {eu_acc:.2%}")
    report.append(f"  Face Accuracy: {eu_face_acc:.2%}")
    report.append(f"  Voice Accuracy: {eu_voice_acc:.2%}")
    report.append(f"  Valid Trials: {eu_data.get('num_valid_trials', 'N/A')}")
    report.append("")
else:
    report.append("EU-Emotion results not found")
    report.append("")

# Load CAM results
if cam_results_file.exists():
    with open(cam_results_file, 'r') as f:
        cam_data = json.load(f)
    cam_metrics = cam_data.get('metrics', {})
    cam_acc = cam_metrics.get('accuracy', 0.0)
    cam_face_acc = cam_metrics.get('face_accuracy', 0.0)
    cam_voice_acc = cam_metrics.get('voice_accuracy', 0.0)
    
    report.append("CAM Replication Results:")
    report.append(f"  Overall Accuracy: {cam_acc:.2%}")
    report.append(f"  Face Accuracy: {cam_face_acc:.2%}")
    report.append(f"  Voice Accuracy: {cam_voice_acc:.2%}")
    report.append(f"  Valid Trials: {cam_data.get('num_valid_trials', 'N/A')}")
    report.append("")
else:
    report.append("CAM results not found")
    report.append("")

# Comparison
if eu_results_file.exists() and cam_results_file.exists():
    report.append("Comparison:")
    report.append(f"  Accuracy Difference: {cam_acc - eu_acc:+.2%}")
    report.append(f"  Face Accuracy Difference: {cam_face_acc - eu_face_acc:+.2%}")
    report.append(f"  Voice Accuracy Difference: {cam_voice_acc - eu_voice_acc:+.2%}")
    report.append("")
    report.append("Note: These are separate replications on different datasets.")
    report.append("      Direct comparison should consider dataset differences.")

report.append("=" * 80)

# Save report
report_path = Path("$OUTPUT_BASE/comparison_report.md")
with open(report_path, 'w') as f:
    f.write("\n".join(report))

print("\n".join(report))
print(f"\nComparison report saved to: {report_path}")
EOF

echo ""
echo "============================================================"
echo "Dual Replication Complete!"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  EU-Emotion: $OUTPUT_BASE/eu_emotion_replication/"
echo "  CAM: $OUTPUT_BASE/cam_replication/"
echo "  Comparison: $OUTPUT_BASE/comparison_report.md"
echo ""
echo "Next steps:"
echo "  1. Review results in comparison report"
echo "  2. Increase epochs for better performance (on HPC)"
echo "  3. Analyze per-emotion/concept performance"
echo ""

