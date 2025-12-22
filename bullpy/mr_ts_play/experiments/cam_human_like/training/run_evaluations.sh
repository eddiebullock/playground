#!/bin/bash
# Evaluate EU-Emotion fine-tuned model on EU-Emotion test set
# This completes the EU-Emotion replication (train on EU-Emotion, test on EU-Emotion)

set -e

cd /Users/eb2007/playground/bullpy/mr_ts_play

# Configuration
EU_MODEL_PATH="results/eu_emotion_replication/model_checkpoints_v3/best_model"
EU_TRIALS="results/eu_emotion_replication/eu_emotion_trial_definitions_test.json"
EU_DATA_ROOT="/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions"
DEVICE="mps"
NUM_FRAMES=8

# Detect Python
if [ -f "venv/bin/python3" ]; then
    PYTHON_CMD="venv/bin/python3"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

echo "============================================================"
echo "EU-Emotion Replication: Evaluation"
echo "============================================================"
echo ""
echo "Model: $EU_MODEL_PATH"
echo "Test Set: EU-Emotion test split"
echo "Device: $DEVICE"
echo ""
echo "Note: This evaluates the EU-Emotion model on EU-Emotion test data."
echo "      We do NOT test on CAM because they have different labels."
echo ""

# Evaluate EU-Emotion model on EU-Emotion test set
$PYTHON_CMD experiments/cam_human_like/training/evaluate_on_cam.py \
    --model_path "$EU_MODEL_PATH" \
    --trial_definitions "$EU_TRIALS" \
    --data_root "$EU_DATA_ROOT" \
    --dataset_type eu_emotion \
    --split test \
    --device $DEVICE \
    --num_frames $NUM_FRAMES \
    --use_multiframe

echo ""
echo "============================================================"
echo "EU-Emotion Evaluation Complete!"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  results/eu_emotion_replication/model_checkpoints_v3/eu_emotion_evaluation_test.json"
echo ""
echo "Next steps:"
echo "  1. Review EU-Emotion results (should be ~55-65%)"
echo "  2. Run CAM replication: Train on CAM, test on CAM"
echo "  3. Compare both replications"
echo ""

