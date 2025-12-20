#!/bin/bash
#SBATCH --job-name=cam_finetune
#SBATCH --output=logs/cam_finetune_%j.out
#SBATCH --error=logs/cam_finetune_%j.err
#SBATCH --time=04:00:00          # 4 hours (should be enough for 10 epochs)
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --mem=16G                # 16GB RAM
#SBATCH --cpus-per-task=4        # 4 CPUs

# ============================================================================
# CAM Fine-Tuning SLURM Submission Script
# ============================================================================
# 
# Usage:
#   1. Edit DATA_ROOT, TRAIN_DATA, VAL_DATA paths below
#   2. Adjust module/conda setup for your HPC
#   3. Submit: sbatch submit_finetuning.sh
#
# ============================================================================

# Load modules (ADJUST FOR YOUR HPC)
# Option 1: Using modules
# module load python/3.9
# module load cuda/11.8

# Option 2: Using conda (RECOMMENDED)
# source activate cam_finetune
# OR
# conda activate cam_finetune

# Set working directory (adjust if needed)
cd /path/to/mr_ts_play  # CHANGE THIS to your project path on HPC

# Set paths (ADJUST FOR YOUR HPC)
DATA_ROOT="/path/to/cam/stimuli/on/hpc"  # CHANGE THIS
TRAIN_DATA="data/splits/train.csv"
VAL_DATA="data/splits/val.csv"
OUTPUT_DIR="models/clip_cam_finetuned"

# Create logs directory
mkdir -p logs

# Print environment info
echo "=========================================="
echo "CAM Fine-Tuning Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Data root: $DATA_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo "=========================================="

# Check GPU
nvidia-smi

# Run training
python experiments/cam_human_like/training/finetune_clip_emotions.py \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs 10 \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --device cuda \
    --num_frames 8

echo ""
echo "Training complete! Model saved to: $OUTPUT_DIR/best_model"


