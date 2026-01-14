#!/bin/bash
#SBATCH --job-name=video_finetune
#SBATCH --output=video_finetune_%j.out
#SBATCH --error=video_finetune_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -p icelake

# Fine-tune I3D and TimeSformer video models on HPC with CPU
# Expected time: ~4-6 hours on CPU (similar to MPS, but can run in background)

# CSD3 CPU partition configuration:
# - Partition: icelake (CPU nodes)
# - 8 CPUs per task (for efficient data loading)
# - 32G memory (sufficient for batch processing)
# - 12 hours (should be enough with early stopping)

# Load required modules for CSD3 CPU nodes
module purge
# Load Python module (no CUDA needed for CPU)
if module load python/3.8 2>/dev/null; then
    echo "Loaded python/3.8"
elif module load python/3.11.0-icl 2>/dev/null; then
    echo "Loaded python/3.11.0-icl"
elif module load python/3.11.9/gcc/nptrdpll 2>/dev/null; then
    echo "Loaded python/3.11.9/gcc/nptrdpll"
elif module load python/3.7 2>/dev/null; then
    echo "Loaded python/3.7"
else
    echo "Error: Could not load Python module. Available:"
    module avail python 2>&1 | grep -E "python/3\.[678]|python/3\.1" | head -10
    exit 1
fi

# Activate virtual environment
if [ -f "${HOME}/mr_ts_play/venv/bin/activate" ]; then
    source ${HOME}/mr_ts_play/venv/bin/activate
elif [ -f "${HOME}/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/venv/bin/activate" ]; then
    source ${HOME}/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/venv/bin/activate
elif [ -f "/rds/user/eb2007/rds-autism-research-ePtR33Nsgi4/users/eb2007/venv/bin/activate" ]; then
    source /rds/user/eb2007/rds-autism-research-ePtR33Nsgi4/users/eb2007/venv/bin/activate
elif [ -f "/rds-d7/project/45718/users/eb2007/venv/bin/activate" ]; then
    source /rds-d7/project/45718/users/eb2007/venv/bin/activate
else
    echo "Error: Virtual environment not found!"
    exit 1
fi

# Set environment variables for CPU (match number of CPUs requested)
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

# Change to project directory
cd ${HOME}/mr_ts_play || exit 1

# Print environment info
echo "=========================================="
echo "Video Model Fine-Tuning Job (CPU)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "Partition: icelake (CPU)"
echo "=========================================="

# Run fine-tuning (both I3D and TimeSformer)
# IMPORTANT: use the HPC wrapper (it sets the correct RDS data root).
# The generic finetune_i3d_timesformer.sh may contain a local (macOS) DATA_ROOT.
bash experiments/eu_emotion_model_comparison/training/hpc_finetune_video_models.sh

echo "=========================================="
echo "Fine-tuning completed!"
echo "=========================================="
