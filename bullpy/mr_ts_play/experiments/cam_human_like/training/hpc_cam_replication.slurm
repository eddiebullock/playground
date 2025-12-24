#!/bin/bash
#SBATCH --job-name=cam_replication
#SBATCH --output=cam_replication_%j.out
#SBATCH --error=cam_replication_%j.err
#SBATCH --time=06:00:00
# CPU-optimized replication: 20 epochs, all available files, comprehensive evaluation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -p icelake

# CSD3 CPU partition configuration (user has CPU access only):
# - Partition: icelake (CPU nodes - user has access to this)
# - 8 CPUs per task (for efficient data loading)
# - 32G memory (sufficient for batch processing)
# - 12 hours (CPU training is slower, but will complete)
# Note: Scripts automatically detect CPU and adjust batch size

# Load required modules for CSD3 CPU nodes
module purge
# Load Python module (no CUDA needed for CPU)
if module load python/3.8 2>/dev/null; then
    echo "Loaded python/3.8"
elif module load python/3.11.0-icl 2>/dev/null; then
    echo "Loaded python/3.11.0-icl"
elif module load python/3.7 2>/dev/null; then
    echo "Loaded python/3.7"
else
    echo "Error: Could not load Python module. Available:"
    module avail python 2>&1 | grep -E "python/3\.[678]|python/3\.1" | head -10
    exit 1
fi

# Activate virtual environment
# Try symlink first (points to RDS venv), then check RDS paths directly, then home venv
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
    echo "Please run: bash experiments/cam_human_like/training/setup_rds_venv.sh"
    exit 1
fi

# Set environment variables for CPU (match number of CPUs requested)
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

# Change to project directory
cd ${HOME}/mr_ts_play || exit 1

# Run the CAM replication script
bash experiments/cam_human_like/training/hpc_cam_replication.sh
