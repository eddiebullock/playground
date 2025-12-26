#!/bin/bash
#SBATCH --job-name=hp_tuning_cam
#SBATCH --output=hp_tuning_cam_%j.out
#SBATCH --error=hp_tuning_cam_%j.err
#SBATCH --time=12:00:00
# Hyperparameter tuning: 5 configurations for CAM
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -p icelake
# Using CPU partition (icelake) - user has CPU access only

# CSD3 CPU partition configuration for hyperparameter tuning:
# - Partition: icelake (CPU nodes)
# - 8 CPUs per task (for efficient data loading)
# - 32G memory (sufficient for batch processing)
# - 12 hours (allows for 5 runs: ~2-3 hours each on CPU, with early stopping may be faster)

# Load required modules for CSD3 CPU nodes
module purge

# Load Python module
if module load python/3.11.0-icl 2>/dev/null; then
    echo "Loaded python/3.11.0-icl"
elif module load python/3.8 2>/dev/null; then
    echo "Loaded python/3.8"
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
    echo "Please run: bash experiments/cam_human_like/training/setup_rds_venv.sh"
    exit 1
fi

# Set environment variables for CPU
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

# Set TMPDIR to RDS to avoid filling up /home with temp files
if [ -d "${HOME}/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007" ]; then
    export TMPDIR="${HOME}/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/tmp"
elif [ -d "/rds/user/eb2007/rds-autism-research-ePtR33Nsgi4/users/eb2007" ]; then
    export TMPDIR="/rds/user/eb2007/rds-autism-research-ePtR33Nsgi4/users/eb2007/tmp"
else
    export TMPDIR="${HOME}/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/tmp"
fi
mkdir -p "$TMPDIR"
echo "Using TMPDIR: $TMPDIR"

# Change to project directory
cd ${HOME}/mr_ts_play || exit 1

# Run the CAM hyperparameter tuning script
bash experiments/cam_human_like/training/hpc_hyperparameter_tuning_cam.sh

