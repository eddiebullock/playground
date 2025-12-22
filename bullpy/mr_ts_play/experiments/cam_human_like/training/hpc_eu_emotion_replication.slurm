#!/bin/bash
#SBATCH --job-name=eu_emotion_repl
#SBATCH --output=eu_emotion_repl_%j.out
#SBATCH --error=eu_emotion_repl_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -p icelake

# CSD3 CPU partition configuration (full replication):
# - 12 hours for 10 epochs (estimated 6-10 hours needed)
# - Same resource specs as CAM replication

# Load required modules for CSD3
module purge
# Load available Python module (compute nodes have different modules than login nodes)
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
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# Change to project directory
cd ${HOME}/mr_ts_play || exit 1

# Run the EU-Emotion replication script
bash experiments/cam_human_like/training/hpc_eu_emotion_replication.sh

