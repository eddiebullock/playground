#!/bin/bash
#SBATCH --job-name=cam_replication
#SBATCH --output=cam_replication_%j.out
#SBATCH --error=cam_replication_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --partition=ampere

# Optimized resource requests for CSD3 queue positioning:
# - 4 CPUs (sufficient for data loading, not too many)
# - 16GB RAM (enough for batch_size=16, not excessive)
# - 8 hours (realistic for 10 epochs, shorter = faster queue)
# - ampere partition (GPU nodes on CSD3)

# Partition selection:
# - ampere: Main GPU partition (use if ukaea-amp not accessible)
# - If ukaea-amp accessible, change to: --partition=ukaea-amp

# Load required modules for CSD3
module purge
# Try Python 3.11.9 first, fallback to 3.9.12 (commonly available on CSD3)
if module load python/3.11.9/gcc/nptrdpll 2>/dev/null; then
    echo "Loaded python/3.11.9/gcc/nptrdpll"
elif module load python/3.9.12/gcc/pdcqf4o5 2>/dev/null; then
    echo "Loaded python/3.9.12/gcc/pdcqf4o5"
elif module load python/3.9.12 2>/dev/null; then
    echo "Loaded python/3.9.12"
else
    echo "Error: Could not load Python module. Available:"
    module avail python 2>&1 | grep -E "python/3\.[89]|python/3\.1" | head -5
    exit 1
fi
# module load cuda/11.8  # May be needed, check: module avail cuda
# module load cudnn/8.6  # May be needed

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

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4

# Change to project directory
cd ${HOME}/mr_ts_play || exit 1

# Run the CAM replication script
bash experiments/cam_human_like/training/hpc_cam_replication.sh
