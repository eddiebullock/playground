#!/bin/bash
#SBATCH --job-name=hp_tuning
#SBATCH --output=hp_tuning_%j.out
#SBATCH --error=hp_tuning_%j.err
#SBATCH --time=24:00:00
# Hyperparameter tuning: 5 configurations × 2 datasets = 10 runs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -p ukaea-amp

# CSD3 GPU partition configuration for hyperparameter tuning:
# - Partition: ukaea-amp (GPU nodes with good availability)
# - 1 GPU per job (NVIDIA A100 or similar)
# - 8 CPUs per task (for efficient data loading)
# - 32G memory (sufficient for batch processing)
# - 24 hours (allows for 10 runs: 5 configs × 2 datasets)

# Load required modules for CSD3 GPU nodes
module purge
# Load CUDA and cuDNN for GPU support
if module load cuda/11.8 2>/dev/null; then
    echo "Loaded cuda/11.8"
elif module load cuda/12.0 2>/dev/null; then
    echo "Loaded cuda/12.0"
elif module load cuda/11.4 2>/dev/null; then
    echo "Loaded cuda/11.4"
else
    echo "Warning: Could not load CUDA module. Available:"
    module avail cuda 2>&1 | head -10
fi

if module load cudnn/8.6 2>/dev/null; then
    echo "Loaded cudnn/8.6"
elif module load cudnn/8.9 2>/dev/null; then
    echo "Loaded cudnn/8.9"
else
    echo "Warning: Could not load cuDNN module. Available:"
    module avail cudnn 2>&1 | head -10
fi

# Load Python module
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

# Set environment variables for GPU
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

# Verify GPU is available
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" || echo "Warning: Could not verify GPU availability"

# Change to project directory
cd ${HOME}/mr_ts_play || exit 1

# Run the hyperparameter tuning script
bash experiments/cam_human_like/training/hpc_hyperparameter_tuning.sh

