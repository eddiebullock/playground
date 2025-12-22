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

# Alternative SLURM script using 'ampere' partition
# Use this if ukaea-amp is not accessible to your account
# Note: ampere currently has 0 idle nodes (50 allocated), so queue time may be longer

# Load required modules for CSD3
module purge
module load python/3.11.9/gcc/nptrdpll  # Adjust version as needed
# module load cuda/11.8  # May be needed, check: module avail cuda
# module load cudnn/8.6  # May be needed

# Activate virtual environment
source ${HOME}/mr_ts_play/venv/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4

# Change to project directory
cd ${HOME}/mr_ts_play || exit 1

# Run the CAM replication script
bash experiments/cam_human_like/training/hpc_cam_replication.sh

