#!/bin/bash
#SBATCH --job-name=rf_xgb_hyperparam
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=/home/eb2007/c4/logs/hyperparam_%j.out
#SBATCH --error=/home/eb2007/c4/logs/hyperparam_%j.err
#SBATCH --partition=icelake

module load python/3.9.12/gcc/pdcqf4o5
mkdir -p /home/eb2007/c4/logs
source /home/eb2007/c4/venv/bin/activate
cd /home/eb2007/c4
python experiments/hyperparam_tuning.py