#!/bin/bash
#SBATCH --job-name=autism_eda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/eda_%j.out
#SBATCH --error=logs/eda_%j.err
#SBATCH --partition=icelake

module load python/3.9.12/gcc/pdcqf4o5
source venv/bin/activate

python src/exploratory_analysis.py --data_path data/processed/features_full.csv --output_dir results/eda --target_col autism_any --sample_size 10000 