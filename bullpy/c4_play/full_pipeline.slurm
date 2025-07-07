#!/bin/bash
#SBATCH --job-name=autism_full_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/full_pipeline_%j.out
#SBATCH --error=logs/full_pipeline_%j.err
#SBATCH --partition=medium

# Load required modules
module load python/3.9

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment
source autism_env/bin/activate

# Set working directory
cd ~/autism_project

# Run full pipeline test
echo "Starting full pipeline test at $(date)"
echo "Step 1: Data loading..."

python src/data_loader.py --sample_size 10000 --output_dir hpc_test
if [ $? -ne 0 ]; then
    echo "ERROR: Data loading failed"
    exit 1
fi

echo "Step 2: Preprocessing..."
python src/preprocessing.py --input_dir hpc_test --output_dir hpc_processed
if [ $? -ne 0 ]; then
    echo "ERROR: Preprocessing failed"
    exit 1
fi

echo "Step 3: Feature engineering..."
python src/feature_engineering.py --input_dir hpc_processed --output_dir hpc_features
if [ $? -ne 0 ]; then
    echo "ERROR: Feature engineering failed"
    exit 1
fi

echo "Step 4: Model training..."
python src/training.py \
    --data_path hpc_features/engineered_data.csv \
    --model_type all \
    --test_size 0.2 \
    --random_state 42 \
    --output_dir hpc_results

if [ $? -eq 0 ]; then
    echo "Full pipeline completed successfully at $(date)"
    echo "Results saved in hpc_results/"
    
    # Print summary of results
    echo "=== Pipeline Summary ==="
    echo "Data samples processed: 10,000"
    echo "Models trained: Random Forest, XGBoost, Logistic Regression"
    echo "Results directory: hpc_results/"
    echo "Log files: logs/full_pipeline_$SLURM_JOB_ID.out"
else
    echo "ERROR: Model training failed"
    exit 1
fi 