#!/bin/bash
#SBATCH --job-name=autism_quick_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/quick_test_%j.out
#SBATCH --error=logs/quick_test_%j.err
#SBATCH --partition=cclake

# Load required modules
module load python/3.9

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment
source autism_env/bin/activate

# Set working directory
cd ~/predict_asc_c4

# Run quick validation test
echo "Starting quick test at $(date)"
echo "Testing data loading and preprocessing..."

python src/modules/data_loader.py --sample_size 1000 --output_dir test_output
if [ $? -ne 0 ]; then
    echo "ERROR: Data loading failed"
    exit 1
fi

python src/preprocessing.py --input_dir test_output --output_dir test_processed
if [ $? -ne 0 ]; then
    echo "ERROR: Preprocessing failed"
    exit 1
fi

echo "Testing simple model training..."
python src/training.py \
    --data_path test_processed/processed_data.csv \
    --model_type logistic \
    --test_size 0.2 \
    --random_state 42 \
    --output_dir test_results

if [ $? -eq 0 ]; then
    echo "Quick test completed successfully at $(date)"
    echo "Results saved in test_results/"
else
    echo "ERROR: Model training failed"
    exit 1
fi 