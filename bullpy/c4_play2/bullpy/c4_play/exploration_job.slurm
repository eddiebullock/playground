#!/bin/bash
#SBATCH --job-name=autism_exploration
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:30:00
#SBATCH --output=logs/exploration_%j.out
#SBATCH --error=logs/exploration_%j.err
#SBATCH --partition=icelake
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=eb2007@cam.ac.uk

# Load required modules
module load python/3.9.12/gcc/pdcqf4o5

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment
source venv/bin/activate

# Set working directory
cd ~/predict_asc_c4

# Create results directory
mkdir -p results/exploration

# Run exploratory analysis
echo "Starting exploratory analysis at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"

# Check if data file exists
if [ ! -f "data/data_c4_raw.csv" ]; then
    echo "ERROR: Data file not found at data/data_c4_raw.csv"
    exit 1
fi

# Run the analysis with error handling
python src/exploratory_analysis.py \
    --data_path data/data_c4_raw.csv \
    --output_dir results/exploration \
    --sample_size 5000 \
    --target_col diagnosis_0

if [ $? -eq 0 ]; then
    echo "Exploratory analysis completed successfully at $(date)"
    echo "Results saved in results/exploration/"
    
    # List output files
    echo "Generated files:"
    ls -la results/exploration/
else
    echo "ERROR: Exploratory analysis failed"
    exit 1
fi 