#!/bin/bash
#SBATCH --job-name=autism_quick_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=logs/quick_test_%j.out
#SBATCH --error=logs/quick_test_%j.err
#SBATCH --partition=icelake

# Load required modules
module load python/3.9.12/gcc/pdcqf4o5

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment
source venv/bin/activate

# Set working directory
cd ~/predict_asc_c4

# Run quick validation test
echo "Starting quick test at $(date)"
echo "Testing data loading and basic functionality..."

# Test data loading with small sample
python src/modules/data_loader.py --sample_size 1000 --output_dir test_output
if [ $? -ne 0 ]; then
    echo "ERROR: Data loading failed"
    exit 1
fi

echo "Testing basic data validation..."
# Test that the loaded data has expected structure
python -c "
import pandas as pd
import sys

try:
    df = pd.read_csv('test_output/loaded_data.csv')
    print(f'Data loaded successfully. Shape: {df.shape}')
    print(f'Columns: {list(df.columns)}')
    
    # Check for expected columns
    expected_cols = ['userid', 'age', 'sex', 'diagnosis_0', 'spq_1', 'eq_1', 'aq_1']
    missing_cols = [col for col in expected_cols if col not in df.columns]
    
    if missing_cols:
        print(f'WARNING: Missing expected columns: {missing_cols}')
    else:
        print('All expected columns present')
    
    # Check data types
    print(f'Data types: {df.dtypes.value_counts()}')
    
    print('Basic validation passed')
    
except Exception as e:
    print(f'ERROR: Validation failed - {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "ERROR: Data validation failed"
    exit 1
fi

echo "Quick test completed successfully at $(date)"
echo "Data loading and validation working correctly"
echo "Results saved in test_output/" 