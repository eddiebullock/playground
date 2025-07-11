# HPC Test Run Checklist

## Pre-HPC Setup

### 1. Code Organization ✅
- [ ] All scripts are in `src/` directory
- [ ] Data loading functions in `src/data_loader.py`
- [ ] Preprocessing functions in `src/preprocessing.py`
- [ ] Feature engineering in `src/feature_engineering.py`
- [ ] Model training in `src/training.py`
- [ ] Command-line interfaces working locally
- [ ] All dependencies listed in `requirements.txt`

### 2. Data Preparation
- [ ] Data files uploaded to HPC storage
- [ ] Data paths updated in scripts for HPC environment
- [ ] Data validation scripts ready
- [ ] Sample data subset for quick testing

### 3. Environment Setup for Cambridge HPC

#### Module Loading
```bash
# Load required modules
module load python/3.9
module load cuda/11.8  # if using GPU
module load gcc/9.3.0  # for some packages
```

#### Python Environment
```bash
# Create virtual environment
python -m venv autism_env
source autism_env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Test environment
python -c "import pandas, numpy, sklearn, xgboost; print('Environment ready')"
```

#### Data Storage
- [ ] Upload data to `/home/username/` or `/shared/` directory
- [ ] Update data paths in scripts
- [ ] Verify data accessibility

## HPC Test Run Steps

### 1. Quick Validation Test (5-10 minutes)
```bash
# Test data loading and basic preprocessing
python src/data_loader.py --sample_size 1000 --output_dir test_output
python src/preprocessing.py --input_dir test_output --output_dir test_processed
```

### 2. Small Model Test (15-30 minutes)
```bash
# Test with small dataset and simple model
python src/training.py \
    --data_path test_processed/processed_data.csv \
    --model_type logistic \
    --test_size 0.2 \
    --random_state 42 \
    --output_dir test_results
```

### 3. Full Pipeline Test (1-2 hours)
```bash
# Test complete pipeline with 10K samples
python src/data_loader.py --sample_size 10000 --output_dir hpc_test
python src/preprocessing.py --input_dir hpc_test --output_dir hpc_processed
python src/feature_engineering.py --input_dir hpc_processed --output_dir hpc_features
python src/training.py \
    --data_path hpc_features/engineered_data.csv \
    --model_type all \
    --test_size 0.2 \
    --random_state 42 \
    --output_dir hpc_results
```

## SLURM Job Scripts

### Quick Test Job (`quick_test.slurm`)
```bash
#!/bin/bash
#SBATCH --job-name=autism_quick_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/quick_test_%j.out
#SBATCH --error=logs/quick_test_%j.err

# Load modules
module load python/3.9

# Activate environment
source autism_env/bin/activate

# Run quick test
python src/data_loader.py --sample_size 1000 --output_dir test_output
python src/preprocessing.py --input_dir test_output --output_dir test_processed
python src/training.py \
    --data_path test_processed/processed_data.csv \
    --model_type logistic \
    --test_size 0.2 \
    --output_dir test_results
```

### Full Pipeline Job (`full_pipeline.slurm`)
```bash
#!/bin/bash
#SBATCH --job-name=autism_full_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/full_pipeline_%j.out
#SBATCH --error=logs/full_pipeline_%j.err

# Load modules
module load python/3.9

# Activate environment
source autism_env/bin/activate

# Run full pipeline
python src/data_loader.py --sample_size 10000 --output_dir hpc_test
python src/preprocessing.py --input_dir hpc_test --output_dir hpc_processed
python src/feature_engineering.py --input_dir hpc_processed --output_dir hpc_features
python src/training.py \
    --data_path hpc_features/engineered_data.csv \
    --model_type all \
    --test_size 0.2 \
    --output_dir hpc_results
```

## Monitoring and Debugging

### 1. Job Monitoring
```bash
# Check job status
squeue -u $USER

# Monitor job output
tail -f logs/quick_test_<job_id>.out
tail -f logs/quick_test_<job_id>.err
```

### 2. Common Issues and Solutions
- [ ] **Module not found**: Check module availability with `module avail`
- [ ] **Memory issues**: Increase `--mem` in SLURM script
- [ ] **Time limit exceeded**: Increase `--time` or optimize code
- [ ] **Data path errors**: Verify paths and permissions
- [ ] **Environment issues**: Recreate virtual environment

### 3. Performance Monitoring
```bash
# Check resource usage
seff <job_id>

# Monitor disk usage
df -h

# Check available modules
module avail python
module avail cuda
```

## Success Criteria

### Quick Test Success
- [ ] Data loads without errors
- [ ] Preprocessing completes in <5 minutes
- [ ] Simple model trains and produces results
- [ ] Output files are created correctly

### Full Pipeline Success
- [ ] All pipeline stages complete
- [ ] Model training produces metrics
- [ ] Results are saved to output directory
- [ ] No memory or time limit issues

## Next Steps After Successful Test

1. **Scale up**: Increase sample size to 50K-100K
2. **Hyperparameter tuning**: Add grid search jobs
3. **Cross-validation**: Implement k-fold CV
4. **Ensemble methods**: Test multiple model combinations
5. **Feature selection**: Add feature importance analysis

## Cambridge HPC Specific Notes

### Storage Locations
- **Home directory**: `/home/username/` (limited space)
- **Shared storage**: `/shared/` (larger capacity)
- **Temporary storage**: `/tmp/` (fast, but temporary)

### Module System
- **Python**: `module load python/3.9`
- **CUDA**: `module load cuda/11.8` (for GPU jobs)
- **GCC**: `module load gcc/9.3.0` (for compiled packages)

### Job Submission
```bash
# Submit job
sbatch quick_test.slurm

# Check queue
squeue -u $USER

# Cancel job if needed
scancel <job_id>
```

### File Transfer
```bash
# Upload files to HPC
scp -r src/ username@login.hpc.cam.ac.uk:~/autism_project/
scp requirements.txt username@login.hpc.cam.ac.uk:~/autism_project/

# Download results
scp -r username@login.hpc.cam.ac.uk:~/autism_project/results/ ./
``` 