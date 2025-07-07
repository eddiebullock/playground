# Cambridge HPC Quick Reference Guide

## Connection and Access

### SSH Connection
```bash
ssh username@login.hpc.cam.ac.uk
```

### File Transfer
```bash
# Upload files
scp -r src/ username@login.hpc.cam.ac.uk:~/autism_project/
scp data_file.csv username@login.hpc.cam.ac.uk:~/autism_project/

# Download results
scp -r username@login.hpc.cam.ac.uk:~/autism_project/results/ ./
```

## Environment Setup

### Initial Setup
```bash
# Run setup script
chmod +x setup_hpc.sh
./setup_hpc.sh
```

### Manual Setup
```bash
# Load modules
module load python/3.9

# Create environment
python -m venv autism_env
source autism_env/bin/activate

# Install packages
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
pip install imbalanced-learn optuna joblib
```

## Job Management

### Submit Jobs
```bash
# Quick test (30 minutes)
sbatch quick_test.slurm

# Full pipeline (4 hours)
sbatch full_pipeline.slurm
```

### Monitor Jobs
```bash
# Check job status
squeue -u $USER

# Check job details
scontrol show job <job_id>

# Check resource usage
seff <job_id>
```

### Cancel Jobs
```bash
scancel <job_id>
```

## Storage and File Management

### Storage Locations
- **Home**: `/home/username/` (limited space, ~10GB)
- **Shared**: `/shared/` (larger capacity)
- **Temporary**: `/tmp/` (fast, temporary)

### File Operations
```bash
# Check disk usage
df -h
du -sh ~/autism_project/

# Clean up old files
rm -rf test_output/ test_processed/ test_results/
```

## Troubleshooting

### Common Issues

#### Module Not Found
```bash
# Check available modules
module avail python
module avail cuda

# Load specific version
module load python/3.9
```

#### Memory Issues
- Increase `--mem` in SLURM script
- Use smaller sample sizes for testing
- Monitor with `seff <job_id>`

#### Time Limit Exceeded
- Increase `--time` in SLURM script
- Use `--partition=long` for longer jobs
- Optimize code for faster execution

#### Data Path Errors
```bash
# Check file permissions
ls -la ~/autism_project/

# Verify data files exist
find ~/autism_project/ -name "*.csv"
```

### Debugging Commands
```bash
# Check job logs
tail -f logs/quick_test_<job_id>.out
tail -f logs/quick_test_<job_id>.err

# Check system resources
htop
nvidia-smi  # if using GPU

# Check Python environment
which python
pip list
```

## Performance Optimization

### Resource Allocation
- **CPU**: Start with 4-8 cores for testing
- **Memory**: 8-32GB depending on dataset size
- **Time**: 30min for quick test, 4hr for full pipeline

### Partition Selection
- **short**: < 1 hour, quick tests
- **medium**: 1-4 hours, full pipeline
- **long**: > 4 hours, large datasets

### Best Practices
1. Test with small datasets first
2. Monitor resource usage
3. Clean up temporary files
4. Use appropriate partitions
5. Check logs for errors

## Data Management

### Upload Strategy
```bash
# Create project structure
mkdir -p ~/autism_project/{src,data,results,logs}

# Upload code
scp -r src/ username@login.hpc.cam.ac.uk:~/autism_project/

# Upload data (if small)
scp data.csv username@login.hpc.cam.ac.uk:~/autism_project/data/

# For large data, use rsync or scp with compression
rsync -avz --progress data/ username@login.hpc.cam.ac.uk:~/autism_project/data/
```

### Results Download
```bash
# Download results
scp -r username@login.hpc.cam.ac.uk:~/autism_project/results/ ./

# Download logs
scp -r username@login.hpc.cam.ac.uk:~/autism_project/logs/ ./
```

## Quick Commands Reference

```bash
# Job submission
sbatch quick_test.slurm

# Job monitoring
squeue -u $USER
seff <job_id>

# Environment activation
source autism_env/bin/activate

# Module management
module load python/3.9
module avail python

# File operations
ls -la ~/autism_project/
du -sh ~/autism_project/

# Log monitoring
tail -f logs/quick_test_<job_id>.out
``` 