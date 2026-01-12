# Complete CAM Study Setup on HPC

## Current Status

✅ EU emotions transfer to RDS in progress  
✅ Space freed up in /home (removed ~/data/EU_emotions)  
✅ RDS data folder created: `~/rds/rds-autism-research-ePtR33Nsgi4/data`  
⏳ Need to: Transfer scripts, set up Python environment, run CAM study

## Step 1: Transfer Updated Scripts

From your local machine:

```bash
cd /Users/eb2007/playground/bullpy/mr_ts_play
bash experiments/cam_human_like/training/transfer_to_hpc.sh
```

This will transfer:
- Updated HPC scripts (with RDS support)
- Setup scripts (setup_rds_venv.sh)
- Transfer scripts (transfer_eu_emotions_to_rds.sh)

## Step 2: Set Up Python Environment on RDS

On HPC:

```bash
ssh eb2007@login-cpu.hpc.cam.ac.uk
cd ~/mr_ts_play

# Run the RDS venv setup script
bash experiments/cam_human_like/training/setup_rds_venv.sh
```

This will:
- Create venv on RDS (not in /home)
- Install PyTorch with CUDA
- Install all dependencies
- Create symlink in project directory

## Step 3: Verify Setup

```bash
# Check venv location
ls -la ~/mr_ts_play/venv  # Should be a symlink to RDS

# Activate and test
source ~/mr_ts_play/venv/bin/activate
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## Step 4: Run CAM Replication Study

### Option A: Interactive Session (for testing)

```bash
# Request GPU node
srun --gres=gpu:1 --time=2:00:00 --cpus-per-task=4 --mem=16G --partition=ampere --pty bash

# Activate environment
cd ~/mr_ts_play
source venv/bin/activate

# Run CAM replication
bash experiments/cam_human_like/training/hpc_cam_replication.sh
```

### Option B: SLURM Job (recommended)

```bash
cd ~/mr_ts_play
sbatch experiments/cam_human_like/training/hpc_cam_replication.slurm

# Monitor
squeue -u eb2007
tail -f cam_replication_*.out
```

## Data Paths

The scripts will automatically detect:
- **CAM data**: `/home/eb2007/data/CAM` (already on HPC)
- **EU emotions**: `~/rds/rds-autism-research-ePtR33Nsgi4/data/EU_emotions` (transferring)
- **Venv**: `~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/venv` (on RDS)

## Expected Results

- **Training time**: ~2-4 hours for 10 epochs on GPU
- **Validation accuracy**: 65-75% (vs 37% zero-shot baseline)
- **Model saved**: `results/cam_replication/model_checkpoints/best_model/`
- **Evaluation**: `results/cam_replication/model_checkpoints/cam_evaluation_test.json`

## Troubleshooting

### If scripts fail to transfer

Check quota:
```bash
quota -s
```

If still over quota, clean up more:
```bash
# Remove Python cache
find ~/mr_ts_play -name "*.pyc" -delete
find ~/mr_ts_play -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

# Remove old logs
find ~ -name "*.log" -size +10M -delete
```

### If venv setup fails

Use manual setup:
```bash
module load python/3.11.9/gcc/nptrdpll
python3 -m venv --without-pip ~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/venv
source ~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/venv/bin/activate
curl -sS https://bootstrap.pypa.io/get-pip.py | python3
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers pillow opencv-python tqdm numpy pandas
```

## Next Steps After CAM Study

1. Wait for EU emotions transfer to complete
2. Set up EU emotions replication study (similar process)
3. Compare results between CAM and EU emotions replications





