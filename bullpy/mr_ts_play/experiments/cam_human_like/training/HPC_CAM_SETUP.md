# HPC Setup: CAM Replication Study

## Overview

This guide sets up the CAM replication study on HPC to leverage GPU resources for better performance.

## Prerequisites

- HPC account with GPU access
- CAM data already on HPC at `/home/eb2007/data/CAM`
- Python environment with PyTorch, transformers, etc.

## Step 1: Transfer Code to HPC

### Transfer Project Code

Run these rsync commands from your **local machine**:

```bash
# Transfer entire project (excluding large files and results)
rsync -avh --progress \
  --exclude 'venv/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.git/' \
  --exclude 'results/' \
  --exclude 'models/' \
  --exclude '*.pth' \
  --exclude '*.safetensors' \
  /Users/eb2007/playground/bullpy/mr_ts_play/ \
  eb2007@login-cpu.hpc.cam.ac.uk:~/mr_ts_play/

# Transfer trial definitions (small, important files)
rsync -avh --progress \
  /Users/eb2007/playground/bullpy/mr_ts_play/data/cam_trial_definitions_20concepts.json \
  eb2007@login-cpu.hpc.cam.ac.uk:~/mr_ts_play/data/

# Transfer any existing CAM splits if you have them
rsync -avh --progress \
  /Users/eb2007/playground/bullpy/mr_ts_play/results/cam_replication/ \
  eb2007@login-cpu.hpc.cam.ac.uk:~/mr_ts_play/results/cam_replication/ \
  || echo "No existing splits to transfer (will be created on HPC)"
```

### Verify Transfer

SSH to HPC and verify:

```bash
ssh eb2007@login-cpu.hpc.cam.ac.uk
cd ~/mr_ts_play
ls -la experiments/cam_human_like/training/
ls -la data/cam_trial_definitions_20concepts.json
```

## Step 2: Set Up Python Environment on HPC

### Automated Setup (Recommended)

```bash
# SSH to HPC
ssh eb2007@login-cpu.hpc.cam.ac.uk

# Run setup script
cd ~/mr_ts_play
bash experiments/cam_human_like/training/setup_hpc_python.sh
```

This will:
1. Check available Python modules
2. Create virtual environment
3. Install PyTorch with CUDA support
4. Install all dependencies
5. Verify installation

### Manual Setup

```bash
# SSH to HPC
ssh eb2007@login-cpu.hpc.cam.ac.uk

# Check available Python modules
module avail python

# Load Python module
module purge
module load python/3.11.9/gcc/nptrdpll  # or latest available

# Create virtual environment
cd ~/mr_ts_play
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers pillow opencv-python tqdm numpy pandas

# Verify CUDA
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

See `HPC_PYTHON_SETUP.md` for detailed instructions.

## Step 3: Verify CAM Data Location

```bash
# On HPC, verify CAM data is accessible
ls -la /home/eb2007/data/CAM
# Should see directories: 01, 02, 03, ..., 24, Audio, definitions, Scenarios
```

## Step 4: Run CAM Replication

### Option A: Interactive Session (recommended for initial testing)

**Advantages**: Faster queue time, immediate feedback, can test before full run

```bash
# Request GPU node interactively (optimized resources for faster queue)
srun --gres=gpu:1 --time=2:00:00 --cpus-per-task=4 --mem=16G --partition=ampere --pty bash

# Activate environment
cd ~/mr_ts_play
source venv/bin/activate  # or your conda env

# Test with fewer epochs first
# Edit hpc_cam_replication.sh: NUM_EPOCHS=2
bash experiments/cam_human_like/training/hpc_cam_replication.sh
```

### Option B: SLURM Job (recommended for full run)

**Optimized for queue positioning**: Minimal resource requests, realistic time limits

**Partition Selection** (based on current HPC status):
- **`ukaea-amp`** (recommended): 22 idle nodes available → faster queue
- **`ampere`**: 0 idle nodes (50 allocated) → longer wait time

```bash
# Check current partition availability
sinfo -p ukaea-amp
sinfo -p ampere

# Submit job (try ukaea-amp first for faster queue)
cd ~/mr_ts_play
sbatch experiments/cam_human_like/training/hpc_cam_replication.slurm

# If ukaea-amp not accessible, use ampere partition:
sbatch experiments/cam_human_like/training/hpc_cam_replication_ampere.slurm

# Monitor job
squeue -u eb2007

# Check output
tail -f cam_replication_*.out

# Check job efficiency after completion
seff <job_id>
```

**Resource requests** (optimized for queue):
- 1 GPU (sufficient)
- 4 CPUs (enough for data loading)
- 16GB RAM (sufficient for batch_size=16)
- 8 hours (realistic for 10 epochs)
- `ukaea-amp` partition (22 idle nodes = faster queue)

## Step 5: Monitor Progress

```bash
# Watch log file
tail -f cam_replication_*.out

# Check results directory
ls -lh results/cam_replication/model_checkpoints/

# Check training progress
tail -f results/cam_replication/model_checkpoints/training_log.txt
```

## Expected Results

- **Training time**: ~2-4 hours for 10 epochs on GPU (A100)
- **Queue time**: Faster with optimized resource requests (4 CPUs, 16GB, 8h)
- **Validation accuracy**: 65-75% (vs 37% zero-shot baseline)
- **Model saved to**: `results/cam_replication/model_checkpoints/best_model/`
- **Evaluation results**: `results/cam_replication/model_checkpoints/cam_evaluation_test.json`

## Resource Optimization

The SLURM script is optimized for **queue positioning**:
- **Minimal resources**: 4 CPUs, 16GB RAM (sufficient, not excessive)
- **Realistic time**: 8 hours (not 24h, faster queue)
- **Correct partition**: `ampere` (GPU nodes on CSD3)

This improves queue time compared to requesting maximum resources.

See `HPC_RESOURCE_OPTIMIZATION.md` for detailed optimization guide.

## Troubleshooting

### GPU Not Available

If GPU is not available, the script will fall back to CPU (slower). Check:

```bash
nvidia-smi  # Should show GPU
python3 -c "import torch; print(torch.cuda.is_available())"  # Should be True
```

### Out of Memory

If you get OOM errors, reduce batch size in `hpc_cam_replication.sh`:

```bash
BATCH_SIZE=8  # Instead of 16
```

### Module Errors

Make sure all dependencies are installed:

```bash
pip install -r requirements.txt  # If you have one
# Or install manually:
pip install torch transformers pillow opencv-python tqdm numpy pandas
```

### Path Issues

If paths don't work, check:

```bash
# Verify CAM data path
ls /home/eb2007/data/CAM

# Verify project structure
cd ~/mr_ts_play
ls experiments/cam_human_like/training/
```

## Transferring Results Back

After training completes, transfer results back to local:

```bash
# From local machine
rsync -avh --progress \
  eb2007@login-cpu.hpc.cam.ac.uk:~/mr_ts_play/results/cam_replication/ \
  /Users/eb2007/playground/bullpy/mr_ts_play/results/cam_replication/
```

## Next Steps

After CAM replication completes:
1. Transfer results back to local
2. Run EU-Emotion replication (once data transfer completes)
3. Compare results from both replications

