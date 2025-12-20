# HPC Transfer Guide: Moving CAM Dataset and Code

## Overview

Transfer CAM dataset and project code to HPC for:
- ✅ Save local laptop storage
- ✅ Faster training on HPC GPUs
- ✅ Better resource management

## Step 1: Create Directory Structure on HPC

**On HPC (in your SSH session):**

```bash
# Create directory structure
mkdir -p ~/data/cam
mkdir -p ~/mr_ts_play/experiments/cam_human_like
mkdir -p ~/mr_ts_play/data/splits
mkdir -p ~/mr_ts_play/models
mkdir -p ~/mr_ts_play/configs

# Verify structure
ls -la ~/data/
ls -la ~/mr_ts_play/
```

## Step 2: Transfer CAM Dataset

**From your local machine (in a NEW terminal, keep HPC SSH open):**

```bash
# Navigate to your local project
cd /Users/eb2007/playground/bullpy/mr_ts_play

# Transfer CAM dataset (this will take time - dataset is large)
rsync -avz --progress \
    "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/mindreading_transporter_files/Mindreading emotions library/Emotions/" \
    eb2007@login-cpu.hpc.cam.ac.uk:~/data/cam/

# This will:
# - Show progress (-progress)
# - Preserve permissions (-a)
# - Compress during transfer (-z)
# - Be verbose (-v)
```

**Note**: This is a large dataset. The transfer may take 30-60 minutes depending on network speed.

## Step 3: Transfer Project Code

**From your local machine:**

```bash
# Transfer project code (excluding large files)
rsync -avz --progress \
    --exclude 'venv/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.git/' \
    --exclude 'models/' \
    --exclude 'results/' \
    --exclude '*.ipynb_checkpoints' \
    /Users/eb2007/playground/bullpy/mr_ts_play/ \
    eb2007@login-cpu.hpc.cam.ac.uk:~/mr_ts_play/
```

## Step 4: Transfer Essential Data Files

**From your local machine:**

```bash
# Transfer splits and configs
rsync -avz \
    /Users/eb2007/playground/bullpy/mr_ts_play/data/splits/ \
    eb2007@login-cpu.hpc.cam.ac.uk:~/mr_ts_play/data/splits/

rsync -avz \
    /Users/eb2007/playground/bullpy/mr_ts_play/configs/ \
    eb2007@login-cpu.hpc.cam.ac.uk:~/mr_ts_play/configs/

# Transfer trial definitions if they exist
rsync -avz \
    /Users/eb2007/playground/bullpy/mr_ts_play/data/cam_trial_definitions*.json \
    eb2007@login-cpu.hpc.cam.ac.uk:~/mr_ts_play/data/ 2>/dev/null || echo "Trial definitions not found, that's OK"
```

## Step 5: Verify Transfer on HPC

**Back on HPC (in your SSH session):**

```bash
# Check CAM data
ls -lh ~/data/cam/ | head -20
du -sh ~/data/cam/  # Check total size

# Check project code
ls -la ~/mr_ts_play/
ls -la ~/mr_ts_play/experiments/cam_human_like/training/

# Check splits
ls -la ~/mr_ts_play/data/splits/
```

## Step 6: Update Paths for HPC

**On HPC, create/update a config file:**

```bash
# Create HPC-specific config
cat > ~/mr_ts_play/configs/cam_config_hpc.yaml << 'EOF'
# CAM Face-Voice Battery Experiment Configuration - HPC Version
data:
  root: "/home/eb2007/data/cam"
  splits_dir: "data/splits"
  trial_definitions_file: "data/cam_trial_definitions_20concepts.json"

model:
  type: "clip"
  name: "openai/clip-vit-base-patch32"
  num_frames: 8
  aggregation: "mean"

device: "cuda"  # Use GPU on HPC

evaluation:
  temperature: 1.0
  calibration:
    enabled: false

seed: 42

output:
  results_dir: "results/cam_human_like"
EOF
```

## Step 7: Set Up Python Environment on HPC

**On HPC:**

```bash
# Load Python module (adjust for your HPC)
module load python/3.9  # or whatever version is available

# Create virtual environment
cd ~/mr_ts_play
python -m venv venv_hpc
source venv_hpc/bin/activate

# Install dependencies (you may need to create requirements.txt first)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers pillow opencv-python pandas tqdm numpy

# Or if you have a requirements.txt:
# pip install -r requirements.txt
```

## Step 8: Test Fine-Tuning Script on HPC

**On HPC:**

```bash
cd ~/mr_ts_play
source venv_hpc/bin/activate

# Test with 1 epoch first
python experiments/cam_human_like/training/finetune_clip_emotions.py \
    --train_data data/splits/train.csv \
    --val_data data/splits/val.csv \
    --data_root "/home/eb2007/data/cam" \
    --output_dir models/clip_cam_finetuned_hpc \
    --num_epochs 1 \
    --batch_size 16 \
    --device cuda \
    --num_frames 8
```

## Step 9: Create SLURM Submission Script

**On HPC, create submission script:**

```bash
cat > ~/mr_ts_play/submit_cam_finetuning.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=cam_finetune
#SBATCH --output=logs/cam_finetune_%j.out
#SBATCH --error=logs/cam_finetune_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Load modules
module load python/3.9
module load cuda/11.8

# Activate environment
cd ~/mr_ts_play
source venv_hpc/bin/activate

# Run training
python experiments/cam_human_like/training/finetune_clip_emotions.py \
    --train_data data/splits/train.csv \
    --val_data data/splits/val.csv \
    --data_root "/home/eb2007/data/cam" \
    --output_dir models/clip_cam_finetuned_hpc \
    --num_epochs 10 \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --device cuda \
    --num_frames 8
EOF

chmod +x ~/mr_ts_play/submit_cam_finetuning.sh
mkdir -p ~/mr_ts_play/logs
```

## Quick Reference Commands

### Transfer Dataset (from local):
```bash
rsync -avz --progress \
    "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/mindreading_transporter_files/Mindreading emotions library/Emotions/" \
    eb2007@login-cpu.hpc.cam.ac.uk:~/data/cam/
```

### Transfer Code (from local):
```bash
rsync -avz --progress \
    --exclude 'venv/' --exclude '__pycache__/' --exclude '*.pyc' \
    --exclude '.git/' --exclude 'models/' --exclude 'results/' \
    /Users/eb2007/playground/bullpy/mr_ts_play/ \
    eb2007@login-cpu.hpc.cam.ac.uk:~/mr_ts_play/
```

### Submit Job (on HPC):
```bash
cd ~/mr_ts_play
sbatch submit_cam_finetuning.sh
```

### Check Job Status (on HPC):
```bash
squeue -u eb2007
```

### View Output (on HPC):
```bash
tail -f logs/cam_finetune_<job_id>.out
```

## Troubleshooting

### If transfer is slow:
- Use `--partial` flag to resume interrupted transfers
- Transfer during off-peak hours
- Consider compressing first (though rsync -z already does this)

### If paths are wrong:
- Update `data_root` in config to `/home/eb2007/data/cam`
- Check your HPC username: `whoami` on HPC

### If modules don't load:
- Check available modules: `module avail`
- Ask HPC support for correct module names

## Next Steps

After transfer completes:
1. ✅ Verify files on HPC
2. ✅ Set up Python environment
3. ✅ Test with 1 epoch
4. ✅ Submit full training job (10 epochs)
5. ✅ Monitor progress
6. ✅ Download results back to local

