# HPC Setup Guide for CAM Fine-Tuning

## Should You Use HPC?

**Yes!** If your HPC cluster has GPUs, training time will be:
- **CPU**: ~2-4 hours per epoch (10 epochs = 20-40 hours total)
- **GPU (CUDA)**: ~5-10 minutes per epoch (10 epochs = 50-100 minutes total)

**10-20x speedup** is typical with GPUs.

## HPC Considerations

### 1. Data Transfer
- **Option A**: Transfer CAM data to HPC storage (if not already there)
- **Option B**: Use network-mounted storage (if available)
- **Check**: Where is your CAM data located? Can HPC access it?

### 2. Environment Setup
HPC clusters typically use:
- **Conda environments** or **module system**
- **SLURM/PBS job scheduler** (not interactive Python)
- **Different Python paths**

### 3. Job Submission
You'll need a **SLURM script** (or PBS script) to submit the job.

## Quick Checklist

Before using HPC:
- [ ] Do you have GPU access? (Check with `nvidia-smi` or ask admin)
- [ ] Is CAM data accessible from HPC? (Network path or transferred?)
- [ ] What job scheduler? (SLURM, PBS, or other?)
- [ ] Python environment? (Conda, modules, or system Python?)

## HPC-Ready Training Script

The training script should work on HPC as-is, but you'll need:

1. **SLURM submission script** (see below)
2. **Environment setup** (conda or modules)
3. **Data path verification** (ensure HPC can access CAM data)

## SLURM Script Example

Create `submit_finetuning.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=cam_finetune
#SBATCH --output=logs/cam_finetune_%j.out
#SBATCH --error=logs/cam_finetune_%j.err
#SBATCH --time=04:00:00          # 4 hours (should be enough)
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --mem=16G                # 16GB RAM
#SBATCH --cpus-per-task=4        # 4 CPUs

# Load modules (adjust for your HPC)
module load python/3.9
module load cuda/11.8
# OR use conda:
# source activate your_env_name

# Set paths (adjust for your HPC)
DATA_ROOT="/path/to/cam/stimuli/on/hpc"
TRAIN_DATA="data/splits/train.csv"
VAL_DATA="data/splits/val.csv"
OUTPUT_DIR="models/clip_cam_finetuned"

# Run training
python experiments/cam_human_like/training/finetune_clip_emotions.py \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs 10 \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --device cuda \
    --num_frames 8
```

Submit with:
```bash
sbatch submit_finetuning.sh
```

## Data Path Options

### Option 1: Network Mount (Best)
If your CAM data is on a network drive accessible from HPC:
```bash
# Use the same path (if mounted) or adjust to HPC mount point
DATA_ROOT="/hpc/mount/path/to/cam/stimuli"
```

### Option 2: Transfer Data
If you need to transfer:
```bash
# From your local machine
rsync -avz /path/to/cam/stimuli/ user@hpc:/hpc/scratch/user/cam_data/
```

### Option 3: Symlink
If data is in a different location:
```bash
# On HPC
ln -s /actual/path/to/data /project/path/data
```

## Environment Setup on HPC

### Using Conda (Recommended)
```bash
# On HPC
conda create -n cam_finetune python=3.9
conda activate cam_finetune
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install transformers pillow opencv-python pandas tqdm
```

### Using Modules
```bash
module load python/3.9
module load cuda/11.8
pip install --user torch transformers pillow opencv-python pandas tqdm
```

## Monitoring Training on HPC

### Check Job Status
```bash
squeue -u $USER  # SLURM
# or
qstat -u $USER    # PBS
```

### View Output
```bash
tail -f logs/cam_finetune_<job_id>.out
```

### Cancel Job
```bash
scancel <job_id>  # SLURM
# or
qdel <job_id>     # PBS
```

## Recommended Approach

1. **Test locally first** (1-2 epochs) to verify script works
2. **Transfer to HPC** and run full training
3. **Download fine-tuned model** back to local machine for evaluation

## Time Estimates

| Setup | Time per Epoch | Total (10 epochs) |
|-------|---------------|-------------------|
| Local CPU | 2-4 hours | 20-40 hours |
| HPC GPU | 5-10 min | 50-100 min |
| **Speedup** | **10-20x** | **10-20x** |

## Next Steps

1. **Check HPC access**: Can you log in? Do you have GPU quota?
2. **Verify data access**: Can HPC see your CAM data?
3. **Create SLURM script**: Use template above, adjust for your HPC
4. **Test with 1 epoch**: Verify everything works before full run
5. **Submit full job**: Run 10 epochs on GPU

**Recommendation**: Use HPC if you have GPU access - it's worth the setup time!

