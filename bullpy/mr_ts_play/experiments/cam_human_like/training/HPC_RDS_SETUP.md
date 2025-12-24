# Using RDS Storage on CSD3 HPC

## What is RDS?

**RDS (Research Data Store)** is a shared storage system on CSD3 HPC designed for:
- **Large research datasets** (much larger quotas than `/home`)
- **Project-based storage** (shared with your research group)
- **Persistent data** (survives across sessions)
- **Better performance** for large file I/O

### Your RDS Quotas

From your `quota -s` output:

| Filesystem | Used | Quota | Limit | Project |
|------------|------|-------|-------|---------|
| `/rds-d7` | 0.0 GB | 1099.5 GB | 1209.5 GB | P:45718 |
| `rds-ePtR33Nsgi4` | 12410.7 GB | 23000.0 GB | 23000.0 GB | P:90416 |

**You have ~1100 GB available on `/rds-d7`!** (vs only 50 GB on `/home`)

## Setting Up Project on RDS

### Step 1: Check RDS Access

```bash
# Check if you can access your RDS project directory
ls -la /rds-d7/project/45718/

# Or check the other RDS location
ls -la /rds/project/90416/  # If this path exists
```

### Step 2: Create Project Directory on RDS

```bash
# Create your project directory on RDS
mkdir -p /rds-d7/project/45718/users/eb2007/mr_ts_play

# Or if the structure is different, check what exists:
ls -la /rds-d7/project/45718/
```

### Step 3: Move/Copy Project to RDS

**Option A: Copy project to RDS** (keeps original in /home)
```bash
# Copy project code to RDS
cp -r ~/mr_ts_play /rds-d7/project/45718/users/eb2007/

# Or use rsync for better control
rsync -avh ~/mr_ts_play/ /rds-d7/project/45718/users/eb2007/mr_ts_play/
```

**Option B: Create symlink** (project stays in /home, venv/data on RDS)
```bash
# Create venv on RDS
mkdir -p /rds-d7/project/45718/users/eb2007/venv
cd ~/mr_ts_play
ln -s /rds-d7/project/45718/users/eb2007/venv venv
```

### Step 4: Update Scripts for RDS Paths

Update your SLURM scripts to use RDS paths:

```bash
# In hpc_cam_replication.slurm, change:
cd ${HOME}/mr_ts_play
# To:
cd /rds-d7/project/45718/users/eb2007/mr_ts_play
```

## Recommended Setup

**Best approach**: Keep code in `/home`, but use RDS for:
- Virtual environment (venv)
- Model checkpoints
- Results/outputs
- Large datasets (if not already in `/home/eb2007/data/`)

This way:
- Code stays in `/home` (small, easy to edit)
- Large files (venv, models) on RDS (plenty of space)

## Example Setup

```bash
# On HPC
cd ~/mr_ts_play

# Create venv on RDS
python3 -m venv --without-pip /rds-d7/project/45718/users/eb2007/venv
source /rds-d7/project/45718/users/eb2007/venv/bin/activate

# Install packages (they'll be stored on RDS)
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers pillow opencv-python tqdm numpy pandas

# Create symlink for convenience
ln -s /rds-d7/project/45718/users/eb2007/venv venv
```

## Benefits of RDS

1. **Much larger quota**: 1100 GB vs 50 GB
2. **Shared with group**: Can collaborate easily
3. **Persistent**: Data survives across sessions
4. **Better for large files**: Optimized for datasets/models

## Important Notes

- **RDS is shared**: Other project members can see your files
- **Backup policy**: Check CSD3 documentation for backup policies
- **Performance**: RDS may be slightly slower than `/home` for small files, but better for large I/O
- **Path in scripts**: Update all scripts to use RDS paths for venv/data



