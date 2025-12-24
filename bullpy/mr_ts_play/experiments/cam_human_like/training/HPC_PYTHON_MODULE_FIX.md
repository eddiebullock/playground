# Fix: Python Module Not Found

## Problem

The script tried to load `python/3.11.9/gcc/nptrdpll` but it's not available on your login node. System Python 3.6.8 is too old for PyTorch.

## Solution: Use Available Python Module

On HPC, check available Python modules:

```bash
module avail python | grep -E "python/3\.[89]|python/3\.1"
```

From your output, **Python 3.9.12 is available**: `python/3.9.12/gcc/pdcqf4o5`

## Quick Fix on HPC

Run these commands manually:

```bash
# Load Python 3.9.12
module load python/3.9.12/gcc/pdcqf4o5

# Verify
python3 --version  # Should show 3.9.12

# Continue with venv setup
cd ~/mr_ts_play
bash experiments/cam_human_like/training/setup_rds_venv.sh
```

Or manually complete the setup:

```bash
# Load Python module
module load python/3.9.12/gcc/pdcqf4o5

# Create venv on RDS
python3 -m venv --without-pip ~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/venv

# Activate
source ~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/venv/bin/activate

# Install pip
curl -sS https://bootstrap.pypa.io/get-pip.py | python3

# Install packages
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers pillow opencv-python tqdm numpy pandas

# Create symlink
cd ~/mr_ts_play
ln -s ~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/venv venv
```

## Updated Scripts

I've updated the scripts to try Python 3.9.12 as a fallback. After transferring the updated scripts, they should work automatically.

## Verify Python Version

After loading the module:
```bash
python3 --version  # Should be 3.9.12 or higher
```

Python 3.6.8 is too old - PyTorch requires Python 3.8+.


