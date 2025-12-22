# HPC Python Environment Setup for CSD3

## Quick Setup (Recommended)

### Step 1: Check Available Python Modules

```bash
ssh eb2007@login-cpu.hpc.cam.ac.uk
module avail python
```

### Step 2: Load Python Module and Create Virtual Environment

```bash
# Load Python module (adjust version as needed)
module purge
module load python/3.11.9/gcc/nptrdpll  # or latest available

# Create virtual environment in your project directory
cd ~/mr_ts_play
python -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support (for GPU nodes)
# Check CUDA version on HPC: module avail cuda
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install transformers pillow opencv-python tqdm numpy pandas

# Verify PyTorch CUDA
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
```

### Step 4: Test Installation

```bash
# Test imports
python -c "import torch; import transformers; import PIL; import cv2; print('All imports successful!')"
```

## Alternative: Using Conda

If you prefer Conda:

```bash
# Load Conda module
module load miniconda/3

# Initialize Conda (first time only)
conda init
source ~/.bashrc

# Create environment
conda create --name cam_replication python=3.11
conda activate cam_replication

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other packages
pip install transformers pillow opencv-python tqdm numpy pandas
```

## For SLURM Jobs

Update your SLURM script to activate the environment:

```bash
# In hpc_cam_replication.slurm, add:
module load python/3.11.9/gcc/nptrdpll  # or your Python version
source ${HOME}/mr_ts_play/venv/bin/activate
```

## Troubleshooting

### CUDA Not Available

If `torch.cuda.is_available()` returns False:
1. Check CUDA module: `module avail cuda`
2. Load CUDA: `module load cuda/11.8` (or latest)
3. Reinstall PyTorch with correct CUDA version

### Module Not Found

If Python modules aren't found:
1. Check available modules: `module avail`
2. Try different Python version: `module avail python`
3. Use system Python as fallback (not recommended)

### Out of Space

If installation fails due to space:
1. Check quota: `quota -s`
2. Clean pip cache: `pip cache purge`
3. Use `--no-cache-dir` flag: `pip install --no-cache-dir package_name`

