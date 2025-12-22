#!/bin/bash
# Setup Python virtual environment on RDS storage
# This avoids /home quota issues by storing venv on RDS

set -e

echo "============================================================"
echo "Setting up Python Environment on RDS Storage"
echo "============================================================"
echo ""

# RDS project path - using autism-research RDS (project 90416)
# Try different RDS path formats
if [ -d "${HOME}/rds/rds-autism-research-ePtR33Nsgi4" ]; then
    RDS_PROJECT="${HOME}/rds/rds-autism-research-ePtR33Nsgi4"
    RDS_USER_DIR="${RDS_PROJECT}/users/eb2007"
    RDS_VENV="${RDS_USER_DIR}/venv"
    echo "✅ Found RDS at: $RDS_PROJECT"
elif [ -d "/rds/user/eb2007/rds-autism-research-ePtR33Nsgi4" ]; then
    RDS_PROJECT="/rds/user/eb2007/rds-autism-research-ePtR33Nsgi4"
    RDS_USER_DIR="${RDS_PROJECT}/users/eb2007"
    RDS_VENV="${RDS_USER_DIR}/venv"
    echo "✅ Found RDS at: $RDS_PROJECT"
elif [ -d "/rds-d7/project/45718" ]; then
    RDS_PROJECT="/rds-d7/project/45718"
    RDS_USER_DIR="${RDS_PROJECT}/users/eb2007"
    RDS_VENV="${RDS_USER_DIR}/venv"
    echo "✅ Found RDS at: $RDS_PROJECT"
else
    # Default to the path where user created data folder
    RDS_PROJECT="${HOME}/rds/rds-autism-research-ePtR33Nsgi4"
    RDS_USER_DIR="${RDS_PROJECT}/users/eb2007"
    RDS_VENV="${RDS_USER_DIR}/venv"
    echo "⚠️  Using default RDS path: $RDS_PROJECT"
    echo "   Will create if needed"
fi

# Check RDS access
echo "Step 1: Checking RDS access..."
if [ ! -d "$RDS_PROJECT" ]; then
    echo "❌ Error: Cannot access $RDS_PROJECT"
    echo "   Please verify RDS path on HPC:"
    echo "   ls -la ~/rds/rds-autism-research-ePtR33Nsgi4"
    exit 1
else
    echo "✅ RDS accessible at: $RDS_PROJECT"
fi

# Create user directory if needed
mkdir -p "$RDS_USER_DIR"

# Project directory (code stays in /home)
PROJECT_DIR="${HOME}/mr_ts_play"
cd "$PROJECT_DIR" || { echo "Error: Could not cd to $PROJECT_DIR"; exit 1; }

echo ""
echo "Step 2: Checking available Python modules..."
module avail python 2>&1 | head -20 || echo "Note: module command may not be available"

echo ""
echo "Step 3: Loading Python module..."
# Try to load Python module (check what's available first)
# Common versions on CSD3: 3.11.9, 3.9.12, 3.8.11
if module load python/3.11.9/gcc/nptrdpll 2>/dev/null; then
    echo "✅ Loaded python/3.11.9/gcc/nptrdpll"
elif module load python/3.9.12/gcc/pdcqf4o5 2>/dev/null; then
    echo "✅ Loaded python/3.9.12/gcc/pdcqf4o5"
elif module load python/3.9.12 2>/dev/null; then
    echo "✅ Loaded python/3.9.12"
elif module load python/3.9 2>/dev/null; then
    echo "✅ Loaded python/3.9"
elif module load python/3.8.11/gcc/pqdmnzmw 2>/dev/null; then
    echo "✅ Loaded python/3.8.11/gcc/pqdmnzmw"
else
    echo "❌ Error: Could not load Python module"
    echo "   Available Python modules:"
    module avail python 2>&1 | grep -E "python/3\.[89]|python/3\.1" | head -5
    echo ""
    echo "   Please load a Python 3.8+ module manually:"
    echo "   module load python/3.9.12/gcc/pdcqf4o5"
    exit 1
fi

echo ""
echo "Step 4: Creating virtual environment on RDS..."
echo "Venv location: $RDS_VENV"
if [ -d "$RDS_VENV" ]; then
    echo "⚠️  Virtual environment already exists on RDS."
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$RDS_VENV"
        echo "✅ Removed existing venv"
    else
        echo "Using existing venv"
    fi
fi

if [ ! -d "$RDS_VENV" ]; then
    # Create venv WITHOUT pip (workaround for HPC issues)
    echo "Creating virtual environment (without pip, will install manually)..."
    python3 -m venv --without-pip "$RDS_VENV" || python3 -m venv "$RDS_VENV"
    
    # Activate venv
    source "$RDS_VENV/bin/activate"
    
    # Install pip manually
    echo "Installing pip manually..."
    curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    python3 /tmp/get-pip.py --quiet || {
        echo "⚠️  get-pip.py failed, trying alternative method..."
        python3 -m ensurepip --upgrade --default-pip 2>/dev/null || {
            echo "⚠️  ensurepip also failed, but continuing..."
        }
    }
    rm -f /tmp/get-pip.py
    
    echo "✅ Virtual environment created on RDS"
else
    source "$RDS_VENV/bin/activate"
    echo "✅ Using existing venv from RDS"
fi

echo ""
echo "Step 5: Verifying virtual environment..."
python3 --version

echo ""
echo "Step 6: Upgrading pip..."
pip install --upgrade pip --quiet

echo ""
echo "Step 7: Installing PyTorch with CUDA support..."
echo "This may take several minutes..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "Step 8: Installing other dependencies..."
pip install transformers pillow opencv-python tqdm numpy pandas

echo ""
echo "Step 9: Verifying installation..."
python3 -c "
import torch
import transformers
import PIL
import cv2
print('✅ All imports successful!')
print(f'✅ PyTorch version: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ CUDA version: {torch.version.cuda}')
    print(f'✅ GPU device: {torch.cuda.get_device_name(0)}')
else:
    print('⚠️  CUDA not available - GPU jobs will not work')
"

echo ""
echo "Step 10: Creating symlink in project directory..."
cd "$PROJECT_DIR"
if [ -L "venv" ]; then
    rm venv
    echo "✅ Removed existing symlink"
elif [ -d "venv" ]; then
    echo "⚠️  Local venv exists. Remove it first: rm -rf venv"
fi
ln -s "$RDS_VENV" venv
echo "✅ Created symlink: ~/mr_ts_play/venv -> $RDS_VENV"

echo ""
echo "============================================================"
echo "Python Environment Setup Complete on RDS!"
echo "============================================================"
echo ""
echo "Virtual environment location: $RDS_VENV"
echo "Symlink created: ~/mr_ts_play/venv"
echo ""
echo "To activate in future sessions:"
echo "  source ~/mr_ts_play/venv/bin/activate"
echo "  OR"
echo "  source $RDS_VENV/bin/activate"
echo ""
echo "To test:"
echo "  python3 -c 'import torch; print(torch.cuda.is_available())'"
echo ""

