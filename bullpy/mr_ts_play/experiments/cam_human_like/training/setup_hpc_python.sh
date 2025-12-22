#!/bin/bash
# Setup Python environment on CSD3 HPC
# Run this script on HPC after transferring code

set -e

echo "============================================================"
echo "Setting up Python Environment on CSD3"
echo "============================================================"
echo ""

# Check if we're on HPC
if [[ ! -f /etc/os-release ]] || ! grep -q "Red Hat\|CentOS" /etc/os-release 2>/dev/null; then
    echo "Warning: This script is designed for CSD3 HPC. Continuing anyway..."
fi

# Project directory
PROJECT_DIR="${HOME}/mr_ts_play"
cd "$PROJECT_DIR" || { echo "Error: Could not cd to $PROJECT_DIR"; exit 1; }

echo "Step 1: Checking available Python modules..."
module avail python 2>&1 | head -20 || echo "Note: module command may not be available"

echo ""
echo "Step 2: Loading Python module..."
# Try to load Python module (adjust version as needed)
if module load python/3.11.9/gcc/nptrdpll 2>/dev/null; then
    echo "✅ Loaded python/3.11.9/gcc/nptrdpll"
elif module load python/3.9 2>/dev/null; then
    echo "✅ Loaded python/3.9"
else
    echo "⚠️  Could not load Python module, using system Python"
    echo "   You may need to: module load python/3.11.9/gcc/nptrdpll"
fi

echo ""
echo "Step 3: Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists. Removing..."
    rm -rf venv
fi

# Try creating venv without pip first (workaround for HPC issues)
echo "Creating virtual environment (without pip, will install manually)..."
python3 -m venv --without-pip venv || python3 -m venv venv

# Activate venv
source venv/bin/activate

# Install pip manually using get-pip.py
echo "Installing pip manually..."
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
python3 /tmp/get-pip.py --quiet || {
    echo "⚠️  get-pip.py failed, trying alternative method..."
    # Alternative: download pip wheel
    python3 -m ensurepip --upgrade --default-pip 2>/dev/null || {
        echo "⚠️  ensurepip also failed, but continuing..."
    }
}
rm -f /tmp/get-pip.py

echo "✅ Virtual environment created"

# Venv already activated in Step 3
echo ""
echo "Step 4: Verifying virtual environment..."
if [ -f "venv/bin/python3" ]; then
    echo "✅ Virtual environment is active"
    python3 --version
else
    echo "❌ Error: Virtual environment creation failed"
    exit 1
fi

echo ""
echo "Step 5: Upgrading pip..."
pip install --upgrade pip --quiet

echo ""
echo "Step 6: Installing PyTorch with CUDA support..."
echo "This may take several minutes..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "Step 7: Installing other dependencies..."
pip install transformers pillow opencv-python tqdm numpy pandas

echo ""
echo "Step 8: Verifying installation..."
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
echo "============================================================"
echo "Python Environment Setup Complete!"
echo "============================================================"
echo ""
echo "To activate the environment in future sessions:"
echo "  source ~/mr_ts_play/venv/bin/activate"
echo ""
echo "To test:"
echo "  python3 -c 'import torch; print(torch.cuda.is_available())'"
echo ""

