#!/bin/bash

echo "=== Fixing Python Environment Issues ==="

# Remove the broken virtual environment
echo "Removing broken virtual environment..."
rm -rf venv

# Check available Python versions
echo "Available Python modules:"
module avail python 2>&1 | head -20

# Check available GCC versions
echo "Available GCC modules:"
module avail gcc 2>&1 | head -20

# Try to load Python 3.6 (or available version)
echo "Loading Python module..."
if module load python/3.6 2>/dev/null; then
    echo "Successfully loaded python/3.6"
    PYTHON_VERSION="3.6"
elif module load python/3.8 2>/dev/null; then
    echo "Successfully loaded python/3.8"
    PYTHON_VERSION="3.8"
elif module load python/3.9 2>/dev/null; then
    echo "Successfully loaded python/3.9"
    PYTHON_VERSION="3.9"
else
    echo "Available Python versions:"
    module avail python
    echo "Please specify which Python version to use"
    exit 1
fi

# Try to load available GCC
echo "Loading GCC module..."
if module load gcc/9.3.0 2>/dev/null; then
    echo "Successfully loaded gcc/9.3.0"
elif module load gcc/8.3.0 2>/dev/null; then
    echo "Successfully loaded gcc/8.3.0"
elif module load gcc/7.3.0 2>/dev/null; then
    echo "Successfully loaded gcc/7.3.0"
else
    echo "No specific GCC version loaded, using system default"
fi

# Create new virtual environment
echo "Creating new virtual environment with Python $PYTHON_VERSION..."
python3 -m venv venv

# Activate and upgrade pip
echo "Activating virtual environment and upgrading pip..."
source venv/bin/activate
pip install --upgrade pip

# Install requirements with compatibility check
echo "Installing requirements..."
pip install -r requirements.txt

echo "=== Environment Setup Complete ==="
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"
echo "Virtual environment: $(which python)"
