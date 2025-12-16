"""
Utilities for setting random seeds for reproducibility.
"""

import random
import numpy as np
import torch
import os


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Sets seeds for:
    - Python random
    - NumPy
    - PyTorch (CPU and CUDA)
    - Environment variables for CUDA determinism
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)


def worker_init_fn(worker_id: int):
    """Initialize worker with different seed for DataLoader."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

