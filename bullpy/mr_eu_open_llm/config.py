from pathlib import Path


# Random seeds
SEED = 42


# Base directories (local)
PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_DATA_DIR = PROJECT_ROOT / "data"
LOCAL_RESULTS_DIR = PROJECT_ROOT / "results"
LOCAL_MODELS_DIR = PROJECT_ROOT / "models"


# HPC paths (CSD3)
HPC_USER = "eb2007"
HPC_LOGIN_HOST = "login.hpc.cam.ac.uk"

HPC_WORK_BASE = Path("/home") / HPC_USER / "rds" / "hpc-work" / "study2"
HPC_MODELS_DIR = HPC_WORK_BASE / "models"
HPC_DATA_DIR = HPC_WORK_BASE / "data"
HPC_RESULTS_DIR = HPC_WORK_BASE / "results"

# Projects / accounts
HPC_GPU_PROJECT = "BARON-COHEN-SL3-GPU"
HPC_CPU_PROJECT = "BARON-COHEN-SL3-CPU"


# Model identifiers and local/HPC paths
MODELS = {
    "qwen2vl": {
        "hf_id": "Qwen/Qwen2-VL-7B-Instruct",
        "local_path": LOCAL_MODELS_DIR / "qwen2vl",
        "hpc_path": HPC_MODELS_DIR / "qwen2vl",
    },
    "internvl2": {
        "hf_id": "OpenGVLab/InternVL2-8B",
        "local_path": LOCAL_MODELS_DIR / "internvl2",
        "hpc_path": HPC_MODELS_DIR / "internvl2",
    },
    "llavanext": {
        "hf_id": "lmms-lab/llava-next-interleave-7b",
        "local_path": LOCAL_MODELS_DIR / "llavanext",
        "hpc_path": HPC_MODELS_DIR / "llavanext",
    },
}


# Dataset paths
DATASETS = {
    "eu_emotions": {
        "local": LOCAL_DATA_DIR / "eu_emotions",
        "hpc": HPC_DATA_DIR / "eu_emotions",
    },
    "mindreading": {
        "local": LOCAL_DATA_DIR / "mindreading",
        "hpc": HPC_DATA_DIR / "mindreading",
    },
    "rmet": {
        "local": LOCAL_DATA_DIR / "rmet",
        "hpc": HPC_DATA_DIR / "rmet",
    },
}


# Evaluation parameters
EVAL = {
    "n_frames_default": 4,
    "n_frames_ablation": [4, 8],
    "frame_sampling_4": [0.0, 0.25, 0.5, 0.75],
    "frame_sampling_8": [0.0, 0.2, 0.35, 0.5, 0.6, 0.7, 0.8, 1.0],
    "temperature": 0.1,
    "top_p": 1.0,
    "seed": SEED,
}


# LoRA hyperparameters (defaults and search space)
LORA_DEFAULT = {
    "r": 16,
    "alpha": 32,
    "target_modules": ["q_proj", "v_proj"],
    "dropout": 0.1,
}

LORA_SWEEP = {
    "learning_rates": [1e-4, 5e-5, 1e-5],
    "ranks": [8, 16, 32],
}

TRAINING_DEFAULTS = {
    "epochs": 5,
    "batch_size": 4,
    "grad_accum_steps": 4,
    "fp16": True,
    "save_every_epoch": True,
}


# Human benchmark values (Golan et al. 2006)
HUMAN_BENCHMARKS = {
    "non_autistic": {
        "accuracy": 0.8629,
        "n": 17,
    },
    "autistic": {
        "accuracy": 0.6805,
        "n": 21,
    },
}


# Chance level for 4AFC
CHANCE_LEVEL = 0.25


__all__ = [
    "SEED",
    "PROJECT_ROOT",
    "LOCAL_DATA_DIR",
    "LOCAL_RESULTS_DIR",
    "LOCAL_MODELS_DIR",
    "HPC_USER",
    "HPC_LOGIN_HOST",
    "HPC_WORK_BASE",
    "HPC_MODELS_DIR",
    "HPC_DATA_DIR",
    "HPC_RESULTS_DIR",
    "HPC_GPU_PROJECT",
    "HPC_CPU_PROJECT",
    "MODELS",
    "DATASETS",
    "EVAL",
    "LORA_DEFAULT",
    "LORA_SWEEP",
    "TRAINING_DEFAULTS",
    "HUMAN_BENCHMARKS",
    "CHANCE_LEVEL",
]

