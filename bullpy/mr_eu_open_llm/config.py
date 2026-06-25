from pathlib import Path


# Random seeds
SEED = 42

PROTOCOL_VERSION = "v2-study1-study2"

# Base directories (local)
PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_DATA_DIR = PROJECT_ROOT / "data"
LOCAL_RESULTS_DIR = PROJECT_ROOT / "results"
LOCAL_MODELS_DIR = PROJECT_ROOT / "models"
LOCAL_CACHE_DIR = LOCAL_DATA_DIR / "cache"


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


# Study 1/2 model set (InternVL2 excluded 2026-06: transformers 5.x load failures on CSD3).
STUDY_MODELS = ("qwen2vl", "llavanext", "gemma4")

# Model identifiers and local/HPC paths
MODELS = {
    "qwen2vl": {
        "hf_id": "Qwen/Qwen2-VL-7B-Instruct",
        "local_path": LOCAL_MODELS_DIR / "qwen2vl",
        "hpc_path": HPC_MODELS_DIR / "qwen2vl",
    },
    "llavanext": {
        "hf_id": "llava-hf/llava-interleave-qwen-7b-hf",
        "local_path": LOCAL_MODELS_DIR / "llavanext",
        "hpc_path": HPC_MODELS_DIR / "llavanext",
    },
    "gemma4": {
        "hf_id": "google/gemma-4-E4B-it",
        "local_path": LOCAL_MODELS_DIR / "gemma4",
        "hpc_path": HPC_MODELS_DIR / "gemma4",
    },
}


# Per-architecture LoRA target module names (exact names may differ per checkpoint)
LORA_TARGET_MODULES = {
    "qwen2vl": ["q_proj", "v_proj"],
    "llavanext": ["q_proj", "v_proj"],
    "gemma4": ["q_proj", "v_proj"],
}


def lora_alpha_for_rank(r: int) -> int:
    return 2 * r


# Dataset paths
DATASETS = {
    "eu_emotions": {
        "local": LOCAL_DATA_DIR / "eu_emotions",
        "hpc": HPC_DATA_DIR / "eu_emotions",
        "manifest_hpc": HPC_DATA_DIR / "eu_emotions_118_manifest.json",
        "manifest_local": LOCAL_DATA_DIR / "eu_emotions_118_manifest.json",
        "root_118_hpc": HPC_DATA_DIR / "eu_emotions_118",
        "root_118_local": LOCAL_DATA_DIR / "eu_emotions_118",
    },
    "mindreading": {
        "local": LOCAL_DATA_DIR / "mindreading",
        "hpc": HPC_DATA_DIR / "mindreading",
    },
    "rmet": {
        "local": LOCAL_DATA_DIR / "rmet",
        "hpc": HPC_DATA_DIR / "rmet",
        "deprecated": True,
    },
}


# Protocol v2 frame policy (all study models)
FRAME_POLICY = {
    "fps": 1,
    "max_frames": 16,
    "ablation_frames": [4, 16],
    "ablation_n_trials": 30,
    "enforce_multi_frame": True,
    # Artifact: report composite-grid (default) AND native multi-frame where supported
    "modes": {
        "composite_grid": {"fps": 1, "max_frames": 16, "enforce_multi_frame": True},
        "native_video": {"fps": 1, "max_frames": 16, "enforce_multi_frame": False},
    },
    "default_mode": "composite_grid",
}

# Semantic entropy (Stage 1 ambiguity index)
N_EU_EMOTIONS_LABELS = 27  # full taxonomy (4-AFC foils); entropy pool excludes neutral
ENTROPY_EXCLUDE_LABELS = ("neutral",)
ENTROPY_USE_RICH_LABEL_EMBEDDINGS = True
ENTROPY_COLLAPSE_INTENSITY = True  # primary H_sem is over base emotions after collapsing intensity
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ENTROPY_TEMPERATURE = 0.1
ENTROPY_LOG_BASE = "e"
# Post-hoc robustness (scripts/entropy_sensitivity.py); does not change primary eval defaults.
ENTROPY_SENSITIVITY_TEMPERATURES = (0.05, 0.1, 0.2)
ENTROPY_SENSITIVITY_EMBEDDING_MODELS = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
)

# Two-stage evaluation
CHAIN_STAGES = False
STAGE1_MAX_NEW_TOKENS = 256
STAGE2_MAX_NEW_TOKENS = 128

# Study 2 layer depth fractions
LAYER_DEPTH_FRACTIONS = [0.125, 0.375, 0.75]

# Set after Study 1 baseline comparison (or via select_best_model.py)
BEST_MODEL_KEY = "gemma4"

# Generation defaults (Stage 2 and legacy alias)
EVAL = {
    "temperature": 0.1,
    "top_p": 1.0,
    "seed": SEED,
    # Deprecated v1 keys (callers should use FRAME_POLICY)
    "n_frames_default": FRAME_POLICY["max_frames"],
    "n_frames_ablation": FRAME_POLICY["ablation_frames"],
}


# LoRA hyperparameters
LORA_DEFAULT = {
    "r": 16,
    "alpha": lora_alpha_for_rank(16),
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
    "mindreading_train_subset": 100,
    "mindreading_val_subset": 50,
}

# Modality-matched EU-Emotions human benchmarks (O'Reilly / Lassalle).
# TODO: fill accuracy/n from O'Reilly et al. (EU-Emotion Stimulus Set) and
# Lassalle et al. (EU-Emotion Voice Database) validation papers.
HUMAN_BENCHMARKS = {
    "eu_emotion": {
        "video_only": {
            "citation": "O'Reilly et al. (EU-Emotion Stimulus Set)",
            "accuracy": None,
            "n": None,
        },
        "audio_only": {
            "citation": "Lassalle et al. (EU-Emotion Voice Database)",
            "accuracy": None,
            "n": None,
        },
        "multimodal": {
            "citation": "O'Reilly et al. (EU-Emotion Stimulus Set)",
            "accuracy": None,
            "n": None,
        },
    },
    "mindreading": {},
}

MODALITY_CONDITIONS = ("video_only", "audio_only", "multimodal")

# Native audio input support in scripts/model_inference.py (audit 2026-06).
# Audio ablations are scheduled only for True entries; infrastructure still validates resolution.
MODEL_AUDIO_CAPABILITIES = {
    "qwen2vl": False,  # Qwen2-VL: vision/video only (use Qwen2-Audio for speech)
    "llavanext": False,
    "gemma4": True,  # Gemma 4 E4B-it: native audio (max ~30s per clip)
}

# LoRA fine-tuning modality (Mindreading item-folder audio; never Emotions/Audio/).
FINETUNE_MODALITY = "multimodal"
FINETUNE_MODALITY_BY_MODEL = {
    "gemma4": "multimodal",
    "qwen2vl": "video_only",
    "llavanext": "video_only",
}

EU_EMOTION_LABELS_FILE = LOCAL_DATA_DIR / "eu_emotion_states_list.txt"

CHANCE_LEVEL = 0.25
CONFIRMATORY_N_MODELS = len(STUDY_MODELS)

# Catastrophic forgetting threshold (percentage points on EU-Emotions)
FORGETTING_THRESHOLD_PP = 5.0


__all__ = [
    "SEED",
    "PROTOCOL_VERSION",
    "PROJECT_ROOT",
    "LOCAL_DATA_DIR",
    "LOCAL_RESULTS_DIR",
    "LOCAL_MODELS_DIR",
    "LOCAL_CACHE_DIR",
    "HPC_USER",
    "HPC_LOGIN_HOST",
    "HPC_WORK_BASE",
    "HPC_MODELS_DIR",
    "HPC_DATA_DIR",
    "HPC_RESULTS_DIR",
    "HPC_GPU_PROJECT",
    "HPC_CPU_PROJECT",
    "STUDY_MODELS",
    "MODELS",
    "LORA_TARGET_MODULES",
    "lora_alpha_for_rank",
    "DATASETS",
    "FRAME_POLICY",
    "N_EU_EMOTIONS_LABELS",
    "EMBEDDING_MODEL",
    "ENTROPY_TEMPERATURE",
    "ENTROPY_LOG_BASE",
    "CHAIN_STAGES",
    "STAGE1_MAX_NEW_TOKENS",
    "STAGE2_MAX_NEW_TOKENS",
    "LAYER_DEPTH_FRACTIONS",
    "BEST_MODEL_KEY",
    "EVAL",
    "LORA_DEFAULT",
    "LORA_SWEEP",
    "TRAINING_DEFAULTS",
    "HUMAN_BENCHMARKS",
    "MODALITY_CONDITIONS",
    "MODEL_AUDIO_CAPABILITIES",
    "FINETUNE_MODALITY",
    "EU_EMOTION_LABELS_FILE",
    "CHANCE_LEVEL",
    "CONFIRMATORY_N_MODELS",
    "FORGETTING_THRESHOLD_PP",
]
