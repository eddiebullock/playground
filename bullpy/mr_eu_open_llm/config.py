from pathlib import Path
import os


# Random seeds
SEED = 42

PROTOCOL_VERSION = "v2-study3-full-eu-6afc"

# Base directories (local)
PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_DATA_DIR = PROJECT_ROOT / "data"
LOCAL_RESULTS_DIR = PROJECT_ROOT / "results"
LOCAL_MODELS_DIR = PROJECT_ROOT / "models"
LOCAL_CACHE_DIR = LOCAL_DATA_DIR / "cache"


# HPC paths (CSD3)
HPC_USER = "eb2007"
HPC_LOGIN_HOST = "login.hpc.cam.ac.uk"

# study3 = data/results home for this experiment. Override with MR_EU_HPC_PROJECT=study2 if needed.
HPC_PROJECT = os.environ.get("MR_EU_HPC_PROJECT", "study3")
HPC_WORK_BASE = Path("/home") / HPC_USER / "rds" / "hpc-work" / HPC_PROJECT
HPC_STUDY2_BASE = Path("/home") / HPC_USER / "rds" / "hpc-work" / "study2"
# Reuse study2 checkpoints/envs (do not retrain for study3 setup).
HPC_MODELS_DIR = Path(os.environ.get("MR_EU_HPC_MODELS_DIR", str(HPC_STUDY2_BASE / "models")))
HPC_DATA_DIR = HPC_WORK_BASE / "data"
HPC_RESULTS_DIR = HPC_WORK_BASE / "results"

# Projects / accounts
HPC_GPU_PROJECT = "BARON-COHEN-SL3-GPU"
HPC_CPU_PROJECT = "BARON-COHEN-SL3-CPU"


# Study 1/2 model set (InternVL2 excluded 2026-06: transformers 5.x load failures on CSD3).
STUDY_MODELS = ("qwen2vl", "llavanext", "gemma4")

# study3 tiers. Kept separate from STUDY_MODELS because study2 results depend on that
# tuple (llavanext membership, and CONFIRMATORY_N_MODELS drives Bonferroni correction).
STUDY3_BENCHMARK_MODELS = ("gemma4", "qwen3vl", "molmo2", "llama4")
STUDY3_MECHANISTIC_MODELS = ("qwen3vl", "molmo2")

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
    "qwen3vl": {
        "hf_id": "Qwen/Qwen3-VL-8B-Instruct",
        "local_path": LOCAL_MODELS_DIR / "qwen3vl",
        "hpc_path": HPC_MODELS_DIR / "qwen3vl",
    },
    "molmo2": {
        "hf_id": "allenai/Molmo2-O-7B",
        "local_path": LOCAL_MODELS_DIR / "molmo2",
        "hpc_path": HPC_MODELS_DIR / "molmo2",
    },
    "llama4": {
        "hf_id": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "local_path": LOCAL_MODELS_DIR / "llama4",
        "hpc_path": HPC_MODELS_DIR / "llama4",
        "gated": True,
    },
}


# Per-architecture LoRA target module names (exact names may differ per checkpoint)
LORA_TARGET_MODULES = {
    "qwen2vl": ["q_proj", "v_proj"],
    "llavanext": ["q_proj", "v_proj"],
    "gemma4": ["q_proj", "v_proj"],
    "qwen3vl": ["q_proj", "v_proj"],
    "molmo2": ["q_proj", "v_proj"],
    "llama4": ["q_proj", "v_proj"],
}


def lora_alpha_for_rank(r: int) -> int:
    return 2 * r


# Dataset paths (study3: full Faces EDITED + Fixed UK Voices; not the 118-trial subset)
DATASETS = {
    "eu_emotions": {
        "local": LOCAL_DATA_DIR / "eu_emotions",
        "hpc": HPC_DATA_DIR / "eu_emotions",
        "manifest_hpc": HPC_DATA_DIR / "eu_emotions_full_manifest.json",
        "manifest_local": LOCAL_DATA_DIR / "eu_emotions_full_manifest.json",
        # Legacy study2 118-trial paths (still on study2 disk; do not use for study3 eval)
        "root_118_hpc": HPC_STUDY2_BASE / "data" / "eu_emotions_118",
        "root_118_local": LOCAL_DATA_DIR / "eu_emotions_118",
        "manifest_118_hpc": HPC_STUDY2_BASE / "data" / "eu_emotions_118_manifest.json",
        "manifest_118_local": LOCAL_DATA_DIR / "eu_emotions_118_manifest.json",
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
N_EU_EMOTIONS_LABELS = 27  # full taxonomy (6-AFC foils in study3); entropy pool excludes neutral
N_OPTIONS = 6  # study3 Stage-2 forced choice (study2 used 4)
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

# RQ1.1b forced-choice response entropy (primary calibration metric).
# Sampled at temperature 1.0, not EVAL["temperature"]=0.1: at 0.1 the response
# distribution is near-degenerate, so model entropy would be ~0 by construction and the
# correlation against human entropy would be floored by the decoding setting.
FORCED_CHOICE = {
    "n_samples": 20,
    "temperature": 1.0,
    "sensitivity_temperatures": (0.7,),
    # Options come from the human study's own per-item set (target + 4 controls +
    # "None of the above") so RQ1.1b compares like with like; see scripts/human_entropy.py.
    "use_human_option_sets": True,
    "human_entropy_lookup": LOCAL_DATA_DIR / "eu_emotions_human_entropy.json",
}

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
# Note: published human figures are often 6AFC — aligned with study3 N_OPTIONS.
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
    "gemma4": True,  # Gemma 4 E4B-it: native audio (audio_config present in checkpoint)
    "qwen3vl": False,  # Qwen3-VL: text/image/video only (audio lives in Qwen3-Omni)
    "molmo2": False,  # Molmo2-O: Olmo3 text + SigLIP2 vision/video; no audio tower
    "llama4": False,  # Llama 4: early-fusion text+image; no audio input
}

# LoRA fine-tuning modality (Mindreading item-folder audio; never Emotions/Audio/).
FINETUNE_MODALITY = "multimodal"
FINETUNE_MODALITY_BY_MODEL = {
    "gemma4": "multimodal",
    "qwen2vl": "video_only",
    "llavanext": "video_only",
    "qwen3vl": "video_only",
    "molmo2": "video_only",
}

EU_EMOTION_LABELS_FILE = LOCAL_DATA_DIR / "eu_emotion_states_list.txt"

CHANCE_LEVEL = 1.0 / float(N_OPTIONS)  # study3 6AFC; study2 was 0.25
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
    "HPC_PROJECT",
    "HPC_WORK_BASE",
    "HPC_STUDY2_BASE",
    "HPC_MODELS_DIR",
    "HPC_DATA_DIR",
    "HPC_RESULTS_DIR",
    "HPC_GPU_PROJECT",
    "HPC_CPU_PROJECT",
    "STUDY_MODELS",
    "STUDY3_BENCHMARK_MODELS",
    "STUDY3_MECHANISTIC_MODELS",
    "MODELS",
    "LORA_TARGET_MODULES",
    "lora_alpha_for_rank",
    "DATASETS",
    "FRAME_POLICY",
    "N_EU_EMOTIONS_LABELS",
    "N_OPTIONS",
    "EMBEDDING_MODEL",
    "ENTROPY_TEMPERATURE",
    "ENTROPY_LOG_BASE",
    "CHAIN_STAGES",
    "STAGE1_MAX_NEW_TOKENS",
    "STAGE2_MAX_NEW_TOKENS",
    "LAYER_DEPTH_FRACTIONS",
    "BEST_MODEL_KEY",
    "EVAL",
    "FORCED_CHOICE",
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
