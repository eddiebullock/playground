# Copy to local/site.sh and fill in real values. site.sh is git-ignored.
# Tracked files, commit messages, branches and PR text must never contain the
# real values. Refer to them only as $HPC_USER, $HPC_HOST, $HPC_ROOT, etc.

# Cluster login (password + TOTP; no SSH keys). The user opens a ControlMaster
# socket by hand. Scripts only reuse it.
HPC_USER="YOUR_HPC_USER"
HPC_HOST="login.example.invalid"
HPC_SSH_CONTROL_PATH="${HOME}/.ssh/cm-esvlm-%C"

# Absolute RDS project root for THIS study. Never point this at another
# project's tree (study2 / study3 / study4_rmet or any AV project).
HPC_ROOT="/absolute/path/to/rds/es_vlm"

# SLURM. GPU account is rejected on CPU partitions and vice versa.
HPC_GPU_ACCOUNT="YOUR_GPU_ACCOUNT"
HPC_CPU_ACCOUNT="YOUR_CPU_ACCOUNT"
HPC_GPU_PARTITION="ampere"
HPC_CPU_PARTITION="icelake"

# Read-only reuse of weights already on disk (check for config.json).
# New downloads go to $HPC_ROOT/models, never into another project's tree.
ESVLM_MODELS_DIR="/absolute/path/to/existing/weights"
ESVLM_NEW_MODELS_DIR="${HPC_ROOT}/models"

# Read-only stimuli already on the cluster. Load in place; copy nothing
# into tracked locations.
ESVLM_RMET_DIR="/absolute/path/to/rmet/data"
ESVLM_RMET_MANIFEST="${ESVLM_RMET_DIR}/stimuli/manifest.json"
ESVLM_RMET_ANSWER_KEY="${ESVLM_RMET_DIR}/answer_key/rmet_adult_answer_key.json"
ESVLM_EU_EMOTIONS_DIR="/absolute/path/to/eu_emotions/data"
ESVLM_EU_EMOTIONS_MANIFEST="/absolute/path/to/eu_emotions_full_manifest.json"
ESVLM_EU_EMOTIONS_ENTROPY="/absolute/path/to/eu_emotions_human_entropy.json"
ESVLM_MINDREADING_DIR="/absolute/path/to/mindreading/data"

# Public, redistributable S-domain sets. Never mix with licensed trees.
# Large video sets can live on a project store via ESVLM_CLEVRER_DIR.
ESVLM_PUBLIC_DIR="${HPC_ROOT}/data/public"
ESVLM_PHYSBENCH_DIR="${ESVLM_PUBLIC_DIR}/physbench/raw"
ESVLM_PHYSBENCH_MANIFEST="${ESVLM_PUBLIC_DIR}/physbench/manifest.json"
ESVLM_CLEVRER_DIR="${ESVLM_PUBLIC_DIR}/clevrer/raw"
ESVLM_CLEVRER_MANIFEST="${ESVLM_PUBLIC_DIR}/clevrer/manifest.json"
ESVLM_PIQA_DIR="${ESVLM_PUBLIC_DIR}/piqa/raw"
ESVLM_PIQA_MANIFEST="${ESVLM_PUBLIC_DIR}/piqa/manifest.json"
# Paper 1/2 public annotations. Videos are optional; missing mp4 is
# the NExT-QA text control. Never point MMToM at a 257GB video zip.
ESVLM_NEXTQA_DIR="${ESVLM_PUBLIC_DIR}/nextqa/videos"
ESVLM_NEXTQA_MANIFEST="${ESVLM_PUBLIC_DIR}/nextqa/pairs_val.json"
ESVLM_MMTOM_MANIFEST="${ESVLM_PUBLIC_DIR}/mmtom/manifest.json"

# Caches must be absolute paths on RDS. A quoted tilde over SSH creates a
# literal ~ directory on $HOME (over quota).
ESVLM_CACHE_ROOT="${HPC_ROOT}/cache"

# Memory caps used by scripts/submit.sh. SLURM silently inflates CPU count
# if these are exceeded.
HPC_GPU_MEM_PER_CPU_MAX_MB=8000
HPC_CPU_MEM_PER_CPU_MAX_MB=3370

# Optional extra strings that must never appear in tracked files.
# Comma-separated. Include hostnames, CRSid, institution names, account
# names, and absolute cluster paths.
SITE_DENYLIST="login.example.invalid,YOUR_HPC_USER,YOUR_GPU_ACCOUNT,YOUR_CPU_ACCOUNT"

export HPC_USER HPC_HOST HPC_SSH_CONTROL_PATH HPC_ROOT
export HPC_GPU_ACCOUNT HPC_CPU_ACCOUNT HPC_GPU_PARTITION HPC_CPU_PARTITION
export ESVLM_MODELS_DIR ESVLM_NEW_MODELS_DIR
export ESVLM_RMET_DIR ESVLM_RMET_MANIFEST ESVLM_RMET_ANSWER_KEY
export ESVLM_EU_EMOTIONS_DIR ESVLM_EU_EMOTIONS_MANIFEST ESVLM_EU_EMOTIONS_ENTROPY
export ESVLM_MINDREADING_DIR
export ESVLM_PUBLIC_DIR
export ESVLM_PHYSBENCH_DIR ESVLM_PHYSBENCH_MANIFEST
export ESVLM_CLEVRER_DIR ESVLM_CLEVRER_MANIFEST
export ESVLM_PIQA_DIR ESVLM_PIQA_MANIFEST
export ESVLM_NEXTQA_DIR ESVLM_NEXTQA_MANIFEST ESVLM_MMTOM_MANIFEST
export ESVLM_CACHE_ROOT
export HPC_GPU_MEM_PER_CPU_MAX_MB HPC_CPU_MEM_PER_CPU_MAX_MB
export SITE_DENYLIST
