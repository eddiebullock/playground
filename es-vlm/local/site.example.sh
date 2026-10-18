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
ESVLM_EU_EMOTIONS_DIR="/absolute/path/to/eu_emotions/data"
ESVLM_MINDREADING_DIR="/absolute/path/to/mindreading/data"

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
export ESVLM_RMET_DIR ESVLM_EU_EMOTIONS_DIR ESVLM_MINDREADING_DIR
export ESVLM_CACHE_ROOT
export HPC_GPU_MEM_PER_CPU_MAX_MB HPC_CPU_MEM_PER_CPU_MAX_MB
export SITE_DENYLIST
