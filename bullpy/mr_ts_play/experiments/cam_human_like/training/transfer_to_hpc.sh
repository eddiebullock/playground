#!/bin/bash
# Transfer project code to HPC for CAM replication
# Run this from your LOCAL machine

set -e

HPC_USER="eb2007"
HPC_HOST="login-cpu.hpc.cam.ac.uk"
HPC_PATH="~/mr_ts_play"
LOCAL_PROJECT="/Users/eb2007/playground/bullpy/mr_ts_play"

# SSH ControlMaster: Reuse SSH connection to avoid multiple authentications
SSH_CONTROL_DIR="${HOME}/.ssh/control"
mkdir -p "$SSH_CONTROL_DIR"
SSH_OPTS="-o ControlMaster=auto -o ControlPath=${SSH_CONTROL_DIR}/%r@%h:%p -o ControlPersist=300"

echo "============================================================"
echo "Transferring Project Code to HPC"
echo "============================================================"
echo ""
echo "Local: $LOCAL_PROJECT"
echo "HPC: ${HPC_USER}@${HPC_HOST}:${HPC_PATH}"
echo ""
echo "Note: You will only need to authenticate ONCE (SSH connection reuse enabled)"
echo ""

# Transfer main project code (excluding large files)
echo "Step 1: Transferring project code..."
rsync -avh --progress -e "ssh $SSH_OPTS" \
  --exclude 'venv/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.git/' \
  --exclude 'results/' \
  --exclude 'models/' \
  --exclude '*.pth' \
  --exclude '*.safetensors' \
  --exclude '.DS_Store' \
  --exclude '*.log' \
  "${LOCAL_PROJECT}/" \
  "${HPC_USER}@${HPC_HOST}:${HPC_PATH}/"

echo ""
echo "Step 2: Transferring trial definitions..."
rsync -avh --progress -e "ssh $SSH_OPTS" \
  "${LOCAL_PROJECT}/data/cam_trial_definitions_20concepts.json" \
  "${HPC_USER}@${HPC_HOST}:${HPC_PATH}/data/"

echo ""
echo "Step 3: Transferring HPC scripts..."
rsync -avh --progress -e "ssh $SSH_OPTS" \
  "${LOCAL_PROJECT}/experiments/cam_human_like/training/hpc_cam_replication.sh" \
  "${LOCAL_PROJECT}/experiments/cam_human_like/training/hpc_cam_replication.slurm" \
  "${LOCAL_PROJECT}/experiments/cam_human_like/training/hpc_cam_replication_ampere.slurm" \
  "${LOCAL_PROJECT}/experiments/cam_human_like/training/setup_rds_venv.sh" \
  "${LOCAL_PROJECT}/experiments/cam_human_like/training/transfer_eu_emotions_to_rds.sh" \
  "${HPC_USER}@${HPC_HOST}:${HPC_PATH}/experiments/cam_human_like/training/"

# Clean up SSH control socket
ssh $SSH_OPTS -O exit "${HPC_USER}@${HPC_HOST}" 2>/dev/null || true

echo ""
echo "============================================================"
echo "Transfer Complete!"
echo "============================================================"
echo ""
echo "Next steps on HPC:"
echo "  1. SSH to HPC: ssh ${HPC_USER}@${HPC_HOST}"
echo "  2. cd ~/mr_ts_play"
echo "  3. Set up Python environment (if needed)"
echo "  4. Run: bash experiments/cam_human_like/training/hpc_cam_replication.sh"
echo "  OR submit SLURM job: sbatch experiments/cam_human_like/training/hpc_cam_replication.slurm"
echo ""

