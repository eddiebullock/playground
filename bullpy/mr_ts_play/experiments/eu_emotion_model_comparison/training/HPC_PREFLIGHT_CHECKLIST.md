# Pre-Flight Checklist for Video Model Fine-Tuning on HPC

Before submitting the job, verify these items on HPC:

## 1. Verify Files Were Transferred

```bash
# SSH to HPC
ssh eb2007@login.hpc.cam.ac.uk

# Check training scripts exist
ls -la ~/mr_ts_play/experiments/eu_emotion_model_comparison/training/hpc_finetune_video_models.*
ls -la ~/mr_ts_play/experiments/eu_emotion_model_comparison/training/finetune_video_models_task_specific.py

# Check model files exist
ls -la ~/mr_ts_play/experiments/eu_emotion_model_comparison/models/video_model_wrappers.py
ls -la ~/mr_ts_play/experiments/eu_emotion_model_comparison/models/video_utils.py
```

## 2. Verify Data Path

```bash
# Check EU emotions data exists at the expected location
ls -la ~/rds/rds-autism-research-ePtR33Nsgi4/data/EU_emotions | head -10

# Verify it has the expected structure
ls -la ~/rds/rds-autism-research-ePtR33Nsgi4/data/EU_emotions/emotions\ */ | head -5
```

## 3. Verify Trial Definitions

```bash
# Check train/val trial files exist
ls -la ~/mr_ts_play/data/trial_definitions/eu_emotion_train.json
ls -la ~/mr_ts_play/data/trial_definitions/eu_emotion_val.json

# Quick check that they have content
head -20 ~/mr_ts_play/data/trial_definitions/eu_emotion_train.json
```

## 4. Verify Data Path in Script

```bash
# Check the data path in the HPC script matches actual location
grep "DATA_ROOT" ~/mr_ts_play/experiments/eu_emotion_model_comparison/training/hpc_finetune_video_models.sh

# Should show:
# DATA_ROOT="${HOME}/rds/rds-autism-research-ePtR33Nsgi4/data/EU_emotions"
```

## 5. Check Python Environment

```bash
# Activate venv and check Python version
source ~/mr_ts_play/venv/bin/activate
python --version

# Check required packages are installed
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import pytorchvideo; print('pytorchvideo: OK')" 2>/dev/null || echo "pytorchvideo: NOT INSTALLED"
python -c "from transformers import TimesformerForVideoClassification; print('transformers: OK')" 2>/dev/null || echo "transformers: NOT INSTALLED"
```

## 6. Test Script Syntax

```bash
# Check script syntax (should not error)
bash -n ~/mr_ts_play/experiments/eu_emotion_model_comparison/training/hpc_finetune_video_models.sh

# Check SLURM script syntax
bash -n ~/mr_ts_play/experiments/eu_emotion_model_comparison/training/hpc_finetune_video_models.slurm
```

## 7. Verify Output Directory

```bash
# Check models directory exists (or will be created)
ls -la ~/mr_ts_play/models/ 2>/dev/null || echo "models/ directory will be created"
```

## 8. Quick Test Run (Optional)

```bash
# Test that the script can at least start (will fail quickly if paths wrong)
cd ~/mr_ts_play
source venv/bin/activate

# Dry run - just check imports and paths (will fail fast if wrong)
python -c "
import sys
sys.path.insert(0, '.')
from experiments.eu_emotion_model_comparison.training.finetune_video_models_task_specific import *
print('✅ Imports successful')
"
```

## 9. Check HPC Resources

```bash
# Check available partitions
sinfo | grep -E "PARTITION|icelake"

# Check your job limits
sacctmgr show user $USER -p

# Check current queue
squeue -u $USER
```

## 10. Final Verification

```bash
# Make sure scripts are executable
chmod +x ~/mr_ts_play/experiments/eu_emotion_model_comparison/training/hpc_finetune_video_models.sh
chmod +x ~/mr_ts_play/experiments/eu_emotion_model_comparison/training/hpc_finetune_video_models.slurm
```

## Quick One-Liner Check

Run this to check most things at once:

```bash
cd ~/mr_ts_play && \
echo "=== Files ===" && \
ls -1 experiments/eu_emotion_model_comparison/training/hpc_finetune_video_models.* && \
echo "=== Data ===" && \
ls -d ~/rds/rds-autism-research-ePtR33Nsgi4/data/EU_emotions && \
echo "=== Trials ===" && \
ls -1 data/trial_definitions/eu_emotion_{train,val}.json && \
echo "=== Python ===" && \
source venv/bin/activate && python -c "import torch; print('PyTorch OK')" && \
echo "✅ All checks passed!"
```

## If Something is Missing

- **Missing files**: Re-run `./transfer_video_finetuning_to_hpc.sh` from local
- **Missing data**: Data should be at `~/rds/rds-autism-research-ePtR33Nsgi4/data/EU_emotions`
- **Missing trials**: Transfer from local: `rsync -avz data/trial_definitions/eu_emotion_*.json eb2007@login.hpc.cam.ac.uk:~/mr_ts_play/data/trial_definitions/`
- **Missing packages**: Install in venv: `pip install pytorchvideo transformers`
