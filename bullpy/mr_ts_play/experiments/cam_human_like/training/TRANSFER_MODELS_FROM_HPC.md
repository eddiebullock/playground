# Transfer Fine-Tuned Models from HPC to Local

This guide explains how to transfer your fine-tuned models from the HPC cluster to your local computer for use in local experiments.

## 📍 Model Locations on HPC

Your models are saved on RDS at:

### CAM Model
```
~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/mr_ts_play_results/cam_replication/model_checkpoints/best_model/
```

### EU-Emotion Model
```
~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/mr_ts_play_results/eu_emotion_replication/model_checkpoints/best_model/
```

### What's in Each Model Directory?

Each `best_model/` directory contains:
- `config.json` - Model configuration
- `pytorch_model.bin` - Model weights (or `model.safetensors`)
- `preprocessor_config.json` - CLIP processor configuration
- `tokenizer_config.json` - Tokenizer configuration
- `vocab.json` - Vocabulary file
- `merges.txt` - BPE merges (for tokenizer)

## 🚀 Transfer Commands

### Option 1: Transfer Both Models (Recommended)

From your **local computer**, run:

```bash
# Set your HPC hostname (adjust if different)
HPC_HOST="eb2007@login.hpc.cam.ac.uk"  # or your HPC login node

# Transfer CAM model
rsync -avz --progress \
  ${HPC_HOST}:~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/mr_ts_play_results/cam_replication/model_checkpoints/best_model/ \
  models/cam_finetuned_best/

# Transfer EU-Emotion model
rsync -avz --progress \
  ${HPC_HOST}:~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/mr_ts_play_results/eu_emotion_replication/model_checkpoints/best_model/ \
  models/eu_emotion_finetuned_best/

# Also transfer evaluation results (optional but useful)
rsync -avz --progress \
  ${HPC_HOST}:~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/mr_ts_play_results/cam_replication/model_checkpoints/cam_evaluation_test.json \
  results/cam_evaluation_test.json

rsync -avz --progress \
  ${HPC_HOST}:~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/mr_ts_play_results/eu_emotion_replication/model_checkpoints/eu_emotion_evaluation_test.json \
  results/eu_emotion_evaluation_test.json
```

### Option 2: Transfer Individual Models

**CAM Model Only:**
```bash
rsync -avz --progress \
  eb2007@login.hpc.cam.ac.uk:~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/mr_ts_play_results/cam_replication/model_checkpoints/best_model/ \
  models/cam_finetuned_best/
```

**EU-Emotion Model Only:**
```bash
rsync -avz --progress \
  eb2007@login.hpc.cam.ac.uk:~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/mr_ts_play_results/eu_emotion_replication/model_checkpoints/best_model/ \
  models/eu_emotion_finetuned_best/
```

### Option 3: Transfer All Checkpoints (Including Epoch Checkpoints)

If you want all epoch checkpoints too:

```bash
# Transfer entire CAM checkpoints directory
rsync -avz --progress \
  ${HPC_HOST}:~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/mr_ts_play_results/cam_replication/model_checkpoints/ \
  models/cam_replication_checkpoints/

# Transfer entire EU-Emotion checkpoints directory
rsync -avz --progress \
  ${HPC_HOST}:~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/mr_ts_play_results/eu_emotion_replication/model_checkpoints/ \
  models/eu_emotion_replication_checkpoints/
```

## 📂 Local Directory Structure

After transfer, your local structure will be:

```
mr_ts_play/
├── models/
│   ├── cam_finetuned_best/
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   ├── preprocessor_config.json
│   │   ├── tokenizer_config.json
│   │   ├── vocab.json
│   │   └── merges.txt
│   └── eu_emotion_finetuned_best/
│       ├── config.json
│       ├── pytorch_model.bin
│       └── ... (same structure)
└── results/
    ├── cam_evaluation_test.json
    └── eu_emotion_evaluation_test.json
```

## 🔍 Verify Transfer

After transferring, verify the models are complete:

```bash
# Check CAM model
ls -lh models/cam_finetuned_best/
# Should see: config.json, pytorch_model.bin, preprocessor_config.json, etc.

# Check EU-Emotion model
ls -lh models/eu_emotion_finetuned_best/
# Should see: config.json, pytorch_model.bin, preprocessor_config.json, etc.

# Check file sizes (pytorch_model.bin should be ~150-200MB)
du -sh models/cam_finetuned_best/
du -sh models/eu_emotion_finetuned_best/
```

## 💻 Loading Models in Local Experiments

### Example: Load and Use CAM Model

```python
from transformers import CLIPModel, CLIPProcessor
from pathlib import Path

# Load the fine-tuned model
model_path = Path("models/cam_finetuned_best")
model = CLIPModel.from_pretrained(str(model_path))
processor = CLIPProcessor.from_pretrained(str(model_path))

# Use the model for inference
# ... your inference code here ...
```

### Example: Evaluate Model Locally

```python
# Evaluate CAM model
python experiments/cam_human_like/training/evaluate_on_cam.py \
    --model_path models/cam_finetuned_best \
    --trial_definitions_file data/cam_trial_definitions_test.json \
    --data_root /path/to/cam/stimuli \
    --dataset_type cam \
    --device cpu

# Evaluate EU-Emotion model
python experiments/cam_human_like/training/evaluate_on_cam.py \
    --model_path models/eu_emotion_finetuned_best \
    --trial_definitions_file data/eu_emotion_trial_definitions_test.json \
    --data_root /path/to/eu_emotion/stimuli \
    --dataset_type eu_emotion \
    --device cpu
```

## 📊 Transfer Trial Definitions (Optional)

You may also want to transfer the trial definitions for consistency:

```bash
# Transfer CAM trial definitions
rsync -avz --progress \
  ${HPC_HOST}:~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/mr_ts_play_results/cam_replication/cam_trial_definitions_*.json \
  data/cam_trials/

# Transfer EU-Emotion trial definitions
rsync -avz --progress \
  ${HPC_HOST}:~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/mr_ts_play_results/eu_emotion_replication/eu_emotion_trial_definitions_*.json \
  data/eu_emotion_trials/
```

## ⚠️ Troubleshooting

### Issue: "Permission denied" or "Connection refused"
- Make sure you're connected to the HPC cluster via SSH
- Verify the path exists on HPC: `ssh eb2007@login.hpc.cam.ac.uk "ls -la ~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/mr_ts_play_results/"`

### Issue: "No such file or directory"
- Check the exact path on HPC first:
  ```bash
  ssh eb2007@login.hpc.cam.ac.uk "ls -la ~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/mr_ts_play_results/cam_replication/model_checkpoints/"
  ```

### Issue: Transfer is very slow
- Model files are ~150-200MB each, so transfer may take a few minutes
- Use `--progress` flag to see transfer progress
- Consider transferring during off-peak hours

### Issue: Model won't load locally
- Ensure you have the same versions of `transformers` and `torch` installed locally
- Check that all files were transferred (especially `pytorch_model.bin` or `model.safetensors`)

## 📝 Quick Reference

**HPC Model Paths:**
- CAM: `~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/mr_ts_play_results/cam_replication/model_checkpoints/best_model/`
- EU-Emotion: `~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/mr_ts_play_results/eu_emotion_replication/model_checkpoints/best_model/`

**Local Model Paths (after transfer):**
- CAM: `models/cam_finetuned_best/`
- EU-Emotion: `models/eu_emotion_finetuned_best/`

**Model Size:** ~150-200MB per model (pytorch_model.bin is the largest file)


