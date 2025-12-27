# Next Steps: Basic Emotions Experiment

## Step 1: Transfer Experiment Code to HPC (Do This First!)

From your **local machine**:

```bash
cd /Users/eb2007/playground/bullpy/mr_ts_play
bash experiments/basic_emotions_recognition/training/transfer_to_hpc.sh
```

This will transfer:
- The entire `experiments/basic_emotions_recognition/` directory
- Basic emotion mapping files
- Configuration files

**Expected output**: Shows progress of files being transferred to HPC.

---

## Step 2: Verify Data is on HPC

SSH to HPC and check:

```bash
ssh eb2007@login-cpu.hpc.cam.ac.uk

# Check CAM data (should already be there)
ls -la /home/eb2007/data/CAM

# Check EU-Emotion data (should be on RDS)
ls -la ~/rds/rds-autism-research-ePtR33Nsgi4/data/EU_emotions

# If EU-Emotion data is missing, transfer it from local:
# (Exit HPC first, then from local machine:)
# bash experiments/cam_human_like/training/transfer_eu_emotions_to_rds.sh
```

**If EU-Emotion data is missing**, exit HPC and run from local:
```bash
bash experiments/cam_human_like/training/transfer_eu_emotions_to_rds.sh
```

---

## Step 3: Generate Basic Emotion Trials (On HPC)

SSH to HPC:

```bash
ssh eb2007@login-cpu.hpc.cam.ac.uk
cd ~/mr_ts_play
```

### For CAM:

```bash
python experiments/basic_emotions_recognition/training/create_basic_emotion_trials.py \
    --dataset_type cam \
    --input_trials data/trial_definitions/cam_test.json \
    --mapping_file data/basic_emotion_mapping.json \
    --output_dir ~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/mr_ts_play_results/basic_emotions_cam \
    --train_ratio 0.8 \
    --seed 42
```

### For EU-Emotion:

```bash
python experiments/basic_emotions_recognition/training/create_basic_emotion_trials.py \
    --dataset_type eu_emotion \
    --input_trials data/trial_definitions/eu_emotion_test.json \
    --mapping_file experiments/basic_emotions_recognition/data/basic_emotion_mappings/eu_emotion_basic_mapping.json \
    --output_dir ~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/mr_ts_play_results/basic_emotions_eu_emotion \
    --train_ratio 0.8 \
    --seed 42
```

**Note**: Outputs go to RDS to avoid `/home` quota issues.

---

## Step 4: Train Models on HPC (Submit SLURM Jobs)

Still on HPC:

### For CAM:

```bash
sbatch experiments/basic_emotions_recognition/training/hpc_basic_emotions_cam.slurm
```

### For EU-Emotion:

```bash
sbatch experiments/basic_emotions_recognition/training/hpc_basic_emotions_eu_emotion.slurm
```

### Check Job Status:

```bash
squeue -u eb2007
```

### Monitor Progress:

```bash
# Watch output files
tail -f basic_emotions_cam_*.out
tail -f basic_emotions_eu_*.out
```

**Expected time**: 8-12 hours per dataset (CPU training)

---

## Step 5: Verify Training Completed

After jobs finish, check results:

```bash
# On HPC
ls -la ~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/mr_ts_play_results/basic_emotions_cam/
ls -la ~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/mr_ts_play_results/basic_emotions_eu_emotion/

# Check for best model
ls -la ~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/mr_ts_play_results/basic_emotions_cam/model_checkpoints/best_model/
```

---

## Step 6: Transfer Models to Local (For LLM Augmentation)

From your **local machine**:

```bash
# Transfer CAM model
rsync -avz --progress \
    eb2007@login-cpu.hpc.cam.ac.uk:~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/mr_ts_play_results/basic_emotions_cam/model_checkpoints/best_model/ \
    models/basic_emotions_cam_finetuned/

# Transfer EU-Emotion model
rsync -avz --progress \
    eb2007@login-cpu.hpc.cam.ac.uk:~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/mr_ts_play_results/basic_emotions_eu_emotion/model_checkpoints/best_model/ \
    models/basic_emotions_eu_emotion_finetuned/
```

---

## Step 7: Run LLM Augmentation (Local)

On your **local machine**:

### Generate LLM Cache:

```bash
cd /Users/eb2007/playground/bullpy/mr_ts_play

python experiments/basic_emotions_recognition/llm_augmentation/scripts/generate_basic_emotions_llm_cache.py \
    --provider openai \
    --model text-embedding-3-small \
    --cache_dir experiments/basic_emotions_recognition/llm_augmentation/data/llm_cache
```

### Run LLM Experiment:

```bash
python experiments/basic_emotions_recognition/llm_augmentation/scripts/run_basic_emotions_llm_experiment.py \
    --config experiments/basic_emotions_recognition/llm_augmentation/configs/basic_emotions_llm_config.yaml \
    --dataset cam \
    --device cpu \
    --num_frames 8
```

---

## Quick Reference: File Locations

### On HPC:
- **Code**: `~/mr_ts_play/experiments/basic_emotions_recognition/`
- **CAM Data**: `/home/eb2007/data/CAM`
- **EU-Emotion Data**: `~/rds/rds-autism-research-ePtR33Nsgi4/data/EU_emotions`
- **Outputs**: `~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/mr_ts_play_results/basic_emotions_{dataset}/`

### On Local:
- **Code**: `/Users/eb2007/playground/bullpy/mr_ts_play/experiments/basic_emotions_recognition/`
- **Models** (after transfer): `models/basic_emotions_{dataset}_finetuned/`
- **LLM Results**: `results/basic_emotions_{dataset}/`

---

## Troubleshooting

**"Experiment code not found on HPC"**
→ Run Step 1 (transfer code)

**"Data not found"**
→ Check Step 2, transfer EU-Emotion data if needed

**"Trial definitions not found"**
→ Run Step 3 (generate trials)

**"Job failed"**
→ Check SLURM output files: `cat basic_emotions_*_*.err`

**"Model path not found"**
→ Wait for training to complete (Step 4), then transfer (Step 6)

---

## Timeline Estimate

- **Step 1-2**: 10-30 minutes (transfer + verification)
- **Step 3**: 5-10 minutes (generate trials)
- **Step 4**: 8-12 hours per dataset (training - can run in parallel)
- **Step 6**: 10-30 minutes (transfer models)
- **Step 7**: 1-2 hours (LLM augmentation)

**Total**: ~10-15 hours (mostly waiting for training)

