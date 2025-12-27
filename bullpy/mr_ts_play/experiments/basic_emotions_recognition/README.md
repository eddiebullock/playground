# Basic Emotions Recognition Experiment

This experiment fine-tunes CLIP models on Ekman's 6 basic emotions (+ neutral = 7 classes) for both CAM and EU-Emotion datasets, and runs LLM augmentation experiments to compare performance on basic emotions vs complex emotions.

## Overview

**Basic Emotion Categories (7 total):**
1. **happy** - positive emotions
2. **sad** - negative/sorrowful emotions
3. **angry** - anger-related emotions
4. **fear** - fear/anxiety-related emotions
5. **surprise** - surprise/shock-related emotions
6. **disgust** - disgust/revulsion-related emotions
7. **neutral** - calm, thinking, or ambiguous emotions

**Key Difference from Fine-Grained Experiments:**
- **7-way classification** (not 4-option forced-choice)
- Model selects from all 7 basic emotions for each trial
- No foil selection needed
- Standard multi-class classification

## Expected Results

**Basic Emotions (7-way classification):**
- Random baseline: ~14.3% (1/7)
- Expected CLIP-only: 60-80% (much better than fine-grained)
- Expected LLM-augmented: 70-85% (approaching human performance)

**Comparison to Fine-Grained:**
- Fine-grained CAM: ~60-75% (20-405 classes, forced-choice)
- Basic emotions CAM: Expected 70-85% (7 classes, easier task)
- This shows that basic emotions are easier to recognize than complex emotions

## Project Structure

```
experiments/basic_emotions_recognition/
├── __init__.py
├── README.md
├── training/
│   ├── __init__.py
│   ├── hpc_basic_emotions_cam.sh          # HPC training script for CAM
│   ├── hpc_basic_emotions_cam.slurm        # SLURM script for CAM (CPU/icelake)
│   ├── hpc_basic_emotions_eu_emotion.sh    # HPC training script for EU-Emotion
│   ├── hpc_basic_emotions_eu_emotion.slurm # SLURM script for EU-Emotion (CPU/icelake)
│   ├── create_basic_emotion_trials.py     # Generate trials with basic emotion labels
│   ├── finetune_basic_emotions.py          # Fine-tune CLIP on basic emotions
│   └── evaluate_basic_emotions.py         # Evaluate on basic emotion test sets
├── llm_augmentation/
│   ├── __init__.py
│   ├── models/                             # Reused from llm_augmented_emotion_recognition
│   ├── evaluation/                         # Reused from llm_augmented_emotion_recognition
│   ├── scripts/
│   │   ├── run_basic_emotions_llm_experiment.py      # Main LLM augmentation script
│   │   └── generate_basic_emotions_llm_cache.py     # Pre-generate LLM embeddings
│   └── configs/
│       └── basic_emotions_llm_config.yaml   # LLM config for basic emotions
├── data/
│   ├── basic_emotion_mappings/
│   │   ├── cam_basic_emotion_mapping.json  # CAM fine-grained → basic (reuse existing)
│   │   └── eu_emotion_basic_mapping.json   # EU-Emotion → basic (created)
│   └── trial_definitions/                  # Generated basic emotion trials
└── configs/
    └── basic_emotions_config.yaml          # Training config for basic emotions
```

## Data Paths

**CAM Data Location:**
- `/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/mindreading_transporter_files/Mindreading emotions library/Emotions`

**EU-Emotion Data Location:**
- Root: `/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions`
- Structure: Within this folder, navigate to `emotions*/HD Version - Face, Body, Social/Faces - HD Version/EDITED/` or `Original/`
- **IMPORTANT**: Only use face videos from these paths, ignore voice/body/other modalities

**Existing Basic Emotion Mapping:**
- CAM mapping: `/Users/eb2007/playground/bullpy/mr_ts_play/data/basic_emotion_mapping.json`
  - Maps all 405 CAM fine-grained emotions to 7 basic categories
  - **Reused** - already created

**Trained CLIP Models:**
- Will be created by training scripts: `models/basic_emotions_cam_finetuned/` and `models/basic_emotions_eu_emotion_finetuned/`

## EU-Emotion Basic Emotion Mapping

The EU-Emotion dataset has 27 emotions that are mapped to 7 basic categories using professional judgment based on emotion recognition literature.

**EU-Emotion Emotions (27 total):**
afraid, afraid low intensity, angry, angry low intensity, ashamed, bored,
disappointed, disgusted, disgusted low intensity, excited, frustrated,
happy, happy low intensity, hurt, interested, jealous, joking, kind,
neutral, proud, sad, sad low intensity, sneaky, surprised,
surprised low intensity, unfriendly, worried

**Mapping Rationale:**
- `frustrated` → `angry` (anger-related, common in emotion taxonomy)
- `jealous` → `angry` (envy/anger, typically classified as anger in basic emotion models)
- `worried` → `fear` (anxiety/fear, core component of fear category)
- `bored`, `sneaky` → `neutral` (low arousal, ambiguous emotional state)
- `unfriendly` → `angry` (negative valence, hostility-related)
- Intensity variants map to same base emotion (strip " low intensity" suffix)

See `data/basic_emotion_mappings/eu_emotion_basic_mapping.json` for the complete mapping.

## Prerequisites: Data Transfer

**IMPORTANT**: Before running experiments, you need to transfer data and code to HPC.

### Step 0: Transfer Experiment Code to HPC

From your local machine:
```bash
cd /Users/eb2007/playground/bullpy/mr_ts_play
bash experiments/basic_emotions_recognition/training/transfer_to_hpc.sh
```

This transfers:
- Experiment code (`experiments/basic_emotions_recognition/`)
- Basic emotion mapping files
- Configuration files

### Step 0.5: Verify Data is on HPC

**CAM Data:**
- Should already be at: `/home/eb2007/data/CAM`
- If not, transfer it separately (see `experiments/cam_human_like/training/` for transfer scripts)

**EU-Emotion Data:**
- Should be at: `~/rds/rds-autism-research-ePtR33Nsgi4/data/EU_emotions`
- If not, transfer using: `bash experiments/cam_human_like/training/transfer_eu_emotions_to_rds.sh`

**Verify on HPC:**
```bash
ssh eb2007@login-cpu.hpc.cam.ac.uk
ls -la /home/eb2007/data/CAM  # Should exist
ls -la ~/rds/rds-autism-research-ePtR33Nsgi4/data/EU_emotions  # Should exist
```

## HPC vs Local: Which Should You Use?

**Recommendation: Use HPC** for the following reasons:

1. **Storage**: All outputs go to RDS (1100GB available) vs local storage constraints
2. **Consistency**: Same infrastructure as fine-grained experiments
3. **CPU Resources**: HPC has better CPU resources for training
4. **Time**: CPU training takes 8-12 hours - better to run on HPC than tie up local machine

**Local is OK if:**
- You have sufficient local storage
- You want to test/debug quickly
- You have a powerful local machine with good CPU

**For this experiment, HPC is recommended** since:
- Outputs are large (model checkpoints, trial definitions)
- Training time is long (8-12 hours)
- You already have HPC infrastructure set up

## Usage

### Phase 1: Data Preparation (On HPC)

#### 1. Generate Basic Emotion Trials

**For CAM (on HPC):**
```bash
ssh eb2007@login-cpu.hpc.cam.ac.uk
cd ~/mr_ts_play

python experiments/basic_emotions_recognition/training/create_basic_emotion_trials.py \
    --dataset_type cam \
    --input_trials data/trial_definitions/cam_test.json \
    --mapping_file data/basic_emotion_mapping.json \
    --output_dir ~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/mr_ts_play_results/basic_emotions_cam \
    --train_ratio 0.8 \
    --seed 42
```

**For EU-Emotion (on HPC):**
```bash
python experiments/basic_emotions_recognition/training/create_basic_emotion_trials.py \
    --dataset_type eu_emotion \
    --input_trials data/trial_definitions/eu_emotion_test.json \
    --mapping_file experiments/basic_emotions_recognition/data/basic_emotion_mappings/eu_emotion_basic_mapping.json \
    --output_dir ~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/mr_ts_play_results/basic_emotions_eu_emotion \
    --train_ratio 0.8 \
    --seed 42
```

**Note**: Outputs go to RDS (`~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/mr_ts_play_results/`) to avoid `/home` quota issues.

This creates:
- `cam_basic_emotions_train.json` / `cam_basic_emotions_test.json`
- `eu_emotion_basic_emotions_train.json` / `eu_emotion_basic_emotions_test.json`

Each trial has:
- `all_candidate_labels`: All 7 basic emotions (7-way classification)
- `basic_emotion`: Correct basic emotion label
- `correct_idx`: Index of correct label (0-6)

### Phase 2: CLIP Fine-Tuning (HPC - CPU)

#### 2. Train CLIP Models on Basic Emotions

**For CAM (HPC):**
```bash
# On HPC login node
ssh eb2007@login-cpu.hpc.cam.ac.uk
cd ~/mr_ts_play

# Submit SLURM job
sbatch experiments/basic_emotions_recognition/training/hpc_basic_emotions_cam.slurm

# Check job status
squeue -u eb2007
```

**For EU-Emotion (HPC):**
```bash
# On HPC login node
ssh eb2007@login-cpu.hpc.cam.ac.uk
cd ~/mr_ts_play

# Submit SLURM job
sbatch experiments/basic_emotions_recognition/training/hpc_basic_emotions_eu_emotion.slurm

# Check job status
squeue -u eb2007
```

**All outputs go to RDS**: `~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/mr_ts_play_results/basic_emotions_{dataset}/`

**Local Training (NOT RECOMMENDED - use HPC instead):**
```bash
# CAM
python experiments/basic_emotions_recognition/training/finetune_basic_emotions.py \
    --dataset_type cam \
    --train_trials experiments/basic_emotions_recognition/data/trial_definitions/cam_basic_emotions_train.json \
    --val_trials experiments/basic_emotions_recognition/data/trial_definitions/cam_basic_emotions_test.json \
    --data_root "/path/to/cam/data" \
    --output_dir models/basic_emotions_cam_finetuned \
    --num_epochs 20 \
    --batch_size 4 \
    --device cpu
```

**Training Configuration:**
- Device: CPU (icelake partition)
- Batch size: 4 (CPU-optimized)
- Learning rate: 5e-5
- Num frames: 16
- Early stopping: patience=5, min_delta=0.001

#### 3. Evaluate Models

```bash
# CAM
python experiments/basic_emotions_recognition/training/evaluate_basic_emotions.py \
    --model_path models/basic_emotions_cam_finetuned/best_model \
    --trial_definitions experiments/basic_emotions_recognition/data/trial_definitions/cam_basic_emotions_test.json \
    --data_root "/path/to/cam/data" \
    --device cpu \
    --num_frames 16 \
    --output_file results/basic_emotions_cam/evaluation_results.json
```

### Phase 3: LLM Augmentation (Local - After Models Trained)

**Note**: LLM augmentation can be run locally after models are trained on HPC. First, transfer models from HPC to local.

#### 3.5: Transfer Models from HPC to Local (Optional)

After training completes on HPC, transfer models to local for LLM augmentation:

```bash
# From local machine
rsync -avz --progress \
    eb2007@login-cpu.hpc.cam.ac.uk:~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/mr_ts_play_results/basic_emotions_cam/model_checkpoints/best_model/ \
    models/basic_emotions_cam_finetuned/

rsync -avz --progress \
    eb2007@login-cpu.hpc.cam.ac.uk:~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/mr_ts_play_results/basic_emotions_eu_emotion/model_checkpoints/best_model/ \
    models/basic_emotions_eu_emotion_finetuned/
```

#### 4. Generate LLM Embeddings for Basic Emotions (Local)

```bash
# On local machine
cd /Users/eb2007/playground/bullpy/mr_ts_play

python experiments/basic_emotions_recognition/llm_augmentation/scripts/generate_basic_emotions_llm_cache.py \
    --provider openai \
    --model text-embedding-3-small \
    --cache_dir experiments/basic_emotions_recognition/llm_augmentation/data/llm_cache
```

This generates embeddings for all 7 basic emotions (much smaller cache than fine-grained).

#### 5. Run LLM Augmentation Experiment (Local)

```bash
# On local machine
python experiments/basic_emotions_recognition/llm_augmentation/scripts/run_basic_emotions_llm_experiment.py \
    --config llm_augmentation/configs/basic_emotions_llm_config.yaml \
    --dataset cam \
    --device cpu \
    --num_frames 8
```

This runs three conditions:
- **CLIP-only**: Fine-tuned CLIP model predictions
- **LLM-only**: LLM embedding similarity predictions
- **LLM-augmented**: Weighted average of CLIP and LLM scores

Results are saved to `results/basic_emotions_{dataset}/`.

## Trial Structure (7-Way Classification)

Unlike forced-choice experiments, basic emotion trials use **7-way classification**:

```json
{
  "trial_id": "basic_trial_001",
  "stimulus_path": "...",
  "modality": "face",
  "fine_grained_emotion": "humiliated",  // Original emotion (for reference)
  "basic_emotion": "sad",                 // Correct basic emotion label
  "all_candidate_labels": [
    "happy", "sad", "angry", "fear", 
    "surprise", "disgust", "neutral"
  ],  // All 7 options
  "correct_label": "sad",                 // Correct answer
  "correct_idx": 1,                       // Index of correct label (0-6)
  "actor": "M",
  "scenario_id": "0100104"
}
```

**Key differences:**
- Model must select from all 7 basic emotions (not 4 options)
- No foil selection needed
- Evaluation: Standard accuracy (predicted emotion matches ground truth)
- This is standard multi-class classification, not forced-choice

## Configuration Files

### `configs/basic_emotions_config.yaml`
Training configuration for basic emotions fine-tuning:
- CPU training settings
- Batch size: 4 (CPU-optimized)
- Learning rate: 5e-5
- Num frames: 16

### `llm_augmentation/configs/basic_emotions_llm_config.yaml`
LLM augmentation configuration:
- Provider: OpenAI
- Embedding model: text-embedding-3-small
- Fusion method: weighted_average
- CLIP weight: 0.7, LLM weight: 0.3

## Results

**On HPC (RDS storage):**
- Training outputs: `~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/mr_ts_play_results/basic_emotions_{dataset}/`
  - Model checkpoints: `model_checkpoints/best_model/`
  - Trial definitions: `{dataset}_basic_emotions_{train|test}.json`
  - Evaluation: `evaluation_results.json`

**On Local (after transfer):**
- Models: `models/basic_emotions_{dataset}_finetuned/` (transferred from HPC)
- LLM augmentation: `results/basic_emotions_{dataset}/{condition}_results.json`

## Comparison to Fine-Grained Experiments

| Aspect | Fine-Grained | Basic Emotions |
|--------|--------------|-----------------|
| **Classes** | 20-405 | 7 |
| **Task Type** | 4-option forced-choice | 7-way classification |
| **Random Baseline** | ~25% (1/4) | ~14.3% (1/7) |
| **Expected CLIP** | 60-75% | 70-85% |
| **Expected LLM-Augmented** | 70-80% | 75-90% |

**Key Finding:** Basic emotions are easier to recognize than complex emotions, demonstrating that simpler emotion categories improve model performance.

## Notes

1. **CPU Training**: All training uses CPU (icelake partition) with batch_size=4
2. **7-Way Classification**: Unlike forced-choice, model selects from all 7 emotions
3. **Actor Independence**: Train/test splits ensure no actor overlap
4. **Reused Code**: LLM wrapper and evaluation code reused from `llm_augmented_emotion_recognition/`
5. **EU-Emotion Mapping**: Professional judgment mapping based on emotion recognition literature

## Troubleshooting

**Issue: Trial definitions not found**
- Solution: Run `create_basic_emotion_trials.py` first on HPC to generate basic emotion trials
- Check: Outputs should be in RDS, not `/home`

**Issue: Model path not found**
- Solution: Train models first on HPC using SLURM scripts, then transfer to local if needed
- Check: Models are in RDS: `~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/mr_ts_play_results/basic_emotions_{dataset}/model_checkpoints/best_model/`

**Issue: Data not found on HPC**
- CAM: Should be at `/home/eb2007/data/CAM` (already there)
- EU-Emotion: Should be at `~/rds/rds-autism-research-ePtR33Nsgi4/data/EU_emotions`
- Solution: Transfer EU-Emotion data using `experiments/cam_human_like/training/transfer_eu_emotions_to_rds.sh`

**Issue: Experiment code not on HPC**
- Solution: Run `bash experiments/basic_emotions_recognition/training/transfer_to_hpc.sh` from local machine

**Issue: `/home` quota exceeded**
- Solution: All outputs go to RDS automatically. Check scripts use RDS paths for outputs.

**Issue: CPU training too slow**
- Solution: This is expected. CPU training takes 8-12 hours for 20 epochs. Use HPC for better resources.

## References

- Ekman, P. (1992). An argument for basic emotions. *Cognition & Emotion*, 6(3-4), 169-200.
- Golan, O., et al. (2006). The "Reading the Mind in the Voice" Test-Revised. *Journal of Autism and Developmental Disorders*, 36(8), 1099-1115.

