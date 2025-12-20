# Fine-Tuning Strategy: CAM Data vs External Dataset

## The Question

**Should you fine-tune CLIP on:**
1. CAM dataset itself (train split)?
2. External emotion dataset (FER2013, AffectNet)?
3. Both (two-stage fine-tuning)?

## Answer: Use External Dataset First (Best Practice)

### Recommended Approach: Two-Stage Fine-Tuning

**Stage 1: Fine-tune on External Dataset (FER2013)**
- **Purpose**: Learn general emotion recognition
- **Data**: FER2013 (external, no overlap with CAM)
- **Expected**: 60-65% on CAM
- **Why**: Shows model can learn emotions in general, not just CAM-specific patterns

**Stage 2: Fine-tune on CAM Train Split (Optional)**
- **Purpose**: Adapt to CAM-specific emotions and video format
- **Data**: CAM train split (with proper train/val/test separation)
- **Expected**: 65-75% on CAM
- **Why**: Task-specific adaptation, but acknowledge this in methodology

## Why This Matters

### For PhD Thesis / Publication

**Best Practice:**
1. **Start with external dataset** (FER2013) - shows general emotion recognition ability
2. **Report both results**:
   - Fine-tuned on FER2013: 60-65% (general emotion recognition)
   - Fine-tuned on CAM train: 65-75% (task-specific adaptation)
3. **Acknowledge in methodology**: "Model was fine-tuned on FER2013 for general emotion recognition, then optionally adapted to CAM train split"

### Data Leakage Concerns

**Fine-tuning on CAM train split is OK IF:**
- ✅ You use proper train/val/test splits
- ✅ Test set is never seen during training
- ✅ You acknowledge this in methodology
- ✅ You report both external and CAM-finetuned results

**However, for a replication study:**
- **More rigorous**: Use external dataset only
- **Shows generalizability**: Model works on CAM without seeing CAM data
- **More comparable**: Closer to "zero-shot" evaluation

## Recommended Workflow

### Option A: External Dataset Only (Most Rigorous) ⭐ RECOMMENDED

```bash
# 1. Fine-tune on FER2013 (external dataset)
python experiments/cam_human_like/training/finetune_clip_emotions.py \
    --fer2013_dir fer2013/ \
    --output_dir models/clip_fer2013_finetuned \
    --num_epochs 10

# 2. Evaluate on CAM (zero-shot on CAM, but fine-tuned on emotions)
python experiments/cam_human_like/run_experiment.py \
    --config configs/cam_config.yaml \
    --split all --no-actor-filtering
# (Update config to use models/clip_fer2013_finetuned/best_model)
```

**Expected**: 60-65% face accuracy
**Advantage**: No data leakage, shows general emotion recognition

### Option B: Two-Stage (Best Performance)

```bash
# Stage 1: Fine-tune on FER2013
python experiments/cam_human_like/training/finetune_clip_emotions.py \
    --fer2013_dir fer2013/ \
    --output_dir models/clip_fer2013_finetuned \
    --num_epochs 10

# Stage 2: Fine-tune on CAM train (starting from FER2013 model)
python experiments/cam_human_like/training/finetune_clip_emotions.py \
    --train_data data/splits/train.csv \
    --val_data data/splits/val.csv \
    --data_root "/path/to/cam/stimuli" \
    --model_name models/clip_fer2013_finetuned/best_model \
    --output_dir models/clip_cam_finetuned \
    --num_epochs 5  # Fewer epochs (already trained on emotions)
```

**Expected**: 65-75% face accuracy
**Advantage**: Best performance, but acknowledge CAM-specific training

## Comparison

| Approach | Accuracy | Rigor | Best For |
|----------|----------|-------|----------|
| External dataset only | 60-65% | ⭐⭐⭐ Highest | PhD thesis, publication |
| CAM train split only | 65-75% | ⭐⭐ Medium | Task-specific optimization |
| Two-stage (external → CAM) | 65-75% | ⭐⭐ Medium | Best performance |

## My Recommendation

**For your PhD thesis/replication study:**

1. **Start with FER2013 fine-tuning** (external dataset)
   - Shows general emotion recognition ability
   - No data leakage concerns
   - Expected: 60-65% (much better than 37%)

2. **Report this as main result**
   - "CLIP fine-tuned on FER2013 achieves 60-65% on CAM"
   - This is a valid, rigorous result

3. **Optionally add CAM fine-tuning** (as supplementary)
   - "With additional fine-tuning on CAM train split: 65-75%"
   - Acknowledge this is task-specific adaptation
   - Shows upper bound of performance

## Implementation

I'll update the fine-tuning script to support both approaches. The key is:
- **External dataset**: FER2013, AffectNet (no overlap with CAM)
- **CAM dataset**: Use train split only, keep test separate

Both are valid - external is more rigorous, CAM gives better performance.

