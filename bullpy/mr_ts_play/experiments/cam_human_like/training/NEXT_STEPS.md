# Next Steps After EU-Emotion Fine-Tuning

## ✅ Completed
- **EU-Emotion fine-tuning**: 59.26% validation accuracy (best at epoch 3)
- **Training data**: 213 trials (10 per emotion)
- **Model saved**: `results/eu_emotion_replication/model_checkpoints_v3/best_model/`

## 📋 Next Steps

### Step 1: Evaluate EU-Emotion Model on Its Own Test Set

**Purpose**: Verify the EU-Emotion replication worked correctly

Test how well the model performs on EU-Emotion test data:

```bash
cd /Users/eb2007/playground/bullpy/mr_ts_play
source venv/bin/activate

python3 experiments/cam_human_like/training/evaluate_on_cam.py \
    --model_path results/eu_emotion_replication/model_checkpoints_v3/best_model \
    --trial_definitions results/eu_emotion_replication/eu_emotion_trial_definitions_test.json \
    --data_root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
    --dataset_type eu_emotion \
    --split test \
    --device mps \
    --num_frames 8 \
    --use_multiframe
```

**Expected**: Should match or exceed 59.26% (validation accuracy)

**Why**: This completes the **EU-Emotion replication** - we trained on EU-Emotion, so we test on EU-Emotion.

---

### Step 2: Run CAM Fine-Tuning (Second Part of Dual Replication)

**Purpose**: Complete the **CAM replication** - train on CAM, test on CAM (separate from EU-Emotion)

Since EU-Emotion and CAM have **zero label overlap**, we do **two separate replications**:
1. ✅ **EU-Emotion replication**: Train on EU-Emotion → Test on EU-Emotion (DONE)
2. ⏳ **CAM replication**: Train on CAM → Test on CAM (NEXT)

Fine-tune on CAM dataset to learn CAM-specific concepts:

```bash
# First, create CAM train/test splits (if not already done)
python3 experiments/cam_human_like/training/create_cam_splits.py \
    --trial-definitions data/cam_trial_definitions_20concepts.json \
    --output-dir results/cam_replication \
    --split-method concept_balanced \
    --train-ratio 0.8 \
    --seed 42

# Then fine-tune on CAM
python3 experiments/cam_human_like/training/finetune_clip_emotions.py \
    --task_specific \
    --dataset_type cam \
    --train_trials results/cam_replication/train_trials.json \
    --val_trials results/cam_replication/test_trials.json \
    --data_root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/mindreading_transporter_files/Mindreading emotions library/Emotions" \
    --output_dir results/cam_replication/model_checkpoints \
    --num_epochs 5 \
    --batch_size 8 \
    --learning_rate 1e-5 \
    --device mps \
    --num_frames 8
```

**Expected**: 
- **Training**: ~50-100 minutes for 5 epochs
- **Validation accuracy**: 50-70% (CAM-specific concepts)

---

**Note**: We do NOT test EU-Emotion model on CAM because they have different labels. The dual replication means two separate, independent experiments.

---

### Step 3: Evaluate CAM Model on CAM Test Set

After CAM fine-tuning completes, evaluate it:

```bash
python3 experiments/cam_human_like/training/evaluate_on_cam.py \
    --model_path results/cam_replication/model_checkpoints/best_model \
    --trial_definitions results/cam_replication/test_trials.json \
    --data_root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/mindreading_transporter_files/Mindreading emotions library/Emotions" \
    --dataset_type cam \
    --split test \
    --device mps \
    --num_frames 8 \
    --use_multiframe
```

**Expected**: 
- **Baseline (zero-shot)**: 37%
- **After CAM fine-tuning**: 60-75% (CAM-specific training)
- **Goal**: Significantly better than baseline

---

### Step 4: Compare Results

Create a comparison report:

```bash
python3 << EOF
import json
from pathlib import Path

results = {}

# EU-Emotion on EU-Emotion
eu_eu_file = Path("results/eu_emotion_replication/model_checkpoints_v3/eu_emotion_evaluation_test.json")
if eu_eu_file.exists():
    with open(eu_eu_file) as f:
        data = json.load(f)
        results['EU-Emotion → EU-Emotion'] = data.get('metrics', {}).get('accuracy', 0)

# EU-Emotion on CAM (transfer)
eu_cam_file = Path("results/eu_emotion_replication/model_checkpoints_v3/cam_evaluation_test.json")
if eu_cam_file.exists():
    with open(eu_cam_file) as f:
        data = json.load(f)
        results['EU-Emotion → CAM (transfer)'] = data.get('metrics', {}).get('accuracy', 0)

# CAM on CAM
cam_cam_file = Path("results/cam_replication/model_checkpoints/cam_evaluation_test.json")
if cam_cam_file.exists():
    with open(cam_cam_file) as f:
        data = json.load(f)
        results['CAM → CAM'] = data.get('metrics', {}).get('accuracy', 0)

print("\n" + "="*60)
print("RESULTS COMPARISON")
print("="*60)
print(f"\nBaseline (zero-shot): 37.00%")
for name, acc in results.items():
    print(f"{name}: {acc:.2%}")
print("\n" + "="*60)
EOF
```

---

## 🎯 Expected Results Summary

| Experiment | Train Set | Test Set | Expected Accuracy | Status |
|------------|-----------|----------|------------------|--------|
| **EU-Emotion Replication** | EU-Emotion | EU-Emotion | 55-65% | ✅ To evaluate |
| **CAM Replication** | CAM | CAM | 60-75% | ⏳ To train |
| **Baseline (zero-shot)** | None | CAM | 37% | Reference point |

**Note**: These are TWO SEPARATE experiments. We don't test EU-Emotion model on CAM because they have different labels.

---

## 🚀 Quick Run: Evaluate EU-Emotion Model

```bash
cd /Users/eb2007/playground/bullpy/mr_ts_play
source venv/bin/activate

# Evaluate EU-Emotion model on EU-Emotion test set
python3 experiments/cam_human_like/training/evaluate_on_cam.py \
    --model_path results/eu_emotion_replication/model_checkpoints_v3/best_model \
    --trial_definitions results/eu_emotion_replication/eu_emotion_trial_definitions_test.json \
    --data_root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
    --dataset_type eu_emotion \
    --split test \
    --device mps \
    --num_frames 8 \
    --use_multiframe
```

---

## 📊 What to Look For

1. **EU-Emotion → EU-Emotion**: Should be ~55-65% (matches validation accuracy of 59.26%)
2. **CAM → CAM**: Should be 60-75% (after CAM fine-tuning completes)

**Why no EU-Emotion → CAM test?**
- They have **zero label overlap** (different emotions/concepts)
- Testing would be meaningless - model learned wrong labels
- Dual replication means **two separate experiments**, not transfer learning

---

## 🔄 After Local Testing

Once you've verified everything works locally:

1. **Move to HPC** for longer training (10-20 epochs)
2. **Hyperparameter tuning** (learning rate, batch size)
3. **Ensemble methods** (train multiple models)
4. **Final evaluation** on full test sets

---

## 📝 Notes

- **Training time**: Each evaluation takes ~5-10 minutes
- **Model size**: ~577MB per checkpoint
- **Best model**: Saved at epoch 3 (59.26% validation accuracy)
- **Loss progression**: 1.2851 → 0.9005 (excellent learning curve)

