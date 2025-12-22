# Re-running EU-Emotion Training

## Current Configuration
- **267 trials total**: 213 train, 54 test (10 trials per emotion)
- **5 epochs** (increased from 2)
- **Prompt templates**: "a photo of a person feeling [emotion]"
- **Multi-frame processing**: 8 frames per video
- **Task-specific training**: 4-option forced-choice

## Quick Run Command

Run this in your terminal (from the project root):

```bash
cd /Users/eb2007/playground/bullpy/mr_ts_play

# Activate venv (if needed)
source venv/bin/activate

# Run training
python3 experiments/cam_human_like/training/finetune_clip_emotions.py \
    --task_specific \
    --dataset_type eu_emotion \
    --train_trials results/eu_emotion_replication/eu_emotion_trial_definitions_train.json \
    --val_trials results/eu_emotion_replication/eu_emotion_trial_definitions_test.json \
    --data_root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
    --output_dir results/eu_emotion_replication/model_checkpoints_v3 \
    --num_epochs 5 \
    --batch_size 8 \
    --learning_rate 1e-5 \
    --device mps \
    --num_frames 8 \
    2>&1 | tee results/eu_emotion_replication/training_log_v3.txt
```

## Or Use the Script

```bash
cd /Users/eb2007/playground/bullpy/mr_ts_play
bash experiments/cam_human_like/training/rerun_eu_emotion_training.sh
```

## Monitor Progress

While training runs, you can monitor progress:

```bash
# Watch the log file
tail -f results/eu_emotion_replication/training_log_v3.txt

# Or check periodically
tail -20 results/eu_emotion_replication/training_log_v3.txt
```

## Expected Output

You should see:
- Loading datasets... (213 train, 54 test trials)
- Multi-frame processing: Enabled (8 frames per video)
- Epoch 1/5, 2/5, etc.
- Training loss decreasing
- Validation accuracy improving (target: 50-60%+)

## Time Estimate

- **MPS (MacBook Air 2025)**: ~50-100 minutes for 5 epochs
- **CPU**: ~4-8 hours

## What to Look For

1. **Training loss**: Should decrease over epochs
2. **Validation accuracy**: Should improve (target: 50-60%+)
3. **Best model**: Saved to `results/eu_emotion_replication/model_checkpoints_v3/best_model/`

## After Training

Evaluate the model:

```bash
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

