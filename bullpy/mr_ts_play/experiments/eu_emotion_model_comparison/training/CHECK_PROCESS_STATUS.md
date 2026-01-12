# Check if Training is Running

## The log is empty - this could mean:

1. **Process just started** - Output is buffered (Python buffers stdout)
2. **Process crashed** - Error happened before any output
3. **Output going elsewhere** - Check stderr or other logs

## Check if process is running:

```bash
# Check if process exists
ps aux | grep 12604

# Or check all finetune processes
ps aux | grep finetune_vision_models | grep -v grep
```

## Check for errors:

```bash
# Check if there's any output at all
ls -lh resnet50_test.log

# Try running in foreground to see errors
python experiments/eu_emotion_model_comparison/training/finetune_vision_models.py \
    --model resnet50 \
    --train_trials data/trial_definitions/eu_emotion_train.json \
    --val_trials data/trial_definitions/eu_emotion_val.json \
    --data_root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
    --output_dir models/resnet50_emotion_finetuned_test \
    --num_epochs 1 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --device auto
```

## Force unbuffered output:

If you want to see output immediately, use Python's `-u` flag:

```bash
python -u experiments/eu_emotion_model_comparison/training/finetune_vision_models.py \
    --model resnet50 \
    --train_trials data/trial_definitions/eu_emotion_train.json \
    --val_trials data/trial_definitions/eu_emotion_val.json \
    --data_root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
    --output_dir models/resnet50_emotion_finetuned_test \
    --num_epochs 1 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --device auto > resnet50_test.log 2>&1 &
disown
```

The `-u` flag forces unbuffered output so you'll see logs immediately.
