# How to Monitor Training Progress

## Check Current Status

**View the log file in real-time:**
```bash
tail -f resnet50_test.log
```

**Or view last 50 lines:**
```bash
tail -n 50 resnet50_test.log
```

**Check if process is running:**
```bash
ps aux | grep finetune_vision_models | grep -v grep
```

## What to Look For

**Good signs:**
- "Epoch 1/1" or "Epoch X/10"
- Progress bars showing training
- "Training: XX%|████..." 
- "Validation: XX%|████..."
- "✅ Saved best model"
- "Training completed!"

**Bad signs:**
- "Error" or "Exception"
- "Out of memory"
- "File not found"
- Process stopped/crashed

## Check All Logs

**If running the test script, check each model's log:**
```bash
# ResNet50
tail -f resnet50_test.log

# ViT (when it starts)
tail -f vit_test.log

# EfficientNet (when it starts)
tail -f efficientnet_test.log
```

## Quick Status Check

**See what's currently happening:**
```bash
# Last 20 lines of current log
tail -n 20 resnet50_test.log

# Check if still running
ps aux | grep python | grep finetune
```
