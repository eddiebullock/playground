# Which Fine-Tuning Approach Should You Use?

## Summary: Two Different Approaches

### ❌ **Current Script (Standard Classification)**
- Trains on 27 emotion classes
- Doesn't match evaluation format
- **Lower performance**

### ✅ **CLIP's Approach (Task-Specific)**
- Trains on 4-option forced-choice (matches evaluation)
- Each video has 4 candidate labels
- **Better performance** (matches evaluation format)

## Key Difference Explained

### CLIP's Task-Specific Approach

**How it works:**
1. Each video is paired with **4 candidate labels** (1 correct + 3 foils)
2. CLIP computes similarity between:
   - Video features (from image encoder)
   - Text embeddings of 4 candidate labels (from text encoder)
3. Loss is **cross-entropy over 4 similarities**
4. Model learns to distinguish between the 4 options

**Why this works for CLIP:**
- CLIP has **both image and text encoders**
- Can compute similarity between video and text directly
- Natural fit for the task

### For ResNet/ViT/EfficientNet

**Challenge:**
- These models only have **image encoders** (no text encoder)
- Can't directly compute similarity to text labels

**Two Options:**

#### Option 1: Standard Classification (Easier) ✅ **RECOMMENDED FOR NOW**

**What it does:**
- Train on 27 emotion classes
- Model outputs 27-class probabilities
- During evaluation: map 27-class predictions to 4 candidate labels

**Pros:**
- ✅ Simple to implement
- ✅ Works immediately
- ✅ Can fine-tune all models quickly

**Cons:**
- ⚠️ Doesn't match evaluation format exactly
- ⚠️ Might have slightly lower performance

**Use this if:**
- You want to get results quickly
- You want to fine-tune all models
- You're okay with slightly lower performance

#### Option 2: Task-Specific (More Complex)

**What it would do:**
- Learn emotion embeddings (one per emotion)
- Extract video features
- Compute similarity between video features and 4 candidate emotion embeddings
- Loss over 4 similarities

**Pros:**
- ✅ Matches evaluation format
- ✅ Better performance (potentially)

**Cons:**
- ⚠️ More complex implementation
- ⚠️ Need to learn emotion embeddings
- ⚠️ More hyperparameters to tune

**Use this if:**
- You want best possible performance
- You have time for more complex implementation
- You want exact match to CLIP's approach

## My Recommendation

### **Start with Standard Classification** ✅

**Why:**
1. **Faster to implement** - Get results quickly
2. **Works for all models** - Can fine-tune ResNet, ViT, EfficientNet easily
3. **Good enough** - Will still improve significantly over random (25% → 45-55%)
4. **Can upgrade later** - If needed, can implement task-specific later

**Then:**
- Fine-tune all models using standard classification
- Compare results
- If performance is good enough, you're done
- If you need better performance, implement task-specific later

## What You Should Do

### **Use the Current Script (Standard Classification)**

```bash
# Fine-tune ResNet50
python experiments/eu_emotion_model_comparison/training/finetune_vision_models.py \
    --model resnet50 \
    --train_trials data/trial_definitions/eu_emotion_train.json \
    --val_trials data/trial_definitions/eu_emotion_val.json \
    --data_root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
    --output_dir models/resnet50_emotion_finetuned \
    --num_epochs 20 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --device auto

# Then fine-tune ViT and EfficientNet the same way
```

**This will:**
- ✅ Work immediately
- ✅ Fine-tune all models
- ✅ Give you good results (45-55% accuracy)
- ✅ Be comparable to CLIP (though CLIP might still be better)

## Expected Results

**Standard Classification:**
- ResNet50: ~45-55% accuracy
- ViT: ~50-60% accuracy
- EfficientNet: ~45-55% accuracy

**CLIP (Task-Specific):**
- CLIP: 55.6% accuracy

**Comparison:**
- Standard classification will be **close** to CLIP
- Might be slightly lower, but still much better than random (25%)
- Good enough for comparison

## Bottom Line

**Use standard classification for now:**
1. ✅ Faster to implement
2. ✅ Works for all models
3. ✅ Good enough results
4. ✅ Can upgrade to task-specific later if needed

**Fine-tune all models:**
- ResNet50 (start here)
- ViT (next)
- EfficientNet (third)

This gives you a comprehensive comparison while keeping things simple!
