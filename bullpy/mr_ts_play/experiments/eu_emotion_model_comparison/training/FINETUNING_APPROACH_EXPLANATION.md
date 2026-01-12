# Fine-Tuning Approach: Task-Specific vs Standard Classification

## Key Difference: How CLIP Was Fine-Tuned

### CLIP Used **Task-Specific Fine-Tuning** ✅

**What this means:**
- Each video is paired with **4 candidate labels** (1 correct + 3 foils)
- Loss is **cross-entropy over the 4 options** (not all 27 emotions)
- Matches the **evaluation format** exactly (4-option forced-choice)
- Model learns to distinguish between the 4 options, not classify into 27 classes

**Why this is better:**
- ✅ **Better alignment** with evaluation task
- ✅ **More efficient** - only needs to distinguish 4 options, not 27
- ✅ **Better performance** - matches what model will do during evaluation

### My Script Uses **Standard Classification** ⚠️

**What this means:**
- Each video has a single emotion label (1 of 27)
- Loss is **cross-entropy over 27 emotion classes**
- Model learns to classify into 27 categories
- During evaluation, must map 27-class predictions to 4 candidate labels

**Why this is less ideal:**
- ⚠️ **Mismatch** with evaluation format
- ⚠️ **Less efficient** - learns all 27 emotions, but only needs to distinguish 4
- ⚠️ **Lower performance** - not optimized for the actual task

## Should We Use Task-Specific Fine-Tuning?

### ✅ **YES - For Better Performance**

**Recommendation:**
1. **Use task-specific fine-tuning** (like CLIP) for best results
2. **Fine-tune all models** (ResNet50, ViT, EfficientNet) using task-specific approach
3. This will give you **better accuracy** and **fairer comparison** to CLIP

### Why Fine-Tune All Models?

**You're right - we should fine-tune all of them!**

**Reasons:**
1. **Fair comparison** - All models fine-tuned the same way
2. **Better results** - Each model will perform better after fine-tuning
3. **Comprehensive evaluation** - More models = stronger paper
4. **Ensemble potential** - Can combine multiple fine-tuned models

**Models to fine-tune:**
- ✅ ResNet50 (easiest, good baseline)
- ✅ ViT (might perform well)
- ✅ EfficientNet (efficient architecture)
- ⚠️ ResNet101 (larger, slower, might not be worth it)
- ⚠️ Video models (I3D, TimeSformer - more complex)

## Two Approaches Available

### Option 1: Standard Classification (Current Script)

**Script:** `finetune_vision_models.py`

**Pros:**
- ✅ Simpler implementation
- ✅ Faster to implement
- ✅ Works for any number of emotions

**Cons:**
- ⚠️ Doesn't match evaluation format
- ⚠️ Lower performance than task-specific

**Use when:**
- Quick baseline needed
- Don't have candidate labels in training data
- Want simpler implementation

### Option 2: Task-Specific (Like CLIP) - **RECOMMENDED**

**Script:** `finetune_vision_models_task_specific.py` (I'll create this)

**Pros:**
- ✅ Matches evaluation format exactly
- ✅ Better performance (like CLIP)
- ✅ Fair comparison to CLIP

**Cons:**
- ⚠️ More complex implementation
- ⚠️ Requires candidate labels in training data

**Use when:**
- Want best performance
- Have candidate labels in training data
- Want fair comparison to CLIP

## Recommendation

### **Use Task-Specific Fine-Tuning (Like CLIP)**

**Why:**
1. CLIP was fine-tuned this way - fair comparison
2. Better performance - matches evaluation format
3. More scientifically rigorous

**Implementation:**
- I'll create `finetune_vision_models_task_specific.py`
- Uses 4-option forced-choice format
- Matches CLIP's fine-tuning approach

### **Fine-Tune All Models**

**Priority order:**
1. **ResNet50** - Start here (easiest, good baseline)
2. **ViT** - Next (might perform well)
3. **EfficientNet** - Third (efficient)
4. **ResNet101** - Optional (larger, might not be worth it)

**Expected time:**
- ResNet50: ~2-4 hours
- ViT: ~3-5 hours
- EfficientNet: ~2-4 hours
- **Total: ~7-13 hours** (can run in parallel if you have multiple GPUs)

## Updated Approach

I'll create a **task-specific fine-tuning script** that:
1. Uses 4-option forced-choice format (like CLIP)
2. Works for all models (ResNet, ViT, EfficientNet)
3. Matches CLIP's training procedure

This will give you:
- ✅ Better performance
- ✅ Fair comparison to CLIP
- ✅ All models fine-tuned the same way

## Next Steps

1. **I'll create task-specific script** (matching CLIP's approach)
2. **Fine-tune ResNet50 first** (test the approach)
3. **Fine-tune ViT and EfficientNet** (if ResNet50 works well)
4. **Evaluate all fine-tuned models** on test set
5. **Compare to CLIP** (all using same fine-tuning approach)
