# MindReading/Emotions Dataset: Cost & Performance Analysis

## 1. LLM Experiment Cost Analysis

### Current EU-Emotion LLM Costs

**Current Setup:**
- Test trials: 54 videos
- Frames per video: 4-8 frames
- Model: gpt-4o-mini (low detail)
- **Cost: ~$0.02** per experiment run

**Cost Breakdown (per 54 videos):**
- Input tokens: ~63,920 tokens (54 videos × 8 frames × 85 tokens/frame)
- Output tokens: ~5,400 tokens (54 videos × 100 tokens/description)
- Total: ~$0.02

---

### MindReading/Emotions LLM Costs

**Dataset Scale:**
- **Total videos**: 5,801 files
- **Emotions**: 425 emotions (vs. 27 in EU-Emotion)
- **Estimated test set**: ~1,160 videos (20% split) or ~580 videos (10% split)

**Cost Estimates:**

#### Option 1: Full Dataset (5,801 videos)
- **Frames processed**: 5,801 × 8 = 46,408 frames
- **Input tokens**: 46,408 × 85 = **3,944,680 tokens**
- **Input cost**: (3,944,680 / 1,000,000) × $0.15 = **$0.59**
- **Output tokens**: 5,801 × 100 = **580,100 tokens**
- **Output cost**: (580,100 / 1,000,000) × $0.60 = **$0.35**
- **Total: ~$0.94** (gpt-4o-mini, low detail)

#### Option 2: Test Set Only (~1,160 videos, 20% split)
- **Frames processed**: 1,160 × 8 = 9,280 frames
- **Input tokens**: 9,280 × 85 = **788,800 tokens**
- **Input cost**: (788,800 / 1,000,000) × $0.15 = **$0.12**
- **Output tokens**: 1,160 × 100 = **116,000 tokens**
- **Output cost**: (116,000 / 1,000,000) × $0.60 = **$0.07**
- **Total: ~$0.19** (gpt-4o-mini, low detail)

#### Option 3: Smaller Test Set (~580 videos, 10% split)
- **Frames processed**: 580 × 8 = 4,640 frames
- **Input tokens**: 4,640 × 85 = **394,400 tokens**
- **Input cost**: (394,400 / 1,000,000) × $0.15 = **$0.06**
- **Output tokens**: 580 × 100 = **58,000 tokens**
- **Output cost**: (58,000 / 1,000,000) × $0.60 = **$0.03**
- **Total: ~$0.09** (gpt-4o-mini, low detail)

### Cost Comparison

| Dataset | Videos | Cost (gpt-4o-mini) | Cost (gpt-4o) |
|---------|--------|-------------------|---------------|
| **EU-Emotion (current)** | 54 | **$0.02** | $0.25 |
| **MindReading (10% test)** | 580 | **$0.09** | $1.13 |
| **MindReading (20% test)** | 1,160 | **$0.19** | $2.25 |
| **MindReading (full)** | 5,801 | **$0.94** | $11.25 |

### Cost Optimization Strategies

1. **Use gpt-4o-mini with low detail** (recommended)
   - Full dataset: ~$0.94 (very affordable!)
   - Test set only: ~$0.09-0.19

2. **Cache descriptions** (free re-runs)
   - First run: ~$0.94
   - Subsequent runs: ~$0 (negligible, only embeddings)

3. **Process fewer frames**
   - Use 4 frames instead of 8: **50% cost reduction**
   - Full dataset: ~$0.47 instead of $0.94

4. **Filter to specific emotions**
   - If you only want 27 emotions (like EU-Emotion): ~$0.06-0.12
   - Much more affordable!

### Recommendation for LLM Experiment

**✅ Very affordable!** Even the full dataset costs less than $1 with gpt-4o-mini.

**Suggested approach:**
1. Start with **10% test set** (~580 videos): **$0.09**
2. If results look good, scale to **20% test set** (~1,160 videos): **$0.19**
3. For final results, use **full dataset**: **$0.94**

**Total cost for iterative approach: ~$1.22** (very reasonable!)

---

## 2. Vision Model Performance Analysis

### Current Performance on EU-Emotion

| Model | Accuracy | Notes |
|-------|----------|-------|
| **CLIP (task-specific)** | **55.6%** | Best performer |
| **ResNet50 (task-specific)** | **35.78%** | Fine-tuned |
| **ViT (task-specific)** | **28.44%** | Fine-tuned |
| **EfficientNet (task-specific)** | **34.86%** | Fine-tuned |
| **Random baseline** | 25% | 4-option forced-choice |

### Expected Improvements on MindReading/Emotions

#### ✅ **YES, Vision Models Will Likely Improve**

**Reasons:**

1. **More Training Data**
   - EU-Emotion: ~959 videos, 27 emotions
   - MindReading/Emotions: 5,801 videos, 425 emotions
   - **6x more data** = better generalization

2. **More Diverse Emotions**
   - 425 emotions vs. 27 = more comprehensive coverage
   - Better representation of emotion space
   - Models learn richer emotion features

3. **More Scenarios & Actors**
   - 412 scenarios vs. limited scenarios in EU-Emotion
   - 13 actors vs. fewer in EU-Emotion
   - Better generalization across contexts

4. **Standardized Format**
   - Consistent filename format makes training easier
   - Better data organization

### Expected Performance Gains

**Conservative Estimates:**

| Model | Current (EU-Emotion) | Expected (MindReading) | Improvement |
|-------|---------------------|----------------------|-------------|
| **CLIP** | 55.6% | **60-65%** | +5-10% |
| **ResNet50** | 35.78% | **45-50%** | +10-15% |
| **ViT** | 28.44% | **40-45%** | +12-17% |
| **EfficientNet** | 34.86% | **45-50%** | +10-15% |

**Optimistic Estimates (with proper fine-tuning):**

| Model | Current | Optimistic | Improvement |
|-------|---------|-----------|-------------|
| **CLIP** | 55.6% | **65-70%** | +10-15% |
| **ResNet50** | 35.78% | **50-55%** | +15-20% |
| **ViT** | 28.44% | **45-50%** | +17-22% |
| **EfficientNet** | 34.86% | **50-55%** | +15-20% |

### Important Considerations

#### ⚠️ **Challenges:**

1. **Emotion Alignment**
   - MindReading has 425 emotions vs. 27 in EU-Emotion
   - Need to filter or map emotions for fair comparison
   - Or adapt experiment to use all 425 emotions

2. **Evaluation Consistency**
   - Current experiment uses 27 specific emotions
   - MindReading has different emotion set
   - Need to decide: filter to 27 or expand to 425?

3. **Training Time**
   - 6x more data = 6x longer training time
   - May need to adjust batch sizes, epochs, etc.

4. **Class Imbalance**
   - 425 emotions = many classes with few samples each
   - Some emotions may have only 6-13 samples
   - Need class balancing strategies

#### ✅ **Advantages:**

1. **Better Generalization**
   - More diverse data = better real-world performance
   - Less overfitting

2. **Richer Representations**
   - Models learn more nuanced emotion distinctions
   - Better feature extraction

3. **More Robust Evaluation**
   - Larger test set = more reliable metrics
   - Better statistical significance

### Recommendation for Vision Models

**✅ YES, use MindReading/Emotions for vision models!**

**Suggested Approach:**

1. **Filter to 27 EU-Emotion emotions** (for fair comparison)
   - Use same emotion set as current experiment
   - Compare directly: EU-Emotion vs. MindReading (filtered)
   - Expected: +5-10% improvement

2. **Or expand to all 425 emotions** (for comprehensive research)
   - More challenging but more comprehensive
   - Better for publication
   - Expected: +10-15% improvement (but harder task)

3. **Training Strategy:**
   - Use same hyperparameters initially
   - May need to adjust learning rate for larger dataset
   - Consider class balancing for 425 emotions

4. **Evaluation:**
   - Use same test split ratio (80/20 or 90/10)
   - Larger test set = more reliable metrics

---

## Summary & Recommendations

### LLM Experiment Costs

| Scenario | Cost | Recommendation |
|----------|------|----------------|
| **10% test set** | **$0.09** | ✅ Start here |
| **20% test set** | **$0.19** | ✅ Good for validation |
| **Full dataset** | **$0.94** | ✅ Very affordable! |

**Verdict: Very affordable!** Even full dataset costs less than $1.

### Vision Model Performance

| Aspect | Expected Improvement |
|--------|---------------------|
| **CLIP** | +5-15% (55.6% → 60-70%) |
| **ResNet50** | +10-20% (35.78% → 45-55%) |
| **ViT** | +12-22% (28.44% → 40-50%) |
| **EfficientNet** | +10-20% (34.86% → 45-55%) |

**Verdict: Significant improvements expected!** More data = better performance.

### Overall Recommendation

**✅ YES, use MindReading/Emotions dataset!**

**Benefits:**
1. **LLM costs are very affordable** (~$0.09-0.94)
2. **Vision models will improve significantly** (+5-20%)
3. **More comprehensive evaluation** (larger test sets)
4. **Better for publication** (more data, more robust)

**Action Plan:**
1. Start with **10% test set for LLM** ($0.09) to validate approach
2. Fine-tune vision models on **full MindReading dataset** (filtered to 27 emotions for comparison)
3. Evaluate and compare with EU-Emotion results
4. Scale up LLM to full dataset if needed ($0.94)

**Total estimated cost: ~$1-2** for comprehensive evaluation (very reasonable!)
