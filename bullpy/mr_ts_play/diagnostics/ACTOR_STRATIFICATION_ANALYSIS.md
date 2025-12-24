# Actor Stratification: Trade-off Analysis

## The Fundamental Question

**Should we use actor-independent splits, or allow actors to appear in multiple splits?**

This is a critical research design decision that affects:
1. **Data availability** (samples per class)
2. **Model validity** (what the model actually learns)
3. **Paper credibility** (rigor of experimental design)

## Current Situation (With Actor Stratification)

### Data Distribution
- **Train**: ~582 samples, ~410 classes = **~1.4 samples per class**
- **Test**: ~1,059 samples, ~410 classes = **~2.6 samples per class**
- **Result**: Few-shot learning problem, very challenging

### Actor Independence
- ✅ **No actor overlap** between train/val/test
- ✅ **Prevents data leakage**
- ✅ **Model must learn emotions, not actors**

### Performance
- Current: ~0.1-0.4% accuracy (at or below random: 0.24%)
- Expected with improvements: >1% (still very low)

## Alternative: Without Actor Stratification

### Data Distribution (Estimated)
- **Train**: ~1,730 samples, ~410 classes = **~4.2 samples per class** (3x improvement)
- **Test**: ~370 samples, ~410 classes = **~0.9 samples per class**
- **Result**: Still few-shot, but more manageable

### Actor Overlap
- ⚠️ **Actors appear in multiple splits**
- ⚠️ **Risk of learning actor-specific features**
- ⚠️ **Performance may be inflated**

### Expected Performance
- Likely: 5-15% accuracy (much better!)
- But: May be learning "this actor's face" not "this emotion"

## The Trade-off

### Option 1: Keep Actor Stratification (Current)

**Pros:**
- ✅ **Scientifically rigorous**: Model learns emotions, not actors
- ✅ **Generalizable**: Should work on new actors
- ✅ **High credibility**: No data leakage concerns
- ✅ **Publishable**: Meets standards for computational psychiatry research

**Cons:**
- ❌ **Extreme data scarcity**: 1-2 samples per class
- ❌ **Very low performance**: Likely <5% even with best methods
- ❌ **May need different methods**: Few-shot learning required
- ❌ **Harder to publish**: Low performance may be seen as "failure"

**When to use:**
- If you want to claim the model generalizes to new actors
- If you want maximum scientific rigor
- If you're willing to use few-shot learning methods
- If you can frame low performance as "expected given data scarcity"

### Option 2: Remove Actor Stratification

**Pros:**
- ✅ **More data**: 3-4x more samples per class
- ✅ **Better performance**: Likely 5-15% accuracy
- ✅ **Standard methods work**: Can use supervised learning
- ✅ **Easier to publish**: Higher numbers look better

**Cons:**
- ❌ **Data leakage risk**: Model may learn actor-specific features
- ❌ **Lower credibility**: Reviewers may question validity
- ❌ **Limited generalization**: May not work on new actors
- ❌ **May undermine paper**: If discovered, could be seen as flawed

**When to use:**
- If you explicitly state this limitation
- If you're comparing to methods that also don't use actor stratification
- If you frame it as "in-domain" performance (same actors)
- If you add analysis showing what the model learned

## Hybrid Approach: Report Both

**Best of both worlds:**
1. **Primary results**: Actor-stratified (rigorous)
2. **Ablation study**: Without actor stratification (shows upper bound)
3. **Analysis**: What does the model learn in each case?

**Example structure:**
```
Results:
- Actor-stratified: 1.2% accuracy (rigorous, generalizable)
- Non-stratified: 12.5% accuracy (upper bound, in-domain)

Analysis:
- Actor-stratified model learns emotion features
- Non-stratified model learns actor+emotion features
- Gap shows challenge of generalization
```

## Recommendations

### For PhD-Level Research

**I recommend keeping actor stratification** because:

1. **Scientific rigor**: Computational psychiatry requires high standards
2. **Generalization claim**: You can claim the model works on new actors
3. **Honest reporting**: Low performance is honest about the challenge
4. **Methodological contribution**: Few-shot learning on emotions is novel

**But also:**
- Report both (stratified and non-stratified) as ablation
- Use few-shot learning methods (not standard classification)
- Frame low performance as expected given data scarcity
- Compare to human performance (if available)
- Use hierarchical or similarity-based evaluation

### How to Frame Low Performance

**Don't say:** "Our model achieves 1% accuracy"
**Do say:** 
- "Given 1-2 samples per class, this is a few-shot learning problem"
- "Our model achieves 1% accuracy, which is 4x better than random (0.24%)"
- "With actor-independent splits ensuring generalization, standard methods struggle"
- "This highlights the challenge of fine-grained emotion recognition with limited data"

### Alternative Framings

1. **Few-shot learning benchmark**: Frame as a challenging few-shot learning dataset
2. **Hierarchical evaluation**: Report accuracy at different granularity levels
3. **Similarity-based metrics**: Use top-k accuracy, similarity scores
4. **Transfer learning**: Show improvement from pretraining on larger datasets

## Decision Framework

Ask yourself:

1. **What is your research question?**
   - "Can models recognize emotions?" → Need actor stratification
   - "Can models recognize emotions from these specific actors?" → Can skip

2. **What do you want to claim?**
   - "Generalizes to new actors" → Need actor stratification
   - "Works on this dataset" → Can skip

3. **What's your target audience?**
   - Computational psychiatry (rigorous) → Need actor stratification
   - Applied ML (practical) → Can be more flexible

4. **What's your contribution?**
   - Novel method → Need rigorous evaluation
   - Application → Can be more practical

## Conclusion

**For a PhD-level computational psychiatry study, I strongly recommend:**

1. ✅ **Keep actor stratification** as primary evaluation
2. ✅ **Report non-stratified** as ablation/upper bound
3. ✅ **Use few-shot learning methods** (not standard classification)
4. ✅ **Frame results appropriately** (expected given data scarcity)
5. ✅ **Use alternative metrics** (top-k, similarity, hierarchical)

**The low performance is a feature, not a bug** - it honestly reflects the challenge of the task. A paper that reports 1% with rigorous evaluation is more credible than one reporting 15% with data leakage.









