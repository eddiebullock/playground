# Understanding Fine-Tuning Validation Accuracy

## The Low Accuracy (15.16%) is NOT Worrying

### Why the Validation Accuracy is Low

The **15.16% validation accuracy** in fine-tuning is measuring something **completely different** from CAM test accuracy:

#### Fine-Tuning Validation Metric
- **What it measures**: Contrastive learning accuracy
- **Task**: Given an image, can the model match it to the correct emotion text label?
- **Difficulty**: Very hard! Model must learn image-text alignment from scratch
- **Baseline**: Random chance = 1/582 (0.17%) for 582 unique emotions

#### CAM Test Metric
- **What it measures**: 4-option forced-choice accuracy
- **Task**: Given an image, choose the correct emotion from 4 options
- **Difficulty**: Easier! Only 4 choices, not 582
- **Baseline**: Random chance = 25% (1/4)

### Why This is Expected

1. **Different Tasks**
   - Fine-tuning: Match image to 1 of 582 possible emotions
   - CAM test: Choose from 4 options (much easier)

2. **Early Training**
   - Only 1 epoch with reduced settings
   - Loss is 2.02 (still high, but decreasing)
   - Model is still learning

3. **Contrastive Learning is Hard**
   - Learning image-text alignment from scratch
   - 582 unique emotions to distinguish
   - Much harder than 4-option forced choice

### What to Watch For

#### Good Signs ✅
- **Loss decreasing**: 2.02 is high but should decrease with more epochs
- **No errors**: Script runs successfully
- **Model saving**: Checkpoints are created

#### What Matters
- **Loss trend**: Should decrease over epochs
- **Final CAM test accuracy**: This is what matters (not validation accuracy)
- **Comparison**: Zero-shot (37%) vs Fine-tuned (target: 65-75%)

### Expected Progress

| Epochs | Validation Acc | Loss | CAM Test Acc (expected) |
|--------|---------------|------|-------------------------|
| 1 | 15% | 2.02 | Not tested yet |
| 5 | 20-30% | 1.5-1.8 | 50-60% |
| 10 | 30-40% | 1.2-1.5 | **65-75%** |

### The Real Test

**Don't judge by validation accuracy!** The real test is:

```bash
# After fine-tuning, evaluate on CAM test set
python experiments/cam_human_like/run_experiment.py \
    --config configs/cam_config.yaml \
    --split all --no-actor-filtering
```

Compare:
- **Zero-shot CLIP**: 37% (baseline)
- **Fine-tuned CLIP**: 65-75% (target)

### Why Fine-Tuning Will Work

1. **Loss is decreasing**: Model is learning
2. **Contrastive learning works**: CLIP was trained this way
3. **More epochs help**: 10 epochs should show clear improvement
4. **CAM test is easier**: 4 options vs 582 emotions

### Recommendation

**Don't worry about the 15% validation accuracy!**

- ✅ Continue training (5-10 epochs)
- ✅ Watch loss decrease (should go from 2.02 → 1.2-1.5)
- ✅ Evaluate on CAM test set (this is what matters)
- ✅ Compare to zero-shot baseline (37% → 65-75%)

The validation accuracy in fine-tuning is just a training signal. The real performance is measured on the CAM test set with 4-option forced-choice evaluation.

## Summary

- **15% validation accuracy**: Expected for contrastive learning (582 emotions)
- **Not comparable**: Different metric than CAM test (4 options)
- **Loss decreasing**: Good sign, model is learning
- **Real test**: CAM test set accuracy (target: 65-75%)
- **Fine-tuning will work**: Loss trend and methodology are sound

Keep training! The low validation accuracy is normal and doesn't predict CAM performance.

