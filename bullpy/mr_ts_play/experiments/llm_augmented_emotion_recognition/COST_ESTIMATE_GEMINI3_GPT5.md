# Cost Estimate: Gemini 3 Flash & GPT-5 Mini

## Test Configuration

- **Test set**: 118 trials (EU-Emotion test_final.json)
- **Frames per trial**: 4 frames
- **Total frames**: 472 frames
- **Current model**: Google Gemini 2.5 Flash
- **Current performance**: 70.34% (83/118)

## Token Usage Estimates

Based on existing cost estimation patterns and Google Gemini API:

### Per Trial Token Breakdown:
- **Input tokens**:
  - 4 frames × 85 tokens/frame (low detail) = 340 tokens
  - Prompt text: ~200 tokens (emotion labels, instructions, reasoning request)
  - **Total input per trial**: ~540 tokens
  
- **Output tokens**:
  - Emotion label: ~5 tokens
  - Reasoning: ~100-150 tokens (with reasoning capture)
  - **Total output per trial**: ~125 tokens

### Total Token Usage (118 trials):
- **Input tokens**: 118 × 540 = **63,720 tokens**
- **Output tokens**: 118 × 125 = **14,750 tokens**

## Pricing (2026)

### Gemini 3 Flash (Google)
- **Input**: $0.50 per 1M tokens
- **Output**: $3.00 per 1M tokens
- **Cached input**: $0.05 per 1M tokens (if using cache)

### GPT-5 Mini (OpenAI)
- **Input**: $0.25 per 1M tokens
- **Output**: $2.00 per 1M tokens

## Cost Calculation

### Gemini 3 Flash

**Single test run (118 trials):**
- Input cost: (63,720 / 1,000,000) × $0.50 = **$0.0319**
- Output cost: (14,750 / 1,000,000) × $3.00 = **$0.0443**
- **Total: $0.0762** (~$0.08)

**With validation set (54 trials) + test set (118 trials) = 172 trials:**
- Input tokens: 172 × 540 = 92,880 tokens
- Output tokens: 172 × 125 = 21,500 tokens
- Input cost: (92,880 / 1,000,000) × $0.50 = $0.0464
- Output cost: (21,500 / 1,000,000) × $3.00 = $0.0645
- **Total: $0.1109** (~$0.11)

**If using cache (subsequent runs):**
- Cached input: (92,880 / 1,000,000) × $0.05 = $0.0046
- Output cost: $0.0645 (same)
- **Total: $0.0691** (~$0.07)

### GPT-5 Mini

**Single test run (118 trials):**
- Input cost: (63,720 / 1,000,000) × $0.25 = **$0.0159**
- Output cost: (14,750 / 1,000,000) × $2.00 = **$0.0295**
- **Total: $0.0454** (~$0.05)

**With validation set (54 trials) + test set (118 trials) = 172 trials:**
- Input tokens: 172 × 540 = 92,880 tokens
- Output tokens: 172 × 125 = 21,500 tokens
- Input cost: (92,880 / 1,000,000) × $0.25 = $0.0232
- Output cost: (21,500 / 1,000,000) × $2.00 = $0.0430
- **Total: $0.0662** (~$0.07)

## Cost Comparison Summary

| Model | Test Set Only (118 trials) | Validation + Test (172 trials) | With Cache (172 trials) |
|-------|---------------------------|--------------------------------|-------------------------|
| **Gemini 3 Flash** | **$0.08** | **$0.11** | **$0.07** |
| **GPT-5 Mini** | **$0.05** | **$0.07** | N/A (no cache discount) |
| **Current (Gemini 2.5 Flash)** | ~$0.08 | ~$0.11 | ~$0.07 |

**Note**: Costs are very similar across models. GPT-5 Mini is slightly cheaper.

## Expected Performance Improvement

### Gemini 3 Flash vs Gemini 2.5 Flash

**Expected improvement: +2-4% accuracy**

**Rationale:**
- Gemini 3 Flash is a newer generation with improved reasoning
- Better handling of nuanced distinctions (afraid vs surprised, etc.)
- Improved vision understanding
- Better instruction following

**Current performance**: 70.34% (test), 74.07% (validation)
**Expected with Gemini 3 Flash**: 72-74% (test), 76-78% (validation)

### GPT-5 Mini vs Gemini 2.5 Flash

**Expected improvement: +1-3% accuracy**

**Rationale:**
- GPT-5 Mini is OpenAI's latest mini model
- Strong performance on classification tasks
- May have different strengths than Gemini (better at some confusions)
- However, "mini" models are typically less capable than full models

**Current performance**: 70.34% (test), 74.07% (validation)
**Expected with GPT-5 Mini**: 71-73% (test), 75-77% (validation)

## Recommendation

### Option 1: Try Gemini 3 Flash (Recommended)
- **Cost**: ~$0.08 per test run
- **Expected improvement**: +2-4% (72-74% test accuracy)
- **Rationale**: Same provider, likely better performance, similar cost
- **Risk**: Low (minimal cost)

### Option 2: Try GPT-5 Mini
- **Cost**: ~$0.05 per test run (slightly cheaper)
- **Expected improvement**: +1-3% (71-73% test accuracy)
- **Rationale**: Different model family, may catch different errors
- **Risk**: Low (minimal cost)

### Option 3: Try Both
- **Total cost**: ~$0.13 for both test runs
- **Benefit**: Compare model families, see which performs better
- **Recommendation**: If budget allows, try both

## Cost-Benefit Analysis

| Model | Cost | Expected Accuracy | Improvement | Cost per % Point |
|-------|------|-------------------|-------------|------------------|
| Gemini 3 Flash | $0.08 | 72-74% | +2-4% | $0.02-0.04 per % |
| GPT-5 Mini | $0.05 | 71-73% | +1-3% | $0.017-0.05 per % |

**Both are very cost-effective** - even if improvement is only 1-2%, the cost is negligible.

## Implementation Notes

1. **Cache usage**: First run will be full cost, subsequent runs use cache (Gemini only)
2. **Token variance**: Actual tokens may vary ±20% depending on prompt length and reasoning length
3. **Multiple runs**: If testing multiple prompt variations, multiply costs accordingly
4. **Budget buffer**: Add 20% buffer for token variance = ~$0.10 per model per test run

## Conclusion

**Both models are very affordable** (~$0.05-0.08 per test run). 

**Recommendation**: 
1. Start with **Gemini 3 Flash** (expected +2-4% improvement)
2. If budget allows, also try **GPT-5 Mini** for comparison
3. Total cost for both: ~$0.13 (very reasonable)

The cost is so low that it's worth trying both to see which performs better, even if improvement is modest.
