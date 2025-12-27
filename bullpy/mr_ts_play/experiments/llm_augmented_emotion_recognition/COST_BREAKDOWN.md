# Cost Breakdown: LLM-Augmented Emotion Recognition Experiment

## Overview

This document provides detailed cost estimates for running the LLM-augmented emotion recognition experiment with proper video processing using vision-language models.

## Experiment Scale

- **CAM test trials**: 40 videos
- **EU-Emotion test trials**: 54 videos
- **Total videos**: 94 videos
- **Frames per video**: 8 frames (default)
- **Total frames**: 752 frames

## Model Options and Pricing

### Vision Models (for Video Description)

| Model | Input Cost | Output Cost | Best For |
|-------|-----------|-------------|----------|
| `gpt-4o-mini` | $0.15 / 1M tokens | $0.60 / 1M tokens | **Cost-effective, recommended** |
| `gpt-4o` | $2.50 / 1M tokens | $10.00 / 1M tokens | Higher quality |
| `gpt-4-turbo` | $10.00 / 1M tokens | $30.00 / 1M tokens | Legacy, not recommended |

### Embedding Models (for Emotion Labels)

| Model | Cost | Notes |
|-------|------|-------|
| `text-embedding-3-small` | $0.02 / 1M tokens | **Recommended** |
| `text-embedding-3-large` | $0.13 / 1M tokens | Higher quality |

## Cost Calculation

### Scenario 1: Using `gpt-4o-mini` (Recommended - Cost-Effective)

**Video Description (Vision Processing):**

- **Frames processed**: 752 frames (94 videos × 8 frames)
- **Tokens per frame (input)**: 
  - Low detail: ~85 tokens/frame
  - High detail: ~765 tokens/frame
- **Using low detail** (recommended for cost):
  - Input tokens: 752 × 85 = **63,920 tokens**
  - Input cost: (63,920 / 1,000,000) × $0.15 = **$0.0096** (~$0.01)
- **Output tokens** (descriptions):
  - Average description: ~100 tokens
  - Total output: 94 × 100 = **9,400 tokens**
  - Output cost: (9,400 / 1,000,000) × $0.60 = **$0.0056** (~$0.01)

**Emotion Embeddings (Label Processing):**

- **Unique emotions**: 47 emotions
- **Tokens per emotion**: ~5 tokens
- **Total tokens**: 47 × 5 = **235 tokens**
- **Cost**: (235 / 1,000,000) × $0.02 = **$0.000005** (negligible)

**Description-to-Emotion Comparison:**

- **Description embeddings**: 94 descriptions × ~100 tokens = **9,400 tokens**
- **Cost**: (9,400 / 1,000,000) × $0.02 = **$0.0002** (negligible)

**Total Cost (gpt-4o-mini, low detail): ~$0.02**

### Scenario 2: Using `gpt-4o` (Higher Quality)

**Video Description:**

- **Input tokens** (low detail): 63,920 tokens
- **Input cost**: (63,920 / 1,000,000) × $2.50 = **$0.16**
- **Output tokens**: 9,400 tokens
- **Output cost**: (9,400 / 1,000,000) × $10.00 = **$0.09**

**Total Cost (gpt-4o, low detail): ~$0.25**

### Scenario 3: Using `gpt-4o-mini` with High Detail

**Video Description:**

- **Input tokens** (high detail): 752 × 765 = **575,280 tokens**
- **Input cost**: (575,280 / 1,000,000) × $0.15 = **$0.086** (~$0.09)
- **Output tokens**: 9,400 tokens
- **Output cost**: (9,400 / 1,000,000) × $0.60 = **$0.0056** (~$0.01)

**Total Cost (gpt-4o-mini, high detail): ~$0.10**

## Cost Comparison Summary

| Configuration | Total Cost | Notes |
|---------------|------------|-------|
| **gpt-4o-mini, low detail** | **~$0.02** | **Recommended - Best value** |
| gpt-4o-mini, high detail | ~$0.10 | Better quality, 5x cost |
| gpt-4o, low detail | ~$0.25 | Higher quality, 12.5x cost |
| gpt-4o, high detail | ~$1.50 | Best quality, 75x cost |

## Cost Optimization Strategies

1. **Use `gpt-4o-mini` with low detail** (recommended)
   - Best balance of cost and quality
   - Total cost: ~$0.02 for entire experiment

2. **Cache video descriptions**
   - Descriptions are cached after first run
   - Subsequent runs cost $0 (only emotion embeddings, negligible)

3. **Process fewer frames**
   - Use 1-2 representative frames instead of 8
   - Reduces cost proportionally

4. **Batch processing**
   - Process multiple videos in one API call (if supported)
   - Reduces overhead

## Running Multiple Experiments

- **First run** (with API calls): ~$0.02
- **Subsequent runs** (cached): ~$0 (negligible, only embeddings)

## Cost Monitoring

Monitor costs in OpenAI dashboard:
- Set up usage alerts
- Track token usage per experiment
- Review costs before scaling up

## Recommendations

1. **Start with `gpt-4o-mini` and low detail** for initial experiments
2. **Cache all responses** to enable free re-runs
3. **Compare results** between low/high detail to assess quality trade-off
4. **Scale up** to `gpt-4o` only if needed for final results

## Example: Full Experiment Run

```bash
# First run (costs ~$0.02)
python scripts/run_llm_augmented_experiment.py \
    --config configs/llm_config.yaml \
    --dataset cam \
    --use_cache

# Second run (costs ~$0, uses cache)
python scripts/run_llm_augmented_experiment.py \
    --config configs/llm_config.yaml \
    --dataset cam \
    --use_cache
```

## Notes

- All costs are estimates based on OpenAI pricing as of 2024
- Actual costs may vary based on:
  - Exact token counts
  - Image resolution
  - Description length
  - API rate limits
- Costs are per experiment run (94 videos)
- Caching makes subsequent runs essentially free

