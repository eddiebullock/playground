# Cost Estimate for EU-Emotion Dataset

## Dataset Size
- **EU-Emotion test trials**: 54 videos
- **Frames per video**: 4 frames (configured)
- **Total frames**: 54 × 4 = **216 frames**

## Cost Breakdown by Model

### Cheapest Options (Recommended)

#### 1. Google Gemini 1.5 Flash (CHEAPEST)
- **Cost**: ~$0.05 per 1,000 images
- **For 216 images**: (216 / 1,000) × $0.05 = **~$0.01**
- **Quality**: Good, fast
- **Setup**: Change provider to "google" and model to "gemini-1.5-flash"

#### 2. OpenAI GPT-4o-mini (Low Detail) - CURRENT SETUP ✅
- **Cost**: ~$0.10 per 1,000 images  
- **For 216 images**: (216 / 1,000) × $0.10 = **~$0.02**
- **Quality**: Good
- **Setup**: Already configured (provider: "openai", vision_model: "gpt-4o-mini", vision_detail: "low")

#### 3. Anthropic Claude 3.5 Haiku
- **Cost**: ~$0.25 per 1,000 images
- **For 216 images**: (216 / 1,000) × $0.25 = **~$0.05**
- **Quality**: Good
- **Setup**: Change provider to "anthropic" and model to "claude-3-5-haiku-20241022"

### More Expensive Options (NOT Recommended for Budget)

| Model | Cost per 1K images | Cost for 216 images | Quality |
|-------|-------------------|---------------------|---------|
| Google Gemini 1.5 Pro | ~$0.15 | ~$0.03 | Best |
| OpenAI GPT-4o-mini (high detail) | ~$0.50 | ~$0.11 | Better |
| Anthropic Claude 3.5 Sonnet | ~$0.50 | ~$0.11 | Best |
| OpenAI GPT-4o (low detail) | ~$1.50 | ~$0.32 | Best |
| Anthropic Claude 3 Opus | ~$2.00 | ~$0.43 | Highest |

## Current Configuration (Cheap ✅)

Your current config is set to the **cheapest OpenAI option**:
- ✅ Provider: `openai`
- ✅ Model: `gpt-4o-mini` (cheapest OpenAI model)
- ✅ Vision detail: `low` (cheapest option)
- ✅ Max frames: `4` (reduces cost)
- ✅ Dataset: `eu_emotion` (updated)

**Estimated cost: ~$0.02 for the entire experiment**

## Cost Savings with Caching

- **First run**: ~$0.02 (processes all videos)
- **Subsequent runs**: ~$0 (uses cached descriptions)
- **Re-running experiments**: Free after first run

## Recommendation

**Keep your current setup** (`gpt-4o-mini` with `low` detail):
- ✅ Already configured
- ✅ Very cheap (~$0.02)
- ✅ Good quality
- ✅ Caching makes re-runs free

If you want even cheaper, switch to Google Gemini 1.5 Flash (~$0.01), but the difference is only $0.01.
