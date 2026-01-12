# Implementation Notes: Video Processing with LLM

## What Changed

The experiment now uses **proper video processing** with vision-language models:

### Before (Limited Approach)
- LLM only processed emotion label words
- No actual video content analysis
- Only semantic relationships between labels

### After (Proper Implementation)
- **LLM processes video frames** using GPT-4o/GPT-4o-mini
- Generates descriptions of facial expressions and emotions
- Compares descriptions to emotion labels using embeddings
- Truly multimodal: vision + language

## How It Works

1. **Video Frame Extraction**: Extract 8 frames from each video
2. **Vision Model Processing**: Send representative frame(s) to GPT-4o/GPT-4o-mini
3. **Description Generation**: Model generates text description of emotional expression
4. **Semantic Comparison**: Compare description embedding to emotion label embeddings
5. **Fusion**: Combine CLIP scores (vision) + LLM scores (description-based)

## Model Configuration

The config supports separate models for different tasks:

```yaml
llm:
  embedding_model: "text-embedding-3-small"  # For text embeddings
  vision_model: "gpt-4o-mini"                 # For video description
  vision_detail: "low"                        # "low" or "high"
```

- **Embedding model**: Used for emotion labels and description embeddings (cheap)
- **Vision model**: Used for video frame analysis (more expensive)
- **Vision detail**: Controls image processing detail (affects cost)

## Cost Considerations

- **First run**: ~$0.02 (with gpt-4o-mini, low detail)
- **Subsequent runs**: ~$0 (cached descriptions)
- See `COST_BREAKDOWN.md` for detailed analysis

## Caching

All video descriptions are cached to JSON files:
- Enables reproducible experiments
- Makes re-runs essentially free
- Cache keyed by video path hash

## Fallback Behavior

If vision model fails, the system falls back to:
- Emotion embedding similarities (old approach)
- Logs warning but continues experiment

## Quality vs Cost Trade-offs

| Configuration | Cost | Quality | Use Case |
|---------------|------|---------|----------|
| gpt-4o-mini, low | $0.02 | Good | **Recommended** |
| gpt-4o-mini, high | $0.10 | Better | Higher quality needed |
| gpt-4o, low | $0.25 | Best | Final results |
| gpt-4o, high | $1.50 | Excellent | Research publication |

## Future Improvements

- Multi-frame analysis (process multiple frames per video)
- Temporal modeling (analyze emotion progression)
- Batch processing (process multiple videos in one call)
- Alternative vision models (Claude, Gemini)


