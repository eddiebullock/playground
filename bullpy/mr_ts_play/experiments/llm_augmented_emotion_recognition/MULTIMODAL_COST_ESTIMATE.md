# Multimodal Cost Estimate: Video + Audio with Gemini 2.5 Flash

## Configuration

- **Test set**: 118 trials
- **Frames per trial**: 4 frames
- **Audio**: 1 audio file per trial (when available)
- **Model**: Gemini 2.5 Flash

## Token Usage Estimates

### Per Trial Breakdown (Video + Audio):

**Input tokens:**
- 4 video frames × 257 tokens/image = **1,028 tokens**
- Audio file: ~**500-1,000 tokens** (estimated, depends on audio length)
  - Gemini processes audio differently than images
  - Typical 2-5 second audio clip: ~500-1,000 tokens
  - Conservative estimate: **750 tokens** per audio file
- Prompt text: ~250 tokens (longer prompt for multimodal)
- **Total input per trial**: ~2,028 tokens

**Output tokens:**
- Emotion label: ~5 tokens
- Reasoning: ~150 tokens (multimodal reasoning may be longer)
- **Total output per trial**: ~155 tokens

### Total Token Usage (118 trials):

**Input tokens:**
- Video-only: 118 × 1,228 = 144,904 tokens
- Audio (assuming 80% match rate): 118 × 0.8 × 750 = 70,800 tokens
- Prompt overhead: 118 × 50 = 5,900 tokens
- **Total input**: ~221,604 tokens

**Output tokens:**
- 118 × 155 = **18,290 tokens**

## Cost Calculation

### Gemini 2.5 Flash Pricing:
- **Input**: $0.50 per 1M tokens
- **Output**: $3.00 per 1M tokens

### Multimodal Cost (Video + Audio):

**Input cost:**
- 221,604 tokens × ($0.50 / 1,000,000) = **$0.1108**

**Output cost:**
- 18,290 tokens × ($3.00 / 1,000,000) = **$0.0549**

**Total: $0.1657** (~**$0.17**)

### Comparison: Video-Only vs Multimodal

| Configuration | Input Tokens | Output Tokens | Cost |
|--------------|--------------|--------------|------|
| **Video-only** | 144,904 | 17,700 | **$0.13** |
| **Multimodal** | 221,604 | 18,290 | **$0.17** |
| **Difference** | +76,700 | +590 | **+$0.04** |

## Cost Breakdown

**Per trial cost:**
- Video-only: ~$0.0011 per trial
- Multimodal: ~$0.0014 per trial
- **Audio adds**: ~$0.0003 per trial

**Total experiment cost:**
- **Multimodal**: **~$0.17** for 118 trials

## Cost Efficiency

**Expected improvement:**
- Current (video-only): 70% accuracy
- Expected (multimodal): 73-78% accuracy (+3-8%)
- **Cost per % point improvement**: ~$0.005-0.013

## Notes

1. **Audio matching**: Not all trials will have matching audio files
   - Estimated 80% match rate (94 trials with audio)
   - Remaining 24 trials use video-only
   - Cost reflects this mixed scenario

2. **Audio token cost**: Conservative estimate
   - Actual cost may vary based on:
     - Audio file length (2-5 seconds typical)
     - Audio encoding quality
     - Gemini's audio processing method
   - If audio costs less: total cost could be ~$0.15
   - If audio costs more: total cost could be ~$0.20

3. **Caching**: Results are cached by default
   - First run: ~$0.17
   - Subsequent runs: Much cheaper (only output tokens)

## Recommendation

**Yes, run it!** 

- **Cost**: Only **$0.17** for the full experiment
- **Expected improvement**: +3-8% accuracy
- **Very affordable** for the potential gain
- **Cached results** make re-runs even cheaper

## Running the Experiment

```bash
python experiments/llm_augmented_emotion_recognition/scripts/run_multimodal_experiment.py \
    --trial-definitions data/trial_definitions/eu_emotion_test_final.json \
    --data-root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
    --audio-dir "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions_faces/audio/Fixed - amplified volume" \
    --output-dir results/multimodal_gemini \
    --provider google \
    --model gemini-2.5-flash \
    --use-audio \
    --skip-failed
```

**Estimated cost: ~$0.17** (very affordable!)
