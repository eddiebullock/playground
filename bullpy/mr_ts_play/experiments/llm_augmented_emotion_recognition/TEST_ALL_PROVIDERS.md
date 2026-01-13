# Testing All LLM Providers

## Quick Start

Run all three providers sequentially:

```bash
bash experiments/llm_augmented_emotion_recognition/scripts/test_all_providers.sh
```

## Manual Testing (One at a Time)

### 1. Test OpenAI

Edit `configs/llm_config.yaml`:
```yaml
llm:
  provider: "openai"
```

Run:
```bash
python experiments/llm_augmented_emotion_recognition/scripts/run_llm_augmented_experiment.py \
  --config experiments/llm_augmented_emotion_recognition/configs/llm_config.yaml \
  --dataset eu_emotion
```

**Cost**: ~$0.02

---

### 2. Test Anthropic (Claude)

Edit `configs/llm_config.yaml`:
```yaml
llm:
  provider: "anthropic"
```

Run:
```bash
python experiments/llm_augmented_emotion_recognition/scripts/run_llm_augmented_experiment.py \
  --config experiments/llm_augmented_emotion_recognition/configs/llm_config.yaml \
  --dataset eu_emotion
```

**Cost**: ~$0.11

---

### 3. Test Google (Gemini)

Edit `configs/llm_config.yaml`:
```yaml
llm:
  provider: "google"
```

Run:
```bash
python experiments/llm_augmented_emotion_recognition/scripts/run_llm_augmented_experiment.py \
  --config experiments/llm_augmented_emotion_recognition/configs/llm_config.yaml \
  --dataset eu_emotion
```

**Cost**: ~$0.03

---

## Cost Summary

| Provider | Model | Cost (EU-Emotion, 54 videos) |
|----------|-------|------------------------------|
| OpenAI | gpt-4o-mini | ~$0.02 |
| Anthropic | claude-3-5-sonnet | ~$0.11 |
| Google | gemini-1.5-pro | ~$0.03 |
| **Total** | All three | **~$0.16** |

## Required API Keys

Make sure your `.env` file has all three keys:

```bash
# At: experiments/cam_human_like/training/.env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

## Results Location

Results are saved to:
```
results/llm_augmented_eu_emotion_weighted_average/
```

Each provider's results are cached separately, so you can compare them.

## Cheaper Options

If you want to reduce costs:

1. **Use cheaper models**:
   - Anthropic: `claude-3-5-haiku-20241022` (~$0.05 instead of $0.11)
   - Google: `gemini-1.5-flash` (~$0.01 instead of $0.03)

2. **Test fewer providers**:
   - Just OpenAI: ~$0.02
   - OpenAI + Google: ~$0.05

3. **Use caching**:
   - First run costs money
   - Subsequent runs are free (uses cache)

## Troubleshooting

### Missing API Key Error

If you get an error about a missing API key:
1. Check `.env` file exists at `experiments/cam_human_like/training/.env`
2. Verify the key is set correctly (no quotes, no spaces)
3. Make sure you have the key for the provider you're testing

### Import Error

If you get an import error:
```bash
pip install anthropic google-generativeai
```
