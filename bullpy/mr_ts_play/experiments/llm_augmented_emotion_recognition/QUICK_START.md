# Quick Start: Testing LLMs on EU-Emotion

## Prerequisites Check

1. **API Keys**: Make sure `.env` file has all keys:
   ```bash
   # File: experiments/cam_human_like/training/.env
   OPENAI_API_KEY=sk-...
   ANTHROPIC_API_KEY=sk-ant-...
   GOOGLE_API_KEY=...
   ```

2. **Dependencies**: Install if needed:
   ```bash
   pip install anthropic google-generativeai
   ```

## Option 1: Test All Three Providers (Automated)

Run this single command to test OpenAI, Anthropic, and Google:

```bash
bash experiments/llm_augmented_emotion_recognition/scripts/test_all_providers.sh
```

**Cost**: ~$0.16 total (~$0.02 OpenAI + ~$0.11 Anthropic + ~$0.03 Google)

---

## Option 2: Test One Provider at a Time

### Test OpenAI (Cheapest - $0.02)

```bash
python experiments/llm_augmented_emotion_recognition/scripts/run_llm_augmented_experiment.py \
  --config experiments/llm_augmented_emotion_recognition/configs/llm_config.yaml \
  --dataset eu_emotion
```

(Config is already set to `provider: "openai"`)

### Test Anthropic (Claude)

1. Edit `configs/llm_config.yaml`: Change `provider: "anthropic"`
2. Run:
```bash
python experiments/llm_augmented_emotion_recognition/scripts/run_llm_augmented_experiment.py \
  --config experiments/llm_augmented_emotion_recognition/configs/llm_config.yaml \
  --dataset eu_emotion
```

### Test Google (Gemini)

1. Edit `configs/llm_config.yaml`: Change `provider: "google"`
2. Run:
```bash
python experiments/llm_augmented_emotion_recognition/scripts/run_llm_augmented_experiment.py \
  --config experiments/llm_augmented_emotion_recognition/configs/llm_config.yaml \
  --dataset eu_emotion
```

---

## What the Experiment Does

The experiment runs **three conditions**:

1. **CLIP-only**: Uses your fine-tuned CLIP model (baseline)
2. **LLM-only**: Uses LLM vision model to classify emotions
3. **LLM-augmented**: Combines CLIP + LLM (70% CLIP + 30% LLM)

## Results Location

Results are saved to:
```
results/llm_augmented_eu_emotion_weighted_average/
```

Each run creates:
- Per-model accuracy scores
- Confusion matrices
- Per-emotion breakdowns
- Comparison report

## Expected Results

| Condition | Expected Accuracy |
|----------|------------------|
| CLIP-only | ~55% |
| LLM-only | ~50-60% |
| LLM-augmented | ~65-70% |

## Troubleshooting

### Missing API Key
```
ValueError: OPENAI_API_KEY not found
```
**Fix**: Check `.env` file at `experiments/cam_human_like/training/.env`

### Import Error
```
ModuleNotFoundError: No module named 'anthropic'
```
**Fix**: `pip install anthropic google-generativeai`

### Config Error
```
KeyError: 'provider'
```
**Fix**: Make sure config file has `provider: "openai"` (or "anthropic"/"google")
