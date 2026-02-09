# EU Emotions: Gemini 3 Pro & GPT-5

Run EU emotion test (118 trials, video + audio where supported) with **Gemini 3 Pro** and **GPT-5** (full).

**Cost (approx.):** ~$0.70 (Gemini 3 Pro) + ~$0.48 (GPT-5) ≈ **$1.20** total. See `COST_ESTIMATE_GEMINI3_PRO_GPT5.md`.

---

## Run both (from project root `mr_ts_play`)

```bash
./experiments/llm_augmented_emotion_recognition/scripts/run_eu_gemini3_pro_gpt5.sh
```

Or run in **two terminals** (faster, one model per terminal):

---

### Terminal 1: Gemini 3 Pro (video + audio)

```bash
python experiments/llm_augmented_emotion_recognition/scripts/run_multimodal_experiment.py \
  --trial-definitions data/trial_definitions/eu_emotion_test_final.json \
  --data-root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
  --audio-dir "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions_faces/audio/Fixed - amplified volume" \
  --output-dir results/eu_emotion_gemini3_pro \
  --provider google \
  --model gemini-3-pro-preview \
  --use-audio
```

---

### Terminal 2: GPT-5 (video-only; OpenAI full model has no audio input in this setup)

```bash
python experiments/llm_augmented_emotion_recognition/scripts/run_multimodal_experiment.py \
  --trial-definitions data/trial_definitions/eu_emotion_test_final.json \
  --data-root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
  --audio-dir "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions_faces/audio/Fixed - amplified volume" \
  --output-dir results/eu_emotion_gpt5 \
  --provider openai \
  --model gpt-5 \
  --use-audio
```

*(GPT-5 does not accept `input_audio` in this codebase; run is video-only. If the API returns a model-not-found error, try `gpt-5-turbo` or check [OpenAI models](https://platform.openai.com/docs/models).)*

---

## Expected performance

**Current EU results (same 118 trials):**
- **Gemini 3 Flash:** 77.12% (video + audio)
- **GPT-5 Mini:** video-only (no audio support); accuracy from your run
- **Anthropic Opus 4.5:** video-only; accuracy from your run

**Rough expectations for Pro/full models:**

| Model | vs current | Expected range | Rationale |
|-------|------------|----------------|-----------|
| **Gemini 3 Pro** | vs Gemini 3 Flash (77.12%) | **78–80%** | Pro is stronger on reasoning and nuanced vision; +1–3 pp typical on hard classification. |
| **GPT-5** | vs GPT-5 Mini (video-only) | **+2–5 pp** | Full model usually gains on fine-grained tasks; exact gain depends on Mini baseline. |

- **Gemini 3 Pro:** Likely **78–80%** (up from 77.12%). Pro improves on complex visual and multimodal reasoning; emotion labels are fine-grained, so a small gain is plausible.
- **GPT-5:** If GPT-5 Mini (video-only) is around 72–75%, GPT-5 might be **74–78%**. If the API supports audio for GPT-5 in future, that could add a few more points.

These are **expectations**, not guarantees; actual results depend on the exact models and randomness.

---

## Results

- **Gemini 3 Pro:** `results/eu_emotion_gemini3_pro/summary.json`, `predictions.json`
- **GPT-5:** `results/eu_emotion_gpt5/summary.json`, `predictions.json`
