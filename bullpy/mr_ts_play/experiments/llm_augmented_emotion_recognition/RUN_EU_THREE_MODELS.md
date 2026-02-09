# EU Emotions: Gemini 3 Flash, GPT-5 Mini, Anthropic Opus 4.5

Paths are the same as used for the Gemini 2.5 Flash EU run (data root and audio dir from `configs/llm_config.yaml` and MULTIMODAL_SETUP.md).

**Paths:**
- **Data root:** `/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions`
- **Audio dir:** `/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions_faces/audio/Fixed - amplified volume`
- **Trials:** `data/trial_definitions/eu_emotion_test_final.json`

**Run all three (from project root `mr_ts_play`):**
```bash
./experiments/llm_augmented_emotion_recognition/scripts/run_eu_three_models.sh
```

**Checking progress when running the script:**
- In the terminal you’ll see: `Processing trial 1/118`, `Processing trial 2/118`, … then `RESULTS` and `Accuracy: X.XX%` for each model. The script also prints start/finish timestamps for each run.
- To save a log: `./experiments/llm_augmented_emotion_recognition/scripts/run_eu_three_models.sh 2>&1 | tee results/eu_three_models.log`
- To run in background and follow the log: `nohup ./experiments/llm_augmented_emotion_recognition/scripts/run_eu_three_models.sh > results/eu_three_models.log 2>&1 &` then `tail -f results/eu_three_models.log`
- To see if a run is done: check that `results/eu_emotion_<model>/summary.json` exists and has an `accuracy` field.

**Running individually (recommended):** Run each model in a **separate terminal**. You get:
- **Faster overall** – all three run in parallel (roughly 3× faster).
- **Clear progress** – each terminal shows one model’s trials (e.g. `Processing trial 45/118`).
- **Fault isolation** – if one model fails (e.g. wrong API key or model ID), the others keep going.

Use the three commands below in three terminals.

---

Or run each model separately (copy-paste one at a time):

---

### 1. Gemini 3 Flash (video + audio)
```bash
python experiments/llm_augmented_emotion_recognition/scripts/run_multimodal_experiment.py \
  --trial-definitions data/trial_definitions/eu_emotion_test_final.json \
  --data-root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
  --audio-dir "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions_faces/audio/Fixed - amplified volume" \
  --output-dir results/eu_emotion_gemini3_flash \
  --provider google \
  --model gemini-3-flash-preview \
  --use-audio
```

---

### 2. GPT-5 Mini (video + audio)
```bash
python experiments/llm_augmented_emotion_recognition/scripts/run_multimodal_experiment.py \
  --trial-definitions data/trial_definitions/eu_emotion_test_final.json \
  --data-root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
  --audio-dir "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions_faces/audio/Fixed - amplified volume" \
  --output-dir results/eu_emotion_gpt5_mini \
  --provider openai \
  --model gpt-5-mini \
  --use-audio
```

---

### 3. Anthropic Opus 4.5 (video only; Claude API has no audio input)
```bash
python experiments/llm_augmented_emotion_recognition/scripts/run_multimodal_experiment.py \
  --trial-definitions data/trial_definitions/eu_emotion_test_final.json \
  --data-root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
  --audio-dir "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions_faces/audio/Fixed - amplified volume" \
  --output-dir results/eu_emotion_opus_4_5 \
  --provider anthropic \
  --model claude-opus-4-5 \
  --use-audio
```

---

**Results:** `results/eu_emotion_gemini3_flash/`, `results/eu_emotion_gpt5_mini/`, `results/eu_emotion_opus_4_5/` (each has `summary.json` and `predictions.json`).

**API keys:** Ensure `GOOGLE_API_KEY`, `OPENAI_API_KEY`, and `ANTHROPIC_API_KEY` are set (e.g. in `.env` or environment). If a model ID is wrong (e.g. `gpt-5-mini` or `claude-opus-4-5`), check the provider’s current model list and replace in the script or commands above.
