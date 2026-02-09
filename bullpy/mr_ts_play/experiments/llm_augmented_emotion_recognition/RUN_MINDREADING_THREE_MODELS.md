# MindReading: Opus 4.5, GPT-5, Gemini 3 Pro

Run the same experiment as EU emotions (multimodal where supported) on **MindReading** with **Anthropic Opus 4.5**, **GPT-5**, and **Gemini 3 Pro**.

**Trials:** 1,263 (mindreading_emotions_test.json). ~583 typically process (rest fail video decode).  
**Worst-case cost (all three):** ~**$17** (see `MINDREADING_THREE_MODELS_COST.md`).

---

## Paths

- **Data root:** `/Volumes/MindReading/Emotions`
- **Audio base:** `/Volumes/MindReading/Emotions/Audio`
- **Audio folder:** `1` (options: 1, 2, 3)
- **Trials:** `data/trial_definitions/mindreading_emotions_test.json`

Adjust `MR_DATA_ROOT` / `MR_AUDIO_BASE` in the script if your MindReading volume is elsewhere.

---

## Run all three (from project root `mr_ts_play`)

```bash
./experiments/llm_augmented_emotion_recognition/scripts/run_mindreading_three_models.sh
```

Or run in **three terminals** (faster, one model per terminal):

---

### Terminal 1: Gemini 3 Pro (video + audio)

```bash
python experiments/llm_augmented_emotion_recognition/scripts/run_mindreading_multimodal_experiment.py \
  --trial-definitions data/trial_definitions/mindreading_emotions_test.json \
  --data-root "/Volumes/MindReading/Emotions" \
  --audio-base-dir "/Volumes/MindReading/Emotions/Audio" \
  --audio-folder 1 \
  --output-dir results/mindreading_gemini3_pro \
  --provider google \
  --model gemini-3-pro-preview \
  --use-audio
```

---

### Terminal 2: GPT-5 (video-only)

```bash
python experiments/llm_augmented_emotion_recognition/scripts/run_mindreading_multimodal_experiment.py \
  --trial-definitions data/trial_definitions/mindreading_emotions_test.json \
  --data-root "/Volumes/MindReading/Emotions" \
  --audio-base-dir "/Volumes/MindReading/Emotions/Audio" \
  --audio-folder 1 \
  --output-dir results/mindreading_gpt5 \
  --provider openai \
  --model gpt-5 \
  --use-audio
```

---

### Terminal 3: Anthropic Opus 4.5 (video-only)

```bash
python experiments/llm_augmented_emotion_recognition/scripts/run_mindreading_multimodal_experiment.py \
  --trial-definitions data/trial_definitions/mindreading_emotions_test.json \
  --data-root "/Volumes/MindReading/Emotions" \
  --audio-base-dir "/Volumes/MindReading/Emotions/Audio" \
  --audio-folder 1 \
  --output-dir results/mindreading_opus_4_5 \
  --provider anthropic \
  --model claude-opus-4-5 \
  --use-audio
```

---

## Modality

- **Gemini 3 Pro:** Video + audio (same as EU).
- **GPT-5 / Opus 4.5:** Video only (no audio sent in this setup).

---

## Results

- **Gemini 3 Pro:** `results/mindreading_gemini3_pro/summary.json`, `predictions.json`, `per_emotion.json`
- **GPT-5:** `results/mindreading_gpt5/summary.json`, `predictions.json`, `per_emotion.json`
- **Anthropic Opus 4.5:** `results/mindreading_opus_4_5/summary.json`, `predictions.json`, `per_emotion.json`

Per-emotion scores are in `summary.json` (per_emotion) and `per_emotion.json` for new runs.
