# EU Emotions: Results summary (Anthropic, Gemini, OpenAI)

You have scores for **all three providers** on the EU emotions test set (118 trials, `eu_emotion_test_final.json`).

---

## Overall accuracy by model

| Provider | Model | Valid N | Correct | Accuracy | Modality |
|----------|--------|---------|---------|----------|----------|
| **Google** | Gemini 3 Flash | 118 | 91 | **77.12%** | Video + audio |
| **Google** | Gemini 3 Pro | 117 | 96 | **82.05%** | Video + audio |
| **OpenAI** | GPT-5 Mini | 118 | 85 | **72.03%** | Video only |
| **OpenAI** | GPT-5 | 113 | 85 | **75.22%** | Video only |
| **Anthropic** | Opus 4.5 | 111 | 81 | **72.97%** | Video only |

**Result dirs:**  
`results/eu_emotion_gemini3_flash/`, `results/eu_emotion_gemini3_pro/`,  
`results/eu_emotion_gpt5_mini/`, `results/eu_emotion_gpt5/`,  
`results/eu_emotion_opus_4_5/`  
(each has `summary.json`, `predictions.json`).

---

## Per-emotion scores

**Existing runs:** The summaries you have were produced *before* per-emotion was added. They do **not** contain per-emotion breakdowns in `summary.json`.

**Two ways to get per-emotion scores:**

### 1. From existing runs (no re-run)

Use the helper script on each `predictions.json`:

```bash
# From project root: mr_ts_play
python experiments/llm_augmented_emotion_recognition/scripts/compute_per_emotion_scores.py results/eu_emotion_gemini3_pro/predictions.json --csv
python experiments/llm_augmented_emotion_recognition/scripts/compute_per_emotion_scores.py results/eu_emotion_gpt5/predictions.json --csv
python experiments/llm_augmented_emotion_recognition/scripts/compute_per_emotion_scores.py results/eu_emotion_gemini3_flash/predictions.json --csv
python experiments/llm_augmented_emotion_recognition/scripts/compute_per_emotion_scores.py results/eu_emotion_opus_4_5/predictions.json --csv
python experiments/llm_augmented_emotion_recognition/scripts/compute_per_emotion_scores.py results/eu_emotion_gpt5_mini/predictions.json --csv
```

This writes `per_emotion.json` (and with `--csv`, `per_emotion.csv`) in the same directory as the predictions. Each entry is: `emotion -> { count, correct, accuracy }` (only trials with valid predictions).

### 2. Future runs (automatic)

New runs of `run_multimodal_experiment.py` now:

- Add **`per_emotion`** to `summary.json` (per-emotion `count`, `correct`, `accuracy`).
- Write **`per_emotion.json`** in the same output dir (same structure).

So any **new** EU (or other) experiment will include per-emotion scores by default.

---

## Quick reference

- **Anthropic:** Opus 4.5, 72.97% (111 valid), video-only.  
- **Gemini:** Flash 77.12% (118), Pro 82.05% (117), both video + audio.  
- **OpenAI:** GPT-5 Mini 72.03% (118), GPT-5 75.22% (113), both video-only.  
- **Per-emotion:** Use `compute_per_emotion_scores.py` on existing `predictions.json`; new runs get `per_emotion` in summary and `per_emotion.json`.
