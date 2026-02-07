# Reproducibility: MindReading Emotion Recognition

Step-by-step instructions to replicate the MindReading multimodal and video-only emotion recognition results.

## Environment

- Python 3.8+
- Dependencies: `opencv-python`, `PIL`, `requests`, and project requirements (see project root).
- Optional: `ffmpeg` (for re-encoding), `scipy` (selection bias), `matplotlib` (plots).

## Random seeds

All scripts use a fixed seed for reproducibility:

- **Seed: 42** (documented in scripts and used for trial generation and candidate label generation).

## Model and API

- **Model name and version:** Google Gemini 2.5 Flash (`gemini-2.5-flash`).
- **Provider:** Google AI (Generative Language API).
- **Inference:** Same model for vision and text; no separate vision model.

## Prompt

- **Prompt version:** Default 4-alternative forced choice with reasoning.
- **Location:** `experiments/llm_augmented_emotion_recognition/models/llm_wrapper.py`, method `_classify_google_gemini`.
- **Format:** "Analyze these video frames [AND the accompanying audio] to identify the emotion... The emotion must be one of these options: {labels}. IMPORTANT: Provide your answer FIRST, then explain your reasoning. Format: EMOTION: [label] REASONING: [text]."
- **With audio:** Same prompt plus "AUDIO CUES: Voice tone and prosody, volume, speech patterns, emotional quality of the voice."

## Preprocessing

- **Video:** 4 frames per video, uniformly sampled; converted to JPEG for API.
- **Audio:** Sent as provided (base64). **Methods note:** "Audio consisted of single-word utterances of the emotion label."

## Step 1: Trial definitions

From project root:

```bash
python experiments/llm_augmented_emotion_recognition/scripts/create_mindreading_trials.py \
  --data-root "/Volumes/MindReading/Emotions" \
  --output-dir data/trial_definitions \
  --seed 42
```

Outputs: `mindreading_emotions_train.json`, `mindreading_emotions_test.json`, `mindreading_emotions_all.json`.

## Step 2: (Optional) Re-encode failing videos

If many videos fail to decode (OpenCV), re-encode failing .mov to H.264 MP4:

```bash
python experiments/llm_augmented_emotion_recognition/scripts/reencode_mindreading_videos.py \
  --trial-definitions data/trial_definitions/mindreading_emotions_test.json \
  --data-root "/Volumes/MindReading/Emotions" \
  --output-root results/mindreading_reencoded
```

(Omitting `--output-root` uses `results/mindreading_reencoded` by default. Updated trial definitions are written to `results/mindreading_reencoded/mindreading_emotions_test_reencoded.json`.)

Then use `--data-root results/mindreading_reencoded` and `--trial-definitions results/mindreading_reencoded/mindreading_emotions_test_reencoded.json` for inference.

Validate only (no re-encode):

```bash
python experiments/llm_augmented_emotion_recognition/scripts/reencode_mindreading_videos.py \
  --trial-definitions data/trial_definitions/mindreading_emotions_test.json \
  --data-root "/Volumes/MindReading/Emotions" \
  --output-root results/mindreading_reencode \
  --validate-only
```

## Step 3: Multimodal run (video + audio)

```bash
python experiments/llm_augmented_emotion_recognition/scripts/run_mindreading_multimodal_experiment.py \
  --trial-definitions data/trial_definitions/mindreading_emotions_test.json \
  --data-root "/Volumes/MindReading/Emotions" \
  --audio-base-dir "/Volumes/MindReading/Emotions/Audio" \
  --audio-folder 1 \
  --output-dir results/mindreading_multimodal \
  --provider google \
  --model gemini-2.5-flash \
  --num-frames 4 \
  --use-audio \
  --skip-failed
```

Results: `results/mindreading_multimodal/summary.json`, `predictions.json`.

## Step 4: Video-only baseline

Same trials, no audio (separate cache):

```bash
python experiments/llm_augmented_emotion_recognition/scripts/run_mindreading_multimodal_experiment.py \
  --trial-definitions data/trial_definitions/mindreading_emotions_test.json \
  --data-root "/Volumes/MindReading/Emotions" \
  --audio-base-dir "/Volumes/MindReading/Emotions/Audio" \
  --output-dir results/mindreading_video_only \
  --provider google \
  --model gemini-2.5-flash \
  --num-frames 4 \
  --video-only \
  --skip-failed
```

Results: `results/mindreading_video_only/summary.json`, `predictions.json`.

## Step 5: Compare modalities

```bash
python experiments/llm_augmented_emotion_recognition/scripts/compare_mindreading_modalities.py \
  --multimodal-summary results/mindreading_multimodal/summary.json \
  --video-only-summary results/mindreading_video_only/summary.json \
  --output-dir results/mindreading_comparison
```

Outputs: comparison table (stdout), `modality_comparison.json`, `modality_comparison.csv` (delta = audio contribution in percentage points).

## Step 6: (Optional) Selection bias analysis

If re-encoding is not used and many trials fail to decode:

```bash
python experiments/llm_augmented_emotion_recognition/scripts/analyze_mindreading_selection_bias.py \
  --trial-definitions data/trial_definitions/mindreading_emotions_test.json \
  --summary results/mindreading_multimodal/summary.json \
  --output-dir results/mindreading_selection_bias \
  --plot
```

Outputs: `failure_by_emotion.csv`, `failure_by_folder.csv`, `failure_by_actor.csv`, `selection_bias_report.json`, `SELECTION_BIAS_DISCUSSION.md`, and optionally `selection_bias_plot.png`.

## Summary of key values

| Item            | Value |
|-----------------|--------|
| Seed            | 42     |
| Model           | gemini-2.5-flash |
| Provider        | google |
| Num frames      | 4      |
| Trial defs      | mindreading_emotions_test.json |
| Audio (multimodal) | Single-word utterances of emotion label |

## Methods sentence (for paper)

- **Audio:** "Audio consisted of single-word utterances of the emotion label."
