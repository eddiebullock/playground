# MindReading Emotions Dataset - Multimodal Experiment Setup

This document describes how to run the multimodal emotion recognition experiment on the MindReading Emotions dataset.

**Methods note (for papers):** "Audio consisted of single-word utterances of the emotion label."

## Dataset Structure

The MindReading dataset has a different structure than the EU-Emotion dataset:

### Videos
- **Location**: `/Volumes/MindReading/Emotions/`
- **Structure**: 
  - 24 numbered folders (`01/`, `02/`, `03/`, ..., `24/`)
  - Each folder contains emotion code subfolders (e.g., `0100104/`, `0200102/`)
  - Each emotion code folder contains multiple `.mov` video files
  - Video filename pattern: `{code}{actor}{emotion}.mov` (e.g., `0100104M1Vhumiliating.mov`)

### Audio
- **Location**: `/Volumes/MindReading/Emotions/Audio/`
- **Structure**:
  - Three folders: `1/emotion/`, `2/emotion/`, `3/emotion/`
  - Each folder contains 421 audio files (`.aif` format)
  - Audio filename pattern: `{code}{emotion}_{folder}w.aif` (e.g., `0100104humiliating_1w.aif`)
  - All three folders contain the same 412 emotion codes (different recordings/versions)

### Dataset Statistics
- **Total videos**: 5,480
- **Total unique emotions**: 409
- **Total unique emotion codes**: 412
- **Video folders**: 24
- **Audio folders**: 3 (each with 421 files)

## Setup Steps

### 1. Generate Trial Definitions

First, create trial definitions from the dataset:

```bash
python experiments/llm_augmented_emotion_recognition/scripts/create_mindreading_trials.py \
  --data-root "/Volumes/MindReading/Emotions" \
  --output-dir data/trial_definitions \
  --seed 42
```

This will create:
- `data/trial_definitions/mindreading_emotions_train.json` (80% of trials)
- `data/trial_definitions/mindreading_emotions_test.json` (20% of trials)
- `data/trial_definitions/mindreading_emotions_all.json` (all trials)

**Options**:
- `--trials-per-emotion`: Number of trials per emotion (default: one per video)
- `--min-videos-per-emotion`: Minimum videos required per emotion (default: 1)
- `--train-ratio`: Train/test split ratio (default: 0.8)

### 2. (Optional) Re-encode Failing Videos

If many videos fail to decode (OpenCV), re-encode failing .mov files to H.264 MP4 so the full test set can run:

```bash
python experiments/llm_augmented_emotion_recognition/scripts/reencode_mindreading_videos.py \
  --trial-definitions data/trial_definitions/mindreading_emotions_test.json \
  --data-root "/Volumes/MindReading/Emotions" \
  --output-root results/mindreading_reencoded
```

(Default `--output-root` is `results/mindreading_reencoded` if omitted.) Then use `--data-root results/mindreading_reencoded` and `--trial-definitions results/mindreading_reencoded/mindreading_emotions_test_reencoded.json` for inference. See **REPRODUCIBILITY.md** for full steps.

**Requirements:** You must have **ffmpeg** installed (e.g. `brew install ffmpeg`). The script checks for ffmpeg before re-encoding and exits with instructions if it is missing.

**If you can't open the failing videos manually:** The 680 files that OpenCV can't read may be corrupted or in a codec/container your system doesn't support. It is still worth trying re-encoding after installing ffmpeg—ffmpeg sometimes can read formats that QuickTime or VLC cannot. If ffmpeg also fails to re-encode them (e.g. "Invalid data" or similar), the files are likely unrecoverable. In that case:
1. Report results on the **583 valid trials only**, with a clear limitation in the paper.
2. Run the **selection bias analysis** (`analyze_mindreading_selection_bias.py`) to check whether failures are random across emotion/folder/actor, and include that in the discussion.

### 3. Run Multimodal Experiment

Run the multimodal (video + audio) experiment:

```bash
python experiments/llm_augmented_emotion_recognition/scripts/run_mindreading_multimodal_experiment.py \
  --trial-definitions data/trial_definitions/mindreading_emotions_test.json \
  --data-root "/Volumes/MindReading/Emotions" \
  --audio-base-dir "/Volumes/MindReading/Emotions/Audio" \
  --audio-folder 1 \
  --output-dir results/mindreading_multimodal \
  --provider google \
  --model gemini-2.5-flash \
  --use-audio \
  --skip-failed
```

**Options**:
- `--audio-folder`: Which audio folder to use (`1`, `2`, or `3`, default: `1`)
- `--num-frames`: Number of frames to extract per video (default: 4)
- `--use-audio`: Include audio files in multimodal input
- `--skip-failed`: Skip trials where video or audio files are missing

### 4. Video-Only Baseline

To run video-only (no audio) for comparison and to quantify audio contribution:

```bash
python experiments/llm_augmented_emotion_recognition/scripts/run_mindreading_multimodal_experiment.py \
  --trial-definitions data/trial_definitions/mindreading_emotions_test.json \
  --data-root "/Volumes/MindReading/Emotions" \
  --audio-base-dir "/Volumes/MindReading/Emotions/Audio" \
  --output-dir results/mindreading_video_only \
  --provider google \
  --model gemini-2.5-flash \
  --skip-failed
```

(Use `--video-only` explicitly; same trials, separate cache.)

### 5. Compare Modalities

After running both multimodal and video-only, compare accuracy and delta (audio contribution):

```bash
python experiments/llm_augmented_emotion_recognition/scripts/compare_mindreading_modalities.py \
  --multimodal-summary results/mindreading_multimodal/summary.json \
  --video-only-summary results/mindreading_video_only/summary.json \
  --output-dir results/mindreading_comparison
```

### 6. (Optional) Selection Bias Analysis

If re-encoding is not used and many trials fail to decode, analyze whether failures are random:

```bash
python experiments/llm_augmented_emotion_recognition/scripts/analyze_mindreading_selection_bias.py \
  --trial-definitions data/trial_definitions/mindreading_emotions_test.json \
  --summary results/mindreading_multimodal/summary.json \
  --output-dir results/mindreading_selection_bias \
  --plot
```

Outputs: CSV/JSON by emotion, folder, actor; chi-square tests; SELECTION_BIAS_DISCUSSION.md template.

### 7. Reproducibility

See **REPRODUCIBILITY.md** in this directory for step-by-step replication (model, prompt, seed, preprocessing, methods note).

## Audio Matching Strategy

The audio matcher (`mindreading_audio_matcher.py`) matches audio files to videos by:

1. **Extracting emotion code**: First 7 digits from the video's parent directory name
2. **Extracting emotion label**: From the video filename (e.g., `humiliating` from `0100104M1Vhumiliating.mov`)
3. **Constructing audio filename**: `{code}{emotion}_{folder}w.aif` (e.g., `0100104humiliating_1w.aif`)
4. **Fallback**: If exact match not found, searches for any audio file with matching code

## Differences from EU-Emotion Experiment

1. **Dataset structure**: Different folder organization (numbered folders vs emotion folders)
2. **File formats**: `.mov` videos and `.aif` audio (vs `.mp4`/`.mp3` in EU-Emotion)
3. **Audio organization**: Three separate audio folders instead of emotion-based folders
4. **Emotion codes**: Uses 7-digit codes to match videos and audio
5. **More emotions**: 409 unique emotions (vs 27 in EU-Emotion)

## Output Files

The experiment generates:

- `predictions.json`: Detailed predictions for each trial
- `summary.json`: Summary statistics (accuracy, counts, etc.)
- `cache/`: Cached LLM responses (to avoid repeated API calls)

## Notes

- The audio folders (1, 2, 3) contain the same emotion codes but different recordings. You can experiment with different folders to see if it affects performance.
- The dataset has many fine-grained emotions (e.g., "humiliating", "detesting", "blaming"), making it more challenging than basic emotion recognition.
- Some emotions have multiple codes (e.g., "hysterical" has 2 codes), which provides more training examples.
