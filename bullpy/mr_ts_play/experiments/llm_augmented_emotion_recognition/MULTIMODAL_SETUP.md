# Multimodal Emotion Recognition with Gemini Flash 2.5

## Overview

This multimodal extension adds **audio input** to the existing video-based emotion recognition experiment. By combining visual (video frames) and audio cues, we aim to improve performance beyond the current 70% accuracy achieved with Gemini Flash 2.5.

## How Audio-Video Matching Works

### File Structure

- **Video files**: Located in emotion folders (e.g., `EU_emotions_faces/Disappointed/EF9_cut.mp4`)
- **Audio files**: Located in `EU_emotions_faces/audio/Fixed - amplified volume/[Emotion]/fix__[filename].mp3`

### Matching Strategy

The `audio_matcher.py` module uses a two-step matching approach:

1. **Same Actor Preference**: If `prefer_same_actor=True`, tries to match audio files with the same actor code (e.g., video `EF9_cut.mp4` → audio `fix__EV9D.mp3`, both actor "E")
2. **Emotion Category Fallback**: If no same-actor match is found, uses any audio file from the same emotion category

**Note**: Audio and video files don't have perfect 1:1 matching. The audio files appear to be different instances/actors of the same emotion, which is still useful for multimodal learning.

## Usage

### Basic Multimodal Experiment

Run the experiment with audio enabled:

```bash
python experiments/llm_augmented_emotion_recognition/scripts/run_multimodal_experiment.py \
    --trial-definitions data/trial_definitions/eu_emotion_test_final.json \
    --data-root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
    --audio-dir "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions_faces/audio/Fixed - amplified volume" \
    --output-dir results/multimodal_gemini \
    --provider google \
    --model gemini-2.5-flash \
    --use-audio \
    --num-frames 4 \
    --skip-failed
```

### Video-Only Baseline (for comparison)

Run without audio to compare:

```bash
python experiments/llm_augmented_emotion_recognition/scripts/run_multimodal_experiment.py \
    --trial-definitions data/trial_definitions/eu_emotion_test_final.json \
    --data-root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
    --audio-dir "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions_faces/audio/Fixed - amplified volume" \
    --output-dir results/video_only_baseline \
    --provider google \
    --model gemini-2.5-flash \
    --num-frames 4 \
    --skip-failed
```

## Key Features

### 1. Audio Matching (`audio_matcher.py`)

- **`find_matching_audio_file()`**: Finds audio file for a single video
- **`find_audio_files_for_trials()`**: Batch matching for all trials
- Handles emotion name variations (e.g., "Afraid Low Intensity" vs "Afraid-Low Intensity")
- Prefers same-actor matches when available

### 2. Multimodal LLM Wrapper (`llm_wrapper.py`)

- **Enhanced `classify_emotion_directly()`**: Now accepts `audio_path` parameter
- **Gemini API Integration**: Sends both video frames and audio to Gemini Flash 2.5
- **Enhanced Prompts**: Prompts explicitly mention both visual and audio cues when audio is present
- **Caching**: Cache keys include audio path to avoid confusion

### 3. Experiment Script (`run_multimodal_experiment.py`)

- Processes trials with optional audio
- Tracks which trials have audio vs video-only
- Saves detailed predictions with audio paths
- Calculates accuracy metrics

## Expected Improvements

Based on multimodal emotion recognition research:

- **Visual-only baseline**: ~70% (current performance)
- **Multimodal (video + audio)**: Expected **+3-8% improvement** (73-78%)

**Why audio helps:**
- Voice tone and prosody provide additional emotional cues
- Some emotions are better expressed through voice (e.g., "Joking", "Excited")
- Audio can disambiguate visually similar emotions (e.g., "Sad" vs "Disappointed")

## Output Files

Results are saved to the output directory:

- **`predictions.json`**: Detailed predictions for each trial
  - Includes `audio_path` field (None if audio not found)
  - Includes `reasoning` from Gemini
  - Includes `scores` for all candidate labels

- **`summary.json`**: Overall statistics
  - Total trials processed
  - Number of trials with audio found
  - Accuracy metrics

- **`cache/`**: Cached LLM responses (for faster re-runs)

## Troubleshooting

### Audio Files Not Found

If many trials show `audio_path: null`:

1. Check audio directory path is correct
2. Verify emotion names match between video and audio folders
3. Check for variations like "Low Intensity" vs "-Low Intensity"

### Gemini API Errors

If you get API errors:

1. Check `GOOGLE_API_KEY` is set in environment
2. Verify audio file format is supported (`.mp3`, `.wav`, `.m4a`, `.ogg`)
3. Check audio file size (very large files may timeout)

### Performance Issues

- **Caching**: Results are cached by default - delete cache to re-run
- **Batch Processing**: The script processes trials sequentially (can be parallelized if needed)
- **Audio Loading**: Large audio files may slow down processing

## Next Steps

1. **Run baseline comparison**: Video-only vs multimodal
2. **Analyze which emotions benefit most** from audio
3. **Experiment with different audio matching strategies**:
   - Random audio from emotion category
   - Best matching audio by actor code
   - Multiple audio files per trial

4. **Fine-tune prompts** for multimodal input:
   - Emphasize audio-visual consistency
   - Handle cases where audio and video disagree
   - Weight different cues appropriately

## References

- Gemini API supports multimodal input (images + audio)
- Research shows multimodal fusion improves emotion recognition by 3-8%
- Audio cues are particularly helpful for prosody-based emotions
