# EU-Emotion Audio Model Comparison Experiment

Comprehensive evaluation of multiple audio models on the EU-Emotion audio dataset for emotion recognition.

## Overview

This experiment replicates the vision-based EU emotion recognition experiment but uses **audio files** instead of video files. It compares the performance of various audio models on the EU-Emotion audio dataset:

- **Audio Models**: Wav2Vec2 (base/large), Whisper (tiny/base/small), HuBERT
- **Dataset**: EU-Emotion Audio Stimulus Set (UK Voices)
- **Emotions**: 27 emotion classes
- **Format**: Forced-choice trials (4 candidate labels per trial)

## Dataset

- **Dataset**: EU-Emotion Audio Stimulus Set
- **Location**: `/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions/EU Emotion - UK Voices/Fixed - amplified volume`
- **Format**: `.mp3` audio files organized by emotion folders
- **Emotions**: 27 emotion classes (same as vision experiment)

## Quick Start

### 1. Generate Audio Trial Definitions

First, create the trial definitions from audio files:

```bash
python experiments/eu_emotion_audio_model_comparison/scripts/create_audio_trials.py \
    --audio-dir "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions/EU Emotion - UK Voices/Fixed - amplified volume" \
    --output-dir data/trial_definitions \
    --trials-per-emotion 10 \
    --train-ratio 0.8 \
    --seed 42
```

This will create:
- `data/trial_definitions/eu_emotion_audio_train.json`
- `data/trial_definitions/eu_emotion_audio_test.json`
- `data/trial_definitions/eu_emotion_audio_all.json`

### 2. Run Experiment

Evaluate all audio models:

```bash
python experiments/eu_emotion_audio_model_comparison/scripts/run_audio_comparison.py \
    --config experiments/eu_emotion_audio_model_comparison/configs/audio_comparison_config.yaml \
    --device auto
```

Evaluate specific models:

```bash
python experiments/eu_emotion_audio_model_comparison/scripts/run_audio_comparison.py \
    --config experiments/eu_emotion_audio_model_comparison/configs/audio_comparison_config.yaml \
    --models wav2vec2_base whisper_base \
    --device auto
```

## Configuration

Edit `configs/audio_comparison_config.yaml` to:
- Adjust which models to evaluate
- Change audio processing settings (sample rate, duration limits)
- Set output directory
- Configure model-specific parameters

## Models

### Audio Models

1. **Wav2Vec2-base** - Self-supervised audio model (facebook/wav2vec2-base)
2. **Wav2Vec2-large** - Larger variant (facebook/wav2vec2-large)
3. **Whisper-tiny** - Smallest Whisper model (openai/whisper-tiny)
4. **Whisper-base** - Base Whisper model (openai/whisper-base)
5. **Whisper-small** - Small Whisper model (openai/whisper-small)
6. **HuBERT-base** - HuBERT model (facebook/hubert-base-ls960)

### How Audio Models Work

All audio models follow a similar approach to CLIP:

1. **Extract audio embeddings**: Process audio file to get feature embeddings
2. **Get text embeddings**: Convert emotion labels to text embeddings (e.g., "a person expressing happy emotion")
3. **Compute similarity**: Use cosine similarity between audio and text embeddings
4. **Score emotions**: Normalize scores to get probability distribution over candidate labels

This allows zero-shot emotion recognition without fine-tuning on the emotion dataset.

## Differences from Vision Experiment

| Aspect | Vision Experiment | Audio Experiment |
|--------|------------------|------------------|
| **Input** | Video files (`.mp4`, `.mov`) | Audio files (`.mp3`) |
| **Models** | CLIP, ResNet, ViT, etc. | Wav2Vec2, Whisper, HuBERT |
| **Processing** | Frame extraction | Audio waveform processing |
| **Trial Definitions** | `eu_emotion_*_test.json` | `eu_emotion_audio_*_test.json` |
| **Output Directory** | `results/eu_emotion_model_comparison/` | `results/eu_emotion_audio_model_comparison/` |

## Dependencies

Required packages:

```bash
pip install torch torchaudio transformers librosa pandas numpy tqdm pyyaml matplotlib seaborn
```

Or install from requirements if available:

```bash
pip install -r requirements.txt
```

## Output

Results are saved to `results/eu_emotion_audio_model_comparison/`:

- **Overall results**: `overall_results.csv` and `comparison_report.md`
- **Per-model directories**: Each model has its own directory with:
  - `predictions.json` - All predictions with scores
  - `metrics.json` - Comprehensive metrics
  - `per_emotion_results.csv` - Per-emotion accuracy, precision, recall, F1
  - `confusion_matrix.png` - Visual confusion matrix
  - `confusion_matrix.csv` - Confusion matrix data

## Example Usage

### Generate trials and run experiment:

```bash
# Step 1: Generate trial definitions
python experiments/eu_emotion_audio_model_comparison/scripts/create_audio_trials.py \
    --audio-dir "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions/EU Emotion - UK Voices/Fixed - amplified volume" \
    --output-dir data/trial_definitions

# Step 2: Run experiment
python experiments/eu_emotion_audio_model_comparison/scripts/run_audio_comparison.py \
    --config experiments/eu_emotion_audio_model_comparison/configs/audio_comparison_config.yaml \
    --models wav2vec2_base whisper_base \
    --device auto
```

### Use custom trial definitions:

```bash
python experiments/eu_emotion_audio_model_comparison/scripts/run_audio_comparison.py \
    --config experiments/eu_emotion_audio_model_comparison/configs/audio_comparison_config.yaml \
    --trial-definitions data/trial_definitions/eu_emotion_audio_test.json \
    --models wav2vec2_base
```

## Troubleshooting

### Audio file loading errors

- Ensure `torchaudio` or `librosa` is installed
- Check that audio files are valid and not corrupted
- Verify file paths in trial definitions are correct

### Model loading errors

- Ensure `transformers` library is installed
- Check internet connection for downloading models (first time only)
- Verify model names in config are correct

### Memory issues

- Use smaller models (e.g., `whisper-tiny` instead of `whisper-base`)
- Process fewer trials at once
- Use CPU if GPU memory is limited

## Comparison with Vision Results

To compare audio vs vision performance:

1. Run both experiments with the same trial structure
2. Compare `overall_results.csv` from both experiments
3. Analyze per-emotion differences in `per_emotion_results.csv`

## Notes

- This experiment is **completely separate** from the vision experiment
- No existing code was modified
- All audio experiment code is in `experiments/eu_emotion_audio_model_comparison/`
- Results are saved to separate output directories
