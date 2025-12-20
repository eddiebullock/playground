# CAM Face-Voice Battery Computational Replication

This directory contains the computational replication of the Cambridge Mindreading (CAM) Face-Voice Battery (Golan et al., 2006), replacing human participants with computer-vision and multimodal models.

## Overview

The CAM Face-Voice Battery is a standardized test for recognizing mental states from facial expressions and vocal intonations. This replication preserves the original experimental structure:

- **Stimuli**: Video clips with audio (face and voice modalities)
- **Trial structure**: One stimulus + four candidate mental-state labels + one correct answer
- **Decision procedure**: 4-option forced-choice (model must select from the four options)
- **Evaluation**: Actor-independent train/val/test splits
- **Metrics**: Overall accuracy, per-emotion accuracy, per-concept accuracy, confusion matrices

## Methodology Mapping

### Original CAM (Human Participants)
- Participants view/hear stimulus (video with audio)
- Four numbered adjectives (1-4) are presented
- Participants press 1, 2, 3, or 4 to select answer
- No feedback given
- Response time unrestricted

### Computational Replication (Models)
- Model processes stimulus (video frames + audio waveform)
- Model scores all four candidate labels
- Decision restricted to four trial options (forced-choice)
- Highest-scoring option selected
- No training on CAM stimuli (zero-shot evaluation)

## Directory Structure

```
cam_human_like/
├── dataset.py              # CAM trial loading and actor-independent splits
├── models/                 # Pretrained model wrappers
│   ├── base.py            # Base model interface
│   ├── clip_wrapper.py    # CLIP-style vision-language models
│   ├── audio_wrapper.py   # Audio-only models (Wav2Vec2, etc.)
│   └── multimodal_wrapper.py  # Combined vision + audio
├── trials/                 # Forced-choice trial logic
│   └── forced_choice.py   # 4-option forced-choice decision procedure
├── evaluation/             # Evaluation metrics
│   └── metrics.py         # Accuracy, confusion matrices, concept recognition
├── run_experiment.py       # Main experiment script
└── README.md              # This file
```

## Usage

### 1. Generate Trial Definitions

The CAM experiment requires trial definitions with proper foil selection following CAM methodology.

**Generate trial definitions using the taxonomy-based script:**

```bash
python experiments/cam_human_like/create_trial_definitions.py \
    --data-root "/path/to/cam/stimuli" \
    --output data/cam_trial_definitions.json \
    --seed 42 \
    --trials-per-concept 5 \
    --validate
```

This script will:
- Discover all stimuli grouped by emotion concept
- Generate 5 trials per concept with counterbalanced face/voice distribution (3+2 or 2+3)
- Select foils using CAM taxonomy (different emotion groups, appropriate difficulty)
- Validate trials against CAM rules
- Save trial definitions in JSON format

**Trial definitions format:**

```json
{
  "trials": [
    {
      "trial_id": "trial_001",
      "stimulus_path": "01/0100104/0100104M1Vhumiliating.mov",
      "modality": "face",
      "correct_label": "humiliating",
      "candidate_labels": ["humiliating", "embarrassed", "ashamed", "proud"],
      "correct_idx": 0,
      "actor": "M",
      "scenario_id": "0100104",
      "concept": "humiliating"
    }
  ]
}
```

### 2. Configure Experiment

Edit `configs/cam_config.yaml`:

```yaml
data:
  root: "/path/to/cam/stimuli"
  splits_dir: "data/splits"
  trial_definitions_file: "path/to/trial_definitions.json"

model:
  type: "clip"  # or "audio", "multimodal"
  name: "openai/clip-vit-base-patch32"
```

### 3. Run Experiment

```bash
python experiments/cam_human_like/run_experiment.py \
    --config configs/cam_config.yaml \
    --split test
```

### 4. View Results

Results are saved to `results/cam_human_like/{model_type}_{timestamp}/`:

- `summary.json`: Overall metrics
- `trial_results.csv`: Per-trial predictions and scores
- `confusion_matrix.csv`: Confusion matrix
- `per_emotion_accuracy.csv`: Per-emotion breakdown
- `per_concept_accuracy.csv`: Per-concept breakdown
- `concept_recognition.csv`: Concept-level recognition (4/5 correct = passed)

## Model Wrappers

### CLIPWrapper
- **Use case**: Face (visual) trials
- **Models**: CLIP, OpenCLIP, etc.
- **Process**: Encodes video frames and emotion labels, computes cosine similarity

### AudioWrapper
- **Use case**: Voice (audio) trials
- **Models**: Wav2Vec2, Whisper, etc.
- **Process**: Extracts audio, encodes waveform, scores emotion labels
- **TODO**: Implement label scoring (currently placeholder)

### MultimodalWrapper
- **Use case**: Both face and voice modalities
- **Process**: Combines vision and audio encoders, fuses scores
- **Fusion methods**: Weighted average, concatenation, attention

## Evaluation Metrics

All metrics match the original CAM analysis:

1. **Overall Accuracy**: Proportion of correct trials across all trials
2. **Face Accuracy**: Accuracy on face (visual) trials only
3. **Voice Accuracy**: Accuracy on voice (audio) trials only
4. **Per-Emotion Accuracy**: Accuracy for each emotion label
5. **Per-Concept Accuracy**: Accuracy for each emotion concept
6. **Concept Recognition Rate**: Proportion of concepts passed (4/5 correct items)
7. **Confusion Matrix**: Detailed error analysis

## Experimental Constraints

- **Actor-independent splits**: No actor appears in multiple splits
- **No test data access**: Test set untouched until final evaluation
- **Zero-shot evaluation**: Models use pretrained weights only
- **Optional few-shot calibration**: Temperature scaling or Platt scaling on validation set
- **No stimulus modification**: CAM stimuli used as-is

## Implementation Status

- ✅ Dataset loading with actor-independent splits
- ✅ Model wrapper interfaces (CLIP, Audio, Multimodal)
- ✅ Forced-choice trial logic
- ✅ Evaluation metrics
- ✅ Main experiment script
- ✅ CAM taxonomy module with emotion groups and foil selection
- ✅ Trial generation with proper CAM methodology (5 trials/concept, counterbalanced)
- ✅ Trial validation functions
- ⏳ Audio label scoring (placeholder implemented)
- ⏳ Few-shot calibration (interface defined, implementation TODO)

## References

Golan, O., Baron-Cohen, S., & Hill, J. (2006). The Cambridge Mindreading (CAM) Face–Voice Battery: Testing complex emotion recognition in adults with and without Asperger Syndrome. *Journal of Autism and Developmental Disorders*, 36(2), 169-183.

