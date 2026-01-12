# Audio Model Fine-Tuning Guide

This guide explains how to fine-tune audio models (Wav2Vec2, Whisper) on the EU-Emotion audio dataset for emotion recognition.

## Why Fine-Tune?

The zero-shot audio models performed poorly (21-28% accuracy, near random chance). Fine-tuning will:
- **Learn emotion-specific features** from the audio data
- **Match the evaluation format** (4-option forced-choice)
- **Significantly improve performance** (expected 50-70% accuracy, similar to vision models)

## Quick Start

### 1. Train All Audio Models

```bash
bash experiments/eu_emotion_audio_model_comparison/training/train_all_audio_models.sh
```

This will train:
- Wav2Vec2-base
- Whisper-base
- Whisper-tiny
- Wav2Vec2-large (optional, may fail due to tokenizer issues)

### 2. Train Individual Models

```bash
# Train Wav2Vec2-base
python experiments/eu_emotion_audio_model_comparison/training/finetune_audio_models_task_specific.py \
    --model wav2vec2_base \
    --train_trials data/trial_definitions/eu_emotion_audio_train.json \
    --val_trials data/trial_definitions/eu_emotion_audio_val.json \
    --data_root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
    --output_dir models/wav2vec2_emotion_finetuned_task_specific \
    --num_epochs 20 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --device auto
```

### 3. Use Fine-Tuned Models

After training, update the config file to use fine-tuned models:

Edit `configs/audio_comparison_config.yaml`:

```yaml
model_configs:
  wav2vec2_base:
    model_name: "facebook/wav2vec2-base"
    sample_rate: 16000
    fine_tuned_path: "models/wav2vec2_emotion_finetuned_task_specific"  # Uncomment this
```

Then run evaluation:

```bash
python experiments/eu_emotion_audio_model_comparison/scripts/run_audio_comparison.py \
    --config experiments/eu_emotion_audio_model_comparison/configs/audio_comparison_config.yaml \
    --models wav2vec2_base \
    --device auto
```

## Training Details

### Task-Specific Approach

The fine-tuning uses a **task-specific** approach (same as vision models):
- Each audio file is paired with **4 candidate labels** (1 correct + 3 foils)
- Loss is **cross-entropy over the 4 options** (not all 27 emotions)
- Model learns to **distinguish between 4 options**, matching evaluation format

### Architecture

1. **Audio Encoder**: Wav2Vec2/Whisper extracts audio embeddings
2. **Projection Layer**: Maps audio embeddings → 4 emotion scores
   - Architecture: `Linear(feature_dim → 256) → ReLU → Dropout(0.1) → Linear(256 → 4)`
3. **Loss**: Cross-entropy over 4 options with class weighting

### Training Features

- **Class-weighted loss**: Handles emotion imbalance
- **Data augmentation**: Noise, volume variation (for training only)
- **Early stopping**: Stops if validation accuracy doesn't improve
- **Learning rate scheduling**: Cosine annealing with warmup

## Expected Results

After fine-tuning, you should see:
- **Wav2Vec2-base**: 50-65% accuracy (vs 28% zero-shot)
- **Whisper-base**: 45-60% accuracy (vs 21% zero-shot)
- **Whisper-tiny**: 40-55% accuracy (vs 21% zero-shot)

This is comparable to vision models:
- **CLIP-finetuned**: 53.7% accuracy
- **ResNet50-finetuned**: 33.3% accuracy

## Troubleshooting

### Out of Memory

- Reduce batch size: `--batch_size 4` or `--batch_size 2`
- Use smaller models: `whisper_tiny` instead of `whisper_base`

### Slow Training

- Audio processing is slower than vision (no GPU acceleration for audio loading)
- Consider using fewer epochs: `--num_epochs 10`
- Use CPU if MPS is slow: `--device cpu`

### Model Loading Errors

- Ensure fine-tuned model directory exists
- Check that `best_model/` subdirectory contains model files
- Verify `score_projection.pth` exists

## Comparison: Zero-Shot vs Fine-Tuned

| Model | Zero-Shot | Fine-Tuned (Expected) |
|-------|-----------|----------------------|
| Wav2Vec2-base | 28.57% | 50-65% |
| Whisper-base | 21.43% | 45-60% |
| Whisper-tiny | 21.43% | 40-55% |

Fine-tuning should **double or triple** the accuracy!
