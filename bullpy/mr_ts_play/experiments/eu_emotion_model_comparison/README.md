# EU-Emotion Model Comparison Experiment

Comprehensive evaluation of multiple vision and LLM models on the EU-Emotion dataset for emotion recognition.

## Overview

This experiment compares the performance of various models on the EU-Emotion dataset:
- **Vision Models**: I3D, TimeSformer, ResNet, ViT, EfficientNet, CLIP, FER2013, InstructBLIP
- **LLM Models**: GPT-4o-mini, GPT-4o (with cost tracking)

## Dataset

- **Dataset**: EU-Emotion Stimulus Set
- **Emotions**: 27 emotion classes
- **Test Trials**: 54 trials (forced-choice format with 4 candidate labels)
- **Data Location**: `/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions`

## Quick Start

### 1. Estimate Costs (Optional)

Before running LLM models, estimate costs:

```bash
python experiments/eu_emotion_model_comparison/scripts/estimate_costs.py \
    --config experiments/eu_emotion_model_comparison/configs/comparison_config.yaml
```

### 2. Run Experiment

Evaluate all models:

```bash
python experiments/eu_emotion_model_comparison/scripts/run_comparison.py \
    --config experiments/eu_emotion_model_comparison/configs/comparison_config.yaml \
    --device auto
```

Evaluate specific models:

```bash
python experiments/eu_emotion_model_comparison/scripts/run_comparison.py \
    --config experiments/eu_emotion_model_comparison/configs/comparison_config.yaml \
    --models clip_finetuned gpt-4o-mini fer2013_vit \
    --device auto
```

## Configuration

Edit `configs/comparison_config.yaml` to:
- Adjust which models to evaluate
- Change video processing settings
- Configure LLM settings (vision detail, cost limits)
- Set output directory

## Models

### Vision Models

1. **I3D** - Inflated 3D ConvNet for video action recognition
2. **TimeSformer** - Transformer-based video model
3. **X3D** - Efficient video model variant
4. **ResNet50/ResNet101** - Standard CNN baselines
5. **ViT** - Vision Transformer
6. **EfficientNet** - Efficient CNN architecture
7. **CLIP-Finetuned** - Fine-tuned CLIP model (uses existing model at `models/eu_emotion_finetuned_best/`)
8. **FER2013** - Emotion recognition model trained on FER2013
9. **InstructBLIP** - Vision-language model for emotion recognition

### LLM Models

1. **GPT-4o-mini** - Cost-effective vision model (~$0.02 per experiment)
2. **GPT-4o** - Higher quality vision model (~$0.25 per experiment)
3. **GPT-4-turbo** - Legacy model (expensive, not recommended)

**Note**: Most vision models (except CLIP and FER2013) need fine-tuning for emotion recognition. They currently return uniform scores as placeholders.

## Output Structure

Results are saved to `results/eu_emotion_model_comparison/`:

```
results/eu_emotion_model_comparison/
├── overall_results.csv              # Summary table of all models
├── comparison_report.md              # Human-readable summary
├── cost_breakdown.json              # LLM API costs
├── clip_finetuned/
│   ├── predictions.json
│   ├── metrics.json
│   ├── per_emotion_results.csv
│   ├── confusion_matrix.png
│   └── confusion_matrix.csv
├── gpt_4o_mini/
│   └── ... (same structure)
└── ... (one directory per model)
```

## Cost Tracking

LLM models track costs automatically:
- Total cost in USD
- Token usage (input/output)
- Number of API calls
- Remaining budget

Costs are saved to `cost_breakdown.json` and displayed in the comparison report.

## Per-Emotion Analysis

Each model generates:
- **Per-emotion accuracy**: Accuracy for each of the 27 emotions
- **Per-emotion precision/recall/F1**: Detailed metrics per emotion
- **Confusion matrix**: Shows which emotions are confused
- **Most confused emotions**: Identifies common errors

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Additional dependencies may be needed for specific models:
- `pytorchvideo` - For I3D and X3D
- `timm` - For EfficientNet variants
- `openai` - For LLM models (already in requirements.txt)

## Troubleshooting

### Model Fails to Load

If a model fails to load, use `--skip-failed` to continue with other models:

```bash
python scripts/run_comparison.py --config configs/comparison_config.yaml --skip-failed
```

### API Key Not Found

Set `OPENAI_API_KEY` in environment or `.env` file:
- Location: `experiments/cam_human_like/training/.env`
- Format: `OPENAI_API_KEY=your_key_here` (no quotes, no spaces)

### Cost Exceeds Budget

LLM models will stop if cost limit is reached. Adjust `cost_limit_usd` in config or use `--skip-failed` to skip expensive models.

### CUDA Out of Memory

Use `--device cpu` to run on CPU instead of GPU.

## Notes

- Most vision models need fine-tuning for emotion recognition (currently return uniform scores)
- CLIP and FER2013 models are ready to use
- LLM models use direct classification (choose from candidate labels)
- All API responses are cached to avoid repeat costs
- Results are saved incrementally (can resume interrupted runs)

## Future Improvements

- Fine-tune vision models on emotion data
- Add more emotion-specific models (AffectNet, EmoVLM-KD)
- Implement temporal pooling strategies
- Add ensemble methods
- Support for additional LLM providers (Anthropic, Google)
