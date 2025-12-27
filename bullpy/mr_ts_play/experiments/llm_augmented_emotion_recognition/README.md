# LLM-Augmented Emotion Recognition Experiment

This experiment combines CLIP vision features with LLM semantic embeddings to improve emotion recognition performance through multimodal fusion.

## Overview

The experiment runs three conditions:
1. **CLIP-only**: Uses trained CLIP model only (baseline)
2. **LLM-only**: Uses vision-language model (GPT-4o/GPT-4o-mini) to describe videos, then compares descriptions to emotion labels
3. **LLM-augmented**: Combines CLIP + LLM using configured fusion method (hybrid approach)

**Important**: The LLM now **actually processes video frames** using vision models, not just emotion labels. This provides a scientifically valid comparison.

## Project Structure

```
experiments/llm_augmented_emotion_recognition/
├── __init__.py
├── README.md
├── models/
│   ├── __init__.py
│   ├── llm_wrapper.py              # LLM API integration and caching
│   ├── llm_augmented_wrapper.py    # Combines CLIP + LLM features
│   └── clip_model_loader.py        # Loads trained CLIP models
├── evaluation/
│   ├── __init__.py
│   ├── three_way_comparison.py     # Runs all three conditions
│   └── metrics.py                  # Evaluation metrics
├── data/
│   └── llm_cache/                  # Cached LLM responses
├── scripts/
│   ├── run_llm_augmented_experiment.py  # Main experiment runner
│   └── generate_llm_cache.py      # Pre-generate all LLM responses
└── configs/
    └── llm_config.yaml             # Configuration file
```

## Setup

### 1. Install Dependencies

Dependencies are already in `requirements.txt`. Install:
```bash
pip install openai python-dotenv pyyaml
```

**Note**: The experiment uses vision models (GPT-4o/GPT-4o-mini) for video processing. See `COST_BREAKDOWN.md` for cost estimates (~$0.02 per experiment run with caching).

### 2. Configure API Keys

Create/update `.env` file at:
```
experiments/cam_human_like/training/.env
```

Format (no spaces around `=`, no quotes):
```
OPENAI_API_KEY=your_openai_api_key_here
```

**Important .env file rules:**
- No spaces around the `=` sign
- No quotes around values (unless the value itself contains spaces)
- One key-value pair per line
- Lines starting with `#` are comments

### 3. Configure Paths and Models

Edit `configs/llm_config.yaml` to set:
- Data root paths (CAM and EU-Emotion)
- Model paths (trained CLIP models)
- Trial definition paths
- Fusion method and weights
- **Vision model**: Choose `gpt-4o-mini` (cost-effective, ~$0.02) or `gpt-4o` (higher quality, ~$0.25)
- **Vision detail**: `low` (recommended, cheaper) or `high` (better quality, 5x cost)

See `COST_BREAKDOWN.md` for detailed cost analysis.

## Usage

### Step 1: Pre-generate LLM Cache (Recommended)

Before running experiments, pre-generate all LLM embeddings to ensure reproducibility:

```bash
python scripts/generate_llm_cache.py \
    --trial_definitions \
        data/trial_definitions/cam_test.json \
        data/trial_definitions/eu_emotion_test.json \
    --provider openai \
    --model text-embedding-3-small \
    --cache_dir data/llm_cache
```

This will cache all emotion embeddings, making experiments run without API calls.

### Step 2: Run Experiment

Run the three-way comparison:

```bash
python scripts/run_llm_augmented_experiment.py \
    --config configs/llm_config.yaml \
    --dataset cam \
    --fusion_method weighted_average \
    --clip_weight 0.7 \
    --use_cache \
    --device cpu
```

Command-line arguments:
- `--dataset`: Dataset type (`cam` or `eu_emotion`)
- `--fusion_method`: Fusion method (`weighted_average` or `attention`)
- `--clip_weight`: Weight for CLIP scores (0.0 to 1.0)
- `--use_cache`: Use cached LLM responses
- `--device`: Device to run on (`cpu` or `cuda`)
- `--num_frames`: Number of frames per video (default: 8)

## Fusion Methods

### 1. Weighted Average (Primary Method)

Simple weighted combination:
```
score = α * clip_score + (1-α) * llm_score
```

**Advantages:**
- Simple and interpretable
- No training needed
- Most common in evaluation experiments

**Reference:** Atrey et al. (2010) "Multimodal fusion for multimedia analysis: a survey"

**Usage:**
```yaml
fusion:
  method: "weighted_average"
  clip_weight: 0.7
  llm_weight: 0.3
```

### 2. Attention-Based Fusion (Secondary Method)

Learn attention weights that adapt to input:
```
w = softmax([clip_feature, llm_feature] @ W)
score = w[0] * clip_score + w[1] * llm_score
```

**Advantages:**
- Adapts to input
- Learns which modality to trust
- More sophisticated

**Reference:** Zadeh et al. (2017) "Multimodal Language Analysis in the Wild"

**Usage:**
```yaml
fusion:
  method: "attention"
  attention_dim: 128
```

## Expected Outputs

After running experiments, results are saved to:
```
results/
├── clip_only/
│   ├── predictions.json
│   ├── metrics.json
│   ├── confusion_matrix.csv
│   ├── per_emotion_accuracy.csv
│   └── summary.txt
├── llm_only/
│   └── ... (same structure)
├── llm_augmented_weighted/
│   └── ... (same structure)
└── comparison_report.json
└── comparison_report.md
```

## Data Paths

### CAM Data
- Location: `/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/mindreading_transporter_files/Mindreading emotions library/Emotions`
- Trial definitions: `data/trial_definitions/cam_test.json`

### EU-Emotion Data
- Root: `/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions`
- Structure: `emotions*/HD Version - Face, Body, Social/Faces - HD Version/EDITED/` or `Original/`
- Trial definitions: `data/trial_definitions/eu_emotion_test.json`

## Trained CLIP Models

- CAM model: `models/cam_finetuned_best/`
- EU-Emotion model: `models/eu_emotion_finetuned_best/`

## Reproducibility

All LLM API calls are cached to JSON files in `data/llm_cache/`. Once cached:
- Experiments run without API calls
- Results are fully reproducible
- Cache files are version-controlled (with `.gitkeep`)

Cache version can be set in config for invalidation if needed.

## Error Handling

The code handles:
- Missing API keys (checks `.env` file)
- Network errors (exponential backoff retries)
- Missing cached responses (fallback to API, then save)
- Missing model files (clear error messages)
- Corrupted video files (skips with error message)

## Literature References

- **Weighted Average**: Atrey et al. (2010) "Multimodal fusion for multimedia analysis: a survey"
- **Attention Fusion**: Zadeh et al. (2017) "Multimodal Language Analysis in the Wild"
- **Emotion Recognition**: Poria et al. (2017) "Context-Dependent Sentiment Analysis in User-Generated Videos"

## Future Extensions

- Support for Anthropic and Google LLM providers
- Video description generation for LLM-only baseline
- Concatenation + MLP fusion (requires training)
- Per-trial fusion weight learning

## Troubleshooting

### API Key Not Found
- Check `.env` file exists at `experiments/cam_human_like/training/.env`
- Verify `OPENAI_API_KEY` is set correctly (no quotes, no spaces around `=`)
- Check file permissions

### Model Not Found
- Verify model paths in `configs/llm_config.yaml`
- Ensure models have been trained and saved
- Check path permissions

### Video Loading Errors
- Verify data root paths in config
- Check video file permissions
- Ensure video files are not corrupted (minimum 50KB)

### Cache Issues
- Clear cache directory to regenerate: `rm -rf data/llm_cache/*.json`
- Update `cache_version` in config to invalidate old cache

