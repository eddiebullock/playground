# Multi-Provider LLM Setup Guide

## Overview

The LLM-augmented emotion recognition experiment now supports three providers:
- **OpenAI** (GPT-4o, GPT-4o-mini)
- **Anthropic** (Claude 3.5 Sonnet, Claude 3 Opus)
- **Google** (Gemini 1.5 Pro, Gemini 1.5 Flash)

## Setup

### 1. Install Dependencies

```bash
pip install anthropic google-generativeai
```

Or update your requirements:
```bash
pip install -r requirements.txt
```

### 2. Set API Keys

Add your API keys to the `.env` file at:
```
experiments/cam_human_like/training/.env
```

Required keys:
```bash
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_API_KEY=your_google_key_here
# OR
GEMINI_API_KEY=your_google_key_here  # Alternative name
```

**Note**: You only need the keys for the providers you want to use.

### 3. Configure Provider

Edit `configs/llm_config.yaml`:

```yaml
llm:
  provider: "openai"  # Options: "openai", "anthropic", "google"
  
  # OpenAI configuration
  openai:
    vision_model: "gpt-4o-mini"  # or "gpt-4o"
    model: "gpt-4o-mini"
    api_key_env: "OPENAI_API_KEY"
  
  # Anthropic Claude configuration
  anthropic:
    vision_model: "claude-3-5-sonnet-20241022"
    model: "claude-3-5-sonnet-20241022"
    api_key_env: "ANTHROPIC_API_KEY"
  
  # Google Gemini configuration
  google:
    vision_model: "gemini-1.5-pro"  # or "gemini-1.5-flash"
    model: "gemini-1.5-pro"
    api_key_env: "GOOGLE_API_KEY"
  
  # Common settings
  cache_dir: "data/llm_cache"
  use_cache: true
  cache_version: "1.2"
  vision_detail: "low"  # OpenAI only
  max_frames_per_video: 4
```

## Usage

### Running Experiments

The experiment automatically uses the provider specified in the config:

```bash
python experiments/llm_augmented_emotion_recognition/scripts/run_llm_augmented_experiment.py \
  --config experiments/llm_augmented_emotion_recognition/configs/llm_config.yaml \
  --dataset eu_emotion
```

### Testing Different Providers

To test different providers, simply change the `provider` field in the config:

```yaml
llm:
  provider: "anthropic"  # Change from "openai" to "anthropic" or "google"
```

### Available Models

#### OpenAI
- `gpt-4o` - Best quality, higher cost
- `gpt-4o-mini` - Good quality, lower cost (recommended)
- `gpt-4-turbo` - Legacy

#### Anthropic
- `claude-3-5-sonnet-20241022` - Best quality (recommended)
- `claude-3-opus-20240229` - Highest quality, highest cost
- `claude-3-5-haiku-20241022` - Fastest, lower cost

#### Google
- `gemini-1.5-pro` - Best quality (recommended)
- `gemini-1.5-flash` - Faster, lower cost
- `gemini-pro` - Legacy

## Cost Comparison

Approximate costs per 1000 images (4 frames per video, 54 videos = 216 images):

| Provider | Model | Cost per 1K images |
|----------|-------|-------------------|
| OpenAI | gpt-4o-mini | ~$0.10 |
| OpenAI | gpt-4o | ~$1.50 |
| Anthropic | claude-3-5-sonnet | ~$0.50 |
| Anthropic | claude-3-opus | ~$2.00 |
| Google | gemini-1.5-pro | ~$0.15 |
| Google | gemini-1.5-flash | ~$0.05 |

**Note**: Costs are approximate and may vary. Caching makes subsequent runs essentially free.

## Caching

All providers use the same caching system. Results are cached by:
- Provider name
- Model name
- Video path hash
- Cache version

Cache files are stored in: `data/llm_cache/`

## Troubleshooting

### Missing API Key Error

```
ValueError: ANTHROPIC_API_KEY not found in environment variables
```

**Solution**: Add the API key to your `.env` file.

### Import Error

```
ImportError: Anthropic package not installed
```

**Solution**: Install the package:
```bash
pip install anthropic google-generativeai
```

### Model Not Found Error

```
ValueError: Model xyz does not support vision
```

**Solution**: Check that you're using a vision-capable model name. See "Available Models" above.

## Example: Testing All Three Providers

1. **Test OpenAI**:
   ```yaml
   provider: "openai"
   ```
   ```bash
   python experiments/llm_augmented_emotion_recognition/scripts/run_llm_augmented_experiment.py --dataset eu_emotion
   ```

2. **Test Anthropic**:
   ```yaml
   provider: "anthropic"
   ```
   ```bash
   python experiments/llm_augmented_emotion_recognition/scripts/run_llm_augmented_experiment.py --dataset eu_emotion
   ```

3. **Test Google**:
   ```yaml
   provider: "google"
   ```
   ```bash
   python experiments/llm_augmented_emotion_recognition/scripts/run_llm_augmented_experiment.py --dataset eu_emotion
   ```

## Notes

- All providers use the same caching system, so switching providers won't duplicate cache
- The `classify_emotion_directly` method works with all providers
- Provider-specific settings (like `vision_detail` for OpenAI) are ignored for other providers
- Error handling and retry logic work the same across all providers
