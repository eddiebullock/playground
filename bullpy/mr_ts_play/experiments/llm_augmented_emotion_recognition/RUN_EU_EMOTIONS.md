# Running EU-Emotion Experiment

## Quick Answer

The script **already supports EU emotions**! Just change the `--dataset` argument.

## Command

```bash
python experiments/llm_augmented_emotion_recognition/scripts/run_llm_augmented_experiment.py \
    --config experiments/llm_augmented_emotion_recognition/configs/llm_config.yaml \
    --dataset eu_emotion \
    --fusion_method weighted_average \
    --clip_weight 0.7 \
    --use_cache \
    --device cpu
```

## What Changes

- **Dataset**: `cam` → `eu_emotion`
- **Data root**: Uses `eu_emotion_data_root` from config
- **Trial definitions**: Uses `eu_emotion_test_trials` from config
- **CLIP model**: Uses `eu_emotion_clip_model_path` from config
- **Output directory**: `results/llm_augmented_eu_emotion_weighted_average/`

## Config Check

Make sure your `llm_config.yaml` has:

```yaml
data:
  eu_emotion_data_root: "/path/to/EU_emotions"
  eu_emotion_test_trials: "data/trial_definitions/eu_emotion_test.json"

models:
  eu_emotion_clip_model_path: "/path/to/eu_emotion_finetuned_best"
```

## Expected Results

Based on previous EU-Emotion results:
- CLIP-only: ~55.6%
- LLM-only: ~37-47% (depending on method)
- LLM-augmented: ~46-65% (depending on fusion)

## Notes

- Uses same cache version (1.1) as CAM experiment
- Will create new cache entries for EU-Emotion videos
- Cost: ~$0.01-0.02 per experiment run


