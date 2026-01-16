#!/bin/bash
# Test adaptive fusion on FULL test set for fair comparison

python experiments/llm_augmented_emotion_recognition/scripts/test_adaptive_fusion.py \
    --clip_predictions results/eu_emotion_model_comparison/clip_finetuned/predictions.json \
    --llm_predictions results/llm_only_eu_emotion_google/results.json \
    --val_trials data/trial_definitions/eu_emotion_val_for_fusion.json \
    --test_trials data/trial_definitions/eu_emotion_test.json \
    --normalize
