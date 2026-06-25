# EU-Emotions artifact master table

## Baselines (118 trials)

| Model / condition | Strict 4AFC | Tolerant 4AFC | Free-response judge | ECE | AUROC | Mean H_sem | n |
|-------------------|-------------|---------------|---------------------|-----|-------|------------|---|
| gemma4_multimodal | 43.2% | 45.8% | 22.9% | 0.347 | 0.499 | 2.503 | 118 |
| llavanext_video_only | 25.4% | 27.1% | 5.9% | 0.242 | 0.507 | 2.859 | 118 |
| qwen2vl_video_only | 41.5% | 43.2% | 16.9% | 0.339 | 0.583 | 2.597 | 118 |

## Frame policy ablation

| Model / condition | Strict 4AFC | Tolerant 4AFC | Free-response judge | ECE | AUROC | Mean H_sem | n |
|-------------------|-------------|---------------|---------------------|-----|-------|------------|---|
| gemma4_multimodal | 36.7% | 40.0% | 16.7% | 0.328 | 0.474 | 2.489 | 30 |
| gemma4_multimodal/native_video | 26.7% | 26.7% | 10.0% | 0.208 | 0.545 | 2.695 | 30 |

## Fine-tuning (supplementary; not headline)

| Model / condition | Strict 4AFC | Tolerant 4AFC | Free-response judge | ECE | AUROC | Mean H_sem | n |
|-------------------|-------------|---------------|---------------------|-----|-------|------------|---|
| gemma4_multimodal/finetuned | 1.7% | 1.7% | — | — | — | — | 118 |
| llavanext_video_only/finetuned | 25.4% | 27.1% | 7.6% | 0.240 | 0.501 | 2.858 | 118 |
| qwen2vl_video_only/finetuned | 44.9% | 46.6% | 8.5% | 0.370 | 0.555 | 2.692 | 118 |

_Human EU-Emotions benchmarks often report 6AFC group accuracy; this harness uses deterministic 4AFC foils (protocol v2). Do not headline strict model-vs-human superiority without paradigm alignment._
