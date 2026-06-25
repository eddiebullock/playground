# Manuscript / methods alignment (Protocol v2 modality ablation)

## Removed

- All autistic vs non-autistic benchmark comparisons (Golan et al., 2006 CAM framing).
- Mindreading audio sourced from `MindReading/Emotions/Audio/` (spoken-label leakage).
- Claims that models "met" or "exceeded" an autistic benchmark.

## Human benchmarks (EU-Emotions only)

Modality-matched validation literature:

| Condition   | Citation                                      | accuracy / n |
|-------------|-----------------------------------------------|--------------|
| video_only  | O'Reilly et al. (EU-Emotion Stimulus Set)    | TODO         |
| audio_only  | Lassalle et al. (EU-Emotion Voice Database)   | TODO         |
| multimodal  | O'Reilly et al. (EU-Emotion Stimulus Set)    | TODO         |

Mindreading: no direct human benchmark comparisons in this study.

## Secondary research questions

- Basic vs complex mental states (per-state accuracy; Neutral excluded).
- High vs low intensity EU-Emotions variants (mean high − low accuracy per model).

## Modality ablation methods

Each trial uses a two-stage protocol:

1. **Stage 1 (free text + semantic entropy):** runs under all conditions; prompts and stimuli match the modality (video, audio, or both). Vision-only models use `video_only`; Gemma 4 multimodal uses face video + UK Voices in stage 1 and stage 2.
2. **Stage 2 (4AFC):** condition-aware prompts for `video_only`, `audio_only`, or `multimodal`.

**EU-Emotions:** face video at 1 fps (max 16 frames); multimodal pairs face clips with UK Voices by normalized emotion label; audio-only uses UK Voices for the same 118 trials.

**Mindreading:** V-marker face videos; audio from item-folder T-files (7-digit prefix + tail match), extracted to mono 16 kHz WAV via ffmpeg. `/Emotions/Audio/` is blocked.

**4AFC foils:** deterministic at runtime (`sha256(trial_id|seed)`); exact Golan item-level foils unavailable.

## Spoken-label confound

Where audio-inclusive conditions approach ceiling accuracy, interpret cautiously as potential spoken-label confounds rather than improved theory-of-mind.

## Fine-tuning

LoRA on Mindreading for **Gemma 4** uses **multimodal** face video + item-folder T-file audio (`config.FINETUNE_MODALITY_BY_MODEL["gemma4"]`). `/Emotions/Audio/` remains blocked. Post-FT evaluation re-runs EU-Emotions **multimodal** (118 trials) to test transfer and forgetting.

## Semantic entropy (label embeddings)

EU-Emotions validation-set definitions (O'Reilly et al.) embed as `data/eu_emotion_definitions.json`. Primary entropy excludes **neutral** and collapses intensity variants to base emotions.

## Open VLM audio capability audit (2026-06)

| Model        | Native audio in eval pipeline |
|--------------|-------------------------------|
| Qwen2-VL     | No (vision only)              |
| LLaVA-NeXT   | No                            |
| Gemma 4      | Yes (E4B-it native audio, ~30s max) |

InternVL2-8B was excluded from the study (see `DEVIATIONS.md`).

Audio ablations run for Gemma 4 only until other models gain audio support in `model_inference.py`.
