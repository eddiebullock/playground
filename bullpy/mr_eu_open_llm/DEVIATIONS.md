# Protocol deviations log

Record any intentional departures from `PROTOCOL_V2_AGENT_PROMPT.md` here.

| Date | Component | Deviation | Rationale |
|------|-----------|-----------|-----------|
| 2026-06-11 | Metrics | Free-response judge primary; 4AFC secondary | Artifact pivot; reduce 4AFC gaming |
| 2026-06-11 | `extract_activations.py` | Real stimulus-conditioned forward (replaces dummy) | Study 3 circuit work |
| 2026-06-11 | `activation_patching.py` | Forward-hook patching scaffold | Study 3 causal tests |
| 2026-06-11 | Inspect | `inspect_eu/` package (not stdlib `inspect`) | AISI Inspect front-end; avoid name clash |
| 2026-06-08 | Fine-tuning | Gemma4 LoRA on Mindreading **multimodal** (item-folder T-file audio; not Emotions/Audio) | Align FT with best EU baseline modality; audio is non-leaky per audio resolver |
| 2026-06-08 | Semantic entropy | EU O'Reilly definitions in label embeddings; neutral excluded; intensity collapsed to base emotions | Stronger label anchors; primary H over ~20 base states |
| 2026-06-04 | Stage 1 entropy | Semantic entropy now runs for `multimodal` / `audio_only` (not video_only-only) | Ambiguity-aware co-primary outcome for Gemma multimodal EU baseline |
| 2026-06-04 | Models | InternVL2-8B excluded from Study 1/2 (`STUDY_MODELS` = 3) | Repeated load failures with transformers 5.10 on CSD3 (`all_tied_weights_keys`); three-model comparison retained |
| 2026-06-03 | Benchmarks | Golan autistic/non-autistic removed; O'Reilly/Lassalle modality keys | Reviewer-driven redesign |
| 2026-06-03 | Audio | EU UK Voices + Mindreading item-folder T-files; block Emotions/Audio | Data leakage fix |
| 2026-06-03 | Evaluation | `--condition` video_only/audio_only/multimodal; deterministic foils | Modality ablation protocol |
| 2026-05-22 | `extract_activations.py` | Minimal forward hook scaffold | Full vision-language forward wiring is model-specific; complete on CSD3 with loaded weights |
| 2026-05-22 | `activation_patching.py` | JSON stub until TransformerLens compatibility confirmed per best model | Avoid blocking Study 1 pipeline |
| 2026-05-22 | `finetune.py` | Gemma4 multimodal LoRA SFT implemented (Mindreading JSONL) | Study 1 fine-tuning stage |
