# Project pivot: eval artifact + interpretability flagship

This document captures the repositioning from a broad "do VLMs have ToM?" benchmark toward:

1. **Workstream A** — shippable EU-Emotions evaluation artifact (NeurIPS Datasets & Benchmarks target)
2. **Workstream B** — circuit-level interpretability of mental-state inference (ICML/NeurIPS main / strong workshop)

Existing `scripts/evaluate.py`, SLURM jobs, and trial logic are **preserved**. New modules wrap and extend.

## Ordering (risk-managed)

| Priority | Item | Status |
|----------|------|--------|
| 1 | **B0** `scripts/verify_finetune_eval.py` — gating rescoring | Implemented |
| 2 | **A2** `data/eu_emotion_synonyms.json` + `scripts/free_response_judge.py` | Implemented |
| 3 | **A3** `scripts/selective_prediction.py` | Implemented |
| 4 | **A1** `inspect_eu/` Inspect task wrapper | Implemented (optional `inspect-ai` dep) |
| 5 | **B1** `scripts/activation_forward.py` + real `extract_activations.py` | Implemented |
| 6 | **B1** `scripts/activation_patching.py` hook patching scaffold | Implemented |
| 7 | **A4/A5** docs, CI, frame policy modes | Partial — see `docs/ARTIFACT.md` |
| 8 | **B2–B7** SAE, path patching, killer figure | Not started — requires GPU + peak layer |

## B0: verify fine-tune collapse (run on HPC)

```bash
python -m scripts.verify_finetune_eval \
  --eval_json results/finetune/eu_post_ft/eval_v2_eu_emotions_gemma4_multimodal_finetuned_seed42.json \
  --baseline_json results/baseline/eu_emotions/gemma4/eval_v2_eu_emotions_gemma4_multimodal_seed42.json \
  --output results/stats/b0_finetune_verification.json
```

Interpret `verdict`:
- `GENUINE_DEGRADATION` — tolerant rescore still far below chance
- `LIKELY_PARSE_ARTIFACT` — tolerant score near chance while strict very low
- `PARTIAL_FORMAT_ISSUE` — meaningful gap between strict and tolerant

## Primary vs secondary metrics

| Metric | Role |
|--------|------|
| Free-response judge accuracy (`free_response_judge.py`) | **Primary** (auditable synonym map) |
| Selective prediction / ECE (`selective_prediction.py`) | **Primary axis** (trustworthy deferral) |
| 4AFC accuracy (`parse_emotion`) | **Secondary** (human comparability) |
| Semantic entropy | Co-primary ambiguity index |

## Study 3 framing (fine-tuning demoted)

Fine-tuning is a **causal probe**: representation destroyed vs readout overwritten (probe + patching + optional SAE narrative). Not a headline "catastrophic forgetting" paper.

## Package layout

```
inspect_eu/          # Inspect AI front-end (avoids stdlib `inspect` name clash)
scripts/             # Core pipeline (unchanged entry: evaluate.py)
data/eu_emotion_synonyms.json
docs/ARTIFACT.md     # Release + human paradigm caveats
```

## Next GPU milestones

1. B0 on post-FT JSON
2. `extract_activations --max_trials 5` smoke per model
3. Baseline + finetuned activations on full 118
4. Probes + model-only RSA
5. Patching confused pairs at peak layer
