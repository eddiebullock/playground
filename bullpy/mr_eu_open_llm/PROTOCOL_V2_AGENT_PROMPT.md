# CURSOR AGENT PROMPT: Adapt mr_eu_open_llm to Studies 1 & 2 (Protocol v2)

Use this document as the **source of truth** when implementing or extending this repository. Do not create a new repo; adapt in place with clear versioning so v1 (4-frame, single-stage) results are never mixed with v2.

---

## Your role

You are implementing protocol v2 in `mr_eu_open_llm`. Match the study design below exactly unless a technical blocker forces a documented entry in `DEVIATIONS.md`.

**Constraints:**

- Preserve working model-loading paths in `scripts/evaluate.py` (Qwen2-VL, InternVL2, LLaVA-NeXT, Gemma4) unless multi-frame fairness requires changes.
- Use `SEED = 42` from `config.py` for all stochastic steps (distractors, CV folds, subsets).
- No emojis in code.
- Implement incrementally: smoke test (≤5 trials) before full 118-trial jobs.
- Replace stubs with real implementations for core Study 1/2 paths; do not leave TODOs on the critical path.

---

## Study design (embedded)

### Study 1: Benchmarking and fine-tuning

**Goal:** Evaluate four open multimodal LLMs on mental state recognition; add ambiguity-aware outcomes via semantic entropy on free responses; fine-tune only the best model on Mindreading; compare EU accuracy to modality-matched human benchmarks (O'Reilly / Lassalle). See `MANUSCRIPT_CHANGES.md`.

**Models** (`config.MODELS` keys):

| Key | HF ID |
|-----|-------|
| qwen2vl | Qwen/Qwen2-VL-7B-Instruct |
| internvl2 | OpenGVLab/InternVL2-8B |
| llavanext | llava-hf/llava-interleave-qwen-7b-hf |
| gemma4 | google/gemma-4-E4B-it |

**Frame policy (all four models — must be fair):**

- Videos: 1 frame per second of duration, cap at 16 frames, uniformly spaced in time (`scripts/frame_sampling.py`).
- Static images: single frame.
- Ablation: 30 EU-Emotions trials, **4 vs 16** frames (not 4 vs 8); `slurm_jobs/frame_ablation.sh`.
- **Critical:** Do not silently use 1 frame for 3/4 models. `scripts/multi_frame.py` uses native multi-image for Qwen2-VL and a **composite frame grid** for InternVL2, LLaVA-NeXT, and Gemma4 when `FRAME_POLICY["enforce_multi_frame"]` is true. Document in README.

**Datasets:**

- **EU-Emotions:** 27 labels, 118 trials (`data/eu_emotions_118_manifest.json`, root `data/eu_emotions_118`). Primary benchmark.
- **Mindreading:** train/val/test JSONL (`train_subset_100.jsonl`, `val_subset_50.jsonl`, etc.) via `scripts/prepare_finetune_data.py`.
- **RMET:** Deprecated for Study 1 (keep `DATASETS["rmet"]` optional; do not run in primary pipeline).

**Human benchmarks (EU-Emotions only, keyed by condition):** O'Reilly et al. (video/multimodal), Lassalle et al. (audio-only). Values in `config.HUMAN_BENCHMARKS` (TODO: fill accuracy/n from papers). Mindreading: no human benchmark. Modality ablations: `--condition video_only|audio_only|multimodal`.

#### Two-stage trial procedure

**Stage 1 — Free response (no options):**

- Prompt: `build_free_response_prompt()` in `scripts/evaluate.py`.
- No correct/incorrect scoring.
- Stimulus-level ambiguity index = semantic entropy over 27 label embeddings (`scripts/semantic_entropy.py`).

**Semantic entropy:**

- Labels: canonical sorted list from manifest (27 strings).
- Frozen embedding model: `config.EMBEDDING_MODEL` (default `sentence-transformers/all-MiniLM-L6-v2`).
- Cache: `data/cache/label_embeddings_*.npy`.
- \(p_i = \mathrm{softmax}(\mathrm{sim}(r, \ell_i) / \tau)\), \(\tau\) = `config.ENTROPY_TEMPERATURE` (default 0.1).
- \(H_{\mathrm{sem}} = -\sum_i p_i \log p_i\) with natural log (`ENTROPY_LOG_BASE = "e"`).
- Store: `stage1.free_response_text`, `semantic_entropy`, `label_probs`, `top_labels`.

**Stage 2 — 4AFC:**

- `make_4afc_options`, `build_4afc_prompt`, `parse_emotion` (existing).
- Same stimulus and frames as Stage 1.
- Default: **independent** (`CHAIN_STAGES = False`); Stage 2 does not show Stage 1 text.
- Score: prediction must match one of four options.

**Primary outcomes:** Stage 2 accuracy (Wilson CI, binomial vs 0.25, z-test vs human); Stage 1 mean/median/SD semantic entropy (co-primary).

**Fine-tuning (best model only):**

- Select via `scripts/select_best_model.py` → `results/stats/best_model.json`.
- LoRA sweep: LR \(\{10^{-4}, 5\times10^{-5}, 10^{-5}\}\) × rank \(\{8,16,32\}\), **alpha = 2×rank**, dropout 0.1, architecture-specific `q_proj`/`v_proj` in `config.LORA_TARGET_MODULES`.
- Subset: 100 train / 50 val Mindreading trials; select by validation accuracy.
- Full run: 5 epochs, batch 4, grad accum 4, cross-entropy on correct label (not 4AFC distractors during training).
- After each epoch: re-run two-stage EU-Emotions eval; flag if >5pp drop vs pre-FT baseline.

**Error analysis (feeds Study 2):**

- `scripts/error_analysis.py` → `confused_pairs_{model}.json` (top 5 pairs; prefer high entropy on best model).

---

### Study 2: Mechanistic interpretability

**Models:** All four baselines; best model baseline + fine-tuned.

**Pipeline:** `layer_map.py` → `extract_activations.py` → `probing.py` → `rsa.py` → `activation_patching.py` (or `slurm_jobs/study2_full.sh`).

- Layers at depth fractions 12.5%, 37.5%, 75% + peak probe layer.
- Probing: multinomial logistic regression, stratified 5-fold CV; high/low ambiguity tertiles from Study 1 entropy.
- RSA: Spearman between model RDM and `data/human_rdm.npy` (or `scripts/build_human_rdm.py` pending).
- Patching: top-5 confused pairs at peak layer; TransformerLens or hook fallback in `DEVIATIONS.md`.

---

## Repository map (implementation status)

| Component | Path | Status |
|-----------|------|--------|
| Config v2 | `config.py` | Implemented |
| Frame sampling | `scripts/frame_sampling.py` | Implemented |
| Semantic entropy | `scripts/semantic_entropy.py` | Implemented |
| Multi-frame fairness | `scripts/multi_frame.py` | Implemented |
| Model inference | `scripts/model_inference.py` | Implemented |
| Two-stage eval | `scripts/evaluate.py` | Implemented |
| Select best model | `scripts/select_best_model.py` | Implemented |
| Error analysis | `scripts/error_analysis.py` | Implemented |
| Fine-tune | `scripts/finetune.py` | Config + metadata; full GPU loop pending |
| Prepare MR data | `scripts/prepare_finetune_data.py` | Implemented (scan-based) |
| Layer map | `scripts/layer_map.py` | Implemented |
| Activations | `scripts/extract_activations.py` | Hook scaffold; wire full forward pass on HPC |
| Probing | `scripts/probing.py` | Implemented (single-layer file) |
| RSA | `scripts/rsa.py` | Implemented |
| Patching | `scripts/activation_patching.py` | Stub JSON; implement on HPC |
| Statistics | `scripts/statistics.py` | + Bonferroni |
| SLURM baselines | `slurm_jobs/baseline_eval.sh` | v2 flags |
| SLURM smoke | `slurm_jobs/test_job.sh` | 5 trials, both stages |
| Frame ablation | `slurm_jobs/frame_ablation.sh` | 4 vs 16, 30 trials |
| Study 2 orchestration | `slurm_jobs/study2_full.sh` | Shell chain |

---

## Per-trial JSON schema (v2)

```json
{
  "trial_id": "...",
  "stimulus_path": "...",
  "label": "correct mental state string",
  "frame_indices": [0, 30, 60],
  "n_frames_used": 3,
  "multi_frame_strategy": "native_list | composite_grid | single_first_frame",
  "stage1": {
    "prompt": "...",
    "free_response_text": "...",
    "semantic_entropy": 2.41,
    "label_probs": [0.01],
    "top_labels": [["Label", 0.2]],
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "entropy_temperature": 0.1
  },
  "stage2": {
    "options": ["...", "...", "...", "..."],
    "prediction": "...",
    "correct": true,
    "reasoning": "...",
    "raw_model_output": "..."
  },
  "error": null
}
```

**Top-level metrics must include:** `protocol_version`, `frame_policy`, `primary_outcomes`, `mean_semantic_entropy`, accuracy fields, human comparison p-values.

**Naming example:** `results/baseline/eu_emotions/{model}/eval_v2_{dataset}_{model}_fps1_cap16_two_stage_seed42.json`

---

## Execution order (do not skip)

1. `config.py`, `frame_sampling.py`, `semantic_entropy.py`, `multi_frame.py`
2. Refactor `evaluate.py`; smoke test: `python -m scripts.evaluate --max_trials 5 ...`
3. Verify multi-frame for all four models (short video each)
4. `submit_baselines.sh` → 118 × 4 models
5. `select_best_model.py` + `error_analysis.py`
6. `prepare_finetune_data.py` → hparam sweep → full finetune
7. Post-FT two-stage EU-Emotions on best model
8. Study 2 chain on HPC
9. Notebooks + `summarize_results.py`
10. `frame_ablation.sh` (30 trials)

---

## CLI examples

```bash
# Smoke test (local or HPC)
python -m scripts.evaluate \
  --model qwen2vl \
  --dataset eu_emotions \
  --max_frames 16 --fps 1 --stage both \
  --data_root data/eu_emotions_118 \
  --manifest data/eu_emotions_118_manifest.json \
  --max_trials 5

# Semantic entropy unit tests (no GPU)
python -m pytest tests/test_semantic_entropy.py -q

# Best model selection (after 4 baselines)
python -m scripts.select_best_model

# Error analysis
python -m scripts.error_analysis --results results/baseline/eu_emotions/qwen2vl/eval_v2_*.json
```

---

## Acceptance criteria

**Study 1 complete when:**

- [ ] 4 v2 JSONs for EU-Emotions 118 trials; every scored trial has non-null `stage1.semantic_entropy` and `stage2.correct`
- [ ] `best_model.json` written; full finetune for that model only
- [ ] `confused_pairs_*.json` has ≥5 pairs for best model
- [ ] Frame ablation JSON for 30 trials (4 vs 16)

**Study 2 complete when:**

- [ ] Activations for 4 baselines + best baseline + best finetuned at 3 fractions + peak layer
- [ ] Probe summaries with CV accuracies; high/low ambiguity splits for best model
- [ ] RSA summaries (or `human_rdm_source: pending` documented)
- [ ] Patching results for top-5 pairs at peak layer

---

## Do NOT

- Mix v1 (4-frame, single-stage) JSONs into v2 analysis
- Fine-tune all four models (best only)
- Use RMET in primary pipeline without protocol amendment
- Drop semantic entropy to exploratory-only
- Use fixed `lora_alpha=32` when rank varies (must be 2×rank)
- Run Study 2 patching before confused pairs exist

---

## Open decisions (defaults in `config.py` / README)

| Decision | Default |
|----------|---------|
| Embedding model | all-MiniLM-L6-v2 |
| Entropy τ | 0.1 |
| Stage 2 chained to Stage 1? | No |
| Best model tie-break | Higher accuracy, then alphabetical key |
| Human RDM missing | RSA `pending`; `build_human_rdm.py` |
| TransformerLens incompatible | Hook patching fallback (`DEVIATIONS.md`) |

---

## Commit message prefix

When the user requests commits: `protocol-v2: <description>`
