# CURSOR AGENT PROMPT: Study 2 prep (parallel to Study 1)

Use this when implementing mechanistic interpretability **before** Study 1 fine-tuning is complete. Do not block Study 1; do not run confirmatory patching or fine-tuned comparisons until Study 1 outputs exist.

---

## Context

**Repository:** `mr_eu_open_llm` on CSD3 at `~/rds/hpc-work/study2`

**Study 1 status (check on HPC before assuming):**
- EU-Emotions 118-trial two-stage baselines should exist under `results/baseline/eu_emotions/{model}/eval_v2_*.json`
- Fine-tuning may be in progress or blocked on Mindreading media paths
- `results/stats/best_model.json` and `confused_pairs_*.json` may not exist yet

**Study 2 models (protocol):**
- **Baselines (all):** `qwen2vl`, `llavanext`, `gemma4` (`config.STUDY_MODELS`)
- **Before/after fine-tune:** best model only (baseline + finetuned checkpoint)

**Dataset for activations / probes / RSA:** EU-Emotions 118 trials only
- Manifest: `data/eu_emotions_118_manifest.json`
- Root: `data/eu_emotions_118/`
- Trial order must match activation rows and human RDM rows

---

## What CAN start now (Phase 2a — prep)

| Task | Script / file | GPU? | Depends on Study 1? |
|------|---------------|------|-------------------|
| Layer index mapping | `scripts/layer_map.py` | No | No |
| Fix activation extraction (real VLM forward + hooks) | `scripts/extract_activations.py` | Yes | No (needs manifest + models) |
| Smoke extraction (5 trials, 1 model, 1 layer) | `extract_activations.py --max_trials 5` | Yes | No |
| Multi-layer probe sweep | `scripts/probing.py` | CPU | Eval JSON for labels |
| RSA plumbing (model RDM only) | `scripts/rsa.py` | CPU | Activations |
| Human RDM protocol doc + builder | `scripts/build_human_rdm.py` | No | No |
| SLURM job for extraction only | new `slurm_jobs/study2_extract.sh` | Yes | No |
| Results directory layout | `results/activations/`, `results/probes/`, `results/rsa/` | No | No |

## What MUST wait (Phase 2b — confirmatory)

| Task | Blocked until |
|------|----------------|
| Entropy tertile probes (high vs low ambiguity) | Study 1 `stage1.semantic_entropy` in baseline eval JSON |
| Peak-layer selection for patching | Multi-layer probe sweep complete |
| Activation patching (real) | `confused_pairs_{best_model}.json` from Study 1 |
| Fine-tuned activation extraction | Full LoRA checkpoint on disk |
| RSA vs human (inferential) | `data/human_rdm.npy` from real human data (not placeholder) |
| Baseline vs finetuned RSA/probe comparison | Both checkpoints extracted |

---

## Implementation priorities (order)

### 1. Harden `extract_activations.py` (critical path)

Current code registers hooks but runs a **dummy forward** — activations are not stimulus-conditioned.

**Required behavior:**
- Reuse `scripts/evaluate.py` model loading (`load_hf_model_for_key`, processor, `prepare_images_for_model`)
- For each EU-Emotions trial: load frames via `frame_sampling.load_stimulus_as_images` (same fps/max_frames as Study 1)
- Run **one real inference forward** per trial (Stage 2-style prompt optional; stimulus-only forward is OK for hidden states)
- Register forward hooks at layers from `layer_map.get_layer_indices(model_key, fractions=config.LAYER_DEPTH_FRACTIONS)`
- Mean-pool hidden states over sequence length → vector per trial
- Save per layer:
  - `results/activations/{condition}/{model}/layer{L}_eu_emotions_seed42.npy` shape `(n_trials, hidden_dim)`
  - `layer{L}_trial_ids.json` — list aligned with manifest order
  - `extract_meta.json` — model, condition, layer_map, git commit, frame_policy

**Conditions naming:**
- `baseline_qwen2vl`, `baseline_llavanext`, `baseline_gemma4`
- `finetuned_{best_model}` (later, with `--checkpoint` PEFT path)

**CLI additions:**
- `--max_trials` for smoke tests
- `--checkpoint` for fine-tuned weights
- `--layers` optional override of depth fractions

**Acceptance:** 5-trial smoke produces non-zero, non-identical rows across trials; `trial_ids` matches manifest.

### 2. Extend `probing.py` for all layers

Current code probes **one** `.npy` file.

**Required:**
- Loop all `layer*_eu_emotions_seed42.npy` in an activations directory
- Per layer: stratified 5-fold multinomial logistic regression (`sklearn`, `SEED=42`)
- Save `results/probes/{condition}/{model}/probes_summary.json`:
  ```json
  {"layers": [{"layer_index": 7, "depth_fraction": 0.375, "cv_accuracy": 0.42, ...}], "peak_layer": 14, "peak_accuracy": 0.51}
  ```
- Write `peak_layer.json` for patching
- **If** eval JSON provided: compute `low_ambiguity_accuracy` / `high_ambiguity_accuracy` using entropy tertiles from `stage1.semantic_entropy`

### 3. Extend `rsa.py` for per-layer sweep

- Compute model RDM per layer (cosine distance between trial activation vectors — already in `compute_rdm`)
- If `data/human_rdm.npy` exists: Spearman correlation (upper triangle) per layer
- If missing: `human_rdm_source: "pending"`, skip inferential tests, still save model RDMs to `results/rsa/{condition}/{model}/rdm_layer{L}.npy`

### 4. Human RDM builder (see `docs/HUMAN_RDM.md` or section below)

Extend `build_human_rdm.py` to accept human response CSV/JSON and emit stimulus-aligned `human_rdm.npy`.

### 5. `activation_patching.py` — scaffold only for now

- Document TransformerLens compatibility per model
- Implement native forward-hook patching fallback
- Do **not** run full patching until `confused_pairs` exists

### 6. SLURM: `slurm_jobs/study2_extract.sh`

```bash
# Extract baselines for 3 models; no finetune, no patching
for MODEL in qwen2vl llavanext gemma4; do
  python -m scripts.extract_activations \
    --model "$MODEL" \
    --condition "baseline_${MODEL}" \
    --manifest data/eu_emotions_118_manifest.json \
    --data_root data/eu_emotions_118
done
```

Separate job for probing (CPU, after extraction): `study2_probe.sh`

Do **not** use `study2_full.sh` as-is until extraction is real and Study 1 stats exist.

---

## File layout (target)

```
results/
  activations/
    baseline_qwen2vl/qwen2vl/
      layer3_eu_emotions_seed42.npy
      layer3_trial_ids.json
      extract_meta.json
    baseline_llavanext/llavanext/...
    baseline_gemma4/gemma4/...
    finetuned_qwen2vl/qwen2vl/...   # later
  probes/
    baseline_qwen2vl/qwen2vl/probes_summary.json
    baseline_qwen2vl/qwen2vl/peak_layer.json
  rsa/
    baseline_qwen2vl/qwen2vl/rsa_summary.json
    baseline_qwen2vl/qwen2vl/rdm_layer14.npy
  patching/                        # later
data/
  human_rdm.npy                      # 118 x 118, float32
  human_rdm_source.json              # provenance metadata
  human_behaviour/                   # raw human trial responses (optional)
```

---

## Smoke-test commands (HPC)

```bash
cd ~/rds/hpc-work/study2
module load miniconda
export CONDA_ENVS_PATH="${PWD}/conda_envs"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mr_eu_open_llm

# 5 trials, one model
python -m scripts.extract_activations \
  --model qwen2vl \
  --condition baseline_qwen2vl \
  --manifest data/eu_emotions_118_manifest.json \
  --data_root data/eu_emotions_118 \
  --max_trials 5

# Probe (needs matching eval JSON for labels)
python -m scripts.probing \
  --activations_dir results/activations/baseline_qwen2vl/qwen2vl \
  --eval_json results/baseline/eu_emotions/qwen2vl/eval_v2_*.json \
  --output results/probes/baseline_qwen2vl/qwen2vl/probes_summary.json

# RSA model-only
python -m scripts.rsa \
  --activations results/activations/baseline_qwen2vl/qwen2vl/layer7_eu_emotions_seed42.npy \
  --output results/rsa/baseline_qwen2vl/qwen2vl/rsa_summary.json
```

---

## Constraints

- Same `FRAME_POLICY` (fps=1, cap=16) as Study 1 eval when running forward passes
- `SEED=42` for CV folds and any subsampling
- No emojis in code
- Document deviations in `DEVIATIONS.md`
- Do not mix trial order between activations, human RDM, and eval JSON
- Placeholder human RDM (`--allow_placeholder`) is **dev only** — never for thesis figures

---

## Acceptance criteria (Phase 2a complete)

- [ ] `extract_activations.py` runs real VLM forward on EU-Emotions (smoke + full 118)
- [ ] Baseline activations on disk for all 3 models at 3 depth fractions (+ peak layer post-hoc)
- [ ] `probes_summary.json` per model with CV accuracy vs layer depth
- [ ] Model RDMs saved per layer; RSA runs with `human_rdm_source: pending` OR real Spearman if human RDM ready
- [ ] `data/human_rdm_source.json` documents collection plan or imported source
- [ ] Patching remains stubbed until `confused_pairs_*.json` exists

---

## Study 1 handoff checklist (before Phase 2b)

```bash
test -f results/stats/best_model.json
test -f results/stats/confused_pairs_*.json
ls results/finetune/full_runs/*/checkpoints/
python -m scripts.run_study1_postbaseline
```

Then: finetuned extraction, tertile probes, patching, baseline-vs-FT RSA.
