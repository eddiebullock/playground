# EU-Emotions evaluation artifact (Workstream A)

## What this is

A **reusable mental-state recognition benchmark** for multimodal VLMs on the EU-Emotions stimulus set (118 trials, 27 labels). The harness is public; **stimuli are not redistributable** — users must obtain clips from the original authors and point the manifest at a private path.

## Quick start (local, no GPU)

```bash
pip install -r requirements.txt
pytest tests/test_parse_emotion.py tests/test_free_response_judge.py tests/test_artifact_pipeline.py -q
```

Golden fixture: `tests/fixtures/eval_v2_mini.json` (3 synthetic trials, protocol v2).

## Metrics

### Primary

1. **Free-response judge accuracy** — `artifact_metrics.free_response_judge_accuracy` (rule-based match against `data/eu_emotion_synonyms.json`).
2. **Selective prediction / calibration** — `artifact_metrics.selective_prediction` (low-entropy subset accuracy, AUROC, ECE).

### Secondary (comparability)

- **Strict 4AFC** — top-level `accuracy` (deterministic foils `sha256(trial_id|seed)`).
- **Tolerant 4AFC** — `artifact_metrics.tolerant_rescore.tolerant_4afc_accuracy`.

Do **not** headline strict 4AFC vs published human **6AFC** group accuracy without a paradigm caveat (see below).

## HPC workflow (CSD3)

From `~/rds/hpc-work/study2` after `./sync.sh push` from local:

```bash
module load miniconda || module load miniconda3
export CONDA_ENVS_PATH=$PWD/conda_envs
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mr_eu_open_llm
```

### 1. Multi-model baselines (118 trials)

```bash
cd ~/rds/hpc-work/study2
bash slurm_jobs/submit_baselines.sh
# Smoke: MAX_TRIALS=5 bash slurm_jobs/submit_baselines.sh
```

Expected outputs:

| Model | Condition | Stage |
|-------|-----------|-------|
| qwen2vl | video_only | both |
| llavanext | video_only | both |
| gemma4 | multimodal | both |

Paths: `results/baseline/eu_emotions/{model}/eval_v2_*.json` (protocol v2, `n_scored` ≈ 118).

Then locally: `./sync.sh pull`

### 2. Augment eval JSONs

```bash
sbatch slurm_jobs/augment_all_baselines.sh
# or full post-process (augment + master table + calibration plots):
sbatch slurm_jobs/artifact_postprocess.sh
```

Each `eval_artifact_*.json` gains `artifact_metrics` with judge, selective_prediction, and tolerant_rescore.

Local one-off:

```bash
python -m scripts.augment_eval_artifact --input <eval_v2.json> --output <eval_artifact.json>
```

### 3. Frame policy ablation (`composite_grid` vs `native_video`)

`config.FRAME_POLICY["modes"]` defines both; `scripts/evaluate.py` accepts `--frame_mode composite_grid|native_video`.

```bash
# 30-trial smoke (default)
sbatch slurm_jobs/frame_policy_ablation.sh
# Full 118: sbatch --time=04:00:00 --export=MAX_TRIALS=118 slurm_jobs/frame_policy_ablation.sh
# Optional qwen2vl: sbatch --export=MODEL=qwen2vl,CONDITION=video_only slurm_jobs/frame_policy_ablation.sh
```

Outputs:

- `results/ablation/eval_v2_{model}_{condition}_{mode}_fps1_cap16_n{N}_seed42.json`
- `results/ablation/frame_policy_summary.json`
- `results/ablation/frame_policy_summary.md`

| Mode | `enforce_multi_frame` | Role |
|------|----------------------|------|
| `composite_grid` | True (default) | Fairness path: composite frame grid for non-native multi-image models |
| `native_video` | False | Native multi-frame / temporal sampling where supported |

### 4. Master results table

After augment + `./sync.sh pull`:

```bash
python -m scripts.artifact_results_table
# -> results/stats/artifact_master_table.json
# -> results/stats/artifact_master_table.md
```

Rows: model × condition × metric. Includes paradigm footnote for human comparison.

### 5. Calibration figures (optional)

```bash
python -m scripts.plot_calibration --input results/baseline/eu_emotions/*/eval_artifact_*.json
# -> results/stats/figures/reliability_*.png, ece_by_condition.png
```

### 6. Inspect AI integration (optional)

```bash
pip install inspect-ai
inspect eval inspect_eu/eu_emotions_task.py@eu_emotions_eval --model openai/gpt-4o --limit 3
```

Requires `OPENAI_API_KEY`. The `inspect_eu/` package wraps trial loading and scorers; `scripts/evaluate.py` remains the canonical local/HF path.

## Reference numbers (Gemma4 sanity checks)

| Condition | Strict 4AFC | Notes |
|-----------|-------------|-------|
| Baseline multimodal | ~46.6% | `eval_v2_*_two_stage_seed42.json` |
| Post-FT 4AFC | ~2.5% | B0: `GENUINE_DEGRADATION` |
| Post-FT finetune-prompt | ~1.7% | format not the fix |

B0 verdict: `results/stats/b0_finetune_verification.json`

## Human comparison caveats

- Published EU-Emotions human data often used **6AFC**; this harness uses **4AFC** with runtime-generated foils.
- Do **not** headline strict superiority vs human group accuracy when paradigms differ.
- For representational RSA vs humans, prefer a **matched-paradigm** human RDM (`docs/HUMAN_RDM.md` Option A) — supplementary, not blocking for model-only analyses.

## Modality ablation

Report **video_only vs audio_only vs multimodal** deltas for Gemma 4 (native audio). Treat audio as modality comparison only; do not over-claim prosody mechanisms.

## Licensing

| Component | License |
|-----------|---------|
| Code in this repo | MIT (see LICENSE) |
| EU-Emotions stimuli | **Not included** — user-provided under original dataset terms |
| Model weights | Per upstream HF model licenses |

## Target venue

NeurIPS **Datasets and Benchmarks** — emphasize harness, calibration axis, multi-model baselines, and reproducible scoring.

## CI

GitHub Actions (`.github/workflows/tests.yml`) runs parser, judge, selective-prediction, tolerant-parse, and golden-fixture artifact pipeline tests.
