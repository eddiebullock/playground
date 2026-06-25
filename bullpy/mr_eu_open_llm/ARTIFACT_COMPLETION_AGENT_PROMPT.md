# CURSOR AGENT: Complete Workstream A — EU-Emotions Eval Artifact (NeurIPS D&B)

## Mission

Ship the **benchmark artifact** to ~90% write-up readiness: multi-model baselines, primary/secondary metrics tables, frame-policy ablation, augmented eval JSONs, optional Inspect smoke, CI golden fixtures, and a consolidated results summary. **Do not** block on Study 3 patching or fine-tuning interpretability.

## Repository & HPC

- **Local repo:** `mr_eu_open_llm` (this workspace)
- **HPC root:** `~/rds/hpc-work/study2`
- **Sync:** `./sync.sh push` / `./sync.sh pull`
- **Conda (HPC):**
  ```bash
  cd ~/rds/hpc-work/study2
  module load miniconda || module load miniconda3
  export CONDA_ENVS_PATH=$PWD/conda_envs
  source $(conda info --base)/etc/profile.d/conda.sh
  conda activate mr_eu_open_llm
  ```
- **sbatch:** from `study2` root: `sbatch slurm_jobs/foo.sh`

## Deliverables

1. **Multi-model baselines (118 trials):** `bash slurm_jobs/submit_baselines.sh` → qwen2vl, llavanext, gemma4
2. **Augment all eval JSONs:** `scripts/augment_eval_artifact.py` → `eval_artifact_*.json`
3. **Frame policy ablation:** `composite_grid` vs `native_video` (wire `--frame_mode` in evaluate.py if needed); table in `results/ablation/`
4. **`scripts/artifact_results_table.py`** → `results/stats/artifact_master_table.md`
5. **CI fixtures:** `tests/fixtures/eval_v2_mini.json` + tests
6. **Update `docs/ARTIFACT.md`** with commands and numbers

## Metrics framing

- **Primary:** free-response judge, selective prediction / ECE
- **Secondary:** strict + tolerant 4AFC
- **Caveat:** human 6AFC vs our 4AFC — do not over-claim

## Out of scope

Study 3 patching, SAE, human RDM, new fine-tuning runs

## Known Gemma4 sanity checks

| Condition | Strict 4AFC |
|-----------|-------------|
| Baseline multimodal | ~46.6% |
| Post-FT 4AFC | ~2.5% |
| Post-FT finetune-prompt | ~1.7% |
