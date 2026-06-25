## Theory of Mind in open multimodal LLMs

This repository implements **Protocol v2** for benchmarking EU-Emotions mental-state recognition and mechanistic interpretability on open VLMs. Heavy compute runs on Cambridge CSD3 (NVIDIA A100).

**Pivot (2026-06):** repositioning toward a **shippable eval artifact** + **circuit-level interpretability** — see [`PIVOT.md`](PIVOT.md) and [`docs/ARTIFACT.md`](docs/ARTIFACT.md).

**Agent instructions:** [`PROTOCOL_V2_AGENT_PROMPT.md`](PROTOCOL_V2_AGENT_PROMPT.md)  
**Methods alignment:** [`MANUSCRIPT_CHANGES.md`](MANUSCRIPT_CHANGES.md)

### Evaluation artifact (Workstream A)

- **Primary metric:** free-response judge (`scripts/free_response_judge.py`, synonyms in `data/eu_emotion_synonyms.json`)
- **Calibration axis:** selective prediction / ECE (`scripts/selective_prediction.py`)
- **Secondary:** 4AFC (`parse_emotion`) for human-comparability
- **Inspect AI:** optional front-end in `inspect_eu/` (`pip install inspect-ai`)
- **Canonical local runner:** `scripts/evaluate.py` (unchanged)

```bash
pytest tests/test_parse_emotion.py tests/test_free_response_judge.py -q
python -m scripts.augment_eval_artifact --input results/.../eval_v2_*.json --output results/.../eval_artifact.json
python -m scripts.verify_finetune_eval --eval_json results/finetune/eu_post_ft/....json
```

### Interpretability flagship (Workstream B / Study 3)

Real activation extraction (`scripts/extract_activations.py` + `activation_forward.py`), probing, RSA, hook-based patching. Fine-tuning is a **causal probe** (representation vs readout), not the headline result.

### Models

- **Qwen2-VL-7B-Instruct**
- **InternVL2-8B**
- **LLaVA-NeXT-Interleave-7B**
- **Gemma 4 E4B IT**

### Study 1 (benchmarking + fine-tuning)

Each trial has **two stages**:

1. **Free response** — no label options; model describes expressed mental state(s). Runs under **`video_only`** only (semantic entropy comparability).
2. **4AFC** — deterministic foils (`sha256(trial_id|seed)`); accuracy vs chance (0.25) and modality-matched EU human benchmarks (O'Reilly / Lassalle).

**Modality ablations:** `video_only`, `audio_only`, `multimodal` via `--condition`. EU multimodal pairs face video with UK Voices by emotion label; Mindreading audio uses item-folder T-files (not `Emotions/Audio/`).

From Stage 1 text we compute **semantic entropy** over 27 EU-Emotions label embeddings (co-primary with accuracy).

**Frame policy (all four models):** 1 fps, max 16 frames, uniform in time. Non-Qwen models receive a **composite frame grid** (`scripts/multi_frame.py`).

**Fine-tuning:** best model only; LoRA on Mindreading **video_only** face clips; monitor EU-Emotions retention.

### Study 2 (interpretability)

Linear probing, RSA vs human RDM, and activation patching — baseline four models plus best model before/after fine-tuning.

### Quickstart

From Mac: `./sync.sh push`

On CSD3 (in `~/rds/hpc-work/study2`):

```bash
bash setup_hpc.sh
conda activate mr_eu_open_llm
pip install sentence-transformers
```

Smoke test: `MAX_TRIALS=5 sbatch slurm_jobs/test_job.sh`

Video-only baselines: `sbatch slurm_jobs/submit_baselines.sh`

**Stimulus data on HPC** (not included in `./sync.sh push`):

```bash
# From Mac: UK Voices + Mindreading item folders (skips Emotions/Audio/)
bash scripts/sync_data_hpc.sh
```

You still need EU face videos and manifests under `data/eu_emotions_118/` on CSD3. UK Voices must live at
`data/eu_emotions_118/EU Emotion - UK Voices/` (the sync script places them there). Mindreading T-files
live inside item folders next to V face clips.

Modality ablation suite:

```bash
python run_ablation_suite.py \
  --data-dir-eu data/eu_emotions_118 \
  --data-dir-mr data/mindreading \
  --manifest-eu data/eu_emotions_118_manifest.json \
  --manifest-mr data/mindreading_test_manifest.json
```

Or: `sbatch slurm_jobs/ablation_suite.sh`

### Results (v2 only)

Use names like:

`results/baseline/eu_emotions/{model}/eval_v2_eu_emotions_{model}_video_only_fps1_cap16_two_stage_seed42.json`

Companion CSV: `{model}_{dataset}_{condition}_results.csv`

See [`results/README.md`](results/README.md).

### HPC

- Login: `ssh eb2007@login.hpc.cam.ac.uk`
- Work dir: `~/rds/hpc-work/study2/`
- GPU project: `BARON-COHEN-SL3-GPU`
