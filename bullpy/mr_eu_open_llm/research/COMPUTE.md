# COMPUTE

Mapped 2026-10-30, Europe/London. No serial numbers.

## Local machine

| | |
|---|---|
| Hardware | MacBook Air (Mac15,12), Apple M3 |
| Cores | 8 (4 performance + 4 efficiency) |
| RAM | 16 GB unified |
| Disk | ~29 GB free on `/System/Volumes/Data` (94% used). Do not download 8B VLMs. |
| OS | macOS 14.8.9 (darwin 23.6.0) |
| MPS | Available (`torch.backends.mps.is_available() == True` in conda env) |
| CUDA | Not available locally |

**Conda env:** `mr_eu_open_llm` at `/opt/anaconda3/envs/mr_eu_open_llm`

| package | version |
|---|---|
| python | 3.10.x in the env (import of torch 2.10.0 succeeded) |
| torch | 2.10.0 (MPS yes, CUDA no) |
| numpy | 1.26.4 |
| scipy | 1.15.2 |
| sklearn | 1.7.2 |
| pandas | 2.3.3 |

Homebrew `/opt/homebrew/bin/python3` (3.13.5) does **not** have numpy; always `conda activate mr_eu_open_llm`.

### What can run locally

- Probing, RSA, stats, entanglement null, partial RSA, read-out mixing **if** activation `.npy` are on disk.
- Unit tests, pipeline refactor, preregistration, paper drafting.
- Possibly Gemma 4 E4B smoke on MPS after a disk check. Not attempted in Phase 0.

### What must not run locally

- Qwen3-VL-8B / Molmo2-O-7B full 243-trial video forwards.
- All-layer, multi-position activation extraction.
- Activation patching sweeps.
- Any job that would pull multi-GB weights onto 29 GB free disk.

Long local jobs, if any: `caffeinate -dimsu` and tmux, logs under `research/logs/`.

## CSD3 (Cambridge)

Configured in `config.py` / `sync.sh` / `README.md`. **SSH from this agent session failed** (publickey). Treat as currently unavailable.

| | |
|---|---|
| Login | `eb2007@login.hpc.cam.ac.uk` |
| Study3 root | `~/rds/hpc-work/study3` (override `MR_EU_HPC_PROJECT`) |
| Study2 root (do not touch) | `~/rds/hpc-work/study2` |
| GPU account | `BARON-COHEN-SL3-GPU` |
| CPU account | `BARON-COHEN-SL3-CPU` |
| Partition seen in eval metadata | `ampere` |
| Hosts seen in eval JSON | `gpu-q-20`, `gpu-q-43` |
| GPU walltime cap (docs) | 12 h, `gpu2` QOS |
| Sync | `./sync.sh push` / `pull-study3` (rsync; interactive SSH) |

Historical jobs (from cached eval JSON):

| model | slurm_job_id | created_at_utc |
|---|---|---|
| qwen3vl | 33520827 | 2026-08-13 00:35 |
| gemma4 | 32447858 | 2026-07-31 21:06 |
| molmo2 | 32447860 | 2026-07-31 21:29 |

Eval runtime: CUDA, torch 2.10.0+cu128, transformers 5.10.1 (Molmo 4.57.3).

### What needs GPU (CSD3 A100)

- Missing activation extraction: all layers, last / visual-mean / pre-answer token positions, 243 items, three models. Estimate: a few GPU-hours per model if last-token-only already exists on disk and only extra positions/layers are missing; a full re-extract is longer (prior extract used max_frames=4, 1 fps).
- Phase 3 patching + erasure on 243 items, Qwen first then Gemma. Estimate: patching heatmap tens of GPU-hours; rank-k erasure cheaper if hooks are cached.
- Phase 4 steering sweeps.
- Phase 6 scale (Qwen3-VL 2B/4B/8B, maybe 32B if memory allows).

### Activation cache status (this Mac)

| model | layers saved | pooling | n | `.npy` local |
|---|---|---|---|---|
| qwen3vl | 4, 13, 26 | last_token | 243 | **no** |
| gemma4 | 5, 15, 31 | last_token | 243 | **no** |
| molmo2 | 4, 12, 23 | last_token | 243 | **no** |

Axis vectors (`results/mech/axis_*_qwen3vl_layer4.npy`) **are** local, shape `(4096,)`, unit-norm, float32.

4AFC-era Gemma activations **are** local (~1.2 MB per layer) but are the wrong protocol (118-trial / 4AFC study2). Do not reuse for study3 claims.

Once CSD3 works, pull with something like:

```bash
rsync -avz --include='*/' --include='*.npy' --include='*.json' --exclude='*' \
  eb2007@login.hpc.cam.ac.uk:~/rds/hpc-work/study3/results/activations/baseline_qwen3vl_6afc/ \
  results/activations/baseline_qwen3vl_6afc/
```

(and the gemma4 / molmo2 siblings, plus `results/mech/*ablat*`).

## Estimated GPU ask (for the user, when SSH is back)

Conservative, A100 80 GB, 243 EU-Emotion videos, composite-grid 4–16 frames:

| stage | model | GPU-hours (order of magnitude) |
|---|---|---|
| Pull existing 6AFC last-token npy | — | 0 (rsync) |
| Extra positions / all layers extract | Qwen3-VL-8B | 4–12 |
| Same | Gemma 4 E4B, Molmo2 | 4–12 each |
| Patching map | Qwen | 8–24 |
| Erasure + positive control | Qwen | 4–12 |
| Repeat causal | Gemma | 8–24 |
| Scale family (2B/4B/8B behaviour + gate) | Qwen3-VL | 12–40 |

These are planning estimates, not quotes. Refine after one timed smoke job.

## Models pinned in `config.py` (not yet revision-hashed)

| key | hf_id |
|---|---|
| qwen3vl | `Qwen/Qwen3-VL-8B-Instruct` |
| gemma4 | `google/gemma-4-E4B-it` |
| molmo2 | `allenai/Molmo2-O-7B` |
| llama4 | `meta-llama/Llama-4-Scout-17B-16E-Instruct` (gated; listed, not in the 3-model pilot) |

Phase 6 should pin commit SHAs on Hugging Face before any new eval.
