# Cluster notes

Site-specific values are in git-ignored `local/site.sh`. This file uses
only variable names. An open SSH control socket is not permission to run
commands; propose the exact command and wait for a yes, including
read-only checks.

## Access

Password + TOTP; there are no SSH keys. The user opens a ControlMaster
socket by hand. Scripts reuse it with

```
-o ControlPath="$HPC_SSH_CONTROL_PATH" -o ControlMaster=no -o BatchMode=yes
```

so an expired socket fails fast. Never compute on the login node beyond
seconds-long checks and transfers. Downloads and rsync on the login node
are fine.

## Project tree

Create a new tree `$HPC_ROOT` on RDS for this study only. It holds code,
envs, caches, outputs and logs.

Read-only reuse (never write, rsync or `sbatch` into those trees):

- Weights already on disk: `$ESVLM_MODELS_DIR/<key>/config.json`
  - `qwen3vl` = `Qwen/Qwen3-VL-8B-Instruct`
  - `gemma4` = `google/gemma-4-E4B-it`
  - `molmo2` = `allenai/Molmo2-O-7B`
  - `qwen2vl` = `Qwen/Qwen2-VL-7B-Instruct`
- New weights (Qwen3-VL 2B/32B, Qwen2.5-VL-7B, text-only Qwen3) download
  into `$HPC_ROOT/models`.
- Stimuli: RMET under `$ESVLM_RMET_DIR`; EU-Emotions and Mind Reading
  under `$ESVLM_EU_EMOTIONS_DIR` and `$ESVLM_MINDREADING_DIR`.
- Public S-domain sets: `$ESVLM_PUBLIC_DIR` (PhysBench, CLEVRER
  validation, PIQA). Fetch with `slurm/fetch_sdomain.sbatch`. Never
  write these into a licensed tree.

## OS and Python

- glibc 2.28. Wheels needing glibc >=2.29 fail. If uv's managed Python
  or a wheel fails, fall back to Miniforge under `$HPC_ROOT`, with no
  `conda init`, activated by an explicit `source …/conda.sh`.
- System `python3` is 3.6. Anything that runs before an env is active
  (job preamble, site script) must be bash or 3.6-safe Python: no
  `list[str]`, no `X | None`, no walrus, no
  `from __future__ import annotations`.
- In batch shells, `module` may be undefined; `source /etc/profile.d/modules.sh`
  first. Strip inherited `cuda/11.4` from `CPATH`.
- All caches go to RDS, never `$HOME`. Export absolute paths for
  `HF_HOME`, `HF_MODULES_CACHE`, `TORCH_HOME`, `XDG_CACHE_HOME`,
  `PIP_CACHE_DIR`, `UV_CACHE_DIR`, `MPLCONFIGDIR`, `PYTHONNOUSERSITE=1`,
  and conda pkgs/envs dirs. A quoted `~` over SSH creates a literal `~`
  directory on home.

## SLURM

- GPU: `$HPC_GPU_ACCOUNT` on `$HPC_GPU_PARTITION`, `--gres=gpu:1`.
- CPU: `$HPC_CPU_ACCOUNT` on `$HPC_CPU_PARTITION`.
- 12 h walltime cap. Prefer 30 min to 2 h chunks so jobs backfill.
- `--signal=B:USR1@180` plus checkpoint-on-signal. Every job is
  resumable (`--trial_offset` / `--trial_limit` or shard ids). Chain with
  `--dependency=afterany`.
- One GPU gives 32 CPUs and about 250 GB regardless of request.
  `--mem-per-cpu` must be <=8000M on the GPU partition and <=3370M on
  the CPU partition, otherwise SLURM silently inflates the CPU count.
  `scripts/submit.sh` refuses over-cap memory.
- Always pass `-J <model>_<task>_<shard>` (`%x` in log names). Logs go
  to `logs/%x_%j.out` with `--open-mode=append`.
- Eval jobs set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. Fetch
  weights with `slurm/fetch_weights.sbatch` on the CPU partition.
  Fetch public S-domain sets with `slurm/fetch_sdomain.sbatch`
  (`DATASET=physbench|clevrer|piqa`).

Submit only through `scripts/submit.sh`. Never use plain `sbatch` in
docs or habit.

## Sync

`scripts/sync.sh` supports `status`, `push`, and `pull-results`. macOS
rsync is 2.6.9: write every option as `--opt=value` and single-quote
globs. `push` writes a `VERSION` file and excludes `.git`, `data/`,
`local/`, `.env`, `models/`, caches, envs, `results/` and logs.

Never push code while a job from this project is queued or running.
`sync.sh` checks `squeue` first, which itself requires approval.

## Model pitfalls

- Qwen3-VL OOMs on an A100-80GB when forced-choice sampling is batched
  with video. Score options sequentially and cap frames (start at 8).
- Model load from Lustre takes about 10 minutes. Smoke jobs need >=20
  min walltime but should stay short.
- Llama 4 Scout is gated. Do not assume weights.
- InternVL3.5 is optional. Try it only in a throwaway env; if it fails
  to load twice, drop it and log the decision.
- Gemma 4 needs Transformers 5.x. Molmo2 needs Transformers 4.57
  (`optional_attributes`). InternVL2-8B failed to load on Transformers
  5.10 and was dropped.

## Monitoring (each command needs approval)

Typical checks, with values filled from `local/site.sh`:

```
squeue -u "$HPC_USER" --start
sacct -j <id> --format=JobID,JobName%30,State,ExitCode,Elapsed,NodeList
sprio -j <id>
mybalance
quota -s
```

`df` is wrong on RDS; watch inode count too. Before trusting any
finished GPU job, read the echoed arguments in its log.

API reference models run on the Mac, never on the cluster.
