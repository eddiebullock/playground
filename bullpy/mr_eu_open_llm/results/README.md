## Results directory layout

This folder stores all outputs from baseline evaluation, fine-tuning, and mechanistic interpretability. It is safe to sync between local and CSD3, but **raw datasets, model weights, and large activation arrays** are excluded from git.

### Subdirectories

- `baseline/` – zero-shot evaluation outputs for each model and dataset.
  - `eu_emotions/`
  - `mindreading/`
  - `rmet/`
- `finetune/` – checkpoints and evaluation metrics for LoRA fine-tuning.
  - `hparam_sweep/`
  - `full_runs/`
- `activations/` – `.npy` activation tensors from `extract_activations.py`.
- `probes/` – probe training metrics and weights from `probing.py`.
- `rsa/` – RDMs and RSA correlation results from `rsa.py`.
- `patching/` – activation patching results from `activation_patching.py`.
- `stats/` – aggregated CSV/JSON tables of statistics (CIs, p-values, effect sizes).
- `test_runs/` – outputs from quick smoke tests and SLURM `test_job.sh`.
- `logs/` – SLURM stdout/stderr logs, plus any additional run-time logs.

### File naming conventions

The goal is to keep names **machine-readable** and **self-explanatory**:

- **Baseline evaluation** (`scripts/evaluate.py`)
  - `baseline/{dataset}/{model}/baseline_{dataset}_{model}_frames{n_frames}_seed{seed}.json`
  - Each JSON should include:
    - per-trial predictions and correctness
    - aggregate accuracy and Wilson CI
    - metadata (model, dataset, date, git commit, frame strategy)

- **Hyperparameter sweep** (`scripts/finetune.py` in sweep mode)
  - `finetune/hparam_sweep/{model}/run_lr{lr}_r{rank}_seed{seed}/metrics.json`
  - Optional:
    - `.../train_loss.csv`
    - `.../val_accuracy.csv`

- **Full fine-tuning runs**
  - `finetune/full_runs/{model}/run_{run_id}/`
    - `checkpoints/epoch{epoch}.pt` (or HF-style checkpoint folders)
    - `metrics_epoch{epoch}.json`
    - `eu_emotions_eval_epoch{epoch}.json`

- **Activation extraction** (`scripts/extract_activations.py`)
  - `activations/{model}/{dataset}/layer{layer_idx}_{split}_seed{seed}.npy`
  - If using subsets/batches, append `batch{idx}`.

- **Probing** (`scripts/probing.py`)
  - `probes/{model}/{dataset}/probes_{model}_{dataset}_seed{seed}.json`
  - Optional per-layer weights:
    - `probes/{model}/{dataset}/weights_layer{layer_idx}.joblib`

- **RSA** (`scripts/rsa.py`)
  - `rsa/{model}/{dataset}/rdm_{model}_{dataset}_seed{seed}.npy`
  - `rsa/{model}/{dataset}/rsa_summary_{model}_{dataset}_seed{seed}.json`

- **Activation patching** (`scripts/activation_patching.py`)
  - `patching/{model}/{dataset}/patching_{model}_{dataset}_seed{seed}.json`

- **Statistics and summaries** (`scripts/statistics.py`, `scripts/error_analysis.py`)
  - `stats/summary_{analysis_name}.csv`
  - `stats/summary_{analysis_name}.json`

### Notebook expectations

The notebooks in `notebooks/` assume:

- Results live under the paths described above.
- File names follow the conventions here (model, dataset, n_frames, seed, etc.).

If you change the naming scheme, update both this README and the relevant notebook cells so that analysis remains reproducible.

