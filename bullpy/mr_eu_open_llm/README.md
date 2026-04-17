## Mental State Recognition with Open-Weight Multimodal LLMs

This repository contains the code and analysis for a PhD study on **fine-tuning open-source multimodal LLMs for mental state recognition with mechanistic interpretability**. All heavy compute runs on the Cambridge CSD3 (Wilkes3, NVIDIA A100 GPUs), while this repo is developed locally in Cursor.

### Models

- **Qwen2-VL-7B-Instruct** (`Qwen/Qwen2-VL-7B-Instruct`)
- **InternVL2-8B** (`OpenGVLab/InternVL2-8B`)
- **LLaVA-NeXT-Interleave-7B** (`llava-hf/llava-interleave-qwen-7b-hf`)
- **Gemma 4 E4B (instruction-tuned)** (`google/gemma-4-E4B-it`)

### Datasets

- **EU-Emotions** stimulus set (4AFC, 118 test trials, 27 mental states)
- **Mindreading DVD** (train/val/test splits with videos and images)
- **RMET** Reading the Mind in the Eyes Test (36 trials, held-out zero-shot generalisation)

### Study phases

- **Phase 1 – Baseline evaluation**
  - Zero-shot evaluation of all three models on EU-Emotions, Mindreading test set, and RMET.
  - 4-alternative forced choice with **4 frames per video** (0%, 25%, 50%, 75% of duration).
  - Frame sampling ablation on a 30-trial subset (4 vs 8 frames).

- **Phase 2 – Fine-tuning**
  - LoRA fine-tuning on Mindreading training split:
    - \(r = 16\), \(\alpha = 32\), target modules \([q\_proj, v\_proj]\), dropout \(= 0.1\).
  - Hyperparameter sweep on a 100 train / 50 val subset:
    - LRs \(\{1e{-4}, 5e{-5}, 1e{-5}\}\) × ranks \(\{8, 16, 32\}\).
  - Full runs: 5 epochs, batch size 4, grad accumulation 4 (effective 16), fp16, save each epoch.
  - Monitor EU-Emotions accuracy to detect catastrophic forgetting (>5pp drop).

- **Phase 3 – Mechanistic interpretability**
  - Activation extraction across early/mid/late transformer layers (baseline and fine-tuned).
  - Probing (per-layer logistic regression), RSA, and activation patching (with TransformerLens).
  - Statistical analysis: Wilson CIs, binomial tests vs chance, two-proportion z-tests vs human benchmarks, Fisher’s exact tests with Bonferroni correction, and Cohen’s h.

### Directory structure

- `scripts/`: all runnable Python scripts for preprocessing, evaluation, fine-tuning, and interpretability.
- `slurm_jobs/`: SLURM job scripts for CSD3 (Wilkes3, A100).
- `notebooks/`: Jupyter notebooks for analysis and visualisation.
- `results/`: structured outputs from baseline, fine-tuning, and interpretability.
- `data/`: **local** copies of datasets (not tracked in git).
- `models/`: **local** model weights cache (not tracked in git).

See `results/README.md` for the expected result file layout and naming conventions.

---

### HPC workflow (CSD3, Wilkes3 A100s)

- **HPC username**: `eb2007`
- **Login**: `ssh eb2007@login.hpc.cam.ac.uk`
- **GPU project**: `BARON-COHEN-SL3-GPU` (A100s, `ampere` partition, SL3 12hr limit)
- **CPU project**: `BARON-COHEN-SL3-CPU`
- **Data/model storage**: `~/rds/hpc-work/study2/` (not the home directory)

Environment and paths on CSD3 are configured via `config.py` and `environment.yml`, and the directory structure is bootstrapped by `setup_hpc.sh`.

---

### Quickstart: local setup to first CSD3 test job

#### 1. Create the local conda environment

From the project root:

```bash
conda env create -f environment.yml
conda activate mr_eu_open_llm
python -m ipykernel install --user --name=mr_eu_open_llm
```

Copy your datasets into the local `data/` folder (mirroring the planned HPC layout):

- `data/eu_emotions/`
- `data/mindreading/`
- `data/rmet/`

#### 2. Sync scripts and configs to CSD3

From the project root:

```bash
chmod +x sync.sh
./sync.sh push
```

This will create/update `~/rds/hpc-work/study2/scripts/` and copy `config.py`, `requirements.txt`, and `environment.yml` to `~/rds/hpc-work/study2/` on CSD3.

#### 3. SSH into CSD3 and run HPC setup

```bash
ssh eb2007@login.hpc.cam.ac.uk
cd ~/rds/hpc-work/study2
chmod +x setup_hpc.sh
./setup_hpc.sh
```

This will:

- Create the full HPC directory layout under `~/rds/hpc-work/study2/`.
- Load the `miniconda` module.
- Create the `mr_eu_open_llm` conda environment from `environment.yml`.
- Download the three multimodal models into `~/rds/hpc-work/study2/models/`.

#### 4. Submit the first test job

From the CSD3 login node, in `~/rds/hpc-work/study2`:

```bash
sbatch slurm_jobs/test_job.sh
```

This submits a 1-hour, 1-GPU smoke test (50-trial pipeline) on the `ampere` partition using project `BARON-COHEN-SL3-GPU`. Logs and results will appear under `results/test_runs/` once the job has finished.

