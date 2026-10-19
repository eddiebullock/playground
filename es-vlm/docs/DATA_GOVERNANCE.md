# Data governance

Treat this repository as **public-safe** regardless of GitHub visibility.

## Visibility

Checked 2026-10-18 via `gh repo view eddiebullock/playground --json visibility`
with the `eddiebullock` account active. Visibility is **public**. Treat
every commit as public-safe.

## Never track

- Licensed stimuli: RMET, Multiracial RMET, Mind Reading, EU-Emotions
  faces and voices, EQ-60 / SQ-R item text.
- Human data, including the O'Reilly EU-Emotions item-level workbook
  (Mac-only; never synced).
- Secrets: `.env`, Hugging Face / API / W&B tokens.
- Site-specific details: site or institution name, hostnames, CRSid,
  SLURM account names, absolute cluster paths. These live in git-ignored
  `local/site.sh` and `local/HPC_RUNBOOK.md`. Tracked docs say "the
  cluster" and use `$HPC_USER`, `$HPC_GPU_ACCOUNT`, `$HPC_ROOT`.

## How loaders may read licensed material

Loaders read existing manifests **in place** from paths supplied by
config or `local/site.sh`. They never copy stimuli into the repo.

- RMET: `manifest.json` plus `answer_key/rmet_adult_answer_key.json`.
- EU-Emotions: `eu_emotions_full_manifest.json` plus
  `eu_emotions_human_entropy.json`. Use the **full** manifest, never the
  118-trial subset from an earlier study.
- Mind Reading: item directories.
- Audio: EU UK Voices and Mind Reading item-folder T-files only. Never
  `Emotions/Audio/` (leakage).

Inversion and 2x5 block-scrambling are applied in code from upright
originals. Do not store transformed copies.

## Activations and results

RDS has an inode quota of about 1.05M files. Store **one `.npz` (or
zarr) per model x layer x run**, never one file per item. Write eval
rows incrementally to sharded parquet keyed by item. Only
`results/aggregate/` (small tables and figures) may be committed.

`scripts/sync.sh pull-results` brings back JSON / CSV / parquet and
figures only, never activations or stimuli.

## Enforcement

- `.gitignore` drops `data/`, `local/`, `outputs/`, `cache/`, `.env`,
  weights, media (except `tests/fixtures/synthetic/`), and parquet
  (except `results/aggregate/`).
- `scripts/hooks/forbid_data.py` blocks `data/`, `local/` (except
  `site.example.sh`), media outside fixtures, and files >5 MB.
- `tests/test_no_site_ids.py` greps tracked files for the deny-list in
  `local/site.sh` (skips if that file is absent).
