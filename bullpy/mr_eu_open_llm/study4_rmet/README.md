# Study 4 — CARD-grounded RMET psychometric × mech-interp (PROTOCOL V2)

Self-contained sub-study. **Does not modify study3.**

**Framing:** Do VLMs encode CARD **profile-diagnostic** structure on RMET, or only a **generic ambiguity** route — and is that distinction **causal**?  
Not an RMET accuracy leaderboard; not ToM circuits; not EU-Emotion confusion geometry.

Protocol: `docs/PROTOCOL_V2_CARD_MECH.md` · Decisions: `docs/DECISIONS.md` · Summary: `results/SUMMARY.md`  
Pre-V2 checkpoint: `checkpoints/pre_v2_card_mech_20260806/`

## Isolation guarantees

| Boundary | Rule |
|----------|------|
| Code edits | All study4 work lives under `study4_rmet/` only. |
| Root `scripts/`, `config.py`, `slurm_jobs/study3_*`, root `results/` | **Never edited**. |
| Root `sync.sh` | **Untouched**. Use `study4_rmet/sync.sh`. |
| Imports | Optional read-only parent helpers via `push-repo-readonly`. |

## Quick run (V2 Phases 1–4 CPU)

```bash
./study4_rmet/scripts/run_protocol_v2_phases.sh
```

| Phase | Script | Outputs |
|-------|--------|---------|
| 1 CARD structure | `scripts/build_card_rmet_structure.py` | `results/card_structure/` |
| 2 Behavioural B1/B2 | `scripts/behavioural_profile_alignment.py` | `results/behavioural_v2/` |
| 3 RSA + probes M1 | `scripts/rsa_probe_card_axes.py` | `results/mech/` |
| 4 Causal axes C1 | `scripts/causal_rmet_axes.py` | `results/mech/axis_*.npy` |
| 4b Steer/patch | `scripts/steer_rmet_axes.py` (GPU) | `results/mech/steer_*` |
| Contamination smoke | `scripts/contamination_option_order_smoke.py` | `results/robustness/contamination/` |
| Result 0 (legacy) | prior alignment/robustness | `checkpoints/...` + `results/alignment/` |

## HPC C1 steer (qwen3vl first)

```bash
# after sync + activations + causal_rmet_axes axes exist:
MODEL=qwen3vl LAYER=4 sbatch study4_rmet/slurm_jobs/rmet_causal_steer_smoke.sh
MODE=full MODEL=qwen3vl LAYER=4 sbatch study4_rmet/slurm_jobs/rmet_causal_steer_smoke.sh
```

Steer uses `last_token` and `all_tokens` (distributed-token control). Scores ΔJS toward CARD low-EQ / ASC soft targets; entropy + random axes as controls.

## Legacy evals (still valid)

```bash
python study4_rmet/scripts/human_item_difficulty.py
python study4_rmet/scripts/alignment_analyses.py
./study4_rmet/scripts/run_rmet_api_panel.sh full
# open-weight eval / activations: Slurm under slurm_jobs/
```

For soft-label B1/B2, prefer `--n_samples 20`+ when re-running evals (current full runs use k=10).

## Commercial API arm (behaviour only)

Same 36 eyes-only stimuli + 4AFC prompt as the open-weight eval. No activations.

| Study key | API model | Env var |
|-----------|-----------|---------|
| `gpt5` | `gpt-5` | `OPENAI_API_KEY` |
| `claude_opus` | `claude-opus-4-5` | `ANTHROPIC_API_KEY` |
| `gemini_flash` | `gemini-3-flash-preview` | `GOOGLE_API_KEY` |

Keys from `study4_rmet/.env` (gitignored) or  
`mr_ts_play/experiments/cam_human_like/training/.env`.

```bash
./study4_rmet/scripts/run_rmet_api_panel.sh smoke
./study4_rmet/scripts/run_rmet_api_panel.sh full
./study4_rmet/scripts/run_rmet_api_panel.sh full gpt5
```

Outputs: `study4_rmet/results/model/{gpt5,claude_opus,gemini_flash}/rmet_eval_*_full_seed42.json`

## Robustness layer

```bash
/Users/eb2007/playground/bullpy/mr_ts_play/venv/bin/python -m pytest \
  study4_rmet/scripts/tests/test_card_structure.py study4_rmet/robustness/test_robustness.py -q

/Users/eb2007/playground/bullpy/mr_ts_play/venv/bin/python \
  study4_rmet/robustness/run_robustness_report.py

/Users/eb2007/playground/bullpy/mr_ts_play/venv/bin/python \
  study4_rmet/scripts/contamination_option_order_smoke.py --offline_report
```
