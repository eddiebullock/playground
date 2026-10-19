# es-vlm

Empathizing (E) and systemizing (S) representations inside open-weight
vision-language models. Questionnaires generate hypotheses; they are not
treated as valid trait measures of models.

This folder is a self-contained package inside `eddiebullock/playground`.
It has its own `pyproject.toml` and is not a nested git repo.

## What this tests

- **RQ1 Separability.** E vs S vs valence / emotion / helpfulness vs a
  matched-random null; E-S correlation across simulated personas.
- **RQ2 Double dissociation (core).** Direction x task-domain interaction
  under steering and ablation, swept over layers.
- **RQ3 Self-report vs competence.** Standardised slopes of questionnaire
  log-prob scores vs task accuracy across alpha.
- **RQ4 Human alignment.** RSA / Jensen-Shannon of item-level errors vs
  human EQ / SQ-R tertiles (Mac only; the workbook never leaves this
  machine).

Say "E-direction / S-direction". Never label a model or condition
"autistic" or "low-empathy".

## Layout

See `docs/` for the proposal placeholder, data governance, cluster notes,
preregistration draft, and dated decisions. Site-specific values live in
git-ignored `local/site.sh` (copy `local/site.example.sh`).

## Environments

Two Transformers stacks, each a uv project that installs `esvlm` editable:

| Profile | Path | Transformers | Models |
| --- | --- | --- | --- |
| env-main | `envs/main/` | 5.x | Qwen3-VL, Gemma 4, Qwen2.5-VL, Qwen3 text |
| env-molmo | `envs/molmo/` | 4.57 | Molmo2 only |

Mac (from this folder):

```bash
unset VIRTUAL_ENV
uv sync --project envs/main --extra dev
uv run --project envs/main pytest
```

Cluster jobs must go through `scripts/submit.sh`. Do not call `sbatch`
directly. Every launcher unsets `VIRTUAL_ENV` before activating an env.

## Licensed data

RMET, Multiracial RMET, Mind Reading, EU-Emotions faces/voices, EQ-60 / SQ-R
item text, and all human item-level data stay out of git. Loaders read
existing manifests in place. See `docs/DATA_GOVERNANCE.md`.

## Checkpoints

Steering sweeps do not start until the OSF preregistration is posted.
Prior single-layer pilots at Qwen3-VL layer 4 were null (EU-Emotions job
33776847; RMET job 33536702). This project sweeps many layers with
directions derived from E/S contrasts, not emotion-confusion axes.
