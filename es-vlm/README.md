# es-vlm

Whether open-weight VLMs have separable, causally usable
social-inference and mechanical-inference residual directions
(Baron-Cohen E–S as a computational prediction, not a trait label).

This folder is a self-contained package inside `eddiebullock/playground`.
It has its own `pyproject.toml` and is not a nested git repo.

## What this tests

One paper, see `docs/PROPOSAL.md` and `docs/PLAN.md`.

- **Construct.** Last-token residual while the model is asked the task.
  Say social-inference / mechanical-inference direction. Never "the
  model is empathic", "autistic", or "low-empathy".
- **Design A.** Same RMET faces: mental-state vs appearance, then
  residualize against text-only. If the leftover is chance, it was
  wording.
- **Design B.** EU-Emotions video vs CLEVRER video. Layer-0 AUROC is
  the confound check.
- **Design C (closed).** RMET vs CLEVRER is image-vs-video (AUROC 1.0
  including layer 0). Do not steer it.
- **Causal.** Steer the train direction on hold items at a layer locked
  from the extract. Only if the extract is non-degenerate.
- **Transfer.** 8B first, then 2B and Gemma only if 8B steering works.

Narrative E/S prose, personas, EQ/SQ-R slopes, human RSA, SAEs, and
extra Qwens are not this paper. The prose chapter is a closed causal
null (writing-style leftover, no dissociation).

Questionnaires generate hypotheses; they are not trait measures of
models.

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

OSF preregistration is not part of this project. Prior single-layer
pilots at Qwen3-VL layer 4 were null (EU-Emotions job
33776847; RMET job 33536702). This project sweeps many layers with
directions derived from E/S contrasts, not emotion-confusion axes.
