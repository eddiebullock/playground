# Plan

Live program: `docs/PUBLIC_PAPERS.md` (papers 1 and 2 in parallel).
The licensed E–S / Design A–C chapter is closed below. Do not add
SAEs, EQ/SQ-R slopes, human RSA, extra Qwens, or Mind Reading.

## Live — public items

### Paper 1 (NExT-QA same scene)

1. Fetch val/train CSVs. Pair CW vs DO/DC/DL. No videos in this job.
   Locked: 461 pairs. CPU check: `slurm/check_public.sbatch`.
2. Text extract `DESIGN=same_scene_nextqa` with `--allow-missing-media`.
   This is the wording control. GPU jobs wait on the low-priority QOS.
3. Fetch `{video}.mp4` only after the pair list is locked. Then visual
   extract without `--allow-missing-media`.
4. Steer hold pairs only if `steer_ok` and not `wording_only`.

### Paper 2 (MMToM text control)

1. Fetch `questions.json`. No 257GB video zip. Locked: 600 items.
2. Unsteered 8B score on MMToM text (`PACK=mmtom_text`).
3. Extract `DESIGN=mmtom_vs_piqa` (text vs PIQA).
4. Video ToM stage only if the text leftover is incomplete.

### Stop rules (both)

- Two failed fixes on the same cluster problem, then stop.
- Never write into study2 / study3 / study4_rmet.
- Never commit licensed stimuli, human data, keys, or site IDs.
- Do not re-run narrative add-sweeps or steer Design A/B/C.

## Closed — narrative and licensed-task chapter

Task-conditioned social vs mechanical residual directions in a VLM,
with matched-modality controls, a same-image question control, a
documented cross-modal confound, and a confirmatory steer on held-out
items if the extract is non-degenerate. Transfer is 2B + Gemma only if
8B works. Narrative E/S prose is a closed pilot, cited not re-run.

### Already closed

- Narrative extract + personas: linear leftover in Qwen, chance in
  Gemma, personas not independent.
- Narrative causal at 8B L27: add hurts RMET on every vector; ablate
  is a no-op; position=all is a generic CLEVRER shove.
- Design C (RMET faces vs CLEVRER videos), job 38807101: hold-out
  AUROC 1.0 at every layer including 0. Confound check. Do not steer.

### Gate 1 — 8B extract (closed, job 38817548)

1. Design A: wording only. Raw and text-only AUROC 1.0; unique vs
   text ~0; leftover AUROC chance except a last-layer spike on n=11.
   Do not steer.
2. Design B: layer-0 AUROC 1.0. Content/modality confound. Do not
   steer.

### Gate 2 — 8B causal (skipped)

Would have run only if a design had `steer_ok=true` and not
`wording_only`. Neither did.

### Gate 3 — transfer (skipped)

8B extract did not produce a steerable direction. Do not run 2B or
Gemma on those designs.
