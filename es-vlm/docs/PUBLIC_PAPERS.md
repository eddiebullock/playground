# Public-item papers 1 and 2

**Status:** live program. Two papers, one stack, run in parallel.
The licensed E–S chapter (`docs/PROPOSAL.md`) is closed. Do not
steer Design A/B/C. Do not add narrative sweeps, extra Qwens, or
SAEs on dead leftovers.

Say social-inference / mechanical-inference / belief-inference
direction. Never label a model or condition autistic or low-empathy.

## Why RMET appearance is not Paper 1

Design A (same RMET faces, mental-state vs appearance) already ran
and died as wording. It is not the public same-scene design.

1. **Cannot release.** RMET faces and booklet wording are licensed.
   A highly cited benchmark has to be redistributable. These items
   cannot be the core of that paper.
2. **n=36, hold 11.** A last-layer leftover spike on eleven items is
   not a mechanism.
3. **Author-written foil.** "photograph / line drawing / sculpture /
   medical scan" is not a published physical VQA item. The social
   side is a real test; the mechanical side is ours.
4. **Syntax mismatch.** "what is the person thinking or feeling"
   versus "what kind of image is this" is a different sentence
   frame, not a different demand on the same scene. Text-only AUROC
   was 1.0 at every layer; unique leftover versus text was ~0;
   `wording_only=true`.

CLEVRER-as-social is also out: those clips have no social demand.
EU vs CLEVRER and RMET vs CLEVRER are modality or content confounds
(layer-0 AUROC 1.0). Keep them as the closed negative-control
chapter.

## Paper 1 — same-scene social vs physical VQA

**Claim.** If a VLM has a residual direction that distinguishes
social inference from physical description, it should appear when
the **same video** is asked a published causal-why question and a
published descriptive question, and that leftover should survive
residualizing against the text-only pair.

**Items.** NExT-QA validation CSVs (MIT). One pair per video.
Locked from the published `val.csv` (4996 trials): **461 pairs**
(physical side: 228 DO, 108 DC, 125 DL).

- Social = type `CW` only. `CH` is mixed and is skipped.
- Physical = `DO`, then `DC`, then `DL`. Lowest `qid` wins ties.
- Questions and options are the published strings. Do not rewrite.
- Videos are `{video}.mp4`, fetched later. Missing mp4 loads as
  `media_type=none` plus `meta.media_rel`. The text-only extract
  (`--allow-missing-media`) is the wording control and can run
  before videos exist.

**Design `same_scene_nextqa`.** Paired last-token residuals. Train
direction on a split; projection AUROC on hold. Residualize against
the same questions with no video. Gates are the existing ones:
`wording_only` if text AUROC is ~1 and unique leftover is ~0;
`steer_ok` only if a mid/late leftover survives.

**Causal.** Only if `steer_ok`. Add/ablate the train direction on
hold videos at the extract-locked layer. Predicted: social+ raises
CW more than the descriptive side; physical+ does the reverse.

**Why this can be cited.** Same-video pairing is the control Design
A pretended to be. The items are public. Other labs can rerun the
pairer on `val.csv`.

## Paper 2 — why VLMs fail visual ToM

**Claim.** Published visual theory-of-mind failures may be text
shortcuts. Isolate the text control first. If MMToM-QA questions
already separate from physical commonsense in the residual stream,
the famous video gap is not a purely visual ToM failure.

**Items.** MMToM-QA `questions.json` (MIT, 600 lines). Media is
empty. Official `answer_list` is `["a","b"]` labels; parse the
`(a)` / `(b)` statements from the question. Do **not** download the
257GB video zip. Do not require Ego4D.

**Design `mmtom_vs_piqa`.** Unpaired. MMToM text vs PIQA. Both
sides are text. Layer-0 AUROC is a topic/wording check, not a
visual claim. Unsteered MMToM accuracy is the behavioral baseline.

**Later, only if the text leftover is incomplete.** Video MMToM
versus a matched physical video set. That is a second stage of
this paper, not a reason to fetch 257GB now.

**Why this can be cited.** It is a sharp account of a result people
already quote (VLMs fail visual ToM), with a public text control
anyone can rerun.

## Parallel execution

| Stage | Paper 1 | Paper 2 |
| --- | --- | --- |
| Now | Fetch NExT-QA CSVs; pair val; text extract with missing media | Fetch MMToM questions; score text; extract vs PIQA |
| Next | Fetch NExT-QA videos only; visual extract | Video stage only if text leftover is incomplete |
| Causal | Steer hold pairs if `steer_ok` | Steer only if a non-text leftover exists |

Shared stack: `esvlm-extract-task`, `esvlm-score`, `esvlm-steer`,
confirmatory 8B first. Transfer (2B / Gemma) only if 8B moves.

## Out of both papers

Narrative E/S add-sweeps. Extra Qwens as robustness. SAEs on
collinear leftovers. CLEVRER as a social task. Steering Design
A/B/C. Licensed RMET / EU / Mind Reading as the reusable
benchmark. The MMToM video archive until the text control is
done.
