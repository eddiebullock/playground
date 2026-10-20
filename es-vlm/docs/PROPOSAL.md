# Social versus mechanical residual directions in open-weight VLMs

**Status:** one paper. The core (Design A + B, documented Design C
confound, confirmatory steer if non-degenerate, 2B/Gemma transfer
only if 8B moves) is a single robust claim. Bolting SAEs, old RQ3/RQ4,
extra Qwens, or Mind Reading onto it is a second project, not more
impact. Narrative E/S prose is a closed pilot, not a confirmatory
analysis. Execution order is `docs/PLAN.md`.

## One-sentence claim

If Baron-Cohen E–S theory is treated as a prediction about computation
(social inference and mechanistic inference should be separable and
causally distinct), then a VLM should have a residual-stream direction
that (i) is built from doing those tasks, (ii) is not just image-versus-
video or question-text, and (iii) moves held-out social accuracy more
than mechanical accuracy when steered, and the reverse.

## Why the last study is not this paper

Author-written E/S narratives produced a linear leftover in Qwen that
sorted those stories (AUROC ~0.91) and failed every causal test at the
pre-chosen layer (add, ablate, all-token add). Gemma had no leftover.
Personas were not independent. That chapter is a controlled null on a
writing-style construct.

A first task extract (RMET faces vs CLEVRER videos) gave held-out AUROC
1.0 at every layer, including layer 0. That is a modality confound, not
a social/mechanical mechanism. It is a negative control, not a finding.

## What is in the paper

**Construct.** Last-token residual while the model is asked the
question (with media). Say social-inference direction /
mechanical-inference direction. Never “the model is empathic.”

**Design A (primary, same images).** RMET classic (mental state) versus
RMET appearance (what kind of image) on the same 36 faces. Build the
direction on a train split. Also extract both questions with no image
and residualize the image direction against that text-only direction.
If layer-0 AUROC is ~1.0 and the visual residual is ~0.5, the effect
was the wording. If a mid/late visual residual still separates, we
have a candidate mechanism.

**Design B (primary, same modality).** EU-Emotions video versus CLEVRER
video. Faces-in-video vs objects-in-video is still a content confound,
but it is not image-vs-video. Report layer-0 AUROC as a confound check.

**Design C (negative control).** RMET vs CLEVRER (already AUROC 1.0
everywhere). Do not steer it.

**Causal test.** On held-out items only: add and ablate the train
direction at the candidate layer (not the old emotion-axis layer 4;
not a layer picked after seeing steer results). Alphas {−2, −1, 0, 1,
2}. Tasks: held-out RMET and a CLEVRER/PhysBench image shard. Alpha 0
must match unsteered. Predicted: social+ raises RMET more than the
mechanical task; mechanical+ does the reverse. Random direction is a
control.

**Transfer.** If Design A or B is non-degenerate and the 8B steer
moves as predicted: repeat extract+steer on Qwen3-VL-2B and Gemma 4
E4B. Not a Qwen-family sweep.

**Narrative leftover.** Report cosine of the new direction with the
old E/S prose vectors. Predicted: weak alignment if we actually
changed construct.

## What is not in the paper

- Sparse autoencoders (a second paper if a causal direction exists)
- EQ/SQ-R log-prob slopes
- Human item-level RSA
- Qwen2 / 2.5 / 32B
- Mind Reading (no MCQ)
- More narrative add-sweeps

## Why this is one paper and not a fishing trip

One construct (task-conditioned social vs mechanical), two matched
designs plus a documented confound, one causal recipe, two transfer
models if the primary works. Layer is locked from held-out AUROC on
the extract split before any steer is run.

## Risks

- n=36 RMET is small; appearance control uses the same n. Treat
  AUROC CIs as wide. EU helps Design B sample size.
- Appearance options are author-written, not a published foil set.
- EU accuracy is near floor; residuals can still differ.
- A real visual residual that does not steer is another
  representation-without-causal-role result, which is still the paper.
