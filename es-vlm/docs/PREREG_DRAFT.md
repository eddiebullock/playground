# Preregistration draft

**Status:** scaffold. Unsteered Phase 1 baselines may run without this
being posted. Do not start confirmatory steering sweeps until the OSF
preregistration is posted.

## Title

Empathizing and systemizing representations in open-weight
vision-language models: separability, double dissociation, and human
item-level alignment.

## Hypotheses

1. Residual-stream E-direction and S-direction are each the difference
   of means versus a shared control set (not versus each other). They
   are linearly separable from each other and from valence, emotion,
   helpfulness, and a matched-random null (held-out projection AUROC;
   200-split null). The bipolar `E_minus_S` pair is a named one-axis
   readout, not the definition of E or S.
2. Across >=500 simulated personas the E and S scores are largely
   independent and at most weakly negatively correlated.
3. Steering / ablating the E-direction changes visual mental-state
   inference (RMET, Mind Reading, EU-Emotions / social video) more than
   physical / mechanical reasoning. The S-direction does the reverse
   (direction x domain x alpha interaction).
4. Questionnaire log-prob shifts and competence shifts can dissociate
   (RQ3). Questionnaire scores are hypotheses, not trait measures.
5. Model item-level errors will be compared to human EQ / SQ-R tertiles
   by RSA and Jensen-Shannon divergence (RQ4; Mac only).

## Analysis (locked intent)

- RQ1: E and S are each vs a shared control. Report held-out projection
  AUROC vs control, vs each other (`E_minus_S`), vs valence / emotion /
  helpfulness, and a 200-split matched-random null. Also report
  `||E_⊥S|| / ||E||` and persona `r(E, S)` / `r(E_⊥S, S_⊥E)` across
  500 mundane first-person sketches (not questionnaire items).
- RQ2: `correct ~ direction * domain * alpha + (1|item) + (1|template)`
  with item-bootstrap CIs. Always report generic-VQA accuracy and
  perplexity per alpha.
- RQ3: standardised slopes with bootstrap CIs.
- RQ4: RSA + JSD vs human tertiles. The human workbook stays on the Mac.
- Fairness: accuracy by face ethnicity x gender.

## S-domain tasks

- Primary confirmatory visual S-set: PhysBench dynamics / relationships,
  text options only.
- Optional robustness, not required for the confirmatory contrast:
  CLEVRER validation explanatory / predictive / counterfactual MCQ
  (video), and PIQA validation as a text-only backbone control.

## Exclusions

Molmo2-O-7B is not in the confirmatory panel (infra/processor failure
during setup; no E/S scores exist for it).

Single-layer emotion-confusion steering is not a confirmatory analysis.
Prior nulls at Qwen3-VL layer 4 (jobs 33776847, 33536702) are cited as
the reason for the layer sweep, not as a failed replication target.
