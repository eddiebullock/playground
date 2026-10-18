# Preregistration draft

**Status:** scaffold. Expand during Milestone E. Do not start steering
sweeps until the OSF preregistration is posted.

## Title

Empathizing and systemizing representations in open-weight
vision-language models: separability, double dissociation, and human
item-level alignment.

## Hypotheses

1. Residual-stream E-direction and S-direction are linearly separable
   from each other and from valence, emotion, helpfulness, and a
   matched-random null (held-out projection AUROC; 200-split null).
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

- RQ2: `correct ~ direction * domain * alpha + (1|item) + (1|template)`
  with item-bootstrap CIs. Always report generic-VQA accuracy and
  perplexity per alpha.
- RQ3: standardised slopes with bootstrap CIs.
- RQ4: RSA + JSD vs human tertiles. The human workbook stays on the Mac.
- Fairness: accuracy by face ethnicity x gender.

## Exclusions

Single-layer emotion-confusion steering is not a confirmatory analysis.
Prior nulls at Qwen3-VL layer 4 (jobs 33776847, 33536702) are cited as
the reason for the layer sweep, not as a failed replication target.
