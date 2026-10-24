# Decisions log

Dated, with the evidence. Newest last.

## 2026-10-24

- **New repo `~/playground/gloss`**, separate from `social_ai` (concept
  scenes) and `es-vlm`. It is its own git repo, so the playground branch
  `es-vlm/setup` and its uncommitted changes are untouched.
- **Order of work: language first, reflective first, faces last.** Evidence:
  `docs/research/01_landscape.md` sections 1, 4 and 5 and
  `03_study_spinout.md` section 2. EU AI Act Art. 5(1)(f) covers face and
  voice only. Barrett 2019. The community critiques. No reflective or
  calibrated tool exists. Pending Eddie's agreement and the focus-group
  transcripts.
- **Local-only inference for participant data.** Free cloud tiers either
  exclude UK/EEA apps or allow human review of inputs (`02_tech.md` section 5).
  No paid APIs (standing rule).
- **Confidence comes from first-token option probabilities averaged over
  rotated orders**, not from verbalised confidence. This is the same mechanism
  E1 measures, so the product's ambiguity rating has a measured calibration.
- **The literal or sincere reading is always a candidate.** This guards
  against over-reading and against reinforcing anxious interpretations (the
  sycophancy risk in Ma et al. 2026).
- **E1 result, qwen3:4b direct (Hu et al. items, 169 x 5 orders):** accuracy
  0.548 [0.514, 0.580] vs human 0.863. Confidence barely predicts errors
  (AUROC 0.594, ECE 0.237). 182 of 382 errors (48%) are literal readings.
  Published references on the same items: text-davinci-002 0.697,
  flan-t5-xxl 0.580. **Implication: a 4B on-device model is not good enough
  to show readings with confidence. Study 5 needs a stronger model, or the
  product must lean on abstention.** E2 (8B) is blocked by disk: 5.3 GB free,
  and other sessions are writing.
- **Provisional thresholds:** "clear" when the top reading has p >= 0.75,
  abstain below 0.5. To be re-set from the E1/E2 selective-accuracy curves.
