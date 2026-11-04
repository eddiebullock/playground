# Preregistration v1 — Mental-state ambiguity in VLMs

**Frozen:** 2026-10-30 (Europe/London), before the Phase 2 gate rerun.
**Tag:** `prereg-v1` (commit this file, then tag; do not edit hypotheses after seeing Phase 2 results).
**Pilot data:** EU-Emotion 6AFC video-only, n = 243, seed 42, models Qwen3-VL-8B, Gemma 4 E4B, Molmo2-O-7B. Pilot numbers are archival. Confirmatory analyses are those specified below.

Deviations after freeze go in a dated **Deviations** section at the end. Do not silently rewrite this document.

---

## Hypotheses

**H1 (primary, latent human-uncertainty knowledge).** For at least one model with above-chance 6AFC accuracy, human item-level response entropy is linearly decodable from LM residual-stream activations beyond permutation, category, stimulus/input, and model-confidence baselines (decision rule below).

**H2.** Model *output* entropy does not track human entropy (Spearman ρ CI includes 0, or |ρ| is below the human split-half noise ceiling by a wide margin). This is already the pilot pattern; confirmatory recompute uses the same eval JSON plus bootstrap CIs.

**H3 (causal, both branches).** Decision-relevant sites exist in the residual stream. Erasing a *label* subspace at those sites drops accuracy (positive control). Erasing a human-entropy subspace is then interpretable.

**H4 (methods).** Pair-specific confusion and generic difficulty are entangled. Human–model confusion RSA must be difficulty-matched / partialled.

**Branch A** if H1’s gate passes. Then test whether a held-out entropy probe can be read out or steered into output alignment (Phase 4).

**Branch B** if H1’s gate fails. Paper becomes: above-chance but non-human-like mental-state recognition, with a mechanistic map, entanglement methods, and generalisation.

---

## Primary metrics

- 6AFC accuracy vs chance 1/6, one-sided exact binomial, Wilson 95% CI.
- Spearman ρ(model forced-choice entropy, human entropy), bootstrap 95% CI (≥ 2,000 item resamples).
- Probe: Spearman ρ(held-out predictions, target), nested-CV Ridge, bootstrap 95% CI.
- RSA: Spearman of upper triangles, permutation p over item labels.
- Causal: Δaccuracy, Δoutput entropy, Δρ(model, human entropy), item flips vs matched-rank random subspaces; paired bootstrap.

---

## Phase 2 gate (confirmatory)

**Protocol (all probes).** Ridge; α chosen by nested CV inside each training fold only. Outer loop: leave-one-out **and** grouped CV by actor letter parsed from the EU filename. Standardise features on training folds only. Run every cached layer × token position.

**Targets.** T1 human entropy (primary). T2 1 − p_target (secondary, near-circular). T3 human p_correct / consensus.

**Required baselines.** (1) Permutation of T1, ≥ 1,000 shuffles, permutation p. (2) Category: residualise T1 on emotion label; within-category pooling. (3) Input/stimulus: vision-encoder or layer-0/embedding if cached; else a hand-built set (duration, mean luminance, motion energy, actor one-hot). (4) Model-confidence: residualise T1 on model output entropy and p_target.

**Decision rule — Branch A** if, for at least one *above-chance* model, some layer satisfies all of:

1. T1 ρ permutation p < 0.01, Holm-corrected across layers of that model.
2. T1 ρ exceeds the best input/stimulus baseline by Δρ ≥ 0.10, and the bootstrap CI of Δρ excludes 0.
3. Category-residualised T1 ρ has a CI excluding 0.
4. Model-confidence-residualised T1 ρ has a CI excluding 0.

Otherwise **Branch B**. Log in `DECISIONS.md`. Do not retune the rule.

Pilot 3-layer last-token probes are **exploratory**. They may be reported as such but do not decide the branch.

**Molmo2** is not above chance (48/243, p = 0.116). Its probes are completeness checks, not H1 evidence.

---

## Multiple comparisons

- Holm–Bonferroni within each confirmatory family (layers of one model for T1; models for accuracy vs chance).
- BH-FDR for large exploratory maps (patching heatmaps, all-pair RSA).
- Bored–Unfriendly remains **exploratory** unless it replicates on a held-out dataset (MRMET or CREMA-D) or a pre-registered held-out EU split (none reserved; so only other datasets can promote it).

---

## Causal plan (Phase 3, both branches)

Start with Qwen3-VL, all 243 items.

1. Activation patching layer × position heatmap; FDR; noise-corruption baseline.
2. Positive control at decision sites: ablate/erase the correct-label direction; accuracy must drop.
3. Human-entropy erasure (LEACE and rank-k, k = 1,2,4,8,16) at decision sites and at the best Phase 2 layer; ≥ 20 matched-rank random controls; re-probe after erasure.
4. Repeat the original L4 last-token mean-projection ablation on **all 243** items (pilot was n = 36).
5. Pair axes: exploratory unless this document is amended *before* looking.

Port to Gemma 4 if compute allows. Molmo2 flagged at chance.

---

## Phase 4 (Branch A only)

Held-out entropy probe → temperature / mixture on the output. Baselines: global temperature; logit-only readout; stimulus-feature readout. Steering: α × probe direction vs ≥ 20 random directions.

**Success:** held-out ρ(model entropy, human entropy) beats all baselines, CI excludes 0, accuracy cost < 5 points.

---

## Entanglement (Phase 5)

Report correlation of the *targets* 1−p_target and human entropy. Null: simulate target pairs with that correlation, train probes, expected cosine of directions. Partial RSA controlling a difficulty RDM. Orthogonalise pair axes against the entropy subspace.

---

## Generalisation (Phase 6)

- Scale: ≥ 3 sizes of Qwen3-VL, behaviour + Phase 2 gate; pin HF revisions.
- MRMET: replication, N = 37, directional predictions only. Entropy/RSA N/A if option counts are missing.
- CREMA-D video-only subset, ~600–1,000 clips, grouped CV by actor. Human split-half reliability as noise ceiling.
- EU-Emotion human norms: Face column of the O'Reilly workbook (already verified). State as Face/video-only, not a new claim.

---

## Exploratory (not used for branch decision)

- Difficulty-matched pair RSA, including Bored–Unfriendly.
- n = 36 L4 ablation (already run: Δacc = 0, 5/180 flips, non-selective).
- Steering pilot.
- Option-order robustness (will be reported; if accuracy/entropy move a lot, entropy analyses get a caveat).

---

## Seeds and code

Seed 42 unless a procedure specifies otherwise. Bootstrap ≥ 2,000 item resamples. Nested CV RNG derived from seed 42.

---

## Deviations

### 2026-11-02 — Holm floor with n_perm = 1000 and all layers

The gate used the frozen rule (Holm p < 0.01 across layers of one model; ≥ 1,000 T1 shuffles). All-layer extracts have 36 / 42 / 32 last-token layers. Minimum permutation p is 1/1001, so the Holm floor is n_layers/1001 (0.036 / 0.042 / 0.032), all above 0.01. No layer can pass check 1 at this n_perm. The rule was **not** retuned and n_perm was **not** increased after seeing the result. Branch B is locked. The floor is a limitation, not a silent rewrite of H1.

### 2026-11-04 — Amendment A1: prompt confound found; rotated 6AFC with vision-mismatch control

**Written before any rotated data exists.** Post hoc audit (`research/scripts/audit_prompt_confound.py`, `research/reproduction/prompt_confound_audit.json`) found two defects in the frozen task. The Phase 2 lock and H1/H2 stand as a record of what was run, but their interpretation changes.

1. `human_options` lists the target first on 243/243 items and evaluation used that order. Accuracy therefore equals P(chose option 1), so above-chance accuracy and first-position bias cannot be separated.
2. One option set per target label, so the prompt identifies the category. The stimulus baseline had no text features. Category alone gives grouped T1 rho = 0.503, which matches the peak probes. Last-token activations decode the 21 categories at 0.96 to 1.00 under leave-one-actor-out, which only prompt text can supply.

**Confirmatory task.** Rotate the five named labels cyclically (r = 0..4) with None of the above last, so each target occupies each named position exactly once. Per model (Qwen3-VL, Gemma 4, Molmo2) run each rotation twice: `real` (the item's video) and `mismatch` (the same prompt with a partner item's video, different target label, different actor; `vision_mismatch_partners`, seed 42). Same 20 draws, frames and items as the original.

**Primary estimand.** For item i, a_i = mean correct over 5 real rotations; b_i = the same for mismatch. delta = mean(a_i - b_i): accuracy that requires the matching face.

**Decision.** A model is *vision-dependent* if the one-sided actor-level sign-flip p (100,000 draws) has Holm p < 0.05 across the three models and the actor-bootstrap 95% CI (10,000 draws) of delta excludes 0 from above. No new mental-state claim is made for a model that fails, whatever its accuracy versus 1/6.

**Secondary.** Pooled accuracy vs 1/6; accuracy by target position; Spearman rho and within-category r of model p_target with human p_target. Implemented in `research/scripts/analyse_rotation.py`; analysis code is committed before the data.

**Limits.** Human norms came from one fixed presentation order (not confirmed from the source), so human position effects are unknown. Rotations do not remove defect 2: label sets still identify category, so probes need a separate fix.

**Phase 3 consequence.** Last-token ablation of Phase 2 axes is not requeued. 39709579 and 39709584 were cancelled on 2026-11-02 before running, so no full-N result exists.

### 2026-11-04 — Amendment A1.1: decoder fault; A1 restarted from scratch

**Written before any A1 result was read.** cv2 4.13.0.92, installed in the study environment on 2026-03-16, fails colour conversion on clips flagged interlaced and returns junk frames. Census on first frames: 49 of 243 clips unstable between reads in the conda environment and 51 in the Molmo venv; 0 under cv2 4.11. On a test clip through the unmodified loader, sampled frames were near-black or a stale buffer, and differed between calls. Those trials showed no real face.

**Deviation.** All A1 runs are discarded and rerun. Completed files are moved unread to `quarantine_cv413/` beside each model's results. Design, estimand, decision rule, partners and analysis code are unchanged. Only the decoder changes: cv2 4.11 is pinned ahead of the environment's copy.

**Guard.** `research/scripts/decoder_preflight.py` runs first in every job and stops it unless cv2 is 4.11 and a known interlaced clip decodes stably and is not blank.

**Quarantined files.** They may be read only after the A1 result is locked, solely to measure how far the fault changed results, and that comparison is exploratory.

**Earlier results.** Every run in this environment since 2026-03-16 used the faulty decoder, including the original 6AFC baselines, activations and probes, and vision gates. Results that depend on image content are unreliable on about 20% of items and are not to be cited. Prompt-based audit findings do not depend on decoding: target position, one option set per label, and category-only baselines.

