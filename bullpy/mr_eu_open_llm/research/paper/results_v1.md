# Results (draft v1)

Working title: Mechanisms of mental-state ambiguity in VLMs, and their alignment with human confusion structure.

Every number below is in `research/RESULTS_REGISTRY.md` or the named dump. Do not add estimates. Official Phase 2 lock: **Branch B**, 2026-11-02.

## Behaviour

EU-Emotion 6AFC, video-only, n = 243, chance 1/6, seed 42.

| model | correct | accuracy | Wilson 95% CI | p vs chance (greater) |
|---|---|---|---|---|
| Qwen3-VL-8B | 92/243 | 0.379 | [0.320, 0.441] | 2.3e-15 |
| Gemma 4 E4B | 76/243 | 0.313 | [0.258, 0.374] | 1.5e-8 |
| Molmo2-O-7B | 48/243 | 0.198 | [0.152, 0.252] | 0.116 |

Qwen and Gemma are above chance. Molmo is not; Molmo mechanistic numbers are completeness only.

Output entropy does not track human Face-column entropy (H2). Spearman ρ with 2,000-item bootstrap CIs: Qwen −0.001 [−0.121, 0.120]; Gemma +0.117 [−0.013, 0.235]; Molmo −0.034 [−0.154, 0.097]. All three CIs include 0. File: `research/reproduction/behaviour_summary.json`.

## Official Phase 2 gate (all last-token layers)

Extracts 39709580–82 (Qwen 36/36, Gemma 42/42, Molmo 32/32 npy, 0 errors, max_frames = 4). Probes 39712190–92. Nested Ridge, leave-one-actor-out, n_perm = 1000, n_boot = 2000, seed 42. Stimulus baseline: actor one-hot + duration / luminance / motion, grouped T1 ρ = 0.138.

Dump: `research/reproduction/phase2_gate_all_layers.json`. Layer table: `research/reproduction/phase2_all_layers_summary.csv`. Figure: `research/figures/phase2_grouped_t1_rho.png`.

Peak grouped T1 (human entropy) at last token:

| model | layer | ρ | 95% CI | p_perm | p_Holm | Δρ vs stim (CI) | category-residual ρ (CI) | confidence-residual ρ (CI) |
|---|---|---|---|---|---|---|---|---|
| Qwen | 34 | 0.471 | [0.356, 0.568] | 0.001 | 0.036 | 0.333 [0.174, 0.490] | −0.218 [−0.336, −0.094] | 0.398 [0.278, 0.507] |
| Gemma | 2 | 0.492 | [0.384, 0.584] | 0.001 | 0.042 | 0.354 [0.197, 0.515] | −0.116 [−0.238, 0.012] | 0.322 [0.205, 0.429] |
| Molmo | 22 | 0.518 | [0.410, 0.610] | 0.001 | 0.032 | 0.380 [0.228, 0.530] | −0.046 [−0.171, 0.082] | 0.458 [0.339, 0.557] |

Zero layers pass the pre-registered conjunction. Eligible models = []. **Branch B.**

Qwen L34 fails only Holm: Δρ, Δρ CI, category-residual CI, and confidence-residual CI all pass. Twenty of 36 Qwen layers have that same pattern. Gemma L2 also fails the category-residual CI (includes 0). Molmo is below chance, so it cannot decide the branch.

Holm floors are n_layers/1001 (0.036 / 0.042 / 0.032), all > 0.01, because minimum permutation p is 1/1001. The rule was not retuned and n_perm was not raised after seeing the all-layer result. The 3-layer would-be A (`phase2_gate.json`) remains exploratory.

Leave-one-out ρ exceeds grouped ρ at every peak (Qwen 0.529 vs 0.471; Gemma 0.562 vs 0.492; Molmo 0.578 vs 0.518). Actor leakage is real; grouped CV is the gate statistic.

Category-residual ρ is negative at the peaks. The pre-registered check is two-sided “CI excludes 0”. The positive T1 signal is largely between emotion categories.

## Entanglement (H4)

Spearman ρ(1 − p_target, human entropy) = 0.954 on the 243 items. Probe-direction cosine of entropy vs 1−p_target axes sits *below* a Gaussian null matched to that correlation (Qwen L4 observed 0.808 vs null 0.955). Partial RSA controlling an entropy-difference RDM stays near the raw RSA. File: `research/reproduction/phase5_entanglement.json`. T2 is not an independent discovery.

## Causal (Phase 3, still running)

Pilot n = 36, Qwen L4 mean-projection: Δacc = 0, Δconfusion = 0, 5/180 flips, non-selective. Full-N ablation (jobs 39709579, 39709584) is queued, not cited. Phase 4 is off under Branch B.

## Reading

VLMs are above chance at 6AFC mental-state recognition (Qwen, Gemma) and internally linearly encode which items humans find ambiguous (large grouped ρ, stimulus-beating). That information does not survive the pre-registered Holm gate at n_perm = 1000, and it does not appear in output entropy. The paper is a mechanistic map plus entanglement methods plus generalisation, not an output-alignment claim.
