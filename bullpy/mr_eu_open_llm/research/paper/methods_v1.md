# Methods (draft v1)

Working title: Mechanisms of mental-state ambiguity in VLMs, and their alignment with human confusion structure.

Numbers cited here are only those already in `research/RESULTS_REGISTRY.md`. Do not add estimates.

## Task

EU-Emotion 6AFC, video-only, n = 243, chance 1/6, seed 42. Human norms are the Face column of the O'Reilly workbook. Models: Qwen3-VL-8B-Instruct, Gemma 4 E4B, Molmo2-O-7B.

## Behaviour

Accuracy vs chance: one-sided exact binomial, Wilson 95% CI. Output–human entropy alignment: Spearman ρ with 2,000-item bootstrap CI. H2 is supported if that CI includes 0.

## Phase 2 gate (prereg-v1)

Ridge, α by GCV inside each training fold. Outer loops: leave-one-out and leave-one-actor-out. Standardise on train only. Targets: T1 human entropy (primary), T2 1−p_target, T3 human p_target. Baselines: 1,000 permutations of T1; category-residual T1; stimulus features (actor one-hot, duration, luminance, motion); model-confidence residual (output entropy and p_target).

Branch A if an above-chance model has a layer with T1 grouped permutation p_Holm < 0.01, Δρ vs stimulus ≥ 0.10 with CI excluding 0, and both residual CIs excluding 0. Otherwise Branch B. The 3-layer last-token cache is exploratory and does not lock the branch.

**Locked 2026-11-02: Branch B** (`research/reproduction/phase2_gate_all_layers.json`). All last-token layers, n_perm = 1000. Holm p never fell below 0.01; the permutation floor is n_layers/1001. The rule was not retuned. Phase 4 is omitted. Phase 3 (patching, label-subspace positive control, entropy erasure, full-N ablation) still runs. Results draft: `research/paper/results_v1.md`.

## Entanglement

Report ρ(1−p_target, human entropy). Null: simulate target pairs with that correlation and record probe-direction cosine. Partial RSA residualises RDM upper triangles on an entropy-difference RDM.

## Causal (Phase 3, both branches)

Qwen3-VL first, all 243 items: patching heatmap, label-subspace positive control, human-entropy erasure vs ≥ 20 matched-rank random controls, then the L4 mean-projection ablation on the full set (pilot was n = 36).

## Paper shape under Branch B

Above-chance 6AFC (Qwen, Gemma; Molmo at chance). Output entropy does not track human entropy. Linear probes show large grouped ρ that does not survive Holm at the pre-registered n_perm. Methods: actor-grouped CV, stimulus and residual baselines, entanglement null, difficulty-matched RSA. Generalisation (scale, MRMET, CREMA-D) remains. No output-alignment / steering chapter.
