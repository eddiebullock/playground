# Discussion (draft v1)

**Superseded 2026-11-04;** see `PREREGISTRATION.md` amendment A1.

Numbers only from `research/RESULTS_REGISTRY.md`. Causal full-N results are not in yet; do not preview them.

## What the gate decided

The pre-registered conjunction failed because permutation Holm cannot reach 0.01 when the family is 32–42 layers and n_perm = 1000. That is a resolution floor, not evidence that the probe ρ is zero. Peak grouped T1 ρ is 0.47 (Qwen L34) and 0.49 (Gemma L2), both with raw p_perm = 0.001 and Δρ vs stimulus > 0.33. Twenty Qwen layers fail *only* Holm. Reporting Branch B is the honest reading of the frozen rule. Raising n_perm or switching to BH-FDR after seeing the map would be a silent rewrite of H1.

The 3-layer would-be A (Qwen L26 p_Holm = 0.003) is exploratory. Holm over three tests is more lenient than Holm over 36.

## What still looks real

Above-chance 6AFC is not in doubt for Qwen and Gemma. H2 holds: output entropy does not track human entropy. Actor-grouped CV is the right probe statistic; LOO is larger at every peak, so identity leakage would have inflated an ungrouped gate. Category-residual ρ is negative, so the T1 signal is largely between emotion categories rather than within them. That is compatible with “the model represents how typical / how intense this acted state is,” and is weaker evidence for item-level human confusion structure *inside* a label.

H4: 1 − p_target and human entropy are almost the same target (ρ = 0.954). Probe-direction cosine sits below a Gaussian null matched to that correlation. Difficulty-matched RSA and partial RSA stay near zero in the 3-layer cache. Pair RSA without a difficulty control would have been a methods error.

Molmo’s probe ρ is large and Molmo is at chance. Latent decodability without recognition is not H1.

## What Phase 3 has to do

The n = 36 L4 ablation is non-selective (Δacc = 0, 5/180 flips). Full-N ablation is queued. Under Branch B the causal chapter is a map, not a steering claim: label-subspace positive control, entropy-subspace erasure vs matched-rank random controls, and a layer × position patching heatmap. If full-N ablation remains null, the paper says so.

## Limits

Holm floor; last-token / max_frames = 4 extracts; Face-column human norms rather than a new human sample; public repo, so activations stay off git; CREMA-D and MRMET not run; option-order GPU not yet queued; we did not collect human data.
