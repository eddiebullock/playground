# Introduction (draft v1)

Working title: Mechanisms of mental-state ambiguity in VLMs, and their alignment with human confusion structure.

Numbers live in `research/RESULTS_REGISTRY.md` and `research/paper/results_v1.md`. Do not add estimates.

Vision-language models can name acted mental states from silent video above chance, but it is an open question whether they encode *which items humans find ambiguous*, and whether that information reaches the output. The working hypothesis is that they do encode it internally, and that it does not survive decoding into choice entropy.

We test this on EU-Emotion 6AFC, video-only, n = 243, chance 1/6. Human norms are the Face column of the O'Reilly workbook. Models: Qwen3-VL-8B, Gemma 4 E4B, Molmo2-O-7B. Pre-registration (`prereg-v1`, 2026-10-30) froze a confirmatory probe gate before the all-layer extract. Branch A required a layer with actor-grouped T1 permutation p_Holm < 0.01 plus stimulus and residual baselines. Otherwise Branch B: above-chance but non-human-like recognition, with a mechanistic map, entanglement methods, and generalisation. Phase 4 readout/steering is Branch A only.

The official all-layer gate locked Branch B on 2026-11-02. Qwen and Gemma are above chance; Molmo is not. Output entropy CIs vs human entropy all include 0. Linear probes of last-token residuals give large grouped ρ (peaks 0.47 / 0.49) that beat a stimulus baseline, but Holm floors at n_perm = 1000 sit above 0.01. The rule was not retuned. Pair-specific confusability and generic difficulty are entangled (ρ(1 − p_target, human entropy) = 0.954), so T2 is not an independent discovery.

What remains is the causal map (Phase 3, both branches) and generalisation (scale, MRMET, CREMA-D).
