# REPRODUCTION (Phase 0)

Reproduced 2026-10-30 Europe/London from cached files on this Mac. Script: `research/scripts/reproduce_headline_numbers.py`. Machine-readable dump: `research/reproduction/phase0_numbers.json`.

No model forwards. Independent recomputes use `scipy.stats.binomtest` and `spearmanr` on stored trial-level fields, plus cosine of saved axis vectors.

## Verdict

The project is the right one. Behavioural accuracies, binomial tests, entropy ρ, consensus terciles, probe ρ, raw RSA, and confusability–entropy axis alignment **match** the user brief. Several mechanistic numbers in the brief are **not** in the local artifacts (difficulty-matched RSA, pair-level RSA, causal ablation). One brief range (pair–entropy cosine 0.48–0.65) matches **absolute** cosine and hides a negative signed cosine.

Molmo2 is **not** above chance. The eval JSON already stores the correct p-value; some docs still present three models as a single above-chance panel.

## What matched

### Accuracy, 243 trials, video-only 6AFC, seed 42, chance = 1/6

| model | k / n | accuracy | Wilson 95% CI | stored p (greater) | recomputed p | vs brief |
|---|---|---|---|---|---|---|
| Qwen3-VL | 92 / 243 | 37.9% | [0.320, 0.441] | 2.25e-15 | 2.25e-15 | match |
| Gemma 4 | 76 / 243 | 31.3% | [0.258, 0.374] | 1.54e-08 | 1.54e-08 | match |
| Molmo2 | 48 / 243 | 19.8% | [0.152, 0.252] | 0.1155 | 0.1155 | match, **not above chance** |

Sources: `results/baseline/eu_emotions/{model}/eval_v2_eu_emotions_{model}_video_only_seed42.json`. Protocol `v2-study3-full-eu-6afc`. `n_options` = 6.

### Model vs human per-item entropy (Spearman)

| model | stored ρ | recomputed ρ | p (two-sided) | vs brief |
|---|---|---|---|---|
| Qwen3-VL | −0.001325 | −0.001325 | 0.984 | match (≈ −0.001) |
| Gemma 4 | +0.117246 | +0.117246 | 0.068 | match (≈ +0.117), **not significant at 0.05** |
| Molmo2 | −0.034423 | −0.034423 | 0.593 | match (≈ −0.034) |

Source: `results/stats/rq1_1b_entropy_alignment.json`. None of the three output-entropy alignments survive α = 0.05. The brief reported the point estimates only.

### Accuracy by human-consensus tercile (n = 81 per tercile)

| model | high | mid | low | high vs low p | brief "p < 10⁻⁴" |
|---|---|---|---|---|---|
| Qwen3-VL | 0.543 | 0.383 | 0.210 | 1.20e-05 | yes |
| Gemma 4 | 0.506 | 0.309 | 0.123 | 1.57e-07 | yes |
| Molmo2 | 0.346 | 0.148 | 0.099 | 1.57e-04 | **no** (p = 1.57×10⁻⁴) |

Source: `results/stats/human_calibration.json`. Gradient direction matches for all three; the brief's p-threshold is too tight for Molmo.

### Probes (LOO ridge, 3 layers, last token)

Confusability (1 − p_target) ρ = 0.467–0.611. Human-entropy ρ = 0.441–0.532. Brief 0.44–0.61: match.

| model | layer | probe conf ρ | probe H ρ | RSA ρ | RSA p_perm |
|---|---|---|---|---|---|
| qwen3vl | 4 | 0.520 | 0.501 | −0.079 | 0.045 |
| qwen3vl | 13 | 0.498 | 0.516 | −0.076 | 0.055 |
| qwen3vl | 26 | 0.519 | 0.457 | −0.056 | 0.144 |
| gemma4 | 5 | 0.508 | 0.483 | −0.019 | 0.643 |
| gemma4 | 15 | 0.560 | 0.501 | −0.019 | 0.610 |
| gemma4 | 31 | 0.467 | 0.441 | +0.003 | 0.924 |
| molmo2 | 4 | 0.514 | 0.469 | +0.015 | 0.679 |
| molmo2 | 12 | 0.529 | 0.454 | −0.014 | 0.720 |
| molmo2 | 23 | 0.611 | 0.532 | +0.027 | 0.443 |

Raw RSA range −0.079 to +0.027 matches brief −0.08 to +0.03. **No permutation-corrected claim of a human-like RDM.** Qwen L4 p_perm = 0.045 is uncorrected and the ρ is the wrong sign for "same confusion structure".

Sources: `results/mech/{model}_confusability_probe_rsa.json`. These ρ values were **not** recomputed from activations (tensors missing); they were read from the saved JSON. Trial counts (243) match the sidecars.

### Axes (Qwen L4)

JSON `axis_alignment_conf_vs_entropy` = 0.950086. Cosine of saved unit npy vectors = 0.950085. Brief r = 0.95: match.

Pair own-vs-rest separability: 0.584, 0.528, 0.796. Brief 0.53–0.80: match.

Signed cosine of pair axes vs entropy axis (from npy; **not** stored in the JSON):

| axis | cos vs entropy | |cos| |
|---|---|---|
| Bored–Unfriendly | +0.477 | 0.477 |
| Interested–Kind | +0.652 | 0.652 |
| Disappointed–Worried | **−0.527** | 0.527 |
| random | −0.031 | 0.031 |

Brief "0.48–0.65" matches |cos|, not signed cosine.

### Steering (exploratory)

`results/mech/steer_summary_qwen3vl_layer4.json`: 36 trials, 5 samples, layer 4, last token, alphas ±1. Mean ΔJS vs human is small and not selective vs the random axis. Agrees with "steering ≈ random".

## What did not reproduce / was not present

1. **Difficulty-matched RSA 0.03–0.19** and **Bored–Unfriendly RSA +0.27 / +0.29 / +0.47**. Not in any local JSON. `rsa_difficulty_matched()` exists in code. Saved probe JSON has no such field. **Do not cite until rerun.**
2. **Causal ablation Δacc = 0, Δconfusion = 0, 5/180 flips.** No `qwen3vl_eu_ablation_layer4.json`. `docs/study3_brief_for_claude_agent.md` says the ablation was not run. `results/ablation/` is a different experiment (frame policy / modality). **Unverified.**
3. **Activation tensors.** `extract_meta.json` for all three 6AFC models, 243 trials, last-token pooling, 3 layers each. Zero `.npy` files locally. Probes cannot be rerun with new baselines until CSD3 pull or re-extract.
4. **Brief claim "all models above chance".** Contradicted by the eval JSON itself (Molmo p = 0.12). The JSON is the authority.
5. **Brief "accuracy-by-consensus gradient p < 10⁻⁴" for all models.** Molmo p = 1.57×10⁻⁴.

## Other facts that affect later claims

- Eval JSON field `stage2_prompt_mode` is `"4afc"` while `n_options` is 6. Labeling leftover; the prompt builder is N-AFC (`scripts/prompts.py`).
- Qwen mean forced-choice entropy is 0.179 vs Gemma 0.676 vs Molmo 1.356. Qwen's 6AFC distribution is near-degenerate even though sampling used T = 1.0, n = 20. This is relevant to "models are confident where humans are uncertain" and to option-position bias (Phase 1).
- Human norms: `data/eu_emotions_human_entropy.json` `"modality": "Face"`, 243/243 matched, source `eu_emotions_oreilly_item_level.xlsx`. Builder filters to Face. Actor letter parseable for 17 actors.
- Prompt options are the human study's per-item 6AFC set (target + 4 controls + "None of the above"), not independently sampled foils.
- Activation extract used `max_frames: 4` (not the eval cap of 16). Geometry is from a poorer visual sample than behaviour. Log as a limitation / Phase 1 extract decision.

## How to rerun

```bash
conda activate mr_eu_open_llm
cd /Users/eb2007/playground/bullpy/mr_eu_open_llm
python research/scripts/reproduce_headline_numbers.py
```
