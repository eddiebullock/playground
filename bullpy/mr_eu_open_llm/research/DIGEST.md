# DIGEST

One entry per working day. User-facing; keep to 5–10 lines.

## 2026-10-30 14:47 Europe/London — Phase 2 exploratory gate

Confirmatory protocol on the 3-layer last-token cache (not locked). Would-be **Branch A**: Qwen L26 grouped ρ = 0.44 (p_Holm = 0.003) beats the stimulus baseline by 0.30; Gemma L15 ρ = 0.33 also passes. Molmo layers pass the statistics but Molmo is at chance, so not H1. Category-residual ρ is often negative (between-category signal); the pre-registered check is two-sided and was not retuned. H2 holds: output-entropy vs human-entropy CIs all include 0. Phase 5: 1−p_target vs entropy ρ = 0.954. GPU jobs 39709579–84 still Priority.

## 2026-10-30 afternoon (Europe/London) — CSD3 live, jobs

Reused ControlMaster SSH. Pulled 6AFC last-token `.npy` (3 models × 3 layers) and the missing mech files. Ablation n=36 is real: Δacc = 0, Δconfusion = 0, **5/180 flips**, non-selective (entropy 0, bored 1, disappointed 1, interested 2, random 1). Difficulty-matched RSA on disk (layer-mean ρ Qwen 0.03–0.10, Gemma 0.00–0.09, Molmo 0.16–0.19; Bored–Unfriendly 0.27/0.26/0.47 at the early layer). Archived n=36 as `results/mech/pilot_n36`. Preregistration frozen (`prereg-v1`). GPU jobs in queue: 39709579 ablation full Qwen, 39709580–82 all-layer f4, 39709583 Qwen f16, 39709584 Gemma ablation. Phase 2 confirmatory protocol implemented; running locally on the 3-layer cache (exploratory, will not lock the branch). Option-order GPU jobs written, not submitted.

## 2026-10-30 (Europe/London) — Phase 0

Located the study at `bullpy/mr_eu_open_llm`. Accuracies reproduce: Qwen 37.9% (p=2.3e-15), Gemma 31.3% (p=1.5e-8), Molmo 19.8% (p=0.116, not above chance). Output entropy does not track human entropy (ρ ≈ −0.001, +0.117 n.s., −0.034). Probe JSON ρ ≈ 0.44–0.61; confusability–entropy cosine 0.95. Difficulty-matched RSA and the causal ablation file are missing; not citing them. 6AFC `.npy` not local; CSD3 SSH denied. Tagged `v2-phase0` on `origin/main`. Phase 1 started: metric unit tests pass; Molmo wording fixed in the study3 brief.
