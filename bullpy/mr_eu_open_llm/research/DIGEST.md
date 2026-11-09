# DIGEST

One entry per working day. User-facing; keep to 5–10 lines.

## 2026-11-09 Europe/London — Qwen A1 running, first file in

Both Qwen A1 arrays left the queue overnight. Task 39982137_0 (rotation 0, real) COMPLETED in 7 h 44 m, so the count is 21 of 30; tasks 39982137_1, 39982138_0 and 39982138_1 are past 7 h, and 39982137_2 started 08:33. GPUs at 100% on the check at 04:01; no task has failed. At about 8 h per task and two per array, the last files should land about 01:00 on 10 Nov. Nothing has been read. SSH dropped once (about 07:30) and was re-opened by the user. Quota 1041.9 GB, 996,125 files.

Where the study stands, from locked analyses only: the original 6AFC task was confounded by option order and option set (category alone matches the peak probes, rho .503); swapping in another face lowers accuracy on all three models (A3 pass, Qwen .24, Gemma .22, Molmo .06), but "None of the above" dominates Qwen and Molmo; output entropy does not track human entropy (H2); natural-clip cues are far more collinear than the randomised Snoek design (1/lambda_min 18.4 against 2.1 to 2.3). The face-dependence answer (G1) and the randomised human comparison (G5) are still to come.

## 2026-11-04 Europe/London — Prompt confound found

The 6AFC prompt lists the correct answer first on all 243 items, so accuracy is P(chose option 1), and one option set per label means the prompt reveals the category. Category alone reaches grouped rho 0.50, matching the peak probes, and last-token activations decode category at 0.96 to 1.00. The Phase 2 probe claim is withdrawn pending a rerun. Amendment A1 frozen: 5 rotations, real and vision-mismatch, 3 models. Shuffled 4AFC runs and small within-category human correlations suggest some real signal survives.

## 2026-11-02 11:10 Europe/London — Branch B locked

All-layer last-token extracts finished (Qwen 36, Gemma 42, Molmo 32 layers; n = 243). Official Phase 2 gate **locks Branch B**: no layer has Holm p < 0.01. Peak grouped T1 ρ is still high (Qwen L34 0.47, Gemma L2 0.49) with raw p_perm = 0.001; Holm floors are 0.036 / 0.042 because n_perm = 1000 cannot beat 0.01 across that many layers. Rule not retuned. Paper is above-chance, non-human-like outputs, mechanistic map, entanglement, generalisation. Phase 4 off. Ablation / f16 extract slipped to ~05:15–05:45 tomorrow behind A2. Intro/discussion drafts and an LOO-vs-grouped figure added. All-layer difficulty-matched RSA queued on icelake (does not overwrite the 3-layer RSA file). Option-order GPU still waits.

## 2026-10-30 14:47 Europe/London — Phase 2 exploratory gate

Confirmatory protocol on the 3-layer last-token cache (not locked). Would-be **Branch A**: Qwen L26 grouped ρ = 0.44 (p_Holm = 0.003) beats the stimulus baseline by 0.30; Gemma L15 ρ = 0.33 also passes. Molmo layers pass the statistics but Molmo is at chance, so not H1. Category-residual ρ is often negative (between-category signal); the pre-registered check is two-sided and was not retuned. H2 holds: output-entropy vs human-entropy CIs all include 0. Phase 5: 1−p_target vs entropy ρ = 0.954. GPU jobs 39709579–84 still Priority.

## 2026-10-30 afternoon (Europe/London) — CSD3 live, jobs

Reused ControlMaster SSH. Pulled 6AFC last-token `.npy` (3 models × 3 layers) and the missing mech files. Ablation n=36 is real: Δacc = 0, Δconfusion = 0, **5/180 flips**, non-selective (entropy 0, bored 1, disappointed 1, interested 2, random 1). Difficulty-matched RSA on disk (layer-mean ρ Qwen 0.03–0.10, Gemma 0.00–0.09, Molmo 0.16–0.19; Bored–Unfriendly 0.27/0.26/0.47 at the early layer). Archived n=36 as `results/mech/pilot_n36`. Preregistration frozen (`prereg-v1`). GPU jobs in queue: 39709579 ablation full Qwen, 39709580–82 all-layer f4, 39709583 Qwen f16, 39709584 Gemma ablation. Phase 2 confirmatory protocol implemented; running locally on the 3-layer cache (exploratory, will not lock the branch). Option-order GPU jobs written, not submitted.

## 2026-10-30 (Europe/London) — Phase 0

Located the study at `bullpy/mr_eu_open_llm`. Accuracies reproduce: Qwen 37.9% (p=2.3e-15), Gemma 31.3% (p=1.5e-8), Molmo 19.8% (p=0.116, not above chance). Output entropy does not track human entropy (ρ ≈ −0.001, +0.117 n.s., −0.034). Probe JSON ρ ≈ 0.44–0.61; confusability–entropy cosine 0.95. Difficulty-matched RSA and the causal ablation file are missing; not citing them. 6AFC `.npy` not local; CSD3 SSH denied. Tagged `v2-phase0` on `origin/main`. Phase 1 started: metric unit tests pass; Molmo wording fixed in the study3 brief.
