# DIGEST

One entry per working day. User-facing; keep to 5–10 lines.

## 2026-10-30 (Europe/London) — Phase 0

Located the study at `bullpy/mr_eu_open_llm`. It is the EU-Emotion 6AFC VLM project (Qwen3-VL / Gemma 4 / Molmo2, seed 42, n = 243). Accuracies reproduce: 37.9%, 31.3%, 19.8%. Molmo is not above chance (48/243, p = 0.116). Output entropy does not track human entropy (ρ ≈ −0.001, +0.117 n.s., −0.034). Probe JSON still shows ρ ≈ 0.44–0.61; confusability–entropy axes cosine is 0.95. Difficulty-matched RSA and the causal ablation (Δacc = 0, 5/180) are **not on disk**; not citing them. 6AFC activation `.npy` missing locally; CSD3 SSH denied. Repo is public. Next: git milestone `v2-phase0`, then Phase 1 refactor + Molmo wording fix + preregistration, all on CPU.
