# DECISIONS

Non-trivial decisions. Date, decision, alternatives, why. Do not rewrite history; append.

## 2026-10-30 — Git remote and integration branch

- **Decision:** Treat `origin` (`git@github-personal:eddiebullock/playground.git`) as the personal remote. Create long-lived `paper-v2` from `origin/main`, not from the stale local `main`.
- **Alternatives:** (a) Use local `main` (11 commits behind); (b) add a second remote named `personal`.
- **Why:** `origin` already points at `eddiebullock/playground`. The 11 missing commits are unrelated Study A1/A2 work and should be in the default-branch history before we branch. Local `main` also has unrelated dirty files (`notes/notes.txt`, submodules) that must not enter this paper's commits.

## 2026-10-30 — Public repo + gitignore exceptions

- **Decision:** Keep working on the public repo. Extend `bullpy/mr_eu_open_llm/.gitignore` with targeted `!` exceptions so small result JSON/CSV, study3 figures, human item-level distributions, and `research/` are committable. Do not commit videos, activation tensors, model weights, `.env`, or finetune `tokenizer.json`.
- **Alternatives:** (a) Stop until the user makes the repo private; (b) commit nothing until gitignore is perfect.
- **Why:** Brief forbids sitting idle. Stimuli stay out. Aggregated per-item human response distributions (not videos) are required to reproduce entropy/RSA and are already on disk. Flagged in `NEEDS_INPUT.md` so the user can privatise if they want.

## 2026-10-30 — Molmo is at chance; do not claim otherwise

- **Decision:** Treat only Qwen3-VL and Gemma 4 as above-chance recognisers. Molmo2 48/243 = 19.8%, one-sided binomial p = 0.1155 vs chance 1/6. Mechanistic Molmo results may be reported as a completeness check, never as evidence of mental-state recognition.
- **Alternatives:** Keep the three-model "all above chance" wording from the study3 agent brief table.
- **Why:** Independent `scipy.stats.binomtest` matches the stored `p_binom_gt_chance` in the eval JSON. The brief's instruction to fix this is confirmed, not assumed.

## 2026-10-30 — Ablation headline in the user brief is unverified

- **Decision:** Do not cite `Δacc = 0`, `Δconfusion = 0`, or `5/180 flips` until a result file exists. Local docs (`docs/study3_brief_for_claude_agent.md`) mark the v2 ablation as "code ready — not run". No `qwen3vl_eu_ablation_layer4.json` on disk. HPC is unreachable.
- **Alternatives:** Copy the brief's numbers into the paper.
- **Why:** Integrity rule: never fill in a number. Steering (n = 36, layer 4, last token) **is** on disk and is the causal artifact we can currently discuss.

## 2026-10-30 — Difficulty-matched RSA numbers in the user brief are unverified

- **Decision:** Treat brief values (matched RSA 0.03–0.19; Bored–Unfriendly +0.27/+0.29/+0.47) as **not reproduced**. Probe JSON has only raw RSA. Code for `rsa_difficulty_matched` exists in `scripts/confusability_probe_rsa.py` but was not applied to the saved JSON (or the output was not pulled).
- **Why:** Cannot re-run without activation `.npy`. Will run as soon as caches are present, after preregistration.

## 2026-10-30 — Pair–entropy cosine: report signed values, not only |cos|

- **Decision:** Primary report is signed cosine from the saved unit axes. Qwen L4: Bored–Unfriendly +0.477, Interested–Kind +0.652, Disappointed–Worried **−0.527**. The brief's 0.48–0.65 range matches **absolute** cosine, which hides the sign flip.
- **Why:** Sign is scientifically meaningful (anti-aligned pair axis). Absolute-value summaries stay in a footnote/appendix.

## 2026-10-30 — Human EU-Emotion norms are Face (video-only)

- **Decision:** Phase 6.3 modality check is answered from the builder, not from memory: `scripts/human_entropy.py` filters the O'Reilly workbook to `FACE_MODALITY = "Face"`, and `data/eu_emotions_human_entropy.json` records `"modality": "Face"`. Model evals are `video_only`. Treat as modality-matched for the Face column of EU-Emotion, and still cite the paper for what "Face" meant in the validation study.
- **Why:** This is the documented filter, not an inference from filenames.

## 2026-10-30 — Actor IDs exist; grouped CV is possible

- **Decision:** Pre-register grouped-by-actor CV. Filenames parse to 17 actor letters (B–S, no Q) via the same regex as `parse_stimulus_code`.
- **Why:** Brief flags actor leakage as critical. Metadata does not store an `actor_id` field, but the stimulus code does.

## 2026-10-30 — Continue without CSD3; do not download 8B weights onto this Mac

- **Decision:** Sequence CPU work (Phase 1 refactor, tests, prereg, paper methods, Phase 5 entanglement null once activations exist) while HPC auth is broken. Do not pull Hugging Face 8B checkpoints onto a 16 GB M3 Air with 29 GB free.
- **Why:** Disk and unified memory cannot host the GPU phases. Gemma 4 E4B on MPS is a maybe for smoke tests only, after Phase 1.
