# Phase 0 review

Persona: a conceptually precise senior reviewer who has already seen the pilot and will not let a number through on trust.

## Five strongest critiques

### 1. The "reproduced" probe and RSA numbers are not independently recomputed

The eval accuracies and entropy Spearman were recomputed from trial-level fields. The probe ρ and RSA ρ were only **read** from JSON. The activation tensors that produced them are not on this machine. A reviewer who asks "did you actually rerun the decoder?" gets a no. Until the `.npy` files are pulled, treat probe/RSA as archival, not live.

**Addressed:** `REPRODUCTION.md` and `RESULTS_REGISTRY.md` mark those rows `read-only`. Phase 2 will not start from these ρ values as confirmatory without a fresh run under the preregistered protocol (nested CV, actor grouping, baselines).

### 2. The user brief already contains numbers that this repo cannot support

Difficulty-matched RSA, Bored–Unfriendly RSA, and the ablation null (Δacc = 0, 5/180) are in the brief as if they were results. They are not in the artifact tree. Shipping a paper that repeats them would be fabrication-by-paste.

**Addressed:** Decision logged: do not cite. Local study3 brief says the ablation was not run. If HPC later yields those files, they become data; until then they are hearsay.

### 3. Behaviour is not a clean "three VLMs recognise mental states"

Molmo is at chance. Qwen's 6AFC entropy is 0.18 nats — almost a delta — so the entropy-alignment null for Qwen is partly a measurement floor. Gemma's ρ = 0.12 is the only positive point estimate and is not significant. The interesting claim is already a **dissociation** (accuracy tracks consensus; output entropy does not), but it is carried by two models, not three, and the entropy side has no significant correlation to report.

**Addressed:** Molmo flagged everywhere. Entropy p-values now sit next to ρ. Qwen's near-zero output entropy goes into Phase 1 option-scoring / temperature notes.

### 4. Geometry was extracted under a different visual budget than behaviour

Eval used fps 1, max 16 frames. `extract_meta.json` has `max_frames: 4`. Last-token only, three depth fractions. Layer 4 on Qwen is early. Any "not causal" story from L4 last-token steering is exactly the uninterpretable null the brief already worries about.

**Addressed:** Logged as a Phase 1 extract decision (all layers, several positions, frame policy matched to eval). Do not over-interpret the L4 steer null.

### 5. The public-repo + parent-gitignore combination would have dropped the evidence

`bullpy/.gitignore` ignores `*.json` and `*.csv`. The repo-root ignore drops `*.csv`, `*.png`, `*.pdf`. The numbers above were untracked. A machine wipe loses the paper.

**Addressed:** Project gitignore now has explicit `!` exceptions for small analysis artifacts and `research/`. Stimuli, tensors, weights, `.env` stay ignored. User notified that the remote is public.

## Residual risks (not blocking Phase 1)

- CSD3 credentials. GPU phases wait; CPU phases do not.
- Human Face-column norms are the right modality filter in the workbook; still confirm against O'Reilly 2016 in the paper's limitation paragraph (what "Face" meant, which raters).
- 17 actors, uneven n (9–17 clips). Grouped CV will be noisy; still required.
- `stage2_prompt_mode: "4afc"` leftover will confuse readers; rename in the pipeline, do not rewrite historical JSON.
