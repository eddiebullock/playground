# NEEDS_INPUT

Questions the agent cannot resolve. Each item is standalone. Work continues on everything else.

## 1. CSD3 SSH authentication (blocking GPU phases and activation pull)

**Context.** Heavy study3 compute lives on Cambridge CSD3 (`eb2007@login.hpc.cam.ac.uk`, work dir `~/rds/hpc-work/study3`, GPU account `BARON-COHEN-SL3-GPU`). Local 6AFC activation directories contain `extract_meta.json` and trial-id sidecars but **no** `.npy` tensors. Axis vectors and eval JSON were pulled earlier; the tensors were not. Causal ablation result files are also absent locally.

On 2026-10-30 a batch-mode SSH attempt failed:

```
eb2007@login.hpc.cam.ac.uk: Permission denied (publickey,keyboard-interactive,hostbased).
```

This machine has `~/.ssh/id_ed25519_personal.pub` and `id_ed25519_minimal.pub`. GitHub SSH via `github-personal` works. CSD3 does not.

**Exact question.** Can you restore SSH to CSD3 (agent key, `ssh-copy-id`, or an interactive login + TOTP)? Once that works, the agent can `rsync` only the small 6AFC `.npy` files (on the order of tens of MB, not checkpoints) and any `*ablat*` mech outputs.

**Recommended default.** Continue Phases 1, 5 (code), 6 (adapters), and 7 (methods draft) on CPU. Do not download 8B VLMs onto this Mac.

**Meanwhile.** Mapping compute, reproducing JSON-level numbers, refactoring, preregistration.

## 2. GitHub repo is public

**Context.** `gh repo view eddiebullock/playground --json visibility` → `"visibility": "PUBLIC"`. Paper code, small result tables, and drafts will be pushed here.

**Exact question.** Do you want `eddiebullock/playground` made private before further results and paper text go up? Stimuli and raw videos are not being committed. Item-level human response distributions (`data/eu_emotions_human_entropy.json`) are aggregated choice percentages, not videos; they will be committed so entropy/RSA can be reproduced.

**Recommended default.** Keep pushing; stay strict about videos, activations, weights, `.env`. You can privatise any time.

**Meanwhile.** Gitignore exceptions are scoped to small analysis artifacts.

## 3. HPC activation pull (depends on #1)

**Context.** Phase 2 (the Branch A/B gate) needs last-token activations for all 243 items at the saved layers, and later all layers × several token positions. Those tensors are almost certainly on CSD3 under `~/rds/hpc-work/study3/results/activations/baseline_{qwen3vl,gemma4,molmo2}_6afc/`.

**Exact question.** None beyond #1. Logged so it is not forgotten.

**Recommended default.** `rsync` the `.npy` + manifests as soon as SSH works. Do not re-extract until a cache miss is confirmed.

**Meanwhile.** Probe JSON is enough to reproduce the *reported* ρ values but not to rerun probes with new baselines.

## 4. CREMA-D access form (Phase 6, not blocking now)

**Context.** CREMA-D maintainers ask users to fill a short access form. Videos may be Git LFS. This is a user action.

**Exact question.** When you have a moment, complete the CREMA-D access form so the video-only subset can be downloaded. No need to do this before Phases 1–2.

**Recommended default.** Agent writes the adapter and the pre-registered clip-selection rule against the public `finalResponses.csv` schema without videos.

**Meanwhile.** Adapter scaffolding in Phase 6, not now.

## 5. MRMET option-level human counts (Phase 6)

**Context.** OSF `https://osf.io/ahq6n`. Need per-option counts, not just accuracy. If the download requires a login you must complete, say so; the agent will not email the authors.

**Exact question.** None yet. Will try the OSF download in Phase 6 and only escalate if the archive is gated or lacks option counts.

**Recommended default.** If only accuracy exists, use `p_correct` as the difficulty target and mark entropy/RSA as N/A for MRMET.

**Meanwhile.** Unblocked.

---

No other blocking questions. Local `main` is 11 commits behind `origin/main` with dirty `notes/notes.txt` and two submodules; those are left untouched.
