# NEEDS_INPUT

Questions the agent cannot resolve. Each item is standalone. Work continues on everything else.

## 1. CSD3 SSH — RESOLVED 2026-10-30

Reuse the live ControlMaster socket `~/.ssh/cm-eb2007@login.hpc.cam.ac.uk:22` (process `ssh -fN ... ControlPersist=12h`, plus an interactive `ssh eb2007@login.hpc.cam.ac.uk`). Fresh BatchMode logins without that path still fail. 6AFC `.npy` and the n=36 ablation were pulled. GPU jobs are being submitted through this multiplexed connection.

No remaining question. Keep using `-o ControlPath=/Users/eb2007/.ssh/cm-%r@%h:%p -o ControlMaster=auto`.

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
