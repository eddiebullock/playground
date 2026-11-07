# NEEDS_INPUT

Questions the agent cannot resolve. Each item is standalone. Work continues on everything else.

## 1. CSD3 SSH — RESOLVED 2026-10-30

Reuse the live ControlMaster socket `~/.ssh/cm-eb2007@login.hpc.cam.ac.uk:22` (process `ssh -fN ... ControlPersist=12h`, plus an interactive `ssh eb2007@login.hpc.cam.ac.uk`). Fresh BatchMode logins without that path still fail. 6AFC `.npy` and the n=36 ablation were pulled. GPU jobs are being submitted through this multiplexed connection.

No remaining question. Keep using `-o ControlPath=/Users/eb2007/.ssh/cm-%r@%h:%p -o ControlMaster=auto`. If the socket has gone, as it had on 2026-11-06 between 18:51 and 19:51, checks fail with `Permission denied`, and the user has to log in again.

## 2. GitHub repo is public

**Context.** `gh repo view eddiebullock/playground --json visibility` → `"visibility": "PUBLIC"`. Paper code, small result tables, and drafts will be pushed here.

**Exact question.** Do you want `eddiebullock/playground` made private before further results and paper text go up? Stimuli and raw videos are not being committed. Item-level human response distributions (`data/eu_emotions_human_entropy.json`) are aggregated choice percentages, not videos; they will be committed so entropy/RSA can be reproduced.

**Recommended default.** Keep pushing; stay strict about videos, activations, weights, `.env`. You can privatise any time.

**Meanwhile.** Gitignore exceptions are scoped to small analysis artifacts.

## 3. HPC activation pull — RESOLVED 2026-10-30

3-layer last-token `.npy` for Qwen/Gemma/Molmo 6AFC are local. All-layer extracts are jobs 39709580–39709583. No remaining question.

## 4. CREMA-D access form — RESOLVED

Form done, per STATE. No remaining question.

## 5. MRMET option-level human counts (Phase 6)

**Context.** OSF `https://osf.io/ahq6n`. Need per-option counts, not just accuracy. If the download requires a login you must complete, say so; the agent will not email the authors.

**Exact question.** None yet. Will try the OSF download in Phase 6 and only escalate if the archive is gated or lacks option counts.

**Recommended default.** If only accuracy exists, use `p_correct` as the difficulty target and mark entropy/RSA as N/A for MRMET.

**Meanwhile.** Unblocked.

## 6. A2 sign-off (needed before A2 can freeze, not blocking A1)

**Context.** `research/PREREG_A2_DRAFT.md` lists four open choices, and the agent's recommendations sit under them. They are recommendations only.

**Exact question.** Do you accept each of the four recommendations? In particular, should P14 and P15 be removed, or can you supply the Jack 2014 full text?

**Recommended default.** Hold A2 unfrozen until A1 is read and you have answered.

**Meanwhile.** A1 continues.

