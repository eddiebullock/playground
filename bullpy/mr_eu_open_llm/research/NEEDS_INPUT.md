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

## 6. A2 sign-off — RESOLVED 2026-11-07

The user accepted all four choices, and A2 was frozen before A1 was read (tag `a2-freeze`). No remaining question.

## 7. EU-Emotion norms already public (found 2026-11-07)

`data/eu_emotions_human_entropy.json` has been on `main` of the public repo since commit `2e2af35` (2026-10-30). It holds a human response distribution, options and target label for each of the 243 clips, derived from `eu_emotions_oreilly_item_level.xlsx`. `data/eu_emotions_full_manifest.json` lists clip file paths. These are not stimuli, so the no-stimuli rule was not broken, but we do not know whether the licence allows item-level norms to be public.

**Needs the user, or the stimulus authors:** whether item-level norms may be public, and whether in-memory edits of clips are allowed. Asking the authors is an email, so it is the user's.

**Not done:** nothing was removed. Deleting the file from the latest commit would leave it in history and would break reproduction, and removing it from history means rewriting a public branch, which is the user's call. If it must go, the path is to rewrite history and keep the file off the repo, with reproduction reading it from a private location.

**Done meanwhile:** the paper files give counts instead of the seven unmatched item IDs.
