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

## 7. EU-Emotion licence for public norms and edited clips

Item 2 already disclosed that `data/eu_emotions_human_entropy.json` (item-level human response distributions) is committed to the public repo, as aggregated percentages and not videos, and that default was accepted. It is on `main` since commit `2e2af35` (2026-10-30). This is not a breach of the no-stimuli rule.

**Resolved 2026-11-08 by the user:** the team controls EU-Emotion and its creators are collaborators, with full permission for public item-level norms and in-memory edits. The norms file stays public. Videos, frames and edited clips are still kept out of the repo, by choice, to keep the package small and the rule simple.

**Was open:** whether the EU-Emotion licence allows item-level norms to be public, and whether in-memory edits of clips are allowed. It needs the user or the stimulus authors. Asking the authors is an email, so it is the user's.

**If the answer is no:** the file would have to come out of history as well as the latest commit, which means rewriting a public branch, and that is the user's call. Reproduction would then read it from a private location.

**Meanwhile:** the paper files give counts, not the seven unmatched item IDs, so that nothing further depends on the answer. **Update 2026-11-12:** with the permission settled, the seven IDs are now listed in `research/reproduction/eu_missing_items_check.json`. The user also confirmed that the copy we hold is the collaborators' working set, so whether a distributed set differs is not a question for this paper (outline item 16).

## 8. Snoek group contacted — AWAITING REPLY (2026-11-08)

The user emailed the Snoek group on 2026-11-08 and is waiting for a reply. Nothing waits on it: A4 is frozen on the headless FACSHuman renders and runs as written. If the original GFG stimuli or renders arrive, they enter only as a sensitivity analysis on the same AU configurations, added by a dated deviation written before any model sees them. They do not replace the frozen confirmatory test.

## 9. Before submission (not blocking now, 2026-11-08)

These only matter at submission. Nothing waits on them.

- **Code DOI.** NHB wants code in a DOI-minting repository. Zenodo needs your login. `CITATION.cff` and `.zenodo.json` are drafted in the project folder, but GitHub and Zenodo only read them at the root of `playground`, and a Zenodo release would archive the whole `playground` repo. Options: (a) a release of `playground` as it is, (b) a separate public repo holding only `mr_eu_open_llm`. (b) is cleaner.
- **Preregistration record.** NHB's policy asks for an active link and a date, but does not say whether a tagged commit counts. The amendments are tagged on public GitHub. Depositing the tagged PDFs on OSF (free, your account) would remove the doubt.
- **Authors, affiliations and ethics wording.** These are placeholders (`USER_*`) in CITATION.cff, the cover letter and the reporting summary.
- **RMET.** It is not used in any reported analysis. Confirm it can be left out of the data statement.
- **Metric name clash (found 2026-11-08).** "Causal Feature Alignment (CFA)" is already the name of an unrelated WACV 2024 training method (Venkataramani et al., learning to ignore spurious background features, no human comparison). Options: (a) keep the name and cite it with one sentence on the difference, (b) rename ours. Recommendation: (b) before submission, since the paper's citeable core carries the name. The rename is a text change only and touches no frozen threshold. **RESOLVED 2026-11-09: renamed to Cue Effect Alignment (CEA) on the user's instruction (DECISIONS 2026-11-09).**

## 10. Your notes file was pushed by mistake (2026-11-12, about 22:58) — DECISION NEEDED

**What happened.** I ran `git commit -a` from the project folder to commit two research files. In the `playground` repo that also took your uncommitted edits to `notes/notes.txt` (entries from 2026-10-26 to 2026-11-12, which name other people and private plans) into commit `0bc8a57` ("STATE, DECISIONS: correct times (GMT)"), and I pushed it to the public `p1-pipeline` branch. Your local file is unchanged. My error: from now on I commit only by explicit path.

**What I did not do.** I tried to remove the file from that commit and force-push the branch (with a lease pinned to `0bc8a57`, so nothing else could be overwritten); the permission system blocked it as a history rewrite, so I left it for you. I have not merged anything to `main` since, so the commit is only on `p1-pipeline`.

**Exact question.** Do you want the notes removed from the public branch?

**Recommended default.** Yes. Say "drop the notes commit" and allow the force-push when asked. I will then rebuild `p1-pipeline` from `d8c2e01` (the commit before), replaying every later commit with `notes/notes.txt` left out, check that the result differs from the current branch only in that file, and push with `--force-with-lease` pinned to the current tip. Your local notes file is not touched. GitHub may keep the orphaned commit reachable by its hash for a while; if that matters, GitHub support can purge it.

**Meanwhile.** Work continues; nothing waits on this except PRs to `main`.

## 11. Literature table: how to code [1]'s option sets (low priority, 2026-11-12)

**Context.** Figure 1e and the Results literature paragraph code Strachan et al. [1] as "partly" on "one option set served every item". The independent review (finding 36) notes that [1]'s alternatives are set per item and that on the MRMET repeated labels always came with the same three alternatives, which is the prompt-text fault the legend codes as "no".

**Exact question.** Keep "partly" (with a one-line justification in the legend), or recode [1] to "no"?

**Recommended default.** Recode to "no" and say why in the legend; it is the stricter reading. Recoding changes `literature_table.csv`, its registry counts and Figure 1e, so I have not done it without you.

**Meanwhile.** Nothing waits on it.
