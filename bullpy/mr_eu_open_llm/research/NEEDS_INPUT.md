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
- **Authors, affiliations and ethics wording.** These are placeholders (`USER_*`) in CITATION.cff, the cover letter and the reporting summary. **Ethics, 2026-11-13:** on your instruction, Methods and the reporting summary now carry the recommended statement: no separate ethical approval was required, because the study recruited no participants and used only existing materials under their permission or licence terms. It is marked provisional, for you to confirm with the institution before submission.
- **RMET.** It is not used in any reported analysis. Confirm it can be left out of the data statement.
- **Metric name clash (found 2026-11-08).** "Causal Feature Alignment (CFA)" is already the name of an unrelated WACV 2024 training method (Venkataramani et al., learning to ignore spurious background features, no human comparison). Options: (a) keep the name and cite it with one sentence on the difference, (b) rename ours. Recommendation: (b) before submission, since the paper's citeable core carries the name. The rename is a text change only and touches no frozen threshold. **RESOLVED 2026-11-09: renamed to Cue Effect Alignment (CEA) on the user's instruction (DECISIONS 2026-11-09).**

## 10. Your notes file was pushed by mistake (2026-11-12, about 22:58) — RESOLVED 2026-11-13

**What happened.** I ran `git commit -a` from the project folder to commit two research files. In the `playground` repo that also took your uncommitted edits to `notes/notes.txt` (entries from 2026-10-26 to 2026-11-12, which name other people and private plans) into the commit "STATE, DECISIONS: correct times (GMT)", and I pushed it to the public `p1-pipeline` branch. My error: from now on I commit only by explicit path.

**Resolved 2026-11-13, about 09:20, on your instruction.**
- `p1-pipeline` was rebuilt from `d8c2e01`, the commit before, with `notes/notes.txt` restored to its `d8c2e01` version in all 40 later commits. It was force-pushed with a lease pinned to the old tip. The new tip is `f37beae`.
- Checked before the push:
  - Each rebuilt commit has the same message, author, committer and dates as its original, and differs from it only in `notes/notes.txt`.
  - The leaked version of the file cannot be reached from the new tip.
  - No tag or open PR held the old commits.
- Your local `notes/notes.txt` is byte-identical to before. It shows as a local, uncommitted change, as it did before the mistake.
- The old history is kept on your Mac only, in the local branch `backup/p1-pipeline-before-notes-removal`. Delete it when you no longer want it: `git branch -D backup/p1-pipeline-before-notes-removal`.
- Commit IDs cited in the documents now name the rebuilt commits, whose project files are identical. Two rebuilt commit messages still cite old IDs.

**Left for you, optional.** GitHub keeps the old commits reachable by their IDs until it garbage-collects them, and the branch's activity log names the old tip.
- To remove them sooner, ask GitHub Support to purge the repository's unreachable commits (GitHub's guide "Removing sensitive data from a repository").
- The old IDs are left out of this file for that reason.

## 11. Literature table: how to code [1]'s option sets (low priority, 2026-11-12) — RESOLVED 2026-11-13

**Context.** Figure 1e and the Results literature paragraph code Strachan et al. [1] as "partly" on "one option set served every item". The independent review (finding 36) notes that [1]'s alternatives are set per item and that on the MRMET repeated labels always came with the same three alternatives, which is the prompt-text fault the legend codes as "no".

**Exact question.** Keep "partly" (with a one-line justification in the legend), or recode [1] to "no"?

**Recommended default.** Recode to "no" and say why in the legend; it is the stricter reading. Recoding changes `literature_table.csv`, its registry counts and Figure 1e, so I have not done it without you.

**Meanwhile.** Nothing waits on it.

**Resolved 2026-11-13, on your instruction: recoded to "no".** The table, registry row, pinned counts, Results paragraph and Figure 1e legend were updated, and Figure 1 was rebuilt (DECISIONS 2026-11-13).

## 12. How to run A7 Stage 1 (decision, 2026-11-14) — RESOLVED 2026-11-14

**Context.** A7 Stage 1's route arm (Gemma 4 40795186, Qwen3-VL 40795187; two 59-minute tasks each) is still queued on `gpu2`, where it would wait days. A2, A4 and A9 ran through one-hour `intr` slices that resume a shard where it stopped. A7 cannot run that way as it stands. Its runner (`scripts/a7_run.py`) skips an item once that item's first row is written and flushes an item's rows together. So a time-budget stop in the middle of writing an item could leave that item incomplete for good, and A7's pre-run check would then refuse the shard. A7 has no repair rule for this, as A2 has in A2-2. A7's analysis also needs A2's analysis first (about 12:30 today, once A2's Qwen shards finish).

**Exact question.** Which way should A7 Stage 1 run?
- (a) As one whole `intr` job of four steps, one per task. Each step keeps its 59-minute cap with no slicing, which matches the gpu2 tasks as submitted. It would go in after the last A2 slice, or between slices. This needs only a scheduling DECISIONS note.
- (b) Inside the slices, with a pre-data ruling, like A2-2, to drop an incomplete last item before each resumed run. This needs a dated note in PREREG_A7 and new wrapper code.
- (c) Leave it on `gpu2`.

**Recommended default.** (a): no change to frozen code and no new ruling, and the same cap and settings as submitted (`ARMS=route DECISION_DATE=2026-11-10 N_SHARDS=2 A7_FROZEN=1 STEP=run`). If you agree, I will write the note and submit it after the last A2 slice, so A7's cells are ready when A2's analysis is.

**Meanwhile.** Nothing else waits on it. A7 is the optional route-arm supplement (PREREG_A7, Recommendation).

**Resolved 2026-11-14, about 08:40, on your instruction: option (a).** The wrapper `slurm_jobs/study3_intr_a7_stage1.sh` passed 11 stub checks and is on the cluster. The four `gpu2` tasks are held. A sequencer submits the job once A2's last slice ends (DECISIONS 2026-11-14).
