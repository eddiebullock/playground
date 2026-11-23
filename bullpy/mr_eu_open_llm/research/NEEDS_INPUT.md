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

## 14. A8: neither large model can run the frozen gate script on CSD3's A100s. Should Qwen3-VL-32B continue under a device-only deviation? (decision, 2026-11-15, about 22:30) — RESOLVED 2026-11-17: option (b)

**Resolved 2026-11-17: you chose (b).** The prepared branch was merged into `p1-pipeline` (`d93228d`, `3b814ab`; PR #133) and recorded in DECISIONS 2026-11-17, row 102 (written 2026-11-20). Next: the 8B check on the cluster, which waits on item 16.

**What happened.**
- **Gemma 4 31B stopped twice** in the frozen `gemma4` gate adapter. That adapter assumes the per-layer inputs of Gemma 4 E4B, and 31B has none.
  - A8's lead decision 5 foresaw this case: no gate result, so no A1, A2 or A4.
  - Its A3, delivery and ladder steps still run.
- **Qwen3-VL-32B stopped twice too,** for a reason the design did not foresee:
  - On two GPUs (my setting, row 97), the frozen script's layer hooks index each layer's output with positions held on the first GPU, so the layers on the second GPU fail.
  - On one GPU, as the frozen sibling runs used, 67 GB of weights plus the first forward pass need more than the A100's 80 GB. The run stopped with CUDA out of memory.
- By the rule I set before the one-GPU retry (row 100), Qwen3-VL-32B's gates now count as not tested.
- So no A8 model can enter A1, A2 or A4, and A8 ends after Phase A (DECISIONS 2026-11-15, rows 99 to 101).

**The choice.**
- **(a) Keep the rule.** A8 ends after Phase A, with A3, delivery and ladder for both models.
  - The paper reports A8's scored arms as not tested, with these two reasons, as a scoped limit.
  - No GPU time is spent beyond Phase A.
- **(b) A dated deviation for Qwen3-VL-32B only.** Run the frozen gate script on two GPUs with one change: in its hooks, the position index moves to each layer's own device before indexing. That changes where an index is held, not any value computed.
  - The frozen file stays byte-identical. A small wrapper in A8's code, with a launcher setting for Qwen3-VL-32B's gates, applies the change at run time. So A8's code pins change, by dated deviation; the design block does not.
  - Check first, on the frozen sibling Qwen3-VL-8B, at bfloat16: the frozen gates command twice on one GPU, and the patched one once on two GPUs. The patched file must equal the frozen ones wherever the two frozen runs agree, which is the rule of A8's A1 equivalence test. About 7 GPU-hours: each run takes about 1 h 45 min, as the frozen 8B run's bfloat16 stage did (6,258 s), and the three together use one whole node for about 2 hours.
  - Then Qwen3-VL-32B's gates (about 7 hours on two GPUs). If they pass, A1, A2 and A4 follow under A8's unchanged rules: about 500 to 600 GPU-hours, against about 2,400 left, before the GPU reservation of 2026-12-07.
  - Gemma 4 31B stays not tested, since lead decision 5 is explicit about its adapter.

**Checked overnight (2026-11-16, 02:30), in my scratch folder: nothing committed, nothing submitted.**
- **The wrapper for (b) is written and passes its tests on the Mac.** It reuses the frozen hooks unchanged. Each runs with the positions moved to the device of the tensor it indexes. What it captures stays where the frozen hook puts it, spread over the GPUs as the layers are.
  - Revised at 02:59: my first draft moved every capture to the first GPU. On Qwen3-VL-32B's longest G3 clip (22,440 image tokens), G3's two all-layer captures come to about 29 GB. Beside about 34 GB of weights and the logits, that is close to the A100's 80 GB, so now only the index's device changes.
  - The tests split the gate tests' tiny fake Qwen model over two devices (Apple's GPU and the CPU), as `device_map="auto"` splits a model over GPUs.
  - On that split, the frozen hooks fail with the cluster's error.
  - On one device, the wrapper changes nothing.
  - On the split model, every captured tensor and all 188 values of G2, G3 and G4 equal the frozen hooks' to the bit. Every capture is held on the same device as with the frozen hooks.
  - 8 new tests pass, and so do the gate script's 44.
- **(b) also needs a launcher change.** The launcher runs the frozen script inside its own process, as a fresh copy, so a wrapper cannot reach it from outside. Under (b), a setting in the launcher (`DEVICE_HOOKS`, Qwen3-VL-32B only) runs that model's gates through the wrapper, and the record says so. The design block and its hash stay the same. The files are pinned in PREREG_A8's "Code hashes" by dated deviation.
- **The whole change is ready on a local branch, `prep/a8-gates-device`** (02:59; two commits, not pushed, not merged). It holds the wrapper, the launcher setting, the 8B check's job script and comparison script with its rule, 16 tests, and the re-pinned "Code hashes". The design hash is unchanged. A8's, the gates' and the wrapper's tests pass there: 164 passed, 2 skipped (one needs local data files that a fresh checkout lacks). If you choose (b), I add the dated records, run the full suite, merge, sync the cluster and submit the 8B check (one whole node, about 2 hours).
- **Why the 8B check runs bfloat16 only.** With more free memory per GPU, the float32 stage may run on two GPUs where it ran out of memory on one. The files would then differ for a reason unrelated to the hooks. bfloat16 runs every hook path.
  - Correction (02:46): this saves almost no time, because the frozen 8B run's float32 stage ran out of memory as soon as it loaded. The check stays at about 7 GPU-hours, not the 3.5 I wrote at 02:30.
- **Qwen3-VL-32B's own float32 stage** may run, or run out of memory, on two GPUs. The frozen rule decides either way: out of memory keeps the bfloat16 verdict (DECISIONS 2026-11-05), and a float32 stage that completes must pass too.
- **A stop rule, set before anything runs.** The deviation ends if:
  - the 8B check fails to run;
  - the patched file differs from the frozen pair where the two agree; or
  - Qwen3-VL-32B's patched gates stop on any other device error.

  There is then no second change, and A8 ends after Phase A, as under (a).

**Recommended default.** (b): the change is mechanical, it is shown not to change any output before it is used, and it is decided before any gate result exists.
- Against it: it overrides a rule I set an hour earlier, and a reviewer may weigh any change to frozen code after a failure.
- (a) is the strict reading. It is clean and costs nothing.

**Meanwhile.** Nothing beyond Phase A is submitted.
- Phase A finished at 23:57: delivery, ladder and A3 ended ok for both models.
- A8's A3 summary, its first scored result, waits on this decision, so that the decision is made before any A8 result is known.

## 19. A8 cannot finish before the GPU shutdown on the free tier. Defer its scored steps? (decision, 2026-11-23) — RESOLVED 2026-11-23: option (a)

**Resolved 2026-11-23: you chose (a)** ("great proceed with all jobs", after asking how essential A8 is). The deferral is recorded in DECISIONS 2026-11-23 (row 105) and PREREG_A8's dated note. A8's final checks and the paper's A8 text follow.


**What I found at 09:35 on 2026-11-23** (cluster facts only; no A8 output read):
- **The 8B check has waited 49 hours** (job 41575304, submitted 2026-11-21 08:02). Its priority is 173: 74 for age and 100 for the free tier's QOS, with a fair-share of 0. 391 pending A100 jobs rank above it, and the scheduler has planned no start for it.
- **Capacity is down.** About 30 of the 74 A100 nodes have been in a maintenance reservation (`rebuilds`) since 2026-11-18, with no end date. 548 jobs are pending for the rest.
- **Paid jobs start 400 points ahead.** Paid (SL2) jobs carry a QOS priority of 500, and ours carries 100. Age adds about 36 points a day (weight 1,000 over 28 days). So our job outranks *newly submitted* paid jobs only from about 2 December, and paid jobs submitted before then stay ahead of it. Its earliest realistic start is early December.
- **Every later step restarts at age 0.** Qwen3-VL-32B's gates (about 7 hours on a node) come next, and then A1, A2 and A4 (about 500 to 600 GPU-hours).
- **`intr` cannot carry this.** It allows one job per user at a time, of at most 60 minutes and 3 nodes. The check's frozen runs take about 1 h 44 min each, so they cannot run under it.
- **Projection.** On the free tier, A8's scored steps (gates, then A1, A2 and A4 for Qwen3-VL-32B) cannot run before the shutdown of 2026-12-07 07:00. The 8B check itself may or may not.

**The choice.**
- **(a) Recommended: defer A8's scored steps past the shutdown, by dated note.**
  - Deferral by a dated note is one of the "Not run" reasons A8 fixed before data.
  - Fill the paper's A8 slots now from A8's finished steps: A3 for both larger models (both pass, read on 2026-11-21), and delivery and the ladder once the end-of-A8 pull brings them back to be read by their rules (not read yet). Write the scored steps as not run, with the reason, and infer nothing from them.
  - Keep the 8B check queued, at no cost. If it runs, I read it by its rule and record it, so that any later run (after 2027-01-10, for a revision say) starts with the device question settled.
  - The paper can then be completed now.
- **(b) Hold the paper for A8.** Keep the job queued and wait. The scored steps would resume after 2027-01-10 at the earliest, so submission would move to February or later.
- **(c) Paid GPU time.** The project's free-only rule rules this out, and only you can change that rule.

**Why (a).** It reports what A8 could test, the face dependence of both larger models, and keeps the paper's timeline. A8's Results and Discussion branches for "not run" were drafted before data for this case.

**What I need.** A letter. Nothing changes until you answer, and the job stays queued either way.

## 18. CSD3 SSH: please log in again (2026-11-23) — RESOLVED 2026-11-23

**Resolved 2026-11-23, 09:33: you logged in again.** The 8B check was still pending on priority (item 19).


**Context.** The master connection you opened on 2026-11-21 ended at about 22:05 that evening. This session's watcher on job 41575304 then stalled at a fresh login prompt, and on 2026-11-23 it was stopped. Job 41575304, the 8B check, was still in the queue at the watcher's last good check, about 22:00 on 2026-11-21. The watcher checked only that, not whether the job was pending or running. Its state now is unknown: it may have run, and it may still be waiting. Nothing else of this project was queued or running.

**What waits on it.** Read the 8B check by its rule (DECISIONS 2026-11-17). If it passes, submit Qwen3-VL-32B's gates (one whole `ampere` node, about 7 hours; not `intr`, per row 102's stop rule). Then A1, A2 and A4 for that model by dated note, before the GPU reservation of 2026-12-07 07:00. That is 14 days from 2026-11-23.

**What to do.** In this session, type the line below, then give the password and the one-time code when asked. A watcher in this session is already waiting for the connection, and it reports the job's state as soon as you log in. Keep the Mac awake and on power while jobs run.

`! ssh -fN -o ControlMaster=yes -o ControlPath=~/.ssh/cm-%r@%h:%p -o ControlPersist=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=10 eb2007@login.hpc.cam.ac.uk`

## 17. The reframed paper: your review, the title, and one optional analysis (2026-11-21)

**2026-11-23.** The reframe reached `main` in PR #134, which you merged on 2026-11-21 (`3678113`), and the figures were relabelled on 2026-11-23 (IMPACT_PLAN.md, step 9). The questions below are still open.

**Context.** On your instruction, the paper was reframed around one headline: matching answers cannot show that a vision-language model reads faces as people do, and randomised cues can (`research/paper/v2/IMPACT_PLAN.md`; DECISIONS 2026-11-21). The changes are committed locally on `p1-pipeline` and not pushed.

**What only you can decide.**
1. **Review.** Is the reframe acceptable? Use `git diff bb3f90b -- research/paper/v2`, or read the clean copy from `python research/scripts/manuscript_words.py --out /tmp/manuscript.md`. Once you approve, it goes to `main` through a pull request.
2. **Title.** The working title is "Beyond matching answers: testing whether vision-language models use facial cues as people do", and three alternatives are in `abstract.md`.
3. **An optional exploratory analysis.** Results, Section 5 reports that Gemma 4's answers track people's only weakly where its cue effects align, but that comparison has no noise ceiling. A split-half ceiling from the human data would settle whether the gap is real. It needs CPU only, runs post hoc and would be labelled exploratory. If the gap survives, it becomes a second finding; if not, the text says so (IMPACT_PLAN.md, step 11). Recommendation: run it.

## 16. CSD3 SSH: please log in again (2026-11-20) — RESOLVED 2026-11-21

**Resolved 2026-11-21, about 08:00: you logged in again.** The cluster copy was synced to the 21 pins, and the 8B check was submitted as job 41575304 (STATE, hand-off of 2026-11-21).

**Context.** Item 15's login was not renewed before the session of 2026-11-17 ended, and no master connection exists now (checked 2026-11-20: `Permission denied (publickey,keyboard-interactive,hostbased)`). Nothing of this project is queued or running.

**What waits on it.** Under option (b) (item 14): sync the cluster's `study3` copy to the 21 pins and check them with `sha256sum -c`, then submit the 8B check (`BUDGET=240m sbatch slurm_jobs/a8_gates_device_check.sh`, one whole node, about 2 hours). Then, if it passes, Qwen3-VL-32B's gates; and A8's A3 summary on CPU.

**What to do.** In this session, type:

`! ssh -fN -o ControlMaster=yes -o ControlPath=~/.ssh/cm-%r@%h:%p -o ControlPersist=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=10 eb2007@login.hpc.cam.ac.uk`

then give the password and the one-time code when asked. Keep the Mac awake and on power while the jobs run.

## 15. CSD3 SSH: please log in again (2026-11-16, found at 21:12) — superseded by item 16

**Context.** The SSH master connection you started at 21:25 on 2026-11-15 has ended. The checks at 04:03, 05:03 and 06:03 used it. The next check, due at 07:03, began at 09:19 and found the master gone (`read from master failed: Broken pipe`). That check then stalled until 21:12, when this session resumed; the delays suggest the Mac was asleep. New logins need your one-time code, so cluster work has stopped; local work goes on. The Mac's clock now shows Helsinki time; the times here are London times.

**What waits on it.** Nothing is at risk.
- This project has had nothing queued or running since Phase A ended at 23:57 on 2026-11-15.
- At 06:03, the last check, storage was 1,068.3 GB and 991,641 files, unchanged since 02:19.
- The next cluster steps wait on item 14 anyway: the A3 summary, then, under (b), the 8B check. So log in when you answer item 14.

**What to do.** As in item 13: in this session, type the line below, then give the password and the one-time code when asked.

`! ssh -fN -o ControlMaster=yes -o ControlPath=~/.ssh/cm-%r@%h:%p -o ControlPersist=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=10 eb2007@login.hpc.cam.ac.uk`

The 12-hour `caffeinate` I started on 2026-11-15 has also ended. I have not restarted it, since there is no connection to keep alive.

## 13. CSD3 SSH: please log in again (2026-11-14, 22:36)

**Context.** The Mac went into lid-closed sleep at 21:51:47 and woke at 22:34:52. The SSH master connection exited during the sleep: its keepalives give up after five minutes without an answer. New logins without it are refused (`Permission denied (publickey,keyboard-interactive,hostbased)`), because CSD3 asks for a one-time code. Cluster work has stopped; local work goes on.

**What waits on it.** A8's GPU equivalence test, job 40990285. It started at 21:46 on gpu-q-14 and was expected to end by about 22:15; its outcome is unread. After it come the reading by `a1_gpu_rule`, the cancellation of the held `gpu2` job 40399722, and A8's code-pin step.

**What to do.** In this session, type the line below, then give the password and the one-time code when asked. A master started from your own terminal works just as well, since the socket path is the same.

`! ssh -fN -o ControlMaster=yes -o ControlPath=~/.ssh/cm-%r@%h:%p -o ControlPersist=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=10 eb2007@login.hpc.cam.ac.uk`

**To keep it alive.** The connection survives only while the Mac is awake. On battery, closing the lid puts it to sleep whatever `caffeinate` says; on power with the lid open, `caffeinate -i` keeps it awake.

**Resolved 2026-11-15, 21:25: you logged in again.** The equivalence test had passed (DECISIONS 2026-11-15, row 98), 40399722 was cancelled, the code-pin step was completed on the cluster, and A8's first job began running at 21:35.
