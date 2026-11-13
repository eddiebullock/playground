# STATE

Single source of truth for the paper-v2 agent. Update after every meaningful step.

**Updated:** 2026-11-13 20:00 Europe/London

## Hand-off, 2026-11-13 (written 05:35, last refreshed 20:00)

**New at 17:50: you allowed the `intr` QoS for this project** (DECISIONS 2026-11-13).
- Its limits, read with each field named: jobs of at most 1 h, and one job per user at a time, pending or running, of up to 3 nodes. It adds 10,000 priority points, against 100 for `gpu2`.
- As submitted, only the two A9 timing smokes (30 min) and the four A7 Stage 1 tasks (59 min) fit. A2 (12 h), A4 (6 h), the A9 scored runs (6 h) and the A8 check (4 h) do not.
- It is not an empty lane. At 17:58, 11 other users' `intr` jobs were waiting on the A100s, the oldest since 2026-11-11, but this user's own `intr` jobs (`hel-*` yesterday, `cas_*` today) started within 1 to 8 minutes.
- A9 smoke 40843619 moved to `intr` at 17:55, with the same job, script and 30-minute limit. The rest follow one at a time.
- A4's runner resumes a shard where it stopped (`scripts/a4_run.py`), and the pre-run check needs only the planned cells in each shard file. So A4 can run in one-hour slices on `intr` with no change to frozen code (DECISIONS 2026-11-13, second entry).
- Since 2026-11-12 13:26, no 1-GPU `intr` job on the A100s has started, ours included, while single-node 4-GPU `intr` jobs started within minutes. So at 18:17 the A9 smoke went back to `gpu2`, with its age unchanged, and the first A4 slice went in: job 40920485, Gemma shards 0 to 3 on one node, 52 minutes each, through the unchanged `study3_a4.sh`. Those four shards' `gpu2` tasks (40589226_0 to _3) are held, so no two runs write one file.
- **How the slices continue:** a loop on the Mac (scratchpad `a4_chain.sh`, under `caffeinate -i`, started 18:40) waits for each slice to end, then runs `research/scripts/a4_intr_next_slice.sh` for real. That script reads mechanics only (line counts, non-ok status counts, partial lines, mean seconds; no score field). It holds the `gpu2` task of each shard it slots, cancels the held task of each complete shard, and submits the next slice.
  - The loop stops and tells me on: a slice not ending COMPLETED, a planner STOP, fewer than 1,000 new lines in a slice, SSH failures, or no new job ID.
  - By hand, from the Mac: `ssh <host> 'DRY_RUN=0 bash -l -s' < research/scripts/a4_intr_next_slice.sh`. Only one `intr` job may exist at a time.
  - **Rate (measured over the first full slice):** about 1.6 cells per second per shard, after about 6 minutes of model loading. Shards hold 6,281 to 6,899 cells, so each takes two slices, the second about half-length: about 12 slices, done around 03:30 to 04:00 if each starts within minutes.
  - **Risk:** the loop needs this Mac awake and the SSH connection up. `caffeinate` stops idle sleep, not a closed lid.
- **A2 follows A4 on `intr`** (DECISIONS 2026-11-13, "A2's full-run shards may run in one-hour slices"):
  - `slurm_jobs/study3_intr_slices_a2.sh` runs four A2 shards per slice through the pinned `study3_stimulus_interventions.sh` (sha256 `1c8d22ea...`) with the full run's settings. Before each slice it applies A2's torn-line rule (A2-2), with a log.
  - `research/scripts/a2_intr_next_slice.sh` plans each slice. A2 has six shards of 2,349 cells, 4 to 6 GPU-hours each, so about 9 or 10 slices.
  - A sequencer on the Mac waits for the A4 chain to end. If A4 ends with every shard complete, it starts the A2 chain (scratchpad `intr_chain.sh`). If A4 stops for any other reason, A2 does not start.

**Done this morning on your instructions (about 09:15):**
- **Item 10: your notes file is off the public branch.**
  - `p1-pipeline` was rebuilt without it and force-pushed; the new tip is `f37beae`.
  - Every other change is kept, including all of today's commits, with the same messages and dates. Only the commit IDs changed, and the documents now cite the new ones.
  - Your local `notes/notes.txt` is unchanged.
  - The old history is kept on your Mac only, in the local branch `backup/p1-pipeline-before-notes-removal`.
- **Item 11:** [1] recoded to "no" on option sets. Figure 1 rebuilt (DECISIONS 2026-11-13).
- **Merged into `main`:** PR #114, true merge commit `ee3913a`, then PR #115 (`da95a75`, 11:48) with the later records, then PR #116 (`d1d1246`, 16:58) with the two length cuts and the afternoon records, then PR #117 (`40f38e9`, 17:58) with cut 3. Today's commits now count on your GitHub contribution graph.
- **Ethics:** you said the ethics position is fine, so Methods ("Human data") and the reporting summary now carry the recommended statement. It says no separate ethical approval was required, because the study recruited no participants and used only existing materials under their permission or licence terms. It is marked provisional, for you to confirm before submission.

**Only you can do these** (details in `research/NEEDS_INPUT.md`):
1. Later, before submission: confirm the provisional ethics statement with your institution (Methods, "Human data"; reporting summary; NEEDS_INPUT 9).
2. At submission (item 9): the Zenodo DOI for the code; an OSF deposit of the tagged preregistrations, if wanted; authors and affiliations. The Snoek group's reply is still awaited (item 8). Nothing waits on it.
3. Optional (item 10): GitHub keeps the old commits reachable by their IDs until it purges them. To remove them sooner, ask GitHub Support.
4. To review, not urgent: a reading decision I made overnight for A8 (dated note at the end of `research/PREREG_A8_DRAFT.md`; DECISIONS 2026-11-13).
   - A2's and A4's rulings of 2026-11-12 now apply to A8's runs of the same analyses, so A8 rows are read like the frozen rows above them.
   - One A8-specific choice: A8's hashed command fixes the A4 cells folder, so non-finite rows are filtered in place, with the originals kept unchanged in a subfolder the analysis does not read.
   - It is pre-data with no hash change.
   - An independent review of the night's A8 code then found real problems, all now fixed. One was a regression I had introduced. The addendum at the end of the A8 prereg lists them.
   - A8's next step is its code pins (Freeze checklist item 2). That step also needs the GPU equivalence test (40399722), which is still queued.
   - You said to proceed, so I will take the step when that job has run, unless you object to the note first. If you object after the pins, the change is still possible as a dated pre-data deviation.
5. To review, not urgent: the paper is too long. Once one branch per outcome is kept, the main text should come to about 6,700 to 7,700 words, against the journal's limit of 5,000.
   - The cut plan is in `research/paper/v2/outline.md` ("Length check"). No cut changes a claim, a number or a disclosure.
   - Its biggest choice: the numbered Limitations list (963 words) moves to the Supplementary Information, and the Discussion keeps a paragraph of about 250 words.
   - I will apply the plan when the text is assembled after the results, unless you would rather keep the list in the main text. The cuts would then have to come from elsewhere.

**Cluster: nothing of ours started overnight, and nothing failed.**
- Every job is pending on priority, and Slurm gives no start estimate:
  - A2: 40468625, 40468626
  - A4: 40589226, 40593824
  - A8 equivalence: 40399722
  - A7 Stage 1: 40795186, 40795187
  - A9 smokes: 40843619, 40843620
  - A9 scored runs: 40863756, 40863757
- Why they wait (corrected at 11:20; I had misread the free-tier limit overnight, DECISIONS 2026-11-13):
  - The A100 partition is full, and every tier shares it. At 11:16, 46 of its 74 nodes were busy, 17 were in maintenance, 1 was down, 1 was powering up and 9 were powered down with no recent use. Even interactive jobs at priority 10,000 were waiting.
  - The free tier's 32-GPU limit is per user, not shared by all SL3 users as I wrote overnight. It is not what holds us back.
  - 381 pending jobs ranked above our oldest (2,405 tasks counting each array task). 233 are paid-tier: their tier adds 500 points to priority, against our 100, so a newly submitted paid job ranks above ours until ours has aged about 11 days.
  - Our priority grows only with age, about 1.5 a hour; our oldest jobs were at 208 at 11:16, about 73 h of counted age.
  - Fair share adds little to anyone here: at most 48 points among the jobs above ours. Ours is 0 because your user has used 891 GPU hours on this account since 2026-10-16 (CASCADE `cas_*` 709, this project 168, `hel-*` 14). I have not touched the other projects' jobs.
  - Our past starts all came from backfill; a 12 h task submitted on 2026-11-04 waited about 4.4 days.
- Nothing more could be done for free until 17:50, when you allowed the `intr` QoS (top of this hand-off). The A2 and A4 time limits stay as they are on `gpu2`, because a TIMEOUT would cost days in the queue.
  - The free CPU allocation (199,744 h) is no way round it either (checked 14:29). A8's preregistration pins the A100, and A2's and A7's costs were measured on A100 in bfloat16. By my estimate, CPU inference would be 10 to 100 times slower, and its numbers would not match the GPU runs.
- **Risk:** every GPU node is reserved from 2026-12-07 07:00 to 2027-01-10 (`Dec2026-shutdown`), and A8 can start only after A2 and A4 finish.
  - A8's own dated-note rule defers whatever cannot finish, A4 first.
  - A8 can add only a statement of scope, so no frozen verdict depends on it.

**Done overnight** (dated entries below; reasons in DECISIONS):
- **A9:**
  - Timing smokes and scored runs submitted. The scored runs went in before their smokes ran, to gain queue age; nothing was read (DECISIONS 2026-11-13).
  - A9's pre-run check now enforces replication rule (ii).
- **A7 Stage 1:** added to the Phase 3 runbook, with its own pre-run check.
- **A8:**
  - An audit of its path from launch to paper found three faults in A8's own code. The worst: the natural-clip step could never have finished. All three are fixed with tests, before A8's code is pinned.
  - I also wrote the dated note in item 5, an A8 pre-run check, an A8 runbook (ROADMAP Part 3) and a `pull a8` command.
  - An independent review of that code found more problems, now fixed (DECISIONS 2026-11-13).
  - The pending equivalence job (40399722) runs from its own copy of the A8 sharding code. That copy is byte-identical to the code to be pinned, so its result will count.
- **Phase 3 rehearsal:** run end to end on synthetic cells for A4, A9 and A7. A4's real analysis should take minutes.
- **Found and fixed before any data:**
  - A8's `a4_analyse` step could not have run. The Snoek archive is now on the cluster at its pinned hash.
  - A test-order fault in the suite.
- **Paper:**
  - Independent draft review (41 findings) applied, except citation renumbering (left for the pre-submission pass) and item 11.
  - Results sections 5, 6, 7, 9 and 10 and the Discussion drafted for every outcome, before any of their data.
  - Table 1 and the A7 Stage 0 table built.
  - Deviations table, 80 rows.
  - Submission pack: software versions, ethics draft, code availability.
- **Checks:**
  - Full test suite at `d6dcb93`: 1,096 passed, 26 skipped, 0 failed (09:28). Clean-checkout rerun of the steps changed by the recode at `d6dcb93`: 6 rebuilt and matched, 1 verified by hash, 0 failed.
  - Clean-checkout reproduction at `f4306cb`: 43 rebuilt and matched, 112 verified by hash, 0 failed.

**FYI:** CSD3 `/home` is at 49.8 of 52.4 GB. This project writes nothing there; its jobs write to RDS. The space is in other folders, such as `LEAP3mri`. RDS: 1,067.4 GB of 1,099.5; 991,094 of 1,048,576 files.

**When jobs finish**, the heartbeat follows ROADMAP Part 3: pull unread, pre-run check, one frozen analysis with its run note, then registry rows and figures.

## Current phase

**Phase 2 runs queued; Phase 3 analyses wait on them.** A1, A3 and natural-clip RQ1 are locked (2026-11-09). Every amendment is frozen or closed: A2, A4, A5, A7, A8 and A9 frozen, A6 closed as a limit. The A2, A4, A7 Stage 1, A8 and A9 GPU jobs are pending on priority. History: the prompt confound found 2026-11-04 withdrew the earlier accuracy and probe results (DECISIONS); Branch B stays as a record only.

**2026-11-13 12:30:** Queue unchanged at 12:25 and again at 13:29: nothing started or failed. Oldest jobs at priority 209, then 211; 361, then 358 jobs ranked above ours; RDS 1,067.5 GB, 991,119 files. Length check of the draft (`paper/v2/outline.md`, "Length check"): the main text is set to come to about 6,700 to 7,700 words once one branch is kept, against the limit of 5,000 (Introduction 1,606; Results 2,700 to 3,200; Discussion 2,400 to 2,900). A cut plan is recorded there and in ROADMAP Phase 4, with hand-off item 5 for the one structural choice.

**2026-11-13 14:29:** Nothing of ours started or failed; the oldest jobs are at priority 212. More jobs now rank above ours: 379 (2,677 tasks), against 358 (2,423) at 13:29. RDS 1,067.5 GB, 991,119 files. I checked whether the free CPU allocation could take any of the queued work; it cannot (hand-off, "Nothing more can be done for free").

**2026-11-13 15:30:** Queue unchanged for us: nothing started or failed; oldest jobs at priority 214, with 383 jobs (2,442 tasks) ranked above them. RDS 1,067.5 GB, 991,119 files. Applied the one length cut that depends on no pending result (cut 4, in part; `paper/v2/outline.md`): the Introduction's paper-by-paper list gave way to a pointer to Figure 1e, which codes the same eight papers, and its search-coverage sentences moved to Supplementary Methods, Section S2. Introduction 1,606 to 1,470 words; no claim, number or citation left the paper. The citation check carries a dated note.

**2026-11-13 16:36:** Queue unchanged for us: nothing started or failed. Our oldest jobs are at priority 215, and the count above them keeps growing: 402 jobs (2,919 tasks), against 358 at 13:29. RDS 1,067.5 GB, 991,119 files. Applied length cut 5, which also depends on no pending result: the Discussion's benchmark-audit recommendation went from 341 to 189 words. Its detail moved word for word to Supplementary Methods, Section S2; the recommendation, its exploratory status and the no-field-level-claim scope stay in the Discussion (DECISIONS 2026-11-09). Decisions table row 60 now names both places. Full suite at `f95e012`: 1,096 passed, 26 skipped, 0 failed. Merged into `main` as PR #116, true merge commit `d1d1246` (16:58).

**2026-11-13 17:38:** Queue unchanged for us: nothing started or failed. Our oldest jobs are at priority 217, with 400 jobs (2,668 tasks) ranked above them. RDS 1,067.5 GB, 991,119 files. Applied length cut 3, the last cut that depends on no pending result: about 200 words came out of the Results' locked prose (`paper/v2/outline.md`).
- Section 1: the literature check is now a short summary pointing to Figure 1e; the full coding moved word for word to Supplementary Methods, Section S2.
- Section 4: the natural-clip recovery values gave way to one sentence; all were already in Methods and Section S5.
- Section 5: the run 3 rates and exempt-cell values came out; both were already in Methods and Section S6.

With cuts 4 and 5, the main text is about 490 words shorter. What remains waits on cut 1 (hand-off item 5, your view), cut 2 (after the results) and writing each branch to its budget.

**2026-11-13 18:05:** Full suite at `9181a4d`: 1,096 passed, 26 skipped, 0 failed. Merged into `main` as PR #117, true merge commit `40f38e9` (17:58). You allowed the `intr` QoS at about 17:50: A9 smoke 40843619 moved there at 17:55 and is pending, seventh in the `intr` line at 18:03 (DECISIONS 2026-11-13; hand-off, top). Decisions table row 84 added.

**2026-11-13 18:20:** A4 slices started (DECISIONS 2026-11-13, "A4's scored shards may run in one-hour slices"). The wrapper `slurm_jobs/study3_intr_slices.sh` passed 17 stub checks on the login node and was committed (`eefa487`) before being copied to the cluster; it matches the commit (sha256 `18d1e8ed...`). Held 40589226_0 to _3; A9 smoke 40843619 back to `gpu2`; slice 40920485 (Gemma shards 0 to 3) pending at 18:17. Decisions table row 85. ROADMAP's A4 trigger now names the slice jobs and the waivers.

**2026-11-13 18:40:** Slice 40920485 started at 18:22 on gpu-q-13, about 5 minutes after submission. Model load took about 6 minutes (four processes at once). At 18:35 every shard file held only "ok" cells at about 0.65 s of model time per cell, and lines grew at about 0.9 per second per shard. A planner (`research/scripts/a4_intr_next_slice.sh`, commits `0f2d993` and `a471bfe`) and a local chain loop now carry the slices on; DECISIONS has a dated update (18:30).

**2026-11-13 19:27:** The first automatic handover worked.
- Slice 40920485 ended COMPLETED at 19:14 (52 min). All four Gemma shards stopped cleanly on the budget.
- Each shard got 4,433 to 4,499 cells, all "ok", none partial, at about 0.61 s of model time per cell: 17,892 lines in all.
- The chain loop submitted slice 40925702 (the same shards) at 19:18, and it started at 19:19 on gpu-q-9.
- So each shard takes two slices, the second about half-length. A4 in all is about 12 slices, done around 03:30 to 04:00 if starts stay this quick.
- Queue otherwise unchanged; no failures; RDS 1,067.5 GB, 991,136 files.

**2026-11-13 20:00:** A2 prepared to follow A4 on `intr`, before any A2 cell exists (DECISIONS 2026-11-13, A2 slices; decisions table row 86; commit `e601019`).
- The wrapper passed 23 stub checks, including the A2-2 repair. It is on the cluster and matches the commit (sha256 `f3503d8d...`).
- The planner's dry run shows no A2 cells yet, and would first slot Gemma shards 0 to 2 and Qwen shard 0.
- The sequencer is armed.

**2026-11-13 11:47:** Supplementary decisions table brought up to date: rows 81 to 83 for today's three new decision-log entries, and a superseded-entries row (75 to 83). It again has one row per entry, 83, checked in order against DECISIONS. ROADMAP Phase 4 status refreshed: every step that needs no new GPU output is done. The node counts in the 11:20 correction now include the node that was powering up. Full suite at `62a8b79` with these record edits: 1,096 passed, 26 skipped, 0 failed. Merged into `main` as PR #115, true merge commit `da95a75` (11:48).

**2026-11-13 11:20:** Queue unchanged at 11:16: nothing started or failed. Oldest jobs at priority 208. Correction: the 32-GPU limit of `gpu2` is per user (`MaxTRESPU`), not shared by all SL3 users as written from 03:15 on; at 11:12 `gpu2` had 36 GPUs in use, 32 of them one user's. Our jobs wait because the A100 partition is full (46 of 74 nodes busy, 17 in maintenance, 1 down, 1 powering up, 9 powered down) and 381 jobs rank above ours, 233 of them paid-tier. No decision changes. Corrected in the hand-off and the ROADMAP risk row; a new DECISIONS entry, with a pointer from the A9 entry; a dated note in the A9 prereg, outside its hashed block. Entries below keep their "of 32" counts: the GPUs in use were counted right, only the limit was misread. RDS 1,067.5 GB, 991,118 files.

**2026-11-13 09:50:** Queue unchanged at 09:47: nothing started or failed. Oldest jobs at priority 205; `gpu2` had 28 of 32 GPUs in use; 1,270 `gpu2` tasks rank above ours. RDS 1,067.4 GB, 991,094 files.

**2026-11-13 09:35:** Ethics statement written as recommended, marked provisional, on your instruction (Methods, reporting summary, NEEDS_INPUT 9). Checks at `d6dcb93`: full suite 1,096 passed, 26 skipped, 0 failed. Clean-checkout rerun of the recode's steps: Figures 1 and 3, `SOURCES.md`, the derived counts and the table's hash all matched. `p1-pipeline` merged into `main` as PR #114, true merge commit `ee3913a` (09:33), so today's commits count on your GitHub contribution graph, which counts only the default branch. From now on I merge into `main` after each coherent step, so each day's work shows the same day.

**2026-11-13 09:25:** You returned at about 09:05 and approved NEEDS_INPUT items 10 and 11.
- Item 10: `p1-pipeline` was rebuilt from `d8c2e01` with `notes/notes.txt` restored in all 40 later commits, checked, and force-pushed with a lease. The new tip is `f37beae`.
  - Each rebuilt commit matches its original in message, author, committer and dates, and differs only in the notes file.
  - No tag or PR held the old commits.
  - Cited commit IDs in the documents were moved to the rebuilt ones.
- Item 11: [1] recoded to "no", and Figure 1 rebuilt. Figure 3 rebuilt byte-identically, which confirms the environment.
- Today's work is all committed and pushed. The only untracked project files are legacy `results/stats` outputs from June, left out on purpose.
- Queue unchanged at 09:08: nothing started or failed; `gpu2` had 28 of 32 GPUs in use.

**2026-11-13 08:50:** Queue unchanged at 08:46: nothing started or failed. Oldest jobs at priority 204 (age 104 of 1,000; age grows 1,000 over 28 days, about 1.5 a hour; fair share 0). `gpu2` had all 32 GPUs in use; the first of those jobs is due to end by about 10:44, and 1,261 `gpu2` tasks rank above ours. RDS 1,067.4 GB, 991,082 files.

**2026-11-13 07:50:** Queue unchanged at 07:46: nothing started or failed. Oldest jobs at priority 202; `gpu2` had 28 of 32 GPUs in use; RDS 1,067.4 GB, 991,082 files. The equivalence job's copy of the A8 sharding code (`~/rds/hpc-work/a8_check`) is byte-identical to the local files to be pinned (`a8_rotation_shard.py`, `a8_design.py`, `a8_preflight.py`, `scripts/evaluate.py`), so its result will count for the pinned code.

**2026-11-13 06:55:** Queue unchanged at 06:54: nothing started or failed. Oldest jobs at priority 201; `gpu2` had 24 of 32 GPUs in use; RDS 1,067.4 GB, 991,082 files. Full suite at `86c0441`: 1,096 passed, 26 skipped, 0 failed.

**2026-11-13 06:45:** An independent review of the night's A8 code (commits `b5c4896`, `d4f11d4`, `d0e945e`) found real problems. Two were in the new A4 pre-run check, and one was a regression from `d0e945e`: a crashed gates job could block both models. I checked each against the code and fixed all of them before A8's code pins:
- A8's A4 run now accounts for every eligible or launched model.
- Files move only on a passing check, never onto an existing file.
- The single analyses refuse without a passing check record.
- An interrupted filter can no longer lose a shard.
- A retry still in Slurm's queue is no longer counted as ended.

The A8 note has a same-day addendum, and the block hash is unchanged. `tests/test_a8.py`: 74 passed, 1 skipped, with a regression test per finding (DECISIONS 2026-11-13).

**2026-11-13 06:25:** Queue unchanged at 05:54 (oldest jobs at priority 200; `gpu2` 20 of 32 GPUs in use; RDS unchanged).
- **Reproduction.** Clean-checkout reproduction at `f4306cb`: 43 rebuilt and matched, 112 verified by hash, 0 failed, 6 pending.
- **A8 path audit (ROADMAP local item 4).** An independent agent audited the path; I checked each finding against the code before acting.
  - **Three faults in A8's own code,** which is not pinned yet. All are fixed with tests (`b5c4896`; DECISIONS 2026-11-13):
    - The natural-clip step could never have finished: the frozen script's `sys.exit(main())` ended the launcher before the output was moved.
    - The dated note of 2026-11-09 on out-of-memory retries was not in the code.
    - Provenance could not be checked after a pull.
  - Also: A8's A4 analysis now runs single-threaded, as A4's does.
  - **A2's and A4's rulings of 2026-11-12 carried over to A8** by a dated note, pre-data, with the block hash unchanged (`95dd41b5`). A new `research/scripts/a8_prerun_check.py` applies them to A8's own inputs.
  - **A8 runbook** written (ROADMAP Part 3).
  - The two minor items are done too, with the block hash unchanged:
    - gate and delivery files are read only from jobs that ended ok;
    - the gates file and the A2 output carry provenance.
  - `post_phase3.py pull a8` is written as well. It checks provenance on the cluster, copies the outputs, and records the sha256 of the cells that stay there. Nothing from the audit is open.
  - Full test suite at `d0e945e`: 1,088 passed, 26 skipped, 0 failed (13 min).
  - Tests: `tests/test_a8.py` 66 passed, 1 skipped; `tests/test_a2_declared.py` passes.
- **Paper.** The Figure 5 and 6 markers now say their functions exist and wait only on data, and the Figure 6 legend names the fill-rule input.

**2026-11-13 05:35:** Queue unchanged at 05:22: nothing started or failed. Priority is 199 for the oldest jobs (age 99, fair share 0), and `gpu2` had 24 of its 32 GPUs in use. RDS 1,067.4 GB, 991,082 files. Morning hand-off written at the top of this file. ROADMAP local item 4 is under way: a read-only audit of A8's pull and analysis path against its launcher.

**2026-11-13 04:25:** Queue unchanged at 04:20 (nothing started or failed; RDS 1,067.4 GB, 991,082 files). Submission pack (ROADMAP 4.7): the reporting summary's software row now lists the verified versions (analysis env; torch and transformers as the vision-gate outputs record them; LibreFace 0.2.0, OpenFace 3.0 as `openface-test` 0.1.26, MediaPipe 0.10.14 and ME-GraphAU commit `dda53cba`, read from the cluster environment). Methods now carries a draft ethics statement (with a [USER] marker for the institutional part), the reproduction-script sentence and the journal's standard reporting-summary line. A9's prereg has a dated note for the early submission and the rule (ii) check, and the preregistration links table has the A9 row; the A9 design hash is unchanged (`21341eb4`, 53 A9 tests pass).

**2026-11-13 03:25:** A9 scored arrays submitted before their smokes ran, to gain queue age: Gemma 40863756 and Qwen 40863757, 12 shards each, 6 h cap, `gpu2`. The 03:15 check found `gpu2` capped at 32 GPUs for all SL3 users together (28 in use), with about 870 higher-priority `gpu2` tasks ahead of our oldest job. With fair share 0, our priority grows only by age, about 1.5 a hour. The smokes are older and shorter, so they still run first, and their timing now only lowers the scored limits (DECISIONS 2026-11-13). Before submitting I checked on the cluster: the 33 run-path files, the manifest pin and the 40 render tars. The A9 pre-run check now also requires A4's run note and compares A4's summary with the sha256 recorded there (A9 rule (ii)), so A9's single analysis cannot read another A4 summary. Other jobs unchanged; RDS 1,067.4 GB, 991,082 files. Full test suite at `c88ca73`: 1,076 passed, 26 skipped, 0 failed (22 min).

**2026-11-13 02:15:** Queue unchanged (checks at 01:22 and 02:14; SSH master up; RDS 1,067.3 GB, 991,070 files). Results sections 5, 6, 7, 9 and 10 now carry drafted text for every outcome, written before any of their data, with values as slots for `registry_tables.py`. The A7 pre-run check is now tested against the A2 run note as `post_phase3.analyse` writes it from the real command list.

**2026-11-13 00:55:** Queue unchanged (priority 192, rising about 1.5 a hour). Phase 3 rehearsed end to end on SYNTHETIC cells in the runners' exact format (`research/scripts/rehearse_cells.py`; all in the scratch folder, nothing under `results/`): A4 and A9 manifests rebuilt with their pinned sha256; A4 pre-run check passed; the frozen `a4_analyse` ran in 383 s single-threaded and wrote all eight decisions and both headlines; A9 pre-run check passed and the frozen `a9_analyse` ran in 45 s with the fixed replication wording; `registry_tables.py a4`, Figure 5 (`fig5_from`) and Table 1 all read the real-schema summaries correctly. The A7 chain was rehearsed earlier (`tests/test_a7_main_integration.py`); A2 and A5 were rehearsed by their code audits (2026-11-12). One gap found, in the synthetic rows only (blind rows need `lexicon_mass`, which the real runner writes). A4's real analysis should take minutes, not hours.

**2026-11-13 00:10:** Queue unchanged. Full test suite 1,064 passed, 26 skipped, 0 failed; clean-checkout reproduction at `feefec8`: 40 rebuilt and matched, 112 verified by hash, 1 failure fixed and rechecked (REPRODUCE.md, 2026-11-13). Built from locked outputs and rebuilt by `reproduce.py`: Table 1 (`paper/v2/table1.md`, `research/scripts/make_table1.py`; A2, A4 and A9 cells read pending until their run notes are complete) and the Supplementary A7 Stage 0 table (`make_stage0_table.py`). Results-day aids: `research/scripts/registry_tables.py a2|a4` prints registry tables from the outputs (frozen verdict, fill-rule verdict and .01 reading side by side); `tests/test_a7_main_integration.py` runs the frozen A7 entry point on the inputs file the A7 pre-run check writes. Methods Disclosures 2 and 3 now say which A1 files were opened, and when.

**2026-11-12 23:50:** Queue unchanged at every hourly check (all jobs pending on priority, no start estimate). Found: a cluster-wide GPU shutdown, reservation `Dec2026-shutdown`, 2026-12-07 07:00 to 2027-01-10 (ROADMAP risk register); and A8's `a4_analyse` step could not have run anywhere, so the Snoek archive was placed on the cluster at its pinned sha256 (DECISIONS). A4 smokes ran about 1 s per cell, so A4 shards should take about 2 h of their 6 h; time limits left as submitted. Independent draft review done (`research/reviews/draft_review_2026-11-13.md`, 41 findings, numbers nearly all correct, references all real) and applied, except citation renumbering (pre-submission pass) and the [1] option-set code (NEEDS_INPUT 11). Deviations table built (`paper/v2/supplementary_table_decisions.md`, 74 rows); its four disclosure gaps closed, including A3's declared token-floor sensitivity, now computed with the frozen functions (Gemma 4 still passes on 218 of 243 clips). Figure 6 now plots A2's verdicts after the fill-rule ruling. A test-order fault found and fixed (`tests/conftest.py`): A7's recovery tests swapped A2's sign-flip function in-process; the real analyses run as fresh processes and never see it. Discussion drafted for every open branch before its data. Commits `d3ac428` to `d4cce83` on `p1-pipeline`; no PR to `main` while NEEDS_INPUT 10 (notes file pushed by mistake) is open.

**2026-11-12 22:45 (new agent session; the user is away until the morning of 2026-11-13):** Took over after the previous session closed. The user re-authenticated CSD3 on a new ControlMaster with `ControlPersist=yes` and keepalives (the 12 h persist is what dropped it on 2026-11-11), and `caffeinate` holds the Mac awake. The user has given the `intr` QoS to another project's agent, so this project does not use it; none of its jobs did (all `gpu2`). Queue unchanged: A2, A4, A8 and A7 Stage 1 pending on priority (about 189, fair share 0, no start estimate). RDS 1,067.3 GB, 991,062 files. Submitted the A9 timing smokes (Gemma 40843619, Qwen 40843620; 40 cells, 30 min) after checking run-path hashes, the manifest pin and the 40 render tars on the cluster (DECISIONS 2026-11-12). Added A7 Stage 1 to the Phase 3 runbook with its own pre-run check, `research/scripts/a7_prerun_check.py`, which also builds the inputs file `a7_analyse.py` needs; tests pass (`tests/test_a7_prerun.py`). A self-paced heartbeat checks the cluster about hourly overnight.

**2026-11-12 16:55:** A7 Stage 1 route arm submitted (Gemma 4 40795186, Qwen3-VL 40795187; 2 tasks each, 59 min cap, DECISION_DATE 2026-11-10), before A2's output exists, to gain queue age; its A2 preconditions are applied by A7's analysis code (DECISIONS 2026-11-12). Pending on priority with the A2, A4 and A8 jobs.

**2026-11-12 16:45:** Reconnected after the user's break; SSH master still up, queue unchanged, RDS 1,067.3 GB. A9 pre-run check written (`research/scripts/a9_prerun_check.py`). Phase 3 runbook written and tested: `research/scripts/post_phase3.py` (pull unread with hash checks, then pre-run check and frozen analysis once, with a run note). The Snoek archive is not on the cluster, so analyses run locally after the pull; local and cluster copies of the four analysis scripts have equal sha256.

**2026-11-12 15:50:** Independent code audits of A4, A2 and A5 finished before any of their data exist (`research/reviews/a{4,2,5}_code_audit.md`). None found wrong code; every pin matches its tag. Pre-data rulings appended to each amendment as dated deviations, with no hash change (A2 `0ae22118`, A4 `12c454da`, A5 `fd20495b`), and logged in DECISIONS: A4's Holm family fixed at the two models and non-finite rows unusable; A2's fill rule also governs Opposes; for A4 and A9, a result meeting both Aligned and Not aligned is reported Inconclusive; input errors void a run rather than count as results. Pre-run checks written and tested: `research/scripts/a{2,4,5}_prerun_check.py`, plus `a2_declared_extras.py` and the scripted gates file `research/reproduction/a4_gates.json` (`02b026ea`). A9 got the same readings and a status note (frozen 2026-11-09, tag governs). Cluster copies of A2's gate files checked equal to the committed ones. Supplementary Note 1 now carries the full proofs (Appendix A), and its noise-ceiling simulation is worded as a check on the point estimate, with recovery's boundary rate as the evidence for the rule as applied. Literature recheck: nothing new. The clean-checkout reproduction rerun against the current commit is running (the first run used a stale local main; its 6 mismatches were that, not real). Jobs still pending.

**2026-11-12 15:10:** User confirmed that EU-Emotion's creators are collaborators and our copy is the working set, so outline item 16 is closed; the seven item IDs are now in `eu_missing_items_check.json` (DECISIONS 2026-11-12). Independent code-against-text audits of A4, A2 and A5 started (reports to `research/reviews/a{4,2,5}_code_audit.md`), before any of their data exist. Paper: Methods and Results slots that locked files answer are filled (A7 Stage 0, A6 closed, A9 freeze date, weak labels, literature table, half-shared power, Figure 3b values), with four registry rows added; Figures 2 and 4 drawn from the locked A1, A3 and natural-clip outputs and checked by eye. Methods Disclosure 11 records that A9 was tagged 37 minutes after the A1 analysis ran, with its last design changes from an audit finished before A1. Jobs still pending; fair share 0.000008, no start estimate. RDS 1,067.3 GB, 991,062 files.

**2026-11-12 13:40:** Disk cleanup done with the user's approval. Byte-identical model copies replaced by hard links: `study2/models/qwen3vl` now links to `cascade/weights/Qwen3-VL-8B-Instruct` (17 weight and config files; the 17 small download-metadata files differ and were left), and `mnml/weights/qwen3-vl-2b` links to `es_vlm/models/qwen3vl_2b` (12 files). Removed caches `sim/cache/hf` (4.9G), `cascade/cache/pip` (1.2G) and `social_ai/conda/pkgs` (531M), plus this project's scan and survey temp files. RDS 1,095.6 to 1,067.4 GB, files 998,719 to 991,061. GPU jobs still pending; Slurm estimate 2026-11-14 about 11:30 to 13:25.

**2026-11-12 10:00:** SSH was down from 2026-11-11 about 15:00 until the user re-authenticated this morning. No GPU job has started; A2, A4 and A8 still pending on priority (last Slurm estimate 2026-11-13 about 20:15). RDS 1,093.2 GB of 1,099.5 soft, 998,566 files. Outline item 16(a) settled for our copy: the seven workbook items without a clip are present as unedited originals only (`research/reproduction/eu_missing_items_check.json`). Disk cleanup asked for by the user is under way (caches and byte-identical duplicate weights).

**2026-11-11 10:40:** No GPU job has started. Slurm's estimated starts: A8 equivalence and A2 (both models) 2026-11-13 about 23:30; A4 Gemma 2026-11-14 00:00, Qwen 01:25. GPU fair share 0.000 (account usage 4,513 of 7,121 h). Shorter limits would not help: A4 and A2 shards need hours, and intr allows one job under 1 h. Nothing here is bought; SL2 would cost money. RDS 1,092.2 GB of 1,099.5 soft, growing about 0.8 GB per `hel-c3-intr` run (helsinki_cosmos, not this project), 998,445 files.

**2026-11-10 13:45:** A7 Stage 0 done on both models (exit complete; route arm admitted, token arm not); Stage 1 scope recorded before any A2 output. A6 closed as a limit: its first-frame AU check failed on GraphAU AU10 (.80 > .50), so RQ5 is future work. A4 scored runs submitted for both models. All GPU jobs waiting on priority (fair share 0). PRs 101 to 103 merged today.

**2026-11-10 10:40:** Phase 2 under way. Sync done 2026-11-09 (113 pins OK on the cluster). Submitted: A2 full run Gemma 40468625 and Qwen 40468626 (3 shards each); A4 timing smoke Gemma done (40 cells, about 5 min with load), Qwen smoke 40468637 pending; A4 scored Gemma 40589226 (12 shards); A7 Stage 0 Gemma 40468955, Qwen 40468956 (records via unpinned adapter, sha256 5b10a4d5); A9 render complete (40 tars); A8 equivalence 40399722 pending. A8 checklist items 3 and 6 done. Jobs were held from 2026-11-09 18:22 to 2026-11-10 10:34 while another project's container builds filled the file quota (DECISIONS 2026-11-09). RDS 1,085.2 GB at 10:34, above this project's 1,080 GB stop line, from `helsinki_cosmos` (38 GB, not this project); this project's runs write only small cell files.

**2026-11-09:** A8 (larger models) FROZEN on its design text, tag `a8-freeze`, hash 95dd41b5, before A1 was read; Gemma 4 31B downloaded to the lab project store with user approval; both weight copies verified. Liu et al. 2022 located as a second randomised human dataset; A9 replication being drafted. Benchmark leak audit done (methods note). CFA renamed Cue Effect Alignment. Array throttle raised to 5, all 10 Qwen tasks running, last expected about 17:00. Qwen A1: 24 of 30 rotation files at 14:58, six tasks running, none failed, last expected about 16:30 to 17:00. Nothing read. A9 narrowed by check (DECISIONS-style lead decision 8 in the draft, disclosed as post run 2), recovery run 3 identical to run 2, design hash 41c95dae; independent audit running. A9 FROZEN 17:40, tag `a9-freeze`, design 21341eb4, after audit and re-check. A1 result locked at 17:44 (item 1).

**2026-11-08 evening:** ROADMAP Phase 0 is closed except 0.13. A6 local items closed (PR 87): recovery run 5 matches run 4, freeze prepared, cluster items only. A7 audited (PR 86): numeric complete-cut rule, blocks are not cuts, RQ3 "not testable" where no complete cut is shown; A7 FROZEN 2026-11-08 (tag `a7-freeze`, design 9e40cfcf), before any A2 cell is read; riders descriptive, Stage 0 on the cluster is next. Literature (PR 85): Gold et al. wording fixed; CFA name clash with a WACV 2024 method was NEEDS_INPUT 9; renamed to Cue Effect Alignment (CEA) on 2026-11-09. Qwen A1 arrays still PENDING, estimate 2026-11-09 06:56 to 08:40; 20 of 30 rotation files.

## Next actions

**A4 frozen 2026-11-07:** tag `a4-freeze`, design hash `12c454da`, PR 73 merged as `fc4a327`, before any A1 result was read and before any model saw a render. Residual CFA is confirmatory, recovery 66 of 66 (66 of 68 counting the two withdrawn cells). Render finished 2026-11-07: arrays 40268765 and 40270481 (tasks 0 and 1 failed in 7 s on cpu-q-168 and were rerun on another node), 40 tars, 11,761 images as planned, 2.2 GB, in `results/a4/renders` on the cluster, no model. Scoring waits for RQ0, then a timing smoke per eligible model.

**A5 (RQ4) frozen 2026-11-08:** tag `a5-freeze`, design hash `fd20495b`, PR 80, before any A2 cell existed. After recovery run 6, balance verdicts (Face, Default, Even) are descriptive only, because `shared_family_gain0.5` fails at .094; the evidence classes are confirmatory, and none of their checks fail. A2 test checked under skewed gain nulls (PR 79): per-route FWER up to .14 in one skew direction; verdicts stay as frozen, with a fixed step-down p < .01 sensitivity. Both are in DECISIONS 2026-11-08.

Goals and what done means: `research/GOALS.md`. The user gave full autonomy on 2026-11-07 and accepted all recommendations, including the four A2 choices.

1. **A1 DONE 2026-11-09 (RESULTS_REGISTRY, "A1 rotated 6AFC").** All 30 files pulled unread with hashes checked; frozen analysis run once via `post_a1.sh`. Pass on all three models (delta Qwen .402, Gemma .228, Molmo .071; Holm p .00009 each). One Qwen trial errored (cluster I/O), handled by the pre-set rule. A1 Pass and A3 Pass on all three: face use on both inputs, RQ0 branch R0-a. Natural-clip RQ1 confirmatory for all three, pattern score positive (Qwen .549, Gemma .418, Molmo .543; p .0002 to .0006): aligned with the human pattern, not shown to use the same cues. Molmo's A1 accuracy is at chance (.163); its Pass shows face dependence, not competence. Next: Phase 2 sync, Gate B and A2 on Gemma 4 and Qwen3-VL.
2. A3 done 2026-11-05: all three models pass on `single_face`, Molmo at accuracy .072 (RESULTS_REGISTRY). Pass tests vision dependence, not competence. Full reading needs A1.
3. A2 FROZEN 2026-11-07, tag `a2-freeze`, `design_sha256` 0ae22118, after two independent audits and a third check of the fixes. Gemma and Qwen are eligible subject to RQ0; Molmo is out. The next step needs A1. Then sync A2 code to the cluster, confirm hashes, and run Gate B and the full suite.
4. Same-clip analysis: RQ1 pattern score on model frames is confirmatory, subject to the 5 percent clip rule (DECISIONS). The claim is "aligned with the human pattern". Cells are exploratory.
5. Vision gates: Gemma and Qwen pass G2 to G4, Qwen at bfloat16 only because float32 ran out of memory (DECISIONS 2026-11-05). Molmo fails G2 at bfloat16 and G4, so it is not eligible for A2 as drafted.
6. Scoring path: full forwards for all models unless DECISIONS says otherwise. Never patch Gemma image positions at L24 or later.

Merge with true merge commits, not squash: a squash on PR 50 caused a conflict on the next PR. Cluster quota: 1036.2 GB at 2026-11-07 10:02, under the 1099.5 GB soft limit, hard limit 1209.5. Deleted `es_vlm/models/qwen3vl_32b` that day after a reference scan: CASCADE loads its own copy from the project store, and es-vlm can fetch 32B again from Hugging Face at revision 0cfaf48183f594c314753d30a4c4974bc75f3ccb.

Do not cancel A1/A2/CAS jobs. `cas_*` jobs belong to the separate CASCADE project (`~/rds/hpc-work/cascade/code`), and appear to be resubmitted by a local loop there (`build/chain_loop.sh`), inferred from timing. Ablation 39709579, 39709583, 39709584 were cancelled 2026-11-02 and are not requeued.

SSH: `ssh -o ControlPath=/Users/eb2007/.ssh/cm-%r@%h:%p -o ControlMaster=no -o BatchMode=yes eb2007@login.hpc.cam.ac.uk`

Master (the user runs it, TOTP): `ssh -fN -o ControlMaster=yes -o ControlPath=/Users/eb2007/.ssh/cm-%r@%h:%p -o ControlPersist=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=10 eb2007@login.hpc.cam.ac.uk`. Do not use the `intr` QoS: the user gave it to another project's agent on 2026-11-12.

## Decoder

All video jobs must pin cv2 4.11 and pass `research/scripts/decoder_preflight.py`. See amendment A1.1.

## Access confirmed

EU-Emotion: full access and licence, per user 2026-11-04. Full RMET access. CREMA-D form done. Nothing from these goes in the public repo.

## Branch / git

- Default branch: `main`.
- Integration branch: `paper-v2`.
- Task branch: `p1-pipeline`.
- Tags: `v2-phase0`, `prereg-v1`, `a2-freeze`, `a4-freeze`, `a5-freeze`, `a7-freeze`, `a8-freeze`, `a9-freeze`.
- Repo visibility: **public**.

## Blockers (non-idle)

- Official H1 gate failed Holm (perm-resolution floor). Branch B locked; do not retune.
- MRMET option-count verification waits for Phase 6.
- Phase 4 cancelled under Branch B.
