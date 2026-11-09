# A9 pre-freeze audit

**Written:** 2026-11-09. This is an independent pre-freeze audit of `research/PREREG_A9_DRAFT.md` (design block `41c95dae...`), `research/notes_a9.md`, `scripts/a9_{design,analyse,run,recovery}.py`, `research/scripts/a9_{human_target,manifest}.py`, `slurm_jobs/study3_a9{,_render}.sh`, `tests/test_a9.py` and the A9 reproduction JSONs. It checks them against `research/PREREG_A4_DRAFT.md`, `scripts/a4_analyse.py`, `scripts/a4_recovery.py`, `scripts/a4_design.py` and `research/DECISIONS.md`.

The only file written is this one. Nothing was committed, the cluster was not touched, and no background job was left running. Scratch outputs are in the session scratchpad only.

**What was run**

- `a9_design` import, `check_code_pins(require=True)`, `check_pins()` and `check_liu_files()`.
- sha256 of every file listed under "Code hashes" and "Other A9 files".
- `/opt/anaconda3/bin/pytest tests/test_a9.py`: 42 passed in 25 s.
- Both manifests rebuilt from the `.mat` files.
- `a9_human_target.py target` re-run from the raw `.mat` files (finding 5).
- The three recovery files compared table by table.
- A cross-dataset label check (finding 16).

**Verified as correct**

*Hashes and pins*

- The design block hashes to `41c95dae865b0ee0e3c61051ac6caeb1516dfea1633bd013bf4fe16ab1a9ec29`. That equals `a9_design.DESIGN_SHA256` (`scripts/a9_design.py:27`) and the hash stated in the draft (line 5).
- All six code pins (`PREREG_A9_DRAFT.md:669-676`) equal the current sha256 of their files.
- The other pinned files all match: the two Slurm scripts, the five reproduction JSONs, and `tests/test_a9.py` (`4fefbb29...`).
- Every `a4_pins` and `input_pins` entry matches, as do both Liu `.mat` files.
- Recovery run 3 (`a9_recovery.json`) carries the current design hash. Its `code_sha256` equals the pins exactly.
- Run 3's tables and checks are identical to run 2's in every value. Run 2's `code_sha256` differs from the pins only in `a9_design.py` and `a9_analyse.py`; `a9_recovery.py` is the same.

*Manifests*

- Rebuilt from the `.mat` files, the design B manifest hashes to `4b2f04ae...` (equal to `manifest_sha256`) and the design A manifest to `dac1cd5a...` (equal to the draft, line 535).
- The structure matches the draft's table: 6,200 rows; 5,992 unique images; 600 F and 600 M; 1,200 main, 200 replicate, 1,200 frame_max and 3,600 all_identities rows.

*Liu response codes*

- There is no off-by-one. The block maps code 6 to "sadness", which equals A4's label (`a9_design.py:140`).
- `response_labels` indexes a lookup whose position 0 is a blank (`a9_human_target.py:97-100`), so code k gets label k.
- `load_liu` refuses any code outside 1 to 7 (`:81`). It also checks that `p > 0` equals `stimuli_raw` from the other file (`:79`), so amplitudes and responses are aligned by stimulus.
- `human_configuration_table` keys `liu{i+1:04d}` (`a9_analyse.py:145`) equal `a9_manifest.cfg_name` (`a9_manifest.py:44-45`).
- The human label order (anger to other) and the model option order (Anger to None of the above) match position by position. So `behaviour_metrics` compares like with like.

*AU map*

- `token_columns` (`a9_design.py:222-236`) reproduces the block's map; import refuses any difference (`:240-241`).
- Compounds (`AU25-12`, `AU12-6`, `AU1-2`) and one-sided Open variants resolve as the draft says.

*Analysis against the draft (A4's machinery, used unchanged)*

- `analyse_model` uses `an.estimate`, `an.remove_common`, `an.decide` and `an.apply_rule`, with the human ridge checked against the block (`a9_analyse.py:106-108`).
- The statistic is residual CEA. Signal uses 999 shuffles, the cue null 9,999, the bootstrap 2,000 draws and split-half 20 splits, all read from `a4_design` and checked against the block (`a9_design.py:198-206`).
- Holm runs over `a4_design.MODELS` present in the results.
- The ceilings are divided by `sqrt(R_S)` as in A4.
- Under design B:
  - A configuration is kept only if all four faces have a complete rotation set (`a9_analyse.py:55-58`).
  - The lost share feeds `clips_kept`.
  - Gender is 0.
  - V3 still compares the main face with its same-gender replicate.
  - V4's mean over face-averaged P equals the mean over all four faces' images.
- The face-shared correlation is descriptive. It is written beside the decision and never read by `apply_rule` (test at `tests/test_a9.py:291-303`).

*No human data reaches the model side*

- `a9_run.py` reads only the manifest. The manifest holds AU amplitudes, a hash-assigned gender and a hash-assigned face, and nothing derived from responses.
- The rank (replicate set and the V2 subset), the gender and the face are seeded hashes of the stimulus index (`a9_manifest.py:40-61`).
- The Liu `.mat` files, the Chen file, manifests, renders (`*.tar`, `*.png`), logs and `summary.json` are all gitignored.

*Recovery numbers in the draft*

- Every number in the draft's recovery tables (lines 577-607) and in "Consequences" (line 595) matches `a9_recovery.json`.
- The checks are A4's `verdict`, unchanged: 17 per statistic.
- No A4 withdrawn cell applies, because those keys name A4 datasets (`a4_recovery.py:187`).

Severity: blocking, then should-fix, then note.

---

## Blocking

### 1. The rule that defines "replication" is not stated, although A9 is titled and argued as a replication

**Evidence**

- The title says "replication of A4's confirmatory test" (`PREREG_A9_DRAFT.md:1`).
- "Relation to A4" (lines 423-429) says only:
  - A9 cannot change A4.
  - Disagreements "both stand".
  - A9 cannot make an A4 Inconclusive conclusive.
- Nothing says which A4 outcome a model's A9 decision is compared with:
  - A4 basic_WC, the Western group, as Liu is Western?
  - Or A4's basic headline, which needs WC and EA to agree?
- Nothing says what the paper may call each pair of outcomes, for example:
  - A4 Aligned with A9 Aligned;
  - A4 Aligned with A9 Inconclusive;
  - A4 Mixed or Inconclusive with A9 Aligned.
- `research/paper/v2/results.md:188` has a negative rule (never "confirms" or "overturns") but no positive one.
- This is the one place left where the reading could be chosen after the model output is seen. It is also the first thing a reviewer of a "replication" will ask.

**Fix.** Add a short pre-data table to the draft, outside the block or inside it with a re-hash. It should do three things:

1. Name the A4 comparator. The recommended choice is A4's `basic_WC` decision, with the basic headline reported beside it.
2. For each of the 3 x 3 (or 3 x 4, including Mixed) pairs, give the exact words the paper uses. For example:
   - "same decision on a second sample" for a matching named pair;
   - "A9 inconclusive" or "different decision" otherwise.
3. State that A9 being Inconclusive is never reported as "failed to replicate" unless A9 is Not aligned.

Mirror the table in `results.md`.

---

## Should-fix

### 2. The "fixed before run 2" claim has no record behind it, and the wording overstates how decision 8 relates to the earlier rule

**Evidence: what the record shows**

- The draft says the correlated-noise rhos "and their use were fixed by the lead before run 2" (`PREREG_A9_DRAFT.md:609`; also lines 436 and 442).
- Git has only two A9 commits:
  - `9259fe5` (11:10) has no correlated-noise runs, no decision 2 and no rho.
  - `ac8dc16` (14:59) already contains decision 8.
- So the text that run 2 was hashed against (block `b548bc82...`, with `a9_design.py` `a5f7e004...` and `a9_analyse.py` `22f39ad8...`) was never committed and is not recoverable from the repo.

**Evidence: what that text said**

- `notes_a9.md:34` records what the pre-run operationalisation meant: "any model with face-shared residual correlation above 0 is reported as outside the recovery-validated conditions; in practice that is expected for every model."
- Decision 8 replaced that with "decisions stay confirmatory whatever the correlation".
- The draft discloses the replacement (line 436, the parenthesis; line 442, the timing).
- But the header (line 7), the Disclosures (line 621) and the paper (`methods.md:45`, "The rule itself was stated before that run") describe decision 8 as "applying the narrowing rule stated before it". The pre-run rule was broad, and its stated operationalisation would have scoped every model out. Decision 8 is a new, more favourable operationalisation chosen after the result was seen.

**Fix**

- Say exactly that. Suggested wording: "Decision 2 stated a general narrowing rule before run 2; its first operationalisation (a scope label by largest passing rho, r_star = 0, which would have labelled every model) was replaced after run 2 by narrowing by check."
- State that the pre-run-2 text was not committed, and that `a9_recovery_run2.json` (with its `design_sha256` b548bc82) is the only surviving record of it.
- Make the same change in `methods.md:45` and in the paper's recovery paragraph.

### 3. "Whatever the correlation" is tested only up to rho .5, and the bracketing argument is not made

**Evidence**

- Decision 8(a) and the block's `correlation_use` (line 340) claim decisions stay confirmatory "whatever a model's estimated face-shared correlation".
- Recovery tested rho of 0, .25 and .5 only (line 335).
- `face_shared_correlation` itself "leans high" (`a9_analyse.py:64-67`): it takes OLS residuals of probabilities on the AU design, so any non-additive configuration effect counts as shared. Real models will therefore probably report a rho-hat above .5, in the untested range, right beside their decision.

**Supporting evidence the draft does not use**

- Design A is close to the rho = 1 limit of B: four faces with fully shared noise are approximately one face.
- In run 2 or 3, A passes every decision check on residual CEA:
  - human-like Aligned 1.00;
  - false Aligned at most .02;
  - prior-only signal at most .05;
  - boundary Not aligned at most .024;
  - different-cues Not aligned .98 at 1x;
  - half-shared Aligned .84 at 1x.
- A fails only the same pair-swapped accuracy check.
- The withdrawn error is conservative: the pair-swapped residual CEA is biased low (A .346 against a true .411; B at rho .5, .359 against .396).

**Fix.** Add one paragraph to the Recovery "Consequences" section covering:

- design A as the rho approximately 1 bound, with its decision rates;
- the direction of the 0.25x bias;
- the fact that rho-hat is biased upward.

Alternatively, add a `B_rho1.0` run before the freeze. That changes `recovery.runs` and needs a re-hash and a run 4.

### 4. Two sentences now contradict the withdrawn cell

**Evidence**

- "A4's two withdrawn cells name A4 datasets, so **no A9 cell is withdrawn** ..., which makes A9's rule stricter than A4's as applied" (`PREREG_A9_DRAFT.md:573`; the same in `methods.md`).
- Decision 8 then withdraws `liu_WC|0.25|pair_error` under correlated face noise (block line 341-343). Correlated face noise is the condition A9 expects to hold for real models.
- A4 withdrew the same kind of cell (pair_error at 0.25x).

**Fix.** Reword to say that no cell is withdrawn under independent face noise, and that under correlated face noise one cell (pair_error at 0.25x) is withdrawn, the same kind A4 withdrew. Drop "stricter than A4's".

### 5. Human target and ceiling: reproduced exactly (note, no action needed)

This finding is a note, not a should-fix. It is numbered here only to keep the order of the audit questions.

**Evidence**

- `python -I research/scripts/a9_human_target.py target`, run from the pinned `.mat` files into the scratchpad, took 25 min.
- It wrote `a9_human_target.json` with sha256 `3f93ccbe...6ab0` and `a9_human_ceiling.json` with sha256 `a32e2dbe...e669`. Both are byte-identical to the committed files and to `input_pins`.
- The run's own checks passed:
  - the Liu file hashes;
  - the label list;
  - the shapes;
  - `p > 0` equal to `stimuli_raw`;
  - the codes, all in 1 to 7;
  - the selected ridge, 1.0, equal to the block.
- The values the draft quotes match the file:
  - held-out loss .33818 at 1.0, against .34908 at 0.1, .34352 at 3 and .39617 intercept-only;
  - the Snoek map check: 6,966 configurations, 0 mismatched, 55 collisions;
  - ceiling .9748 raw and .9622 residual;
  - references .8611 and .6901.

**Fix.** None.

### 6. The run table mislabels run 2's code as "final"

**Evidence**

- `PREREG_A9_DRAFT.md:583-590`: the first run-2 row says "pre-decision-8 code", and the next seven rows say "final".
- Run 2's `code_sha256` has `a9_design.py` `a5f7e004...` and `a9_analyse.py` `22f39ad8...`, not the pins.
- Line 12 lists the outputs but omits `a9_recovery_run2.json`.

**Fix**

- Label every run-2 row "pre-decision-8 code (design hash b548bc82)".
- Add `a9_recovery_run2.json` to the output list on line 12.

### 7. `a9_analyse` checks the pinned files but then reads whatever paths it is given

**Evidence**

- `main()` calls `a9.check_pins()`. That hashes the repo copies of `a9_human_target.json`, `a9_human_ceiling.json`, `a4_v1_pilot.csv` and the rest (`a9_analyse.py:175-176`).
- It then loads `args.human`, `args.ceiling`, `args.detector` and `args.recovery` (`:179-184`), which may point anywhere.
- So `--human other.json` passes every pin check and changes the target. The ridge check (`:107`) would catch only a different lambda.
- The summary records neither the hashes of the files actually read, nor `gates.json`, nor the cell shards.
- This is A4's pattern, but A9 adds its own pins, so the gap is new.

**Fix**

- Hash `args.human`, `args.ceiling` and `args.detector` and refuse unless each equals its pin.
- Write into `summary.json` the sha256 of `gates.json`, of the recovery file and of each shard file read.

**Consequence of the fix.** Changing `a9_analyse.py` changes its pin. Run 3's `code_sha256` would then differ from the pins in that one file. Run 2 against run 3 already shows that recovery tables do not depend on `a9_analyse.py`, which recovery never imports. So either rerun recovery as run 4, or state the one-file difference in the text.

### 8. Pinned A9 files are untracked

**Evidence**

- `git status` shows `scripts/a9_run.py`, `slurm_jobs/study3_a9.sh` and `slurm_jobs/study3_a9_render.sh` as untracked (`??`).
- The committed draft (`ac8dc16`) pins `a9_run.py`.
- `research/scripts/__pycache__/` is also untracked and not ignored.

**Fix**

- Add the three files in the tagging commit (freeze checklist item 2 implies this, but it says "Commit A9's files" without listing them).
- Ignore `__pycache__/`.
- Before tagging, check that `git ls-files` covers every path under "Code hashes" and "Other A9 files" except the gitignored manifest.

### 9. The A9 decisions are not in DECISIONS.md, and the paper cites a stale recovery file

**Evidence**

- `research/DECISIONS.md` has only the 2026-11-09 "second randomised human dataset" entry (line 462). That entry still says "No response distribution was examined".
- Lead decisions 1 to 8 are not there, including decision 8, which is a post-result narrowing. A5 and A7's equivalent narrowings were logged there (lines 440-443).
- `research/paper/v2/methods.md:9` cites `a9_recovery.json` as sha256 `79229b70...` with design hash `b548bc82`. That is now `a9_recovery_run2.json`.
- The PENDING note in the recovery paragraph is now resolved, since run 3 is identical.

**Fix**

- Add a dated DECISIONS entry for decisions 1 to 8, with decision 8's timing stated as in finding 2.
- Update `methods.md:9` and the PENDING note to run 3 (`01958263...`, design `41c95dae...`).

---

## Notes

### 10. Seeds: the explanation is wrong, though the conclusion holds

- Line 593 says run 1's seeds differ because "the run name is part of the seed".
- In fact the seed vector is `[SEED, dataset, observer, strength, replicate, design index, round(100 rho)]` (`a9_recovery.py:79-80`). Run 1 had no rho element (`9259fe5:scripts/a9_recovery.py:55`), and run 2 also draws a shared noise matrix first under B.
- The conclusion (independent draws) holds.
- **Fix:** correct the wording.

### 11. The human configuration table can drop silently

- `human_configuration_table` (`a9_analyse.py:132-146`) returns None on `SystemExit`, `FileNotFoundError` or `OSError`, which includes a Liu hash mismatch. The descriptive behaviour metrics then vanish without a message.
- **Fix:** print a warning, and record `"behaviour": "unavailable: <reason>"`.

### 12. Holm family size is not stated for a model that is never run

- `apply_rule` puts into the Holm family only models that have shards (`a4_analyse.py:442`). If one model is never scored, Holm runs over one model and that model is not reported.
- This is inherited from A4.
- **Fix:** state it in the draft (for example, "a model in A4's family with no cells is reported Inconclusive and counted in Holm with p = 1"), or state that the family is the models with cells.

### 13. Differences from A4 that are not listed

- A9 has no one-AU set, so A4's direct contrast and dose sensitivities (A4 draft line 107 and line 173; `a4_analyse.direct_effects`) and the 0.75x and 1.25x amplitude sensitivities are absent. The difference table (lines 458-470) and the Sensitivity section do not say so.
- **Fix:** add a row.

### 14. A render file that is not gitignored

- `a4_stimuli.render_tar` writes `renders_NN.partial` while rendering. `*.partial` is not gitignored, and nor are `results/a9/cells_*.jsonl`. The same holds for A4.
- Renders are made on the cluster, so the risk only arises if results are synced back.
- **Fix:** add `*.partial` to `.gitignore`. Cells hold model outputs only and need no change.

### 15. Recovery does not model face-by-AU interactions

- Design B's synthetic faces differ only by a gender intercept and noise. Real identities render AUs differently, which is a systematic, configuration-independent face-by-AU effect.
- B averages such effects; A does not.
- **Fix:** add one sentence to the recovery disclosures.

### 16. Stronger code-order evidence exists and is not used

- **Liu against Snoek.** Using only human data already in the repo, the cosine of each Liu label row with each Snoek basic WC row is largest on the diagonal for all seven labels:
  - anger .65 (against .54 for disgust);
  - disgust .88;
  - fear .89;
  - happy .93;
  - sadness .81;
  - surprise .95;
  - other .54.
- **Anger and disgust.** Liu's anger has larger AU04, AU07 and AU23 weights than disgust, and disgust larger AU09 and AU10. This settles the one swap the top-3 check cannot: anger and disgust share AU09 and AU10 at the top (`label_corroboration_top3_families_by_beta`).
- **Chen.** The Chen overlap's 1.000 trial-level agreement, under the order happy, surprise, fear, disgust, anger, sad, other (`a9_human_target.py:414`), shows that the lab's coding convention is this order against Snoek's text labels.
- **Fix:** cite both in "Response codes" (outside the block, no re-hash needed). They are corroboration, not a new rule.

### 17. Pair-swapped observers are often decided Aligned

- Pair-swapped observers are decided Aligned in 65 to 97% of replicates, because their true residual CEA is about .40, above the SESOI of .30.
- This is inherited from A4 and the draft states the .38 to .40 value.
- **Fix:** none required. A reviewer may raise it, so the A4 text that addresses it should be cross-referenced.

### 18. Minor wording

- Line 3: "recovery rerun on the final code" is correct for run 3.
- Line 642 is correct.
- `notes_a9.md:9` lists `manifest_designA.csv` and `manifest_designB.csv`. The runner uses `manifest.csv`, which is byte-identical to design B.

---

**Verdict: not freeze-ready.** No code defect, pin mismatch or leakage blocks the freeze. Finding 1 must be added to the text before the tag, because once frozen it cannot be added honestly. Findings 2, 3, 4, 6 and 9 are corrections to the text to be frozen and should go in the same edit; finding 7 is a small code change that will need re-pinning. After those edits, the draft should be ready to freeze:

1. Re-hash the block if it changed, and re-pin the code.
2. Rerun recovery only if the block's recovery fields changed.
3. Commit all A9 files.
4. Tag `a9-freeze`.

---

## Re-check 2026-11-09

This is a re-check of commit `af64e56` on branch p1-pipeline. As before, the only file edited is this one. Nothing was committed, the cluster was not touched, and no job is left running.

### Hashes and pins (point 2)

- The block hashes to `21341eb47a4d5da81e7d672e21c56c83029cbf9ba2b928f98f10fcba35220192`. That equals `DESIGN_SHA256` (`scripts/a9_design.py:27`) and the hash stated in the draft (line 5).
- `check_code_pins(require=True)`, `check_pins()` (A4 pins and input pins) and `check_liu_files()` all pass. Each of the six pinned code files matches its sha256 as committed.
- The other files listed under "Other A9 files" match too: both Slurm scripts, the four recovery JSONs, the human target, the ceiling and the Chen overlap. So does `tests/test_a9.py` (`645de6e0...`).
- pytest gives 53 passed.
- Run 4 (`a9_recovery.json`, sha256 `0ca7bd24...`) carries the design hash `21341eb4...`. Its `code_sha256` equals the pinned lines exactly. Its `runs` are identical to run 3's (`a9_recovery_run3.json`) in every table and check: choice B/residual, r_star 0, and the same single failing cell.
- `a9_run.py` and both Slurm scripts are now tracked.

### Findings 1 to 9 (point 1)

| # | Status | Evidence |
|---|---|---|
| 1 Replication undefined | Fixed | Block `replication` (comparator, wording for each A4/A9 pair, a catch-all, rules); table at `PREREG_A9_DRAFT.md:451-462`; `a9_analyse.py:202-223`; tests `tests/test_a9.py:407-437`. See point 3 for two gaps. |
| 2 Timing claim and spin | Fixed | Header line 7; decision 2 (line 469); decision 8 (line 475); block `correlation_use`; recovery disclosures (line 646); Disclosures (line 658); DECISIONS.md (new A9 entry); `methods.md` item 10 and "The narrowing"; `results.md` 6b "Withdrawn claim". See point 4. |
| 3 Correlation above .5 | Fixed (argued, not tested) | "Beyond the tested correlations" (draft line 632; supplementary S7). It states that design A approximates rho = 1, that A's decision rates hold, that the bias runs low, and that this is "an approximation ..., not a tested condition". |
| 4 "Stricter than A4" | Fixed | Draft Recovery opening paragraph; block `withdrawn_under_independent_face_noise: []`; S7 "Recovery (continued)". |
| 5 (note) | No action needed | |
| 6 Run labels | Fixed | Draft lines 617-626: every run-2 row is labelled "pre-decision-8 code (design `b548bc82`)", and runs 3 and 4 are added. The output list (line 12) names runs 2 and 3. The seed explanation (note 10) is corrected. |
| 7 Path overrides | Fixed | `pinned_input` refuses an unpinned human target, ceiling or detector file. Manifest, recovery, gates, A4 summary and every shard are hashed into `input_sha256` (`a9_analyse.py:245-271`). |
| 8 Untracked files | Fixed | All three are tracked. `__pycache__` is still not ignored: `research/scripts/__pycache__/x.pyc` is un-ignored by `.gitignore:70` (`!research/**`). This is cosmetic. |
| 9 DECISIONS and paper | Fixed | DECISIONS.md has the A9 decisions 1 to 8 with decision 8's timing. `methods.md` line 9 now cites run 4 (`0ca7bd24`, `21341eb4`), and the PENDING note is gone. |

The notes are handled as well:
- Note 11: behaviour metrics now return "unavailable: reason" with a warning.
- Note 12: Holm family size is stated (Relation to A4, item 2).
- Notes 14 and 15: `*.partial` is ignored, and the face-by-AU limit is disclosed.
- Note 16: the cross-dataset and Chen corroboration is added to DECISIONS and S7.
- Note 17: pair-swapped Aligned is disclosed.

### Replication table and its implementation (point 3)

**Fully specified.** I ran `replication_reading` on all nine pairs of A4 basic_WC and A9 decisions. Each returns the block's exact wording:
- Aligned/Aligned gives "replicated on a second human sample (same lab and generator)".
- Not aligned/Not aligned gives "the Not aligned result replicated".
- Each crossed pair gives "did not replicate".
- Any Inconclusive gives "replication not testable", with the failed gate or the signal test named.

So A9 Inconclusive is never read as a failure to replicate. No Mixed case can arise, because the comparator is the per-dataset decision (`a4_analyse.decide`). The headline is only reported beside it.

**Two gaps in the code. Both are should-fix and both are small.**

- **R1. A missing A4 summary is misreported.** When `--a4_summary` is absent, which is the default path and does not exist today, the run does not stop (`a9_analyse.py:253-256`). Every model then reads "replication not testable" with the reason "A4 basic_WC was not run for this model" (`:208-209`). That is false when A4 was run but the file was not supplied. **Fix:** either refuse to run without the A4 summary, or use the reason "A4 summary not supplied" and set a top-level flag.
- **R2. "Frozen summary" is checked only by design hash.** The check is `a4_summary["design_sha256"] == a4_design.design_hash()` (`:259`). That proves the summary came from A4's frozen design and code. It does not prove it is *the* A4 summary: any re-run of `a4_analyse` with a different hand-written `gates.json` would pass. The A4 summary does not exist yet, so its hash cannot be pinned before the freeze. The file's sha256 is written into `input_sha256.a4_summary`. **Fix, with no code change needed:** add one sentence to the draft (Relation to A4) saying that the A4 summary used must be the one whose sha256 was recorded when A4 was analysed (RESULTS_REGISTRY or DECISIONS), and that the A9 summary's `input_sha256.a4_summary` is compared with it in the paper. Optionally, add a `--a4_summary_sha256` argument that must match.

**Note R3.** For Molmo2 (descriptive), the reason reads "Descriptive: neither the Aligned nor the Not aligned criteria met", which is misleading. Relation to A4, item 2, says an unscored model "is reported as not run", but the code simply omits it from `models`. Both are cosmetic.

**Earlier-rule reading.** It uses r_star from the recovery file, 0.0, as written: "outside the validated conditions" if rho-hat is greater than 0, "within" if rho-hat is at most 0, and "not estimable" if rho-hat is None (`a9_analyse.py:182-188`). That matches the operationalisation recorded in `notes_a9.md:34` and in the block's `earlier_rule_reading`.

### Timing disclosure (point 4)

The disclosure is now accurate, and I found no spin anywhere it appears:
- the draft header (line 7);
- decisions 2 and 8 (lines 469 and 475);
- `correlation_use`;
- the recovery disclosures (line 646);
- the Disclosures list (line 658);
- DECISIONS.md;
- `methods.md` (item 10 and "The narrowing, and when it was decided");
- S7 ("Recovery (continued)": "lead decision 8, taken after recovery run 2 was seen");
- `results.md` 6b ("Withdrawn claim").

Each place says four things:
1. The general rule and its first operationalisation were written before run 2.
2. The operationalisation was replaced after run 2 was seen.
3. The replacement is more favourable to confirmatory decisions.
4. Under the earlier rule, with r_star = 0, every model would have been placed outside the validated conditions.

Each place also says the replaced text was never committed and survives only as run 2's design hash. The remaining claim that the rhos and the rule were set before run 2 rests on the lead's word plus that hash, and the text says so.

### New problems introduced (point 5)

- **N1 (note).** S7 "Beyond the tested correlations" says "In every run A passed every decision check: ... prior-only signal at most .05". In run 1 (draft code) A's prior-only signal rate was .07 at 0.5x. That still passes the .10 rule, but ".05" holds for runs 2 to 4 only. **Fix:** say "runs 2 to 4", or ".07".
- **N2 (note).** Draft line 646 lacks a full stop after "(A4 draft, \"Reading Aligned\")".
- **N3 (note).** Freeze checklist "Remaining" item 2 still says to commit the files, though they are committed in `af64e56`; it now only needs the tag. Item 1 can be ticked after this re-check.
- I found no new issue in the hashing code. `pinned_input` reads the same pins as `check_pins()`. Recovery code is unchanged across runs 2 to 4.

### Verdict

R1 and R2 are the only open items, and neither touches the decision rule, the statistic or the human target.
- R2 needs only a sentence of text. That sentence sits outside the block, but it does change the draft.
- R1 is a small fix to a code path that is cosmetic for decisions. If `a9_analyse.py` is edited, re-pin it, and either rerun recovery (run 5) or state that the recovery code was unchanged.

**Verdict: freeze-ready.** Making R1 and R2 before the tag is recommended, not required.
