# A8 pre-freeze audit

**Written:** 2026-11-09, independent adversarial audit of `research/PREREG_A8_DRAFT.md` (design block `49665a15...`), `research/notes_a8_models.md`, `research/scripts/a8_*.py` and `tests/test_a8.py`, against the frozen A1/A2/A3/A4 texts and code. No file other than this one was edited; nothing was committed; the cluster was not touched.

**Test run.** `/opt/anaconda3/bin/pytest tests/test_a8.py`: 27 passed, 2 skipped (the GPU A1 equivalence test and the frozen-A3-file test; the frozen A3 cell files are not in the local checkout), 1 warning, 13 min 50 s. Extra scratch tests written for this audit live only in the session scratchpad: (a) one errored trial in a mismatch shard (finding 1, confirmed failing); (b) one errored trial in a real shard (merges, as intended); (c) a stubbed full 243-item mismatch run in 4 shards against the unsharded frozen run (passed: identical, 243 records; see finding 8).

**Verified as correct.** The block hash recomputes to the stated and pinned value, and is computed exactly as `scripts/a7_design.py` does. Every file in `frozen_code_sha256` matches. `frozen_a3_summary_sha256` equals `stimulus_design.A3_SUMMARY_SHA256` and the committed summary; the A2 and A4 design hashes equal `run.design_hash()` and `a4_design.design_hash()`. `scripts/evaluate.py` has exactly one `enumerate` call (line 680, the trial loop), so the module-level override touches nothing else, and functions from other modules keep the builtin. Per-trial seeding (`seed_generation(seed, trial_id, k)` -> `torch.manual_seed`) and `trial_index=trial_idx` with original indices make a record independent of its shard. The merge refuses gaps, duplicates, mixed runs and edited shard files. The A2 swap is minimal (one function replaced, restored afterwards), and the A3 wrapper is a four-line restatement of the frozen `summarise`. The text does not change any frozen threshold, estimand, readout or verdict. What it adds is new decisions (A8's own Holm families, disclosed as less strict at size 1; the "skip A1" stop rule; strict Gemma gates). None of these touches a frozen family or verdict.

Severity order: high, then medium, then low.

---

## High

### 1. One errored mismatch trial makes a shard impossible to merge, so A1 can never finish

`research/scripts/a8_rotation_shard.py:194-197` checks `t.get("vision_source_trial_id") != partners[...]` for every mismatch record. When a trial fails, `scripts/evaluate.py` (except branch, about lines 983-1001) writes a record **without** `vision_source_trial_id`. So any errored trial (a CUDA OOM on one long clip is the realistic case at 32B in 80 GB) raises `SystemExit` after the shard has run for hours. No sidecar is written, and `merge` cannot use that shard. Rerunning hits the same deterministic error. The frozen rule (DECISIONS 2026-11-08) scores errored trials as wrong and goes on, so the runner is stricter than the frozen pipeline. It also invites a post-data workaround. Confirmed with a stubbed run: `SystemExit: 1 mismatch trials did not use the full-set partner: ['emotions/EDITED/Unfriendly/NF25_cut.mp4']`.

**Fix.** In the partner check, skip records with `error` set, and assert that they carry no `vision_source_trial_id`. Keep them in the shard and in the merge, as the frozen pipeline does. Add a test with one errored trial in each condition.

### 2. Nothing ties the preflight-checked weights to the weights the frozen code loads

- `config.py:27` reads `MR_EU_HPC_MODELS_DIR` when the module is imported. `evaluate.resolve_model_path` (`scripts/evaluate.py:146-151`) silently falls back to `local_path` when the HPC path is missing.
- `a8_preflight.py:147` checks `--models_dir`, which defaults to the design's directory, not the environment variable the job actually uses. It never reads `os.environ["MR_EU_HPC_MODELS_DIR"]`.
- The shard runner sets the variable only when `--models_dir` is passed (`a8_rotation_shard.py:70-71`). Otherwise it inherits the job's value.
- The frozen slurm script it is meant to be copied from sets `export MR_EU_HPC_MODELS_DIR="${ENV_ROOT}/models"` (`slurm_jobs/study3_rotation_eval.sh:64`). That directory holds the **8B** under `qwen3vl` and E4B under `gemma4`.
- No output (eval JSON, sidecar, gate JSON, A2/A3/A4 cells) records the resolved weights path or the loaded config.

**Failure scenario.** An A8 slurm file is copied from the frozen one and keeps line 64, and the runner is called without `--models_dir`. The preflight passes on the design directory, and the frozen code loads Qwen3-VL-8B. The outputs carry key `qwen3vl` and are labelled `qwen3vl_32b`, and nothing downstream can tell. The same applies to every other step (vision gates, delivery, ladder, A3, A2, A4), which all resolve weights through `resolve_model_path` and are not run through any A8 Python wrapper at all.

**Fix.**
- The preflight requires `MR_EU_HPC_MODELS_DIR` to be set and to resolve to the design's `models_dir`, and refuses otherwise.
- Better, add one A8 launcher that runs the preflight and then the frozen module in the same process (`runpy.run_module`). Before that it asserts `config.MODELS[key]["hpc_path"].resolve() == weights_dir` and `evaluate.resolve_model_path(key).resolve() == weights_dir`, and records both in the provenance JSON.
- The shard runner derives `--models_dir` from the design (finding 9), always sets the variable, asserts `ev.resolve_model_path(key).resolve()`, and writes it into the sidecar. The merge then requires it to be equal across shards.

### 3. Step commands in the draft would overwrite or contaminate the small siblings' frozen outputs; the "preflight refuses any other output path" claim is not enforced

The preflight checks only the paths it is handed (`a8_preflight.py:102-113`). Nothing links those paths to the arguments given to the frozen script. Against that, the "Pipelines" table (draft lines 205-214) has these problems:

- **Step 7, A2:** `run_stimulus_interventions --step run` is given no `--output_dir` and no `--reproduction`. The default is `results/stimulus_interventions/cells_qwen3vl_shard{k}.jsonl` (`run_stimulus_interventions.py:1403, 1119-1123`). The run **resumes** from that file (`done_cells`, lines 1083-1097, 1124), so a 32B job either skips every cell the 8B already wrote, or appends 32B cells into the 8B's frozen A2 file. The frozen 8B job then skips those cells in turn. The queue rule only requires the frozen Phase 2 runs to be *submitted* (block `queue`; draft line 272), so the two can run at the same time. The design block's `steps` entry has the right path; the table does not.
- **Step 8, A4 analysis:** `a4_analyse.py --cells_dir results/a8/a4 --gates ...` is given no `--output`. The default `results/a4/summary.json` (`scripts/a4_analyse.py:617, 670-671`) overwrites the frozen A4 summary. The required `--detector` is also missing, which fails safe.
- **Step 6, natural-clip RQ1:** `same_clip_family_analysis.py --model-route` always writes `research/reproduction/same_clip_family_model_{--model}.json` (`research/scripts/same_clip_family_analysis.py:1349`). That is outside the block's declared `research/reproduction/a8/{a8_id}/`, so the output-root rule cannot be met. With `--model qwen3vl` it overwrites the frozen file.

**Fix.**
- Give full explicit output arguments in the table for every step: `--output_dir results/a8/<id>/stimulus_interventions --reproduction research/reproduction/a8/<id>` for A2, `--output results/a8/a4/summary.json` for A4, and `--model <a8_id>` for step 6. For step 6, state the real output path as a declared exception to the roots, or copy the file into the A8 folder after the run.
- Generate each A8 slurm command line from the same variables that are passed to the preflight.
- Add a guard that snapshots the sha256 of every frozen output carrying the key (`results/baseline/eu_emotions/<key>/*`, `research/reproduction/*_<key>.json`, `results/stimulus_interventions/*<key>*`, `results/face_input_validity/cells_<key>.jsonl`, `results/a4/*`) before each A8 job and checks it after.
- Change the queue rule to "frozen Phase 2 runs **completed**", or use a guard that makes overlap harmless.

### 4. The A1 runner does not reproduce the frozen job's environment, and a missing lookup silently changes the estimand

The frozen job runs three steps before `evaluate.py` (`study3_rotation_eval.sh:60-67`):

- `research/scripts/decoder_preflight.py`;
- the cv2 4.11 `PYTHONPATH` pin required by A1.1 ("cv2 4.13 corrupts interlaced clips");
- `python -m paper_v2.option_score --write_rotation_lookups`, which creates `data/eu_emotions_human_entropy_rot{r}.json`. These files are absent in the local checkout.

The runner does none of this. If a lookup is missing, `evaluate.py` only prints a warning and falls back to sampled foils (`scripts/evaluate.py:650-655`). The runner's own `partner_map` falls back the same way (`a8_rotation_shard.py:164`), so its partner check still passes. The result is a "successful" A1 run on the wrong option sets, with partner labels taken from manifest labels instead of human target labels. The design block also pins no data file: not the manifest, not the five lookups, not `eu_emotions_human_entropy.json`.

**Fix.**
- The runner refuses if the lookup file is missing.
- Pin the sha256 of the manifest, of `eu_emotions_human_entropy.json` and of the five rotation lookups in the block, and check them in the runner and preflight. They are deterministic outputs of `--write_rotation_lookups`, so they can be pinned before data.
- The runner calls `decoder_preflight.py`, which is already in `frozen_code_sha256`, or asserts `cv2.__version__` starts with `4.11`. Record the cv2, torch and transformers versions in the sidecar.

---

## Medium

### 5. The family code does not implement the family the text defines

The block says `"a1": "Holm over A8 models that run A1"` and `"a3": "Holm over A8 models that run A3"`. The code instead uses `active_models` (confirmed plus admitted) unconditionally:

- `a8_analyse_rotation.py:55`;
- `a8_face_input_validity_summary.py:76-79`, which requires the names to equal `active` exactly.

Gemma 4 31B is admitted, and the notes (line 25) expect the frozen `gemma4` gate adapter to fail on 31B, which under decision 5 makes it ineligible and skips its A1. In that case the A1 analysis crashes with `FileNotFoundError` on the missing Gemma files. The only way past is `--models`, labelled "tests only", so the family is chosen by hand after the data exist. The A3 wrapper has the same problem if Gemma's `single_face` delivery fails ("not tested").

**Fix.** Make membership mechanical: active models whose stop-rule inputs (gate JSON, delivery JSON, A3 summary) show that they ran the step, computed by code from those files. Alternatively, require a dated note listing the A1 family, written after Phase A and before any A1 output. Remove the "tests only" escape from the production path, or have it refuse when the design gate is on.

### 6. Gemma's admission, and so every family's size, lives outside the hash

`a8_design.admitted()` reads the "Dated notes" section (`a8_design.py:100-124`), which is outside the hashed block. Deleting the 2026-11-09 admit line after results are known changes every A8 family from two models to one, and the design hash does not change. Adding a line can likewise enlarge a family. The block still says `"status": "conditional_on_project_store_approval"` even though the admission was already made before the freeze.

**Fix.** Before the freeze, set `gemma4_31b` to `"status": "confirmed"` (or `"admitted_2026-11-09"`) in the block, re-hash, and drop the note-reading path, or make it refuse any admit line dated after the `a8-freeze` tag.

### 7. Code on the loaded path is not pinned

`frozen_code_sha256` (draft lines 115-148) leaves out modules that the pinned files import and that sit on the A8 path:

- `scripts/activation_forward.py`, which builds forward inputs for `model_inference.py`, `vision_gates.py` and `run_stimulus_interventions.py`;
- `scripts/model_compat.py`, imported by both `evaluate.py` and `model_inference.py`;
- `scripts/semantic_entropy.py`, `scripts/statistics.py`, `scripts/eu_audio_resolver.py` and `scripts/mindreading_audio_resolver.py`, imported by `evaluate.py`;
- `research/scripts/human_au_effects.py`, imported by `a4_analyse.py`;
- `research/scripts/same_clip_cue_analysis.py`, which step 6 uses through `same_clip_family_analysis.py`, itself unpinned;
- `research/scripts/a4_detector.py`, required by A4.

An edit to `activation_forward.py` would change A8 cells and still pass `check_frozen_code`. No software versions are pinned either. The 32B needs a recent transformers, and if the environment is upgraded for Gemma 4 31B, a difference from the small sibling mixes scale with library version.

**Fix.** Pin every repo module reached by import from the A8 entry points. A transitive import walk can generate the list, and a test can check that the list is complete. Record the torch, transformers and cv2 versions used by the sibling's frozen runs, and require them, or state the version difference as a limit.

### 8. The A1 equivalence test, as specified, does not test the property it is there for

- The block's test runs frozen `--max_trials 6` against the shards on the same 6 items. With `max_trials`, `evaluate.py` slices `trials_raw` before building partners (lines 549-551, 663-678), so **both** sides build the partner map over 6 items. The full-243 partner map with a restricted loop, the thing sharding relies on, is never compared with a frozen output.
- The local tests have the same limit: `max_trials` of 6 or 7, and the non-rotated lookup, not a `rot{r}` file.
- The runner's own check against `partner_map` is computed by the same frozen function on the same inputs. It only shows that the loop received the full list.
- No outcome is set in advance in case the GPU test fails because of run-to-run GPU nondeterminism and not because of sharding.

Audit check: a stubbed full 243-item mismatch run in 4 shards, compared with the unsharded frozen run (scratch test). It passed: 243 records, identical. So the mechanism is sound on the frozen code path with stubs. The specified cluster test still does not cover this case.

**Fix.**
- Add the full-243 stubbed comparison to `tests/test_a8.py`. It runs locally on CPU.
- On the cluster, also run the 6-item frozen command twice to establish determinism. State in the block what follows if frozen against frozen differs: compare `fc_predictions` and the parsed fields, not the raw text, or treat it as a stop.
- Optionally, run one full-manifest shard (k of N, N large) on 8B and compare it with the same trial ids in the existing frozen 8B rotation file, using a script that prints only "identical / not". This needs a ruling on whether it counts as reading A1.

### 9. The shard runner is not bound to the design per model

`--key`, `--sampling` and `--label` are free (`a8_rotation_shard.py:345-357`):

- The runner never calls `a8_design.model(...)`, so the admission rule is not enforced here.
- It never checks `sampling == entry["a1_sampling"]` or `key == entry["code_key"]`. A `gemma4_31b` label can run `qwen3vl` with sequential sampling.
- `check_own_code(require=True)` is called only by a test (`tests/test_a8.py:86`), never by the runner, the preflight or any analysis wrapper. So "scored A8 runs wait on the dated pin step" is not enforced anywhere.
- `--no_design_gate` is allowed in production, is not recorded in the sidecar, and is not refused by `merge`.

**Fix.** Replace `--key`/`--sampling`/`--label` with `--a8_id` (keep a separate `--equivalence_label` for the 8B test) and derive the key, sampling and models directory from the block. Call `check_own_code(require=True)` in every scored entry point once the pins exist. Record `no_design_gate` in the sidecar, and have `merge` refuse such shards for any output under `results/a8/<a8_id>/rotation/`.

### 10. The A2 equivalence test is close to a tautology, and the A2 wrapper's A3/A1 inputs are unpinned

`test_a2_loader_swap_gives_the_frozen_output_byte_for_byte` (`tests/test_a8.py:487-507`) replaces `a3_summary` with `lambda s: s` over the very file the frozen loader finds in the same folder. It shows that swapping a function for one returning the same dict leaves the output unchanged. It does not exercise:

- the production view `{key: s[a8_id]}` (`a8_analyse_a2.py:131`);
- the `--model`/`--reproduction` checks;
- `write_a1_view`;
- `main()` itself.

`--a8_a3_sha256` and `--a8_a1_summary` are free command-line inputs. Any hand-made summary passes together with its own hash, and the A1 summary is not hash-checked at all.

**Fix.**
- Add an end-to-end test of `a8_analyse_a2.main()`: rename the frozen summary's `gemma4` entry to an A8 id, run through `main`, and require the output to equal the frozen run byte for byte.
- Require the A3 sha256 to equal a value recorded by `a8_face_input_validity_summary.py` (for example a provenance JSON written next to its output), or recompute the summary from the cell files inside the A2 wrapper.
- Do the same for the A1 summary written by `a8_analyse_rotation.py`.

### 11. The A3 wrapper accepts an incomplete cell file, and the loss share is then understated

`summarise_model` sets `n_items_expected = len({trial_ids present})` (`scripts/face_input_validity.py:150-151`). The frozen A3 relied on complete runs. The A8 A3 job is explicitly multi-job and resumable (table step 4). A summary taken while the 32B file stops at item 200 reports 0% lost and decides on 200 items. `a8_face_input_validity_summary.py:83-88` checks only that the file exists.

**Fix.** Before summarising, require each A8 cells file to hold exactly 243 items x 10 cells as unique `(trial_id, rotation, condition)` keys, with failed cells present, and no partial line.

### 12. "Run settings cannot change any cell" is not true for errored trials

Draft line 270 claims that shard counts and GPU counts cannot change any cell. At 32B (66.7 GB of weights on an 80 GB A100), per-trial CUDA OOMs are caught and scored as wrong (`scripts/evaluate.py:983`), and gate, A2 and A4 cells fail in the same way. Memory headroom depends on GPU count and `device_map`, so the set of errored trials does too. In the mismatch condition the partner's clip is loaded, so OOM incidence can differ between real and mismatch and bias delta.

**Fix.** Add a pre-data rule: any errored trial or cell whose error is an OOM makes the job a failure, rerun with more GPUs before merge or analysis. Errored counts per condition are reported. Reword line 270 to "cannot change any successfully scored cell".

---

## Low

### 13. Fixed sidecar name per directory

`SIDECAR_NAME = "a8_shard.json"` is written to `output.parent` (`a8_rotation_shard.py:48, 214`). Two shards started at the same time in one directory both pass the existence check, and the later sidecar overwrites the earlier one. The merge then refuses (safe), but the compute is lost. `evaluate.py` also writes `{model}_{dataset}_{condition}_results.csv` next to the output, which collides in the same way. **Fix:** name the sidecar `<output stem>.a8_shard.json` and require a fresh directory per shard.

### 14. Merge consistency keys are incomplete

`SAME_ACROSS_SHARDS` (`a8_rotation_shard.py:263-265`) leaves out the evaluate argv, minus `--output`, so a shard run with the test-only `--human_options` override can be merged with production shards. **Fix:** compare `evaluate_argv` with the output value removed, and the resolved weights path (finding 2).

### 15. Relative paths are resolved against ROOT in A8 code but against the cwd in frozen code

`a8_preflight.py:107` and `a8_analyse_a2.py:125-127` resolve against ROOT, while the frozen scripts receive the same relative strings and resolve them against the cwd. A job started from another directory passes the checks and writes or reads elsewhere. **Fix:** `os.chdir(ROOT)` in every A8 entry point, or require absolute paths.

### 16. Storage check details

`--skip_storage` ("local tests only") is accepted in production and not recorded. **Fix:** record it in the provenance and refuse it when `--output` is under the real roots. The quota regex (`a8_preflight.py:116`) fails closed on format changes (grace column with spaces, unit suffixes), which is acceptable.

### 17. Small gaps in the A1 analysis wrapper

`a8_analyse_rotation.py` does not check whether `--errored-out` exists, and `--models` skips the design and frozen-code checks. **Fix:** refuse an existing sidecar, and always run `load_design` and `check_frozen_code`.

### 18. The A3 frozen-file equivalence has not been run

The frozen cell files are not in the local checkout, so `test_a3_wrapper_reproduces_the_frozen_summary_hash` skips. Freeze checklist item 2 lists only the A1 test as a cluster item. **Fix:** add the A3 frozen-file test to the cluster items and record its result before the freeze.

### 19. The float32 gate ruling is generalised beyond its basis

DECISIONS 2026-11-05 let the bfloat16 pass stand partly because Qwen's bfloat16 G2/G3 were exactly zero. The draft's step 1 says the bfloat16 verdict "then stands" in general. The frozen script's rule does decide this. Still, if the 32B shows a non-zero bfloat16 G2/G3, there is no float32 check to tell rounding from leakage. **Fix:** state that limit in Limits.

### 20. Weak "A8 path" check

`a8_face_input_validity_summary.py:81` accepts any path with a component named `a8`. **Fix:** require it under `results/a8/<a8_id>/face_input_validity/`.

### 21. Tests that pass without testing much

- `test_run_settings_are_not_in_the_block` is trivially true.
- `test_a1_analysis_points_the_frozen_module_at_a8_files_and_family` checks configuration only and never runs `main` on merged files. It would not catch the missing-Gemma crash (finding 5).
- `test_argv_matches_the_frozen_rotation_script` checks that flags appear in the slurm text, but not the slurm file's environment steps (finding 4).
- No test covers errored trials (finding 1), weights provenance (finding 2) or the A2 production view (finding 10).

---

## Freeze readiness

Not freeze-ready as drafted. Findings 1 to 4 must be fixed before the freeze: 1 blocks A1 outright on any error; 2 and 3 can silently mix models or overwrite frozen outputs; 4 can silently change the A1 estimand. Findings 5 and 6, and the data and module pins in 4 and 7, touch the hashed block or the family definition, so they must be settled **before** the design hash is tagged. The other medium findings (8 to 12) are code and test changes that fit the dated pre-data code-pin step. The low findings can go into that step too. With these fixed, the design keeps every frozen rule, threshold and family unchanged and cannot rescue a frozen verdict, and it would be freeze-ready.
