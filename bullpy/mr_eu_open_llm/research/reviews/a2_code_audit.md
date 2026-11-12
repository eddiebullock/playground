# A2 code audit against the frozen text (pre-data)

**Date:** 2026-11-12. **Auditor:** independent agent, read-only on the repository; this file is the only one written there.
**Scope:** whether the frozen A2 run and analysis path does what `research/PREREG_A2_DRAFT.md` says when it meets real model output, including the shard merge, Gate B, the sham, the 16 contrasts, the null rule, the .01 reading, the output schema and its downstream readers (A5, A7 Stage 1, Figure 6).
**A2 data:** none exist locally. `results/stimulus_interventions/` is absent, so neither cells nor the pinned `geometry.json` are on this machine. No A2 model cell was opened. The A3 cells in `results/face_input_validity/` are not A2 cells and were not opened.

## Files read (sha256 of the working tree at the time of this report)

| sha256 | File |
|---|---|
| `e464dd7ea610da2139e4d6e487c3b08046958edd9be97ec78609315b986e3dad` | `research/PREREG_A2_DRAFT.md` (all) |
| `aba4f533daad6210e87f7bb4e68fc4627805cb4dfd9d9308df734fada4edc8b1` | `scripts/analyse_stimulus_interventions.py` |
| `89d256c1072e0768fc747294553c5710adaf42184a175866d7bd75bcc6104978` | `scripts/stimulus_design.py` |
| `93c601eb42d6700d4f0fc8ba72317ed5c2e354105d57fcac611c59660125deb3` | `scripts/stimulus_interventions.py` |
| `f7880c94e9671cc294cd4eaeece2108d7cd1d7a4c7e6f9e30a1c495964e6cd82` | `scripts/run_stimulus_interventions.py` |
| `d0c8bf0d634101179c5feacb33cbacc341f998295b820214c008355285ed382b` | `scripts/recovery_stimulus_interventions.py` |
| `1c8d22ea637abedd3a3c2332331639c7e33a7e7ed9cb2e741335d17ce138eec5` | `slurm_jobs/study3_stimulus_interventions.sh` |
| `eafe3d23f9e22c435492aa39b4a0d39ece5bc44e05b08695a93f60e9a003d9cd` | `tests/test_stimulus_interventions.py` |
| `5f3870adc4a01cbb5e21186a53d0b4a88b71d2e6dd9c9dc27ecb431bba255299` | `tests/test_readout_and_analysis.py` (the A2 analysis tests) |
| `88a3d7d6619d7106e20aff505d3e9856abf803a8f5052bf1460c4e975b7a81b6` | `research/paper/v2/results.md` (section 7) |
| `b9fb93ccbc1a6de01fdb4c423b2e00e6e16199cc2638ab5baccda982dae0ee01` | `research/paper/v2/methods.md` (A2 parts) |
| `e773aa46dc452e77b8234ce54711c7956fbed213dee519b98194f05c16cba05e` | `research/DECISIONS.md` (A2, Gate B, sham, readout, A7 scope entries) |
| `17e3731e8b7f30fbb0604aceeff2790042104fe00e5fc12794227008a8212c40` | `scripts/a5_analyse.py` (A2 loading and gate parts) |
| `00350c9a182090943294efb602f46a31e2c4263240cbb8bb764de26476daaae9` | `scripts/a7_analyse.py` (A2 inputs) |
| `5bfb1a74e21596c944df2b946aea835e00a33244a50fd5917c2c52136bcbb665` | `scripts/a7_run.py` |
| `fcebaaa69f9a488acbf5c7c69d83488ea834f3361aecb7be3bf66931208b4015` | `slurm_jobs/study3_a7.sh` |
| `9f69d080044e41af4aa2b5f59309f03ff9db369889a8456e4ffb3549408046d6` | `research/PREREG_A7_DRAFT.md` (A2 dependencies) |
| `98890c5bff65f13abeb0549b61fe414dc9fa30082b84bcb564671683ec12e24f` | `research/scripts/analyse_rotation.py` (`actor_sign_flip_p`, `actor_bootstrap_ci`) |
| `ae949e4c1c3abc6a775a535c4e0a5042bce59ba14f7ac1afeefd476f032a5a2b` | `src/paper_v2/option_score.py` (`vision_mismatch_partners`) |
| `6327b6f88e99409cba5c63d5aca0fe97348fbce4f3b08522158a8d9ba3ce8194` | `scripts/vision_gates.py` (`gate_options`, `strip_pixels`) |
| `d8c6dd5f3138b6b05994eb8ddc74d31c7d0fa5295bd2b06a7beea8895c52d7fb` | `research/scripts/a2_gain_skew_sim.py` (header and test call) |
| `09e0cd4db5c794d56e3bc015fa8706ef6aab8249fb6d006396e69e1b2734458f` | `research/scripts/make_figures.py` (Figure 6) |
| `81500bdd229f88c5aa4038736d4ccc0eb2e04080613dc95380cc8c3dd067072a` | `research/reproduction/rotation_summary.json` (keys and `vision_dependent` only) |
| `c1c64e6975b32356a3314268c054f0dee483ddf74b11cd66d03b2faf972e64a4` | `research/reproduction/face_input_validity_summary.json` (keys and `vision_dependent` only) |
| `b9e5743f72c9a9271aa79d2fa0d2f446030793b4e0959a2acec7bbaa7057d764` | `research/reproduction/vision_gates_gemma4.json` (verdict fields) |
| `2f38aafb41c93505b792bd355c72d2a2b30e841a0e29be067a1d7c5ba01ede8d` | `research/reproduction/vision_gates_qwen3vl.json` (verdict fields) |
| `94f55b9fd96767e0c19c0bbeb34eebd5a47dbbbf5b14058d53d6f15a8b40a8ca` | `research/reproduction/input_delivery_gemma4.json` |
| `802f3231db29f2b27d67e8a6cfd70c78a9ea72c9e3fb4b1cadf7f8c966f1f801` | `research/reproduction/input_delivery_qwen3vl.json` |
| `83c146cdf6f032f314f6ada9ca01428ff192575155d40dda60eac3fd8c8becfe` | `research/reproduction/recovery_stimulus_interventions.json` |
| `5476f7140476981f2696a271a329aa6a083839c18933c0512666105536f7bb43` | `research/STATE.md` (A1, A2 job and sync lines) |
| `11c480454e09805a615307d142f4fbb6c1db465cf363c7e683be31b2c22ce820` | `research/reproduction/REPRODUCE.md` (A2 output path) |

## Tag comparison

- `git -C /Users/eb2007/playground diff --name-status a2-freeze -- bullpy/mr_eu_open_llm` lists no modified code file. The only modified files are text (`.gitignore`, `research/*.md`, one report); everything else is added.
- The six files pinned at `PREREG_A2_DRAFT.md:7` match their stated sha256 and the bytes at the tag: `stimulus_design.py`, `stimulus_interventions.py`, `run_stimulus_interventions.py`, `analyse_stimulus_interventions.py`, `recovery_stimulus_interventions.py`, `study3_stimulus_interventions.sh`. So do `scripts/human_response_predictor.py` (named at `:14`), both test files, and the analysis dependencies `research/scripts/analyse_rotation.py`, `src/paper_v2/option_score.py`, `scripts/prompts.py` and `scripts/vision_gates.py`.
- `PREREG_A2_DRAFT.md` differs from the tag only by the appended "Deviations" section (`:330-357`). The diff is additions only, after line 328.
- `run.design_hash()`, recomputed locally, gives `0ae22118d3b2690c581595890b9bb8b904e1f7860f37be9ac909f925b7058de7`, as stated at `:5` and `:14`.
- The committed A3 summary hashes to the pinned `A3_SUMMARY_SHA256` (`stimulus_design.py:24`).
- The A2 pins inside A5 and A7 (`a2_code_sha256`, `a2_design_sha256`, `a3_summary_sha256`, and A7's evidence pins on the vision-gate and delivery files) all match the working tree.

**Concurrent edits.** During the audit, other sessions edited non-A2 files in the working tree, for example `research/DECISIONS.md` (a new 2026-11-12 entry), `research/paper/v2/{methods,results,outline}.md`, `scripts/reproduce.py` and figure files. None of them is an A2-pinned file. This audit wrote nothing outside its scratchpad except this report.

## Tests and synthetic runs

- `tests/test_stimulus_interventions.py`: **64 passed**. `tests/test_readout_and_analysis.py`: **50 passed**.
  - Both files were run with `/opt/anaconda3/bin/python3 -m pytest -p no:cacheprovider`, with bytecode writing off and `--basetemp` in the scratchpad. That is pytest 8.3.4 and numpy 2.1.3.
  - Disclosure: before the coordinator's instruction not to install anything arrived, I had installed pytest into a scratchpad folder and run both files with the env's Python 3.10 and numpy 1.26.4. The counts were the same, 64 and 50 passed. That folder was then deleted. Nothing was installed into the env or the repository.
- **Synthetic runs**, with the env's Python and the frozen code, writing only to the scratchpad:
  - The frozen `main` ran on 3 synthetic shards: 7,047 rows, the real 243 items and 17 actors, and the real gate files in `research/reproduction`. Only `run.check_geometry` was bypassed, because the pinned geometry is on the cluster. The run finished in 105 s.
  - `load_shards` was run on corrupted shard sets.
  - `analyse_route` was run on an opposing mask effect.
  - The frozen recovery was rerun with 100 replicates. Its output is **byte-identical** to `research/reproduction/recovery_stimulus_interventions.json`.

## Findings

Severity: **V** could change a verdict; **C** would crash, or silently drop or misread data; **R** reporting or text mismatch only; **N** note.

### A2-1 (V). The fill rule on Opposes verdicts is ambiguous, and `fills_agree` is relative to the predicted direction

- **Text.**
  - `PREREG_A2_DRAFT.md:144`: "**Fill rule.** The neutral fill is primary. A mask effect counts only if it passes and the first-frame and blur fills have the same sign, since the occluder is itself an intervention."
  - The decision table at `:190-191` puts "fill rule met" on **Matches prediction** only. Opposes is "The same gates passed; step-down p < .05 two-sided in the opposite direction".
  - `methods.md:226` states the rule as "the first-frame and blur fills share the neutral fill's sign".
- **Code.**
  - `analyse_stimulus_interventions.py:176-181` sets `fills_agree` to whether every other fill's mean actor effect, times `direction`, is above 0. That is the predicted direction, not the neutral fill's sign.
  - `verdict` (`:199-212`) uses `fills_agree` for Matches only (`:205`). Opposes (`:207`) ignores the fills.
  - The output keeps no per-fill signed means, so the `:144` reading cannot be applied to an Opposes row from the output.
- **Scenario.** Run through the frozen `analyse_route`: P1, 8 actors, with the neutral fill raising Happy (opposite to the prediction).
  - (a) The first and blur fills also raise it, so they agree with the neutral fill. Output: `fills_agree: False`, verdict `opposes_prediction`.
  - (b) The first and blur fills lower it, so they disagree with the neutral fill. Output: `fills_agree: True`, verdict `opposes_prediction`.
  - Under `:144`, case (b) is not a mask effect and would be Inconclusive.
  - In case (a), a reader checking `fills_agree` would wrongly conclude that the fill rule failed.
- **Handling.** Add a dated pre-data deviation (clarification) that fixes the reading before any output exists. Either:
  - the decision table governs: the fill rule applies to Matches only, and `fills_agree` means "other fills in the predicted direction" and is not read on Opposes rows; or
  - `:144` also applies to Opposes, computed by a declared script from the cells with the frozen `actor_effects`, comparing each fill's sign with the neutral fill's.

### A2-2 (C). A truncated cells line stops the analysis, and the resume path makes it worse

- **Text.**
  - `:275`: "Analysis takes all of them and refuses an incomplete set, a missing cell, duplicates, anything other than 243 items, or more than one design hash."
  - The slurm header says to "Resubmit a lost task with the same N_SHARDS and --array=<task>".
  - Nothing covers a damaged line.
- **Code.**
  - `run_stimulus_interventions.py:1087-1097`: `done_cells` skips lines that do not parse.
  - `:1160`: the cells file is opened in append mode.
  - `:1194-1195`: each row is written as row plus newline, then flushed.
  - `analyse_stimulus_interventions.py:660`: `json.loads` runs on every non-empty line with no handler.
- **Scenario.**
  - A shard dies mid-write, leaving a partial row with no newline. Triggers include the RDS quota failure of 2026-11-09 (`DECISIONS.md:487`, "while writes failed") or a node failure.
  - On resume, that cell is not "done", so it is recomputed and appended directly after the partial text. Both end up on one unparseable line.
  - Synthetic check: after the resume, the cell is still not seen as done, and `load_shards` raises `JSONDecodeError`. It also raises on a simple truncated last line.
  - Further resumes add a clean copy of the cell but never remove the corrupt line. The single analysis therefore cannot run unless a cells file is edited by hand, and no rule for that exists.
- **Handling.** Add a dated pre-data deviation that fixes a repair procedure, with no code change:
  - Delete only the lines that do not parse, recording their byte offsets and sha256.
  - Resume the shard so the missing cells are recomputed.
  - Keep any backup outside `results/stimulus_interventions/`, or under a name that does not match `cells_{model}_shard*.jsonl`. A5 globs that pattern (`a5_analyse.py:1028-1032`) and would load the backup, then refuse on duplicates.

### A2-3 (C). Cell rows carry no model, so cells from two models can be merged without notice

- **Text.** `:14`: "Analysis refuses cells whose hash differs from the code". `:275` lists the refusals.
- **Code.**
  - `run_stimulus_interventions.py:1162-1170`: the row fields include no model.
  - `analyse_stimulus_interventions.py:661` checks the model on the reports only.
  - `:663-665` checks the design hash, which is the same for both models.
- **Scenario.** `--cells` is given `cells_gemma4_shard0`, `cells_gemma4_shard1` and another model's shard 2, with Gemma's three reports. The synthetic check gave "ACCEPTED, 7047 rows, 243 items": a third of the items would be another model's.
- **Handling.** Add a disclosed check before the single run, recorded with it:
  - every cells path names the model;
  - every ok row's `input.pixel_shapes` fits the model. On `single_face`, `pixel_values` is [1, 2520, 768] on Gemma 4 and [2304, 1536] on Qwen3-VL (`input_delivery_*.json`).

  The likelihood is low, since this needs an operator error.

### A2-4 (R). Equivalence intervals are computed but not written

- **Text.**
  - `:204`: capped contrasts "cap at Inconclusive and report their equivalence interval".
  - `:205`: condition 5 is the 90% actor-bootstrap CI.
  - `results.md:213` has the slot "equivalence intervals for capped contrasts: [PENDING]".
- **Code.**
  - `equivalence_pass` (`:234-237`) computes the 90% interval and returns only a bool.
  - `:511` stores `null_equivalence` as a bool.
  - Each row keeps `ci95` only. The joint-ablation interval (condition 4) is also kept as a bool only (`:306-308`, `:314`).
- **Scenario.** The results.md slot cannot be filled from the output for P5, P6, P9, P10, P14a, P14b, P15a and P15b, nor for P12 and P13 on Qwen3-VL.
- **Handling.** Declare a separate script before data. It calls `an.actor_interval(an.actor_effects(cells, c, actors), 0.90)` with the frozen defaults (10,000 draws, seed 42). That is the same call as inside `equivalence_pass`, so it reproduces the interval exactly.

### A2-5 (R). Decision-class names differ between texts

- **Text.**
  - `PREREG_A2_DRAFT.md:357` says "each Supports or Opposes verdict", and so do `DECISIONS.md:398` ("for every Supports verdict") and `methods.md:218`.
  - The frozen table (`:190`), `methods.md:226`, `results.md:200` and `:212` say "Matches prediction".
- **Code.** `verdict` returns `matches_prediction` (`:206`), and Figure 6 maps that value (`make_figures.py:1095-1099`, at the sha256 in the header).
- **Handling.** Fix the text in the paper: "Supports" means `matches_prediction`. No deviation is needed.

### A2-6 (R). Human-distribution and weak-label outputs are narrower than the text

- **Text.**
  - `:129`: "Each outcome is also computed ... as Jensen-Shannon divergence between the model and the human response distribution".
  - `:210`: "Every result appears twice: against the acted label, and against the human response distribution."
  - `:213`: "with the human modal label beside the acted label".
  - `:214`: "Leakage flag. Model accuracy well above human on these items ...".
- **Code.**
  - `human_tables` (`:402-415`) gives only the mean change in JS divergence per condition, split by weak label and pooled over all labels and items. There is no per-contrast or per-label JS.
  - `descriptive_changes` (`:463-486`) is by acted label only.
  - No accuracy is computed for the flagged items.
  - Per-contrast results against the human modal label do exist (`against_human_modal_label`, `:517`).
- **Handling.** All of this is descriptive (`:216`). Either disclose it as a limit, or declare a descriptive script before data.

### A2-7 (R). There is no "shuffles overall" table

- **Text.**
  - `:178`: "`shuffle` overall | Change from `original`, with `static_apex`".
  - `:261`: the descriptive tables include "shuffles overall".
  - `results.md:214` has the slot "shuffle against `static_apex`".
- **Code.** `descriptive_changes` (`:472-477`) runs one condition at a time, so `shuffle_a`, `shuffle_b` and `static_apex` appear as separate rows. The mean of the two shuffles, with its actor CI, is not written.
- **Handling.** Report the shuffles side by side, or declare a small descriptive script, using the same `actor_effects` with `a=SHUFFLES`.

### A2-8 (R). The frozen text says recovery was and was not rerun

- **Text.** `:9`: "Recovery was rerun on the frozen analysis: 100 replicates, all five observers pass, unchanged." `:253`: "It was not rerun after the audit".
- **Check.** My rerun with 100 replicates on the frozen code is byte-identical to the committed file. The substance holds under either sentence.
- **Handling.** No action, or a one-line note.

### A2-9 (N). Some failures are not written as failed cells, and some failed cells are never retried

- **Text.** `:34`: "Every failed cell stays in the output with its reason ... No item is dropped silently".
- **Code.**
  - `run_stimulus_interventions.py:1189` catches only `EditFailure`. Any other exception, such as a model or CUDA error, stops the shard with no row written. That is fail-safe, because the analysis then refuses on missing cells.
  - Conversely, `done_cells` (`:1087-1097`) counts failed rows as done. A transient I/O error on a clip read (`video_unreadable`, `frames_unavailable`) therefore becomes a permanent, logged exclusion of all 29 cells of that item. A1 had one I/O-errored trial (`STATE.md:35`).
  - The report's `failures` list is reset on every start (`:1151`), so after a resume it under-reports. The analysis does not use it; it counts failures from the cells.
- **Handling.** Add a dated pre-data rule on whether failed cells with I/O reasons may be removed and rerun before the analysis. Otherwise, disclose it.

### A2-10 (N). A missing gate input reads as a failed gate, and the run still completes

- **Code.**
  - `outside_gates` (`:606-614`) reads relative paths: `--reproduction` (`:688`) and `A3_OUTPUT_DIR` (`:586`). A missing file gives `False` (tested at `test_readout_and_analysis.py:417-424`). Every verdict is then Inconclusive and the output looks final.
  - A5 instead stops on a missing gate file (`a5_analyse.py:1046-1052`).
  - With the current files, `rq0`, `g2_g4` and the delivery file are all True for Gemma 4 and Qwen3-VL on both routes. A1 and A3 are `vision_dependent: true`, G2 to G4 pass with `smoke: false`, and delivery passes on `single_face` and `video_face`.
- **Text.** `:193` ("A gate with no result has not passed") agrees with the code. The risk is operational: "the frozen analysis then runs once".
- **Handling.** Before the single run, record a check that:
  - the working directory is the project root;
  - the four gate files and the A3 copy are present, with their sha256;
  - `geometry.json` matches its pin.

  Optionally add a dated rule that a run with a missing input is void.

### A2-11 (N). Delivery fails a whole route if any delivery item cannot be built; B3 uses shard 0's baseline only

- **Code.**
  - `delivery_passed` returns False if any delivery row is a `failed` row (`:624`). Line 34 says a failure "fails only the cells that need it".
  - B3 uses only the first report's image-free row (`:708`).
- **On the pinned geometry, this cannot arise from faces.** `input_delivery_{gemma4,qwen3vl}.json` report `n_clips_for_face_width: 243`, so every clip has a face box. The apex is chosen among frames with landmarks (`stimulus_interventions.py:326-339`), so the original builds on both routes.
- The delivery items are order positions 0 to 5. Positions 0 to 2 are the smoke items, which built on all routes in smoke_v2 (`PREREG_A2_DRAFT.md:304`).
- **Handling.** No action.

### A2-12 (N). Reporting details

- The output does not record `n_flips` or `exact`: `test_family` drops them (`:170-190`). The stated 2^17 exact flips follow from the code defaults (`stimulus_interventions.py:573`, `:590-592`) and the 17 actors.
- No scope tag is written per row (`:195`).
- The output contains non-standard JSON tokens. My synthetic output had 72 `Infinity` (`region_tokens` for non-mask cues) and 45 `NaN`. Python reads them; strict parsers do not.
- A route that fails Gate B or delivery still writes contrast p-values, with verdict Inconclusive, although `:101` and `:117` say such a route "is not analysed" and goes "no further". The paper should not report those p-values.
- In the sensitivity rows, the region-size rule (`region_tokens`, `:258-281`) ignores `exclude`.

### A2-13 (N). Downstream readers

- **A7 token-floor sensitivity.** A7's token-floor sensitivity reads a `below_token_floor` list from its own inputs file (`a7_analyse.py:711`). A2 writes only the share (`share_excluded_under_8_tokens`, `:752`). If the list is absent, A7 skips that sensitivity silently.
- **"Not run" against "Not tested".** `DECISIONS.md:495` says a failed precondition is "reported as not run". A7's code (`a7_analyse.py:423-424`) and prereg (`PREREG_A7_DRAFT.md:424-425`) report **Not tested**.
- Both are A7-side wording and plumbing. The keys A7 conditions on are present and boolean; see "Checked and consistent".

### A2-14 (N). Pins are not checked at analysis time

- The A2 analysis checks the design hash only (`:699-700`, `:663-665`). It checks neither the file hashes at `:7` nor Gate B's dependencies:
  - `analyse_rotation.py` gives the B4 and B5 statistics;
  - `option_score.py` gives the partners.
- Neither dependency is in A2's pin list.
- Both are unchanged since the tag, and they are pinned at the same bytes by A5 (`PREREG_A5_DRAFT.md:141-142`), A8 and A9. A5 and A7 do check A2's code hashes at run time.
- The run report records `script_sha256` for the run script only (`run_stimulus_interventions.py:1141`).
- **Handling.** Record the sha256 of all of these files with the single run.

### A2-15 (N). The P13 risk named in the deviation is present in the data

- **Text.** `:351`: Richoz 2024 is "a risk for P13 if Afraid is among the Low items".
- **Data.** 9 of the 48 Low Intensity items are Afraid. The full split is Disgusted 10, Afraid 9, Angry 8, Surprised 8, Happy 7 and Sad 6, from `is_low_intensity` over `data/eu_emotions_human_entropy.json` and the manifest. The paper can state this fact before data.

## Checked and consistent

1. **Audit fixes 1 to 14 (`:315-328`) are present in the code.**
   - Fix 1: B5 is its own term (`:357-363`).
   - Fixes 2 to 4: six null conditions (`:284-326`); gates must be exactly True (`:194-196`); a significant contrast is never Null (`:209-211`).
   - Fix 5: P6 takes the fill rule on both masks (`stimulus_design.py:74-84`).
   - Fixes 6 and 7: all sensitivities and the smoke items (`:719-759`, `stimulus_design.py:27-31`); CIs, B4 repeats, flip rates, failures and descriptive tables.
   - Fixes 8 and 9: `N_SHARDS` is required and the shard validated (`run_stimulus_interventions.py:198-206`, slurm `:42-47`); every shard is read and gaps are refused (`:657-678`); B1 uses every shard (`:728`).
   - Fixes 10 to 12: design hash coverage (`run_stimulus_interventions.py:172-189`); geometry pin (`:192-195`, `:1107`); the landmarks step refuses to overwrite (`:408-409`); shuffle gating (`:634-636`).
   - Fixes 13 and 14: RQ0 table (`:579-583`); margin per model and route (`:730`); signed dose (`:240-255`); `single_region_null` (`:418-429`); P12 and P13 reach Null where frame order is verified (`:304`); A3 read from either path, held to the pinned bytes (`:594-603`).
2. **Cell path.**
   - Full-run names are `cells_{model}_shard{k}.jsonl` and `run_{model}_shard{k}.json` (`:1121-1124`). The 29 conditions per item are 20, 7, 1 and 1 by route. Shards are assigned by position mod `N_SHARDS`.
   - Every row carries `design_sha256`. Failed rows keep item metadata and their reason.
   - The analysis refuses all of these (tests and my runs): a missing shard or report, a missing cell, a duplicate, a wrong item count, a foreign hash, a smoke report, and mixed `n_shards`.
   - An `EditFailure` cell is kept as failed and held out of the analysis.
3. **Gate B (`:329-399`).**
   - Thresholds `B_RULES` are `.05`, `.25`, `.01` and `1e-3`, with comparisons `>=`, `>=`, `<` and `<=`.
   - B4 is the item's family on its own face minus that family on the donor's face. Donors come from `vision_mismatch_partners` with seed 42 over all 243 ids; the assignment does not depend on input order.
   - B4 and B5 need p < .01 and CI lower bound > 0. They use A1's Monte Carlo sign-flip, as disclosed at `:254`.
   - The B4 repeats are reported and do not decide the gate. `pass` is B1 to B5 together, per model and route.
   - The margin is 10% of the mean absolute B4 item change on that route.
4. **Sham (`:645-654`).** The `single_face` sham and original must send identical media hashes, with at least one pair present. The check is True by construction on `video_face`, as `:193` says.
5. **Outside gates.** The key paths match the files: `<model>.vision_dependent` in the A1 and A3 summaries, `verdict.G2-4.pass` with `smoke`, and `verdicts.<route>.pass`.
6. **Delivery.**
   - The gate requires the run's rows and the delivery file.
   - On `video_face`, every row must show that pixels change under shuffle.
   - Frame order is verified only where every `video_face` row shows it. Gemma's video tensor keeps a frame axis ([1, 8, 2520, 768]); Qwen's has none ([9216, 1536]). Hence 8 of 16 contrasts are capped, or 10 of 16 on Qwen.
7. **Contrasts (`stimulus_design.py:68-95`).**
   - There are 16 contrasts for 14 predictions, with families of 10 on `single_face` and 6 on `video_face`.
   - All directions match `:156-170`. The target is left out for P7, P8, P14 and P15 (`:111`).
   - P6 is an interaction within one actor set. Happy, Afraid and Disgusted share actors C, E, G, I, K, M, O and S. Sad has B, D, F, H, J, N, P and R. Angry and Surprised add L.
   - P13 covers 48 items and 17 actors. The capped contrasts are P5, P6, P9, P10, P14a, P14b, P15a and P15b.
8. **Test.**
   - With 17 actors, `main` gets the exact 2^17 flips (131,072). I checked the step-down maxT algorithm.
   - Floors: 1/256 one-sided and 2/256 two-sided with 8 actors; 7.6e-6 with 17. My synthetic run shows these values.
   - Opposes uses a separate two-sided family. Equivalence is the 90% CI strictly inside the margin.
9. **The .01 sensitivity can be taken from the frozen output.** Every row of `primary_blind_acted` carries `p_maxt` (step-down one-sided, for Matches) and `p_maxt_two_sided` (step-down two-sided, for Opposes). Both are adjusted p-values, so comparing them with .01 is the step-down at .01. No separate script is needed. The results.md "step-down *P*" slot should use `p_maxt_two_sided` on Opposes rows.
10. **Downstream.**
    - A5's globs and keys match A2's files: report `baselines` with `status`, delivery rows, and `tokens_across_view`.
    - A7 reads `single_face.gates` (`rq0`, `g2_g4`, `gate_b`, `delivery`, `sham`, all written as booleans), `single_face.margin`, the top-level `design_sha256` and each row's `design_sha256`. A7's condition names (`original`, `mask_eyesmouth_{neutral,first,blur}`) exist in A2.
    - Figure 6 reads `mean_change`, `ci95`, `direction` and `verdict`. All are present.
11. **results.md section 7.** Every slot exists under these names except A2-4 and part of A2-7:
    - `gate_b.B1` to `B5` and `pass`;
    - per contrast `mean_change`, `ci95`, `p_maxt` and `verdict`;
    - `sensitivity.{without_smoke_items, canonical_word_only, clips_at_8_tokens_or_more}`;
    - `results.against_human_modal_label`.

    The line references at `results.md:196` and `:205` point at the right lines.
12. **Disclosed limits match the code.**
    - The MCQ bridge is empty for P7, P8, P14a, P14b and P15b. P15a is not empty: Disgusted is an option on all 14 Angry items.
    - No item can raise the target-first error in `gate_options`: the target is at index 0 for all 243 items, and rotations are 1 to 4.
    - The post-freeze deviations (`:332-357`) change no code or hash, as they state.

## Not checked

- **The GPU path was not run:** `score_cell`, `delivery_for_item` and `baseline_cells` need models and clips, so they were checked by reading only. `main` ran with `check_geometry` bypassed, because the pinned `geometry.json` is only on the cluster.
- **The cluster copy of `research/reproduction` was not checked.** It is unknown whether it holds `rotation_summary.json` (A2-10), since I did not connect over SSH.
- **A8's wrapper was not audited.** `research/scripts/a8_analyse_a2.py` reruns this analysis for A8 models and is outside this brief.
