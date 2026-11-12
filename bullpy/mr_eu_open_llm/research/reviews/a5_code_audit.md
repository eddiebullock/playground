# A5 code audit: frozen analysis against the final frozen text (pre-data)

**Date:** 2026-11-12.

**Question.** Does `scripts/a5_analyse.py`, as frozen at tag `a5-freeze`, do what the final frozen text says? The final text is `research/PREREG_A5_DRAFT.md`, including "Decision after run 6" and DECISIONS 2026-11-08 (two A5 entries). The check covers the evidence-class definitions, which are confirmatory, and runs on the A2 cell format that `scripts/run_stimulus_interventions.py` writes.

**Data status.** No A2 or A5 output exists locally:
- `results/stimulus_interventions/` and `results/a5/` are absent.
- No A2 `cells_<model>*.jsonl` or `run_<model>*.json` file was found.
- `results/face_input_validity/` holds A3 cell files. They were not opened.

**What was written.** Only this file. Nothing was committed or tagged, and the cluster was not touched. Synthetic files and scratch scripts are in the session scratchpad (`.../scratchpad/audit_a5/`). The test and scenario runs were set up so that nothing was written under `tests/`, `scripts/`, `research/` or `results/`. Other files in the repository changed during the audit, from concurrent work on unrelated modules (for example `scripts/reproduce.py`, `research/STATE.md`, `research/RESULTS_REGISTRY.md` and `research/paper/v2/results.md`). No A5 or A2 file, and no A5 or A2 bytecode cache, was touched.

## Files read, with sha256

| File | sha256 |
|---|---|
| `research/PREREG_A5_DRAFT.md` | `0980db348469758d783060d79406df6270a8410fbb4051e31f136146e489a086` |
| `scripts/a5_analyse.py` | `17e3731e8b7f30fbb0604aceeff2790042104fe00e5fc12794227008a8212c40` |
| `scripts/a5_design.py` | `44ebd027a8dcf41cbe553896564d35d3eee82ffa7605ec1fc64667829b015b58` |
| `scripts/a5_checks.py` | `b38e05fdb26ddc171623eda7f999463e0856938cfe37cd8827fc89cb02fb6b5e` |
| `scripts/a5_recovery.py` | `56d9d195238d50195b9f3ed5919ad6c7e904bde39633e74c34f93f5e34c3f7e8` |
| `scripts/a5_family_diagnostic.py` | `06c0a372a3876f749b5baddacdb6f06bed19fcb3d1f9b46e3eeb5e5103989da3` |
| `scripts/a5_dependence_diagnostic.py` | `6f8ea78a57c46e0604c567cc1aadba805a0a1a52d79170e7c73c9c2496ba8d41` |
| `scripts/a5_skew_diagnostic.py` | `f7493fd4acb978fa988babd64ba066b16d0fa6de9ca8c2b47b53dad095b9ed90` |
| `scripts/a5_gain_guard_diagnostic.py` | `7b6393a4f404ec7afdbe3c98d4813aeeb27e4184cfc9fb1c40f00fa6d59ae31f` |
| `tests/test_a5.py` | `1540afbef3d1891e89c374e72b803c739545095a438886b5abfaeba293410521` |
| `scripts/run_stimulus_interventions.py` | `f7880c94e9671cc294cd4eaeece2108d7cd1d7a4c7e6f9e30a1c495964e6cd82` |
| `scripts/stimulus_design.py` | `89d256c1072e0768fc747294553c5710adaf42184a175866d7bd75bcc6104978` |
| `scripts/stimulus_interventions.py` | `93c601eb42d6700d4f0fc8ba72317ed5c2e354105d57fcac611c59660125deb3` |
| `scripts/analyse_stimulus_interventions.py` | `aba4f533daad6210e87f7bb4e68fc4627805cb4dfd9d9308df734fada4edc8b1` |
| `research/scripts/analyse_rotation.py` | `98890c5bff65f13abeb0549b61fe414dc9fa30082b84bcb564671683ec12e24f` |
| `src/paper_v2/option_score.py` | `ae949e4c1c3abc6a775a535c4e0a5042bce59ba14f7ac1afeefd476f032a5a2b` |
| `slurm_jobs/study3_stimulus_interventions.sh` | `1c8d22ea637abedd3a3c2332331639c7e33a7e7ed9cb2e741335d17ce138eec5` |
| `research/paper/v2/results.md` (working copy, with uncommitted edits in the RQ5 text that appeared during the audit; the RQ4 slot is unchanged) | `88a3d7d6619d7106e20aff505d3e9856abf803a8f5052bf1460c4e975b7a81b6` |
| `research/paper/v2/methods.md` | `b9fb93ccbc1a6de01fdb4c423b2e00e6e16199cc2638ab5baccda982dae0ee01` |
| `research/DECISIONS.md` | `e773aa46dc452e77b8234ce54711c7956fbed213dee519b98194f05c16cba05e` |
| `research/reproduction/rotation_summary.json` (structure and `vision_dependent` only) | `81500bdd229f88c5aa4038736d4ccc0eb2e04080613dc95380cc8c3dd067072a` |
| `research/reproduction/face_input_validity_summary.json` | `c1c64e6975b32356a3314268c054f0dee483ddf74b11cd66d03b2faf972e64a4` |
| `research/reproduction/vision_gates_gemma4.json` (gate fields only) | `b9e5743f72c9a9271aa79d2fa0d2f446030793b4e0959a2acec7bbaa7057d764` |
| `research/reproduction/vision_gates_qwen3vl.json` (gate fields only) | `2f38aafb41c93505b792bd355c72d2a2b30e841a0e29be067a1d7c5ba01ede8d` |
| `research/reproduction/input_delivery_gemma4.json` (verdicts only) | `94f55b9fd96767e0c19c0bbeb34eebd5a47dbbbf5b14058d53d6f15a8b40a8ca` |
| `research/reproduction/input_delivery_qwen3vl.json` (verdicts only) | `802f3231db29f2b27d67e8a6cfd70c78a9ea72c9e3fb4b1cadf7f8c966f1f801` |
| `data/eu_emotions_full_manifest.json`, `data/eu_emotions_human_entropy.json` (item metadata only) | `54e9976f...`, `f9d91b4e...` (equal to draft line 633) |

## Tag comparison

The tag `a5-freeze` is commit `9b127200` (2026-11-08 13:59 UTC).

- **Pinned files.** For the draft and all 35 files in the freeze lists (draft lines 631 and 633), `git diff a5-freeze -- <path>` is empty. The working-tree sha256 equals the tag blob's sha256 in every case.
- **Stated hashes.** All 47 sha256 values stated in the draft equal the current files: 12 under "Code hashes", 17 on line 631 and 18 on line 633.
- **Design block.** It hashes to `fd20495bf411b97bb876a32b0d93994b98bcc73a05aa104c6976f3fe2988efd4`. That equals `a5_design.DESIGN_SHA256` (`a5_design.py:20`) and the stated hash (draft line 5).
- **A2 and A3 pins.** `run.design_hash()` equals the block's `a2_design_sha256`. The four A2 code pins and the A3 summary pin hold.
- **Gate files that A5 reads but does not pin:**
  - `vision_gates_{gemma4,qwen3vl}.json` and `input_delivery_{gemma4,qwen3vl}.json` are unchanged since the tag.
  - `rotation_summary.json` did not exist at the tag. It came with the A1 lock, commit `6052b7c` on 2026-11-09. Its sha256 matches the prefix `81500bdd` recorded at `RESULTS_REGISTRY.md:259`.
- **Run 6 against the frozen file.** Run 6 was produced by `a5_analyse.py` `e726c03a...`, commit `81e2c08`. `git diff 81e2c08 9b12720 -- scripts/a5_analyse.py` adds only `DESCRIPTIVE_NOTE`, `label_claims`, its two call sites and `claim_scope`. No computation changed, which confirms draft line 588 ("The change is labels only").

## Tests

- **The specified command fails.** `/opt/anaconda3/envs/mr_eu_open_llm/bin/python -m pytest tests/test_a5.py -q` gives `No module named pytest`. That env (Python 3.10.20, numpy 1.26.4) has no pytest.
- **The coordinator's command.** `/opt/anaconda3/bin/python3 -m pytest tests/test_a5.py -q` runs on Python 3.13.5, numpy 2.1.3 and pytest 8.3.4. Result: **108 passed** in 1,335 s.
- **The analysis env's own interpreter.** pytest was made importable through a scratchpad folder of symlinks to the base install's pure-Python `pytest`, `_pytest`, `pluggy` and `iniconfig`. Nothing was installed, and numpy is 1.26.4 as in the analysis. Result: **108 passed** in 2,153 s. So the numpy 1.26 and 2.1 runs agree on every test.
- **No writes to the repo.** Both runs used `-p no:cacheprovider`, `PYTHONDONTWRITEBYTECODE=1` and `--basetemp` in the scratchpad.

## Synthetic runs on the real A2 cell format

**How the shards were built.** A generator (`audit_a5/audit_code/gen.py`) writes shards the way `step_run` does:
- file names as at `run_stimulus_interventions.py:1121-1124`;
- report fields as at `:1130-1152`;
- row base fields and failed rows as at `:1162-1194`;
- delivery rows and baselines only on the first `n_delivery_items` = 2 items of each shard, as at `:1196-1199` and `:1407`;
- `blind` fields with the runner's own `family_scores` and its rounding, as at `:806-815`;
- baseline rows as at `:1053-1079`.

The items are the real 243, in `select_items` order and split over 3 shards, as the Slurm job does (`study3_stimulus_interventions.sh:22,37`). Scores are synthetic, from `a5_recovery.World`. Gates come from the real `research/reproduction/` files, which were not modified. Where a scenario edits a gate file, it edits a scratch copy.

All runs used the analysis env's Python. s1 went through `main()` with the frozen default of 1,000 label shuffles. The others called `analyse_runs` with 200 shuffles (s11: 100).

| ID | Input | Outcome |
|---|---|---|
| s1 | Clean, both models, through `main()`. Only `run.check_geometry` was bypassed, because no `geometry.json` exists locally | Ran in 98 s. `rq0`, `g2_g4` and `delivery` are True on all 4 cells from the real gate files. Gate B passed, the families have 4 cells, and evidence is Shown on all 4. Output `design_sha256` is `fd20495b...` |
| s2 | 10 `single_face` originals failed, and all 15 `video_face` originals of one actor | Ran. `n_clips` 233 and 228, with no NaN fault |
| s3 | qwen3vl shards written without `video_face` (`--inputs` without it) | Passed A2's `load_shards`, then stopped with `StopIteration` at `analyse_stimulus_interventions.py:334`. No output |
| s3b | Every qwen3vl `video_face` cell failed | Same `StopIteration`. No output |
| s4 | Shard 0 has `baselines: null` | gemma4: `no_baseline`; B3 False (no input); evidence Unmapped; flags `b3_failed` and `readout_dependence` |
| s5 | Shard 2's `image_free` differs by .002 nats on one family | `no_baseline`; B3 still from shard 0; Shown kept; word share and chain null |
| s6 | Scratch `rotation_summary.json` with `vision_dependent` False | `video_face` Not tested. `single_face` families have 2 cells, so balance is withheld (`balance_small_family`). Evidence is still named |
| s7 | No qwen3vl files | qwen3vl Not tested (`no_a2_cells`). gemma4 families have 2 cells, so balance is withheld |
| s8 | Reproduction folder without the A3 summary | All 4 cells Not tested. The run completes, with `gates.a3: null` |
| s9 | Near-flat scores (T = .0004) | All 4 cells Inconclusive with `no_structure`. The `confirmatory` block reads `evidence: null` |
| s10 | One shard's gemma4 `video_face` delivery row is `{"failed": ...}` | gemma4 `video_face` is Not tested (delivery False). The other three cells form 3-cell families: named balance classes are withheld (`balance_small_family`), and a natively Unresolved one keeps `balance_unresolved` |
| s11 | qwen3vl cells present, `run_qwen3vl_shard*.json` absent | Run completes. qwen3vl is Not tested (`no_a2_cells`), and gemma4 families have 2 cells |
| s11b | qwen3vl reports present, cells absent | Same as s11 |

## Findings

No V finding. On the A2 format and the real gate files, the evidence class follows the text in every rule, value and order checked (see "Checked and consistent"). Three input paths can crash or silently change the result (C). Four reporting gaps (R) and six notes (N) follow.

### A5-1 (C): a model whose cells or reports were not pulled is reported as having no A2 cells

**Text.**
- Line 269: "**Not tested**. Quantities computed for the record and carry no verdict. This covers Molmo2 and any model with no A2 cells."
- Line 413: "A missing `rotation_summary.json`, `vision_gates_*` or `input_delivery_*` stops the analysis rather than reading as a failed gate."

**Code.**
- `a5_analyse.py:1069-1070` returns None when either the cell files or the report files of a model are absent.
- `a5_analyse.py:1088-1092` then writes Not tested rows with flag `no_a2_cells`, and the run completes.
- An incomplete set of shards does stop, in A2's `load_shards` (`analyse_stimulus_interventions.py:661-677`). Only the case where all cell files or all report files are missing passes silently.

**Scenario.** s11 and s11b. The `run_qwen3vl_shard{0,1,2}.json` files are not pulled (or the cell files are not). Then:
- qwen3vl reads Not tested on both routes.
- gemma4's two step-down families shrink from 4 columns to 2.
- Removing columns from a Westfall-Young step-down can only lower or keep the remaining adjusted p-values. So gemma4's E* p-values move in the liberal direction, and its balance is withheld.

So one pull slip turns one model's confirmatory row into Not tested, and can move the other model's evidence class, in the single run. No A2 pull runbook exists yet (`research/scripts/` has `post_a1.sh` only).

**Handling.** Dated pre-data note, no code change:
- Before the single run, confirm that both `cells_<model>_shard<k>.jsonl` and `run_<model>_shard<k>.json` exist for every k below `n_shards`, for gemma4 and for qwen3vl.
- State now that a `no_a2_cells` row for a model whose A2 jobs completed is an input error that voids the run. It is not a result.

### A5-2 (C): a missing or altered A3 summary makes every cell Not tested without stopping

**Text.**
- Line 293: "The gate loader must return True on a fixture built from the real pinned paths, with a test that runs the verdict function end to end. This guards against A2's second audit finding, in which the A3 path made every verdict Inconclusive."
- Line 413, quoted under A5-1, lists three files whose absence stops the run. The A3 summary is not one of them.

**Code.**
- `require_gate_files` (`a5_analyse.py:1046-1052`) stops only for `rotation_summary.json`, `vision_gates_<model>.json` and `input_delivery_<model>.json`.
- The A3 summary is read by A2's `a3_summary` (`analyse_stimulus_interventions.py:594-603`). It returns `{}` unless a file with the pinned bytes is in `results/face_input_validity/` or `--reproduction`.
- `rq0_status` (`:579-583`) then gives False on both routes.
- `check_a3_pin` (`a5_analyse.py:1039-1043`) compares two constants. It never looks at the file.
- The same silent result follows if `rotation_summary.json` exists but lacks the model's key: `a1` is None, so `rq0` is False.

**Scenario.** s8. Run with a `--reproduction` folder that lacks `face_input_validity_summary.json`, or with a copy whose bytes changed. All four cells read Not tested, the run completes, and `gates.a3` is null.

Today the defaults are safe:
- The committed copy matches the pin (`c1c64e69...`) and gives `vision_dependent: true` for gemma4 and qwen3vl.
- `rotation_summary.json` has both keys, also True.
- The default `--reproduction` (`a5_analyse.py:1130`) points at these files.

**Handling.** No code change; the code matches line 413 as written. Add a dated pre-data note: run with the default `--reproduction`. A1 and A3 are already recorded as Pass for both models (`RESULTS_REGISTRY.md:271`: "A1 Pass and A3 Pass on all three models"; the committed files give True). So treat any output row whose `gates.a3` or `gates.a1` is not True as an input error that voids the run, not as a result.

### A5-3 (C): a route with no `original` cell that ran crashes the whole analysis

**Text.**
- Line 270: "| B1 or B2 fails | **Not tested**. The readout is not measured on this route: deterministic scoring or coverage failed |"
- Line 154: "A failed `original` cell is absent from every mean here and is counted."

**Code.**
- `a5_analyse.py:719` calls A2's `gate_b` for every route of every loaded model.
- `analyse_stimulus_interventions.py:334` (`fams = sorted(next(iter(originals.values()))["blind"]["family_logprob"])`) raises `StopIteration` when no `original` cell on the route has status ok.
- A2's `load_shards` builds its expected cells from the routes present in the rows (`:675`). Shards written without a route therefore pass it.

**Scenario.** s3 and s3b: qwen3vl shards written without `video_face`, or with every `video_face` cell failed. Both stop at `:334` with no output for either model.

This is unlikely. The Slurm default `INPUTS` lists all four routes (`study3_stimulus_interventions.sh:37`), and failures are recorded per clip (`run_stimulus_interventions.py:1182-1193`).

**Handling.** No code change, since the function is A2's frozen code. A dated pre-data note fixes the outcome now:
- A model and route with no ok `original` cell is Not tested, because B2 has no input (line 270).
- Before the single run, count ok `original` cells per model and route from the status fields alone. If any count is zero, log a dated deviation before running.

### A5-4 (R): the `descriptive` block has no word share

**Text.**
- `results.md:279`: "[PENDING: `descriptive` block, balance, Phi with 90% CI, word share, `family_gain_log_sd`]".
- Draft line 576: "the word-knowledge result is stated as Absent evidence, with the balance and word share beside it as description."

**Code.**
- `a5_analyse.py:70-75` builds `descriptive` from `verdict`, `balance`, `balance_withheld`, `balance_reasons`, `B`, `Phi`, `Phi_ci90`, `family_gain_log_sd`, `confirmatory: False` and `note`.
- Word share is only at the row's top level (`:755`). `claim_scope` (`:1119-1122`) does not list it either.

**Scenario.** s1: every row's `descriptive` has exactly the keys above, and `word_share` (with `reading: "word_prior"`) sits beside it at row level.

**Handling.** No action on frozen files. Fill the slot from the row's top-level `word_share`, and say so in `results.md`, which is not frozen.

### A5-5 (R): a `no_structure` cell has an empty `confirmatory` block with no reason

**Text.**
- Line 288: "`no_structure` (mean T under the floor). `no_structure` applies to every row, and is checked after the Not tested conditions and before the classes."
- Line 576: "Shown, Unmapped, Absent and Unresolved (with `evidence_unresolved`) ... given in the `confirmatory` block."

**Code.**
- `a5_analyse.py:586-587` returns `evidence: None` with reason `no_structure`.
- `label_claims` (`:67-69`) then writes `{"evidence": null, "evidence_unresolved": false, "not_tested": false}`.
- `no_structure` stays only in the row's top-level `reasons`, and `descriptive.balance_reasons` keeps only `balance_*` (`:72`).
- The flags that qualify an evidence class (`no_baseline`, `b3_failed`, `readout_dependence`; lines 271-273) are also outside the block, in the row's `flags`.

**Scenario.** s9: T = .0004, below `min_total_nats` (.05). All four `confirmatory` blocks read evidence null and not_tested false. A direct `decide` call with T = .04 gives the same.

**Handling.** No code change. Fix the reading rule in a dated note or in `results.md`:
- A null `confirmatory.evidence` with `not_tested: false` takes its reason from the row's `reasons`.
- Each evidence class is reported with the row's `flags`.

### A5-6 (R): the chain is reported without intervals

**Text.**
- Line 196: "**Riders, always reported, with actor-bootstrap 95% CI.**"
- Line 201: "| Chain | Means of W, G, F, E, R, and lexicon mass on real, grey and image-free | ..."

**Code.** `chain` (`a5_analyse.py:647-660`) returns means of actor rows and masses, with no interval. Word share (`:628-644`) and survival (`:663-681`) do carry 95% intervals.

**Scenario.** s1: `chain` is `{W, E, R, G, F, G_F_merged, mass_real_median, mass_grey, mass_image_free}`, all point values.

**Handling.** Dated deviation on reporting: the chain is reported as point means. Any interval added later is labelled as a post-freeze description. No class uses the chain.

### A5-7 (R): lexicon mass is not reported beside every quantity, and disappears when the baseline check fails

**Text.** Line 154: "Lexicon mass is reported beside every quantity."

**Code.**
- Real-clip mass appears only as `chain.mass_real_median` (`a5_analyse.py:658`), and `chain` is null when the baseline check fails (`:649-650`).
- `gate_b` computes `B2_median_lexicon_mass` (`analyse_stimulus_interventions.py:382`). But A5 copies only `B1` to `B5`, `pass`, `B3_rms_nats`, `B4_mean` and `B5_mean` (`a5_analyse.py:752`).
- Sensitivity rows carry no mass.

**Scenario.** s4 and s5: the gemma4 cells with `no_baseline` have no lexicon mass anywhere in the A5 output.

**Handling.** Disclose as a reporting gap. Report the median mass from A2's own output, which keeps the full `gate_b`, including `B2_median_lexicon_mass` (`analyse_stimulus_interventions.py:746`).

### A5-8 (N): the descriptive fields also stay at the top level of each row

**Text.** Line 574: "The output labels them `balance_confirmatory: false` and `verdict_confirmatory: false`, puts them in a `descriptive` block, and states that they are not error-controlled under skewed portrayal gain."

**Code.**
- `label_claims` (`a5_analyse.py:61-76`) adds the labels and copies the fields into `descriptive`. It does not remove the originals at row level (`:602-603`, `:736-761`). `claim_scope` (`:1119-1122`) names those fields as descriptive.
- No field named `confirmatory` holds Face, Default or Even. In s1's full output, balance class and verdict names occur only in `cells[i].verdict`, `cells[i].balance` and `cells[i].descriptive`, and never in `confirmatory` or the sensitivity rows.

**Scenario.** s1, `gemma4|video_face`: the row-level `verdict` is "Inconclusive" (balance unresolved), next to `confirmatory.evidence: "Shown"`. A reader who takes `verdict` as the headline would report Inconclusive where the confirmatory claim is Shown.

**Handling.** No action. `results.md:279` already reads the `confirmatory` block.

### A5-9 (N): some inputs are read with no hash check at run time

**Text.**
- Line 633 lists `data/eu_emotions_human_entropy.json` `f9d91b4e...` as a frozen input.
- Line 635: "The code hashes in "Code hashes" are checked by `a5_analyse.py` and `a5_recovery.py` before they run."

**Code.** `main` checks these before reading any cell:
- the design block, at import (`a5_design.py:46-53`);
- A5 and helper code (`a5_design.py:73-77`);
- A2 code (`:80-85`);
- the A3 pin constant, A2's design hash and the geometry hash (`a5_analyse.py:1137-1142`);
- the design hash carried by every row and report, through `load_shards`.

It reads these with no hash check:
- the human lookup (`:1144`);
- `rotation_summary.json`, `vision_gates_*` and `input_delivery_*` (`analyse_stimulus_interventions.py:606-614`).

The code pins themselves are parsed from the draft's "Code hashes" section (`a5_design.py:56-70`), which lies outside the hashed block. So the run-time check binds only together with the tag.

**Scenario.** None observed. Today the lookup equals `f9d91b4e...`, the gate files equal their tag versions, and `rotation_summary.json` matches the RESULTS_REGISTRY prefix.

**Handling.** No action. Record these files' sha256 in the run log, and run from a checkout whose pinned files equal `a5-freeze`. That holds today.

### A5-10 (N): no test loads the committed gate files' content

**Text.**
- Line 293, quoted under A5-2.
- Line 662: "| Paths that made every verdict Inconclusive | End-to-end test on the real pinned paths |".

**Code.**
- `tests/test_a5.py:589-598` writes synthetic files under the real names into `tmp_path` and monkeypatches `an.A3_SUMMARY_SHA256`.
- The tests at `:601-606` and `:651-669` run on those files.
- `:115-117` checks only the committed A3 summary's hash.

**Scenario.** Nothing fails. Loading the committed files directly:
- `rotation_summary.json` and the A3 summary give `vision_dependent: true` for gemma4 and qwen3vl.
- `vision_gates_*` have `smoke: false` and `pass: true` for G2, G3 and G4.
- `input_delivery_*` pass on `single_face` and `video_face`.
- s1 ran `main()` on these files and got `rq0`, `g2_g4` and `delivery` True on all 4 cells.

**Handling.** No action.

### A5-11 (N): Absent can be returned while E*'s 95% interval lies above 0

**Text.** Line 230: "| Absent | E* not significant, its 90% CI inside the margin, B4 failed, and the positive control holds |"

**Code.** `evidence_class` (`a5_analyse.py:530-536`) reads "not significant" as a step-down `p_E >= .05`. The 95% percentile interval is not consulted for Absent.

**Scenario.** A direct `decide` call with step-down `p_E` .08, E* 95% interval [.01, .05], 90% interval inside ±.10 T and B4 failed returns `evidence: "Absent"`. In a 4-column family the step-down p can exceed .05 where a single test would reject. Under the text as written, Absent then means evidence below the margin, not no evidence.

**Handling.** No action. Report Absent with E*'s interval, and write "below the margin" where the 95% interval excludes 0.

### A5-12 (N): the survival conflict flag depends on the descriptive verdict

**Text.** Line 202: "S at or above `survival_cut` beside an Evidence-led verdict is reported as a conflict".

**Code.** `a5_analyse.py:734-735` adds `survival_conflict` only when `verdict == "Evidence-led"`. That verdict needs a named Face balance in a family of at least 4 cells, and since run 6 it is descriptive.

**Scenario.** A cell has evidence Shown, B above 0 and S at or above .5, but its balance is unresolved or withheld. It is withheld whenever the family has 2 or 3 cells, the sizes reached in s6, s7, s10 and s11. No flag is raised, although S is in `survival`. This follows from the code path; no scenario here produced S at or above .5.

**Handling.** No action. Report S and its interval, which every row carries, beside every Shown cell.

### A5-13 (N): small differences between the text and the output

These are matched text and code:

| Text | Code |
|---|---|
| Line 154, "...is counted" | Failed `original` cells are counted only implicitly, through `n_clips` (the clips that ran, `a5_analyse.py:740`; s2 gives 233 and 228). There is no per-cell failure count or reason |
| Line 269, "This covers Molmo2" | There is no Molmo2 row. `MODELS` comes from the block, `["gemma4", "qwen3vl"]` (`a5_analyse.py:34`) |
| Line 318, "If no clip remains, the row is empty and says so" | The row is `null`, with no message (`a5_analyse.py:950-951`, `:966`) |
| Line 627, "The coordinator reran tests/test_a5.py (108 passed)", with no interpreter named | The analysis env cannot run the tests, because it has no pytest (see Tests). A cache file, `tests/__pycache__/test_a5.cpython-310-pytest-9.2.0.pyc` (2026-11-08 13:50), indicates that the freeze run used CPython 3.10 with pytest 9.2.0, which is neither interpreter available here |

**Handling.** No action, or one sentence in the results text.

## Checked and consistent

**Inputs.**
- Every key A5 reads is written by the runner under the same name:
  - rows: `trial_id`, `actor`, `label`, `human_modal_label`, `weak_label`, `input_name`, `condition`, `status`, `design_sha256`;
  - `blind.family_logprob`: 21 families, from the logsumexp over the 106 words;
  - `blind.word_logprob`: in `lexicon_words()` order, as `canonical_rows` assumes;
  - `blind.lexicon_mass`;
  - reports: `model`, `smoke`, `design_sha256`, `shard`, `n_shards`, the `baselines` rows, and the `delivery` rows per route.
- The names `cells_<model>_shard<k>.jsonl` and `run_<model>_shard<k>.json` match `shard_paths` (`a5_analyse.py:1028-1032`). Smoke files do not.
- Real metadata: 243 items, 17 actors and 21 labels, all mapped by `OPTION_TO_FAMILY`. Every label has at least 7 actors, so every clip has a template. 26 items carry weak labels, and every human option name is mapped.

**Shards and failed cells.** Mixed or incomplete shard sets stop in `load_shards`: wrong design hash, smoke runs, an incomplete shard set, duplicates, an item count other than 243, or missing cells. Failed `original` cells are left out of every mean (s2).

**Gate B.** A5's Gate B equals A2's call:
- same function;
- partners from `vision_mismatch_partners` with seed 42, over the same sorted ids;
- the first shard's `image_free` row, whatever its status;
- the repeat from delivery rows (`a5_analyse.py:1062-1064` equals `analyse_stimulus_interventions.py:727-728`).
- On s1's shards, Gate B built the way A2's `main` builds it (`analyse_stimulus_interventions.py:702-729`) and Gate B from A5's `build_cell` serialise identically, on all four cells. The partner maps are identical too.

**Baselines.**
- Agreement across shards is checked at .001 nats, and the first shard's rows are used.
- A missing, failed or disagreeing row gives `no_baseline` and a null word share and chain.
- B3 with no input fails (s4, s5; line 271).

**Order of tests** (`a5_analyse.py:577-603`), as line 288 requires:
1. RQ0, G2 to G4, delivery, B1 and B2 decide Not tested.
2. Then `no_structure`.
3. Then the classes.

**Identity.** `decompose` (`a5_analyse.py:236-268`) computes T, P, D, E and R, and W, G and F when the baselines hold, exactly as lines 164-176 write them:
- q comes from the softmax of `family_logprob` over the 21 families (`:111-119`; line 154).
- q_def is the mean of actor means, leaving the clip's actor out (`:200-207`; line 158).
- q_tpl is the clip mean over other actors with the same acted label (`:210-222`; line 160).

**Evidence classes, the confirmatory part** (`a5_analyse.py:530-536`):
- **Shown:** step-down `p_E` < .05, the 95% percentile interval's lower end above 0, and Gate B `pass` (B1 to B5).
- **Unmapped:** the same E* test, with Gate B not passed. Only B3 to B5 can fail at this point.
- **Absent:** `p_E` at least .05; the 90% interval inside ±`margin_fraction` (.10) times the cell's mean T over actors; B4 not True; and the positive control (rq0 and g2_g4, which includes G4).
- **Unresolved** otherwise, with `evidence_unresolved`.
- `min_total_nats` is .05.
- These match lines 220, 226-233 and 286-288.

**E\*.**
- Per actor, E minus its mean over 1,000 label shuffles within actor, seed 42 (`:287-321`, `:336-356`).
- Tested one-sided by A2's exact sign-flip maxT over 2^17 flips (`stimulus_interventions.py:590-592`).
- Interval: percentile actor bootstrap, 10,000 draws, seed 42.

**Families.**
- One E* family and one B family over the eligible cells of both models and both routes (at most 4), each corrected only within itself (`a5_analyse.py:472-491`, `:764-769`).
- H1 has its own two-sided family (`:800-816`).
- Each sensitivity is one column with no correction (`:910-921`).

**B (descriptive).**
- Two-sided. Its p is the larger of two maxT tests:
  - a bootstrap-t (10,000 draws, seed 42, actor rows resampled, t centred on the observed mean);
  - the sign-flip.
- Its 95% and 90% intervals are bootstrap-t.
- Even requires all of:
  - p at least .05;
  - a 95% interval that includes 0;
  - both one-sided 95% bootstrap-t lower bounds, of 2P minus D and of 2D minus P, above 0 (`:494-508`, `:539-550`).

**`min_balance_cells` = 4.**
- Below 4 cells, a named class is withheld: balance reads Unresolved, with reason `balance_small_family`, and the class is kept in `balance_withheld`. The evidence class is still named (`:592-600`; s6, s7, s10, s11).
- Below 4 cells, a class that is already Unresolved keeps `balance_unresolved`, as line 288 defines.

**No guard.**
- `family_gain_log_sd` (the mean of the estimable cell estimates over eligible cells, `:463-469`, `:768`) is reported, and so is each cell's `gain_heterogeneity`.
- Neither enters a decision, and `decide` takes no gain argument.
- The 0.4 cut exists only in the diagnostic (`a5_gain_guard_diagnostic.py:25`).

**Labels.**
- The `confirmatory` block holds `evidence`, `evidence_unresolved` and `not_tested`.
- The `descriptive` block holds the verdict, balance, `balance_withheld`, `balance_reasons`, B, Phi, `Phi_ci90`, `family_gain_log_sd`, `confirmatory: false`, and a note with ".094 against .07".
- Each row carries `evidence_confirmatory: true`, `balance_confirmatory: false` and `verdict_confirmatory: false`.
- Of the `results.md` slots, the evidence class, balance, Phi with its 90% interval and `family_gain_log_sd` are present. Word share is missing (A5-4).

**Phi.** D over T over actors, with a 90% percentile actor bootstrap of the ratio of means (10,000 draws, seed 42) on every row with cells (`:511-515`). This matches line 217.

**Riders.**
- Word share (`:628-644`) follows line 200: P not above 0, or more than 2.5% of draws invalid, gives mixed; invalid draws go in the upper tail; the cut is .5.
- Survival (`:663-681`) follows line 202: partner seed 42; floors of 10 items and 5 actors; actor means, then the ratio.

**Sensitivities.** All the ones on lines 315-323 are present. Canonical-word scoring uses each family's first word, in the order the runner writes. The token floor is A2's 8.0, which equals the block's.

**Human comparison.**
- H1: default and template deltas, with a two-sided step-down over eligible cells.
- H2: within-label r with an actor bootstrap.
- Repeats without the 26 flagged items and against the modal label, and the leakage table.
- H3 is reported as "not tested". A6 is now closed as a limit, so H3 stays not tested, which is consistent with line 305.

**Missing gate files.** A missing `rotation_summary.json`, `vision_gates_<model>.json` or `input_delivery_<model>.json` stops the run (`:1046-1052`; tested at `tests/test_a5.py:616-622`).

**Delivery.** A2's rule applies (`analyse_stimulus_interventions.py:621-636`). One `{"failed": ...}` delivery row for a route, among the first 2 items of any shard, makes that route Not tested (s10, line 269).

**Rows without a verdict.** `check_row` refuses a row that has no verdict or no valid reason code (line 295).

## Not checked

- **Real A2 cells.** None exist, so real bf16 agreement of baseline rows across shards (.001 nats) is untested.
- **`main()` with the real `geometry.json`.** It is not local (pinned `2bc76dd6...`). s1 bypassed `run.check_geometry` only, so its token-floor sensitivity ran with no geometry: every clip counted as below the floor, and the row was null.
- **Recovery.** It was not rerun.
