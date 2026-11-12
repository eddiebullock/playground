# A4 code-against-text audit (pre-data)

**Date:** 2026-11-12. Independent audit of the frozen A4 analysis against its frozen text, requested by the coordinator, before any A4 model output exists or is read.

**Model outputs.** None was read. There is no `results/a4/` directory on this machine. A filesystem-wide search by file name for `cells_gemma4*`, `cells_qwen3vl*` and `cells_molmo2*` found only A3's `results/face_input_validity/cells_{gemma4,qwen3vl,molmo2}.jsonl`, which were not opened.

**What was written.** No repository file other than this report was created or edited. Every synthetic check wrote only to `/private/tmp/claude-502/-Users-eb2007/cf56ec16-1588-406d-9db5-6872ba6bedc3/scratchpad/audit_a4/` and ran with `PYTHONDONTWRITEBYTECODE=1`. The tests ran with `-p no:cacheprovider`.

**Naming.** The code and frozen files say CFA, and this report keeps their names when it quotes them. That is deliberate (DECISIONS 2026-11-09) and not a finding.

## Header

### Files read, sha256

All hashes are of the working tree at the end of the audit. "Pinned" means listed in `PREREG_A4_DRAFT.md` lines 7 and 9.

| File | sha256 | Status |
|---|---|---|
| `research/PREREG_A4_DRAFT.md` | `5b5eb82160aa7a8be97234b35a7fa705631e7692b945c6ad97f7bd4df92e6867` | unchanged since `a4-freeze` |
| `scripts/a4_analyse.py` | `30f8f529d128b555ccfbf4d31ac3de70d91f90b1f5bfd52360ae9f39305fe56d` | pinned, matches |
| `scripts/a4_design.py` | `a0bf7bb1c69ec7af1f11fabb5a9f31a6c694a30b92e624f1e1c241f161d062a8` | pinned, matches |
| `scripts/a4_run.py` | `f842e574bf5cdaebbe9057d0ecc0e311c900dd54ff0ca221d1f9baa3768a8dc6` | pinned, matches |
| `scripts/a4_recovery.py` | `12b15e89a366d772078034fd56b105532b347f1b73afe0b4efd5548ad3302526` | pinned, matches |
| `research/scripts/human_au_effects.py` | `c02a945ba578156cc4394e7d3a49761ad6e74495585034fbb059a4ce40199a2b` | pinned, matches |
| `research/scripts/a4_stimuli.py` | `d0dff6930ede7f908e8d7faee2bc647adeb02f366601d04bea47353833715875` | pinned, matches |
| `research/scripts/a4_detector.py` | `3fbd35dee68615a6fe1165df57a4d290c8097e8bf2a5ac714ee73ba15f302e42` | pinned, matches |
| `research/scripts/analyse_rotation.py` | `98890c5bff65f13abeb0549b61fe414dc9fa30082b84bcb564671683ec12e24f` | pinned, matches |
| `scripts/analyse_stimulus_interventions.py` (`rq0_status`, `outside_gates`) | `aba4f533daad6210e87f7bb4e68fc4627805cb4dfd9d9308df734fada4edc8b1` | pinned, matches |
| `scripts/face_input_validity.py` (`read_cells`) | `8a681a652d8e34065372d4f45bce650ba5276b5138e3aa64cf4310d65e93d818` | pinned, matches |
| `scripts/run_stimulus_interventions.py` (scoring path, lexicon) | `f7880c94e9671cc294cd4eaeece2108d7cd1d7a4c7e6f9e30a1c495964e6cd82` | pinned, matches |
| `scripts/vision_gates.py` (`strip_pixels`) | `6327b6f88e99409cba5c63d5aca0fe97348fbce4f3b08522158a8d9ba3ce8194` | pinned, matches |
| `scripts/prompts.py` | `871272d4aa6c86ff809f7631a4db81aeeb41564ba7f0a31809ea08e6dcf64ba7` | pinned, matches |
| `tests/test_a4.py` | `ece6642c4af28e599bf9bce749ca658961ed16568dede1812a67a4fb77e26fe6` | unchanged since `a4-freeze` |
| `slurm_jobs/study3_a4.sh` | `dd9a360b20701742fd70d849b92c964d6b992c966ded744fa4125e0166103c26` | pinned, matches |
| `slurm_jobs/study3_a4_render.sh` | `1988b8d579cf181ef83e8d60c85a595f42fe51740dc73bcaab343be0db463a7b` | pinned, matches |
| `slurm_jobs/study3_a4_detector.sh` | `8f4b03e93888b17d11900ce2186a88f1063175a25907bd1f6b326ea938e453a8` | pinned, matches |
| `research/reproduction/human_au_effects.json` | `58c0f082cbf6aa4827054188ac2fc47da58a254c38da2c7ac202e8f52399310c` | pinned, matches |
| `research/reproduction/a4_human_ceiling.json` | `21fe4f8441532992ecb22b60fba4afc59c38e70aed5407b2d0e8ab48b76cd2eb` | pinned, matches |
| `research/reproduction/a4_v1_pilot.csv` | `c735615a034c281909c96894c700d224638e281f55f995a0959eeef8a6fb3994` | pinned, matches |
| `research/reproduction/a4_recovery.json` | `1ebfc8ea00da4261e3a05e6de09b3fbe7bd259ce0a5d840fe4c7ab03894bedca` | pinned, matches |
| `research/paper/v2/supplementary_note1.md` (sections 1, 2, 4, 5, 7) | `e8fef5d82a3ae87c360a801c4f9cf7b8cab0cd6782e26c2618ba7e576a140fb5` | changed during the audit by commit `9572a18`, which added Appendix A and a numerical-check bullet at line 162 (see A4-13). The other lines of sections 2 and 4 are unchanged |
| `research/paper/v2/results.md` (sections 5, 6) | `33275c08d8bd9e3d7744d69de5da50f1d7b7f995d8afe1411533325ea263a9ce` | changed during the audit; no change to A4 text since `2611f51` |
| `research/paper/v2/methods.md` (A4 parts) | `53b28dc3b3560fb6def4e8a5ec56a37f2b6bef72d5752124f798a90526203faf` | uncommitted edit by another session; no change to A4 text since `2611f51` |
| `research/paper/v2/supplementary_methods.md` (S6) | `27e36a98d191e9028830c68876a82018083bd9360d16b2973d7d35a91f544ddc` | committed |
| `research/DECISIONS.md` (A4 and CFA/CEA entries) | `2c28558fc4aa5cad3548936db170aaa20ba18e353a0dfd440f8b701d34f7fdd5` | changed during the audit (A2 and A5 rulings added); no new A4 entry |
| `research/RESULTS_REGISTRY.md` (A4 rows, "Paper text against these files") | `964f65d31fef70bc248923fe52c051dd7513940e891ca78410f43dad29ee15b5` | changed during the audit (a G9 attenuation row added) |
| `research/PREREG_A2_DRAFT.md` (Deviations section only) | `f50b7159814763e599eb307bb1e62300ae81fee6650748f69c3f2a54d8e8b4de` | changed during the audit (`06c8465`); the Deviations heading cited is still at line 330 |
| `research/reviews/a9_audit.md` (points on A4) | `67b16ff89bd3331dc0fce0d76df4d01d4b924338c1137d511a0186f0c54abf48` | committed |

`render_au_faces.py`, `extract_au_measures.py`, `slurm_jobs/study3_a4_render.sh` and `study3_a4_detector.sh` were checked by hash. The renderer was read only for its column mapping and input checks.

### Tag comparison and reproduction

- **Pins.** All 22 files pinned in the amendment (18 code, 4 data) match their stated sha256 in the working tree and at tag `a4-freeze` (`dd1490e`, commit `568b6f5`). `git diff a4-freeze` is empty for `PREREG_A4_DRAFT.md` and `tests/test_a4.py`.
- **Unpinned code.** Since the tag, no existing file under `scripts/`, `research/scripts/`, `slurm_jobs/`, `tests/` or `src/` has been modified or deleted (`git diff --diff-filter=MDR a4-freeze` is empty); only new files were added. So the unpinned modules the analysis imports are also as at the tag: `scripts/stimulus_interventions.py`, `scripts/stimulus_design.py`, `src/paper_v2/schema.py` and `scripts/__init__.py`.
- **Design hash.** `a4_design.design_hash()` recomputes to `12c454da512bfe4c239e6c3ef3610d80df6fd7c2e9119a9ff8df913a3c1a3e94`, the frozen value.
- **Manifest.** Rebuilt from the Snoek zip (md5 `1c8938c5...`, as recorded in `human_au_effects.json`) by the frozen `a4_stimuli.py manifest` into scratch: 15,225 rows, 11,761 unique images, sha256 `2ef1161e895224d499b360144690a07c5440bb018105e9c2ffc762f68badd2d1`, equal to the pin.
- **Human ceiling.** `write_ceiling` rerun into scratch is byte-identical to `a4_human_ceiling.json` (sha256 `21fe4f84...`).
- **V1.** The frozen `detector_gate` on the pilot CSV gives .7847 (113 of 144), pass.
- **Recovery.** `a4_recovery.json` holds the frozen design hash, `confirmatory_statistic: "residual"`, residual 66 of 66 checks and raw 62 of 66.
- **Tests.** `/opt/anaconda3/bin/python3 -m pytest tests/test_a4.py -q`: 60 passed (Python 3.13.5, numpy 2.1.3). The coordinator noted that pytest is absent from the analysis env. Synthetic checks used the analysis env: numpy 1.26.4, pandas 2.3.3, scipy 1.15.2.
  - What the tests cover: the decision table, Holm over three models with Molmo2 descriptive, smoke-file exclusion, refusal without the detector, the frozen flag or the right hash, rotation completeness, V3 grouping and the recovery verdict rule.
  - What they do not cover: a family model with no cells (A4-1), non-finite responses (A4-2), shard completeness, partial lines or duplicates (A4-3 to A4-5), and the Molmo2 headline (A4-10).

## Findings

Severity:

- **V**: could change a verdict.
- **C**: would crash, or silently drop or misread data.
- **R**: reporting or text mismatch only.
- **N**: note.

Handling options are a dated pre-data deviation, a disclosed limit, a run precondition recorded with a date before the single run, or no action. No finding asks for a threshold to move.

Line numbers are those of `PREREG_A4_DRAFT.md` ("text") and of `scripts/a4_analyse.py` unless another file is named. "Probe" means a call to the frozen function with synthetic input, scripts in the scratch folder named at the end.

### A4-1 (V). The Holm family shrinks to one model when the other has no shard files

**Text.**
- Line 17: "Gemma 4 and Qwen3-VL are the Holm family."
- Line 113: "Holm across the two eligible models, within a dataset."
- Line 141: "missing means not passed".

**Code.**
- `:637-638` keeps only models that have shard files (`if shard_files(args.cells_dir, m)`).
- `:442-443` runs Holm over `[m for m in d.MODELS if m in results]`.
- So a family model with no files is left out of Holm and is absent from `summary.json`.

**Scenario.** The probe calls the frozen `apply_rule` with Gemma's residual p_cue at .03, CFA .40, CI (.32, .48), signal p .001 and all gates passed.
- With Qwen present at p_cue .20, Gemma's p_holm is .06 and the decision is Inconclusive.
- With no Qwen file in `--cells_dir`, Gemma's p_holm is .03 and the decision is Aligned.
- Qwen is then not reported at all, rather than reported as Inconclusive.

The real `main()` path does the same; see "Synthetic end-to-end runs" below.

**Prior record.** The A9 audit flagged this as "inherited from A4" (`research/reviews/a9_audit.md:272-274`). No A4 record carries it.

**Handling.** A dated pre-data deviation stating:
- The family is the two named models, whatever cells exist.
- A family model with no cells is reported Inconclusive and counts in Holm with p = 1.

The code is frozen, so enforce this as a run precondition: the single run starts only when all 12 shard files of both `gemma4` and `qwen3vl` are present and complete (A4-3). With both present, code and text agree.

### A4-2 (V). A non-finite response counts as usable and gets the smallest possible p-value

**Text.** Line 84: "A stimulus without a usable response on every rotation leaves the analysis". "Usable" is not defined further.

**Code.**
- `stimulus_responses` (`:57-63`) keeps any row with `status == "ok"`.
- The runner always writes `status: "ok"` (`scripts/a4_run.py:108`).
- Nothing on the path checks finiteness. There is no `isfinite` or `isnan` in `a4_run.py` or `run_stimulus_interventions.py`. In `a4_analyse.py` such checks apply only to the bootstrap and split-half vectors (`:136, 149`).
- A NaN joint log probability gives a NaN `softmax`, so the fit and `tau` are NaN.
- `cue_null_p` (`:107-113`) then tests `null >= NaN`, which is never true. It returns p = 1/(n+1) = .0001, the smallest p it can give.
- `signal_test` (`:123-124`) likewise returns .001.

**Scenario.** In the probes, one NaN entry in a model matrix gives cosine NaN and `cue_null_p` .0001. One NaN response row gives signal p .001.

Suppose this happens to Qwen on one dataset:
- Qwen itself ends Inconclusive. Its V4 mean is NaN, so `v4_domain` is False.
- Its p_cue of .0001 still enters Holm, so Gemma's adjusted p falls to its raw p.
- In the probe, Gemma at p_cue .03 moves from Inconclusive (.06) to Aligned (.03).

**Handling.** A dated pre-data deviation:
- Define a usable response as one whose joint log probabilities are all finite.
- Apply it as a mechanics check before the single run: a count of non-finite values per model and shard, not a reading of values.
- If any exist, a rule fixed now decides. The suggestion: remove those rows before the run, so their stimuli leave the analysis as line 84 intends and count toward the 5 percent limit.

Whether any real cell is non-finite was not checked, because that needs A4 output.

### A4-3 (C). Shard completeness is not checked; a missing or unfinished shard silently becomes lost stimuli

**Text.**
- Line 264: "Twelve shards per model at about 5 h, against a 6 h limit, resumable".
- Line 272: "5. Analyse once, after all shards finish."

**Code.**
- `shard_files` (`:580-583`) takes whatever `cells_<model>_<digits>.jsonl` files exist.
- Nothing checks the number of shards, or that each holds every planned cell (`planned_cells`, `scripts/a4_run.py:48-77`).
- Missing cells show only as `share_lost` (`:419-428`).

**Scenario.** Synthetic cells were built by the frozen `planned_cells` on the rebuilt manifest. With shard 7 of 12 absent:
- share lost is .0874 on basic WC, .0852 on basic EA, .0850 on conversational WC and .0826 on conversational EA;
- so `clips_kept` is False and that model is Inconclusive on all four datasets;
- no error is raised, and the single run is spent on an operational fault.

Two related cases:
- Empty shard files stop the run only when every file of that model is empty (`read_shards`, `:574-576`). One empty file among others is silent.
- A missing shard that holds the first neutral face crashes the run instead (A4-6).

**Handling.** A run precondition, recorded with a date before the single run. For each model there must be 12 files `cells_<model>_0.jsonl` to `cells_<model>_11.jsonl`, and each must:
- hold exactly the cells that `planned_cells(manifest, k, 12)` lists;
- have unique cell keys;
- have status ok throughout;
- carry the frozen design hash.

This is a mechanics count; no response value is read.

### A4-4 (C). A truncated final line crashes the analysis; the runner's own partial-line rule is not used

**Text.** Silent on partial lines. Line 264 says the run is "resumable".

**Code.**
- `read_shards` (`:569-577`) calls `json.loads` on every line, so the cut-off last line a killed job leaves raises `JSONDecodeError`.
- The runner uses `read_cells` (`scripts/face_input_validity.py:56-78`, called at `scripts/a4_run.py:139-142`), which drops one bad final line and truncates it on resume.

**Scenario.** In the probe, a shard ending in `{"design_sha256": "x", "a"` raises `JSONDecodeError` before any output. It fails loudly and writes nothing, so no result is read.

**Handling.** Covered by the A4-3 precondition (every shard resumed to completion). No code change.

### A4-5 (C). Duplicate cells are not detected; the last one read wins

**Code.**
- `stimulus_responses` (`:61`) overwrites `seen[stim_id][rotation]`.
- `image_rms` (`:473-483`) keys image-free rows by stimulus and averages over every matching rotation-0 row, so a duplicated row counts twice.
- Nothing checks that cell keys are unique across a model's files.

**Scenario.** In the probe, a second rotation-0 row for one stimulus, with reversed log probabilities, silently replaces the first. This happens if a shard is rerun under a different `N_SHARDS`, or a file is copied under two shard numbers.

**Handling.** Part of the A4-3 precondition (unique cell keys).

### A4-6 (C). Two crash paths on incomplete inputs

**Code.**
- `direct_effects` (`:222-228`) sizes its output from the first neutral face only. If that face lacks a complete rotation set, `size` is 0 and `:228` raises `ValueError`. The probe gives "could not broadcast input array from shape (2,) into shape (0,)".
- `analyse_dataset` (`:423`) calls `np.vstack` on the kept stimuli, which raises if none are kept.

**Scenario.** A missing shard that holds `female_a`'s neutral render, which manifest order puts first, stops the whole analysis.

**Handling.** Covered by A4-3. Both fail loudly before any output.

### A4-7 (C). Inputs are not checked against the frozen pins, and the summary records no input hashes

**Text.**
- Line 9 pins `human_au_effects.json`, `a4_human_ceiling.json`, `a4_v1_pilot.csv` and `a4_recovery.json` by sha256.
- Line 268: "`research/reproduction/a4_v1_pilot.csv`, is the file passed to `a4_analyse.py --detector`."
- Line 18: "The manifest is pinned, and checked at run time."

**Code.** What is checked:
- the manifest (`:626`; `a4_design.py:129-132`);
- the design hash embedded in the cells (`:574-576`) and in the recovery file (`:596`).

What is not checked:
- `--human`, `--ceiling`, `--detector`, `--gates` and `--zip` are read from whatever path is given (`:629-634`).
- `design_hash()` hashes the repository copy of `human_au_effects.json` at `ROOT` (`a4_design.py:118-126`), not the file passed as `--human`. The default paths are relative to the working directory.
- `a4_recovery.json` is accepted on its embedded design hash; its own sha256 is not compared with the pin.
- The output carries the design hash and the statistic but no input hashes (`:635-636`).
- The detector job's default output is `results/a4/detector_v1.csv` (`slurm_jobs/study3_a4_detector.sh:22`), not the file line 268 names.

**Scenario.** Running with `--detector results/a4/detector_v1.csv` (a later full V1 run), or from a working directory whose `research/reproduction/a4_human_ceiling.json` differs, gives a different V1 result or ratio. Nothing in `summary.json` would show it.

**Handling.** A run precondition, recorded before the run:
- the exact command line;
- the sha256 of every input (manifest, each shard, `--human`, `--ceiling`, `--recovery`, `--detector`, `--gates`, `--zip`), compared with line 9 where pinned;
- after the run, the output's sha256, which the script prints (`:665`).

The A9 audit asked the same for A9 (`research/reviews/a9_audit.md:376-377`).

### A4-8 (C). RQ0 and G2 to G4 come from a hand-made file, not from the frozen outputs

**Text.** Line 16: "RQ0 is read through the frozen A1 and A3 table by `rq0_status`, the rule A2 uses, at `single_face`. ... Any other outcome, or a missing result, leaves it Inconclusive."

**Code.**
- `rq0_gate` (`:586-590`) applies `rq0_status` to `a1_passed` and `a3_passed` taken from the `--gates` JSON (`:613, 632, 654-655`).
- `g2_g4` must be the JSON value `true`, tested with `is True`.
- Nothing ties these values to the frozen outputs. A2 reads them from the files (`scripts/analyse_stimulus_interventions.py:606-609`). A8 builds A4's gates file by script (`research/scripts/a8_a4_gates.py`). No script builds A4's own file.

**Scenario.**
- `"g2_g4": "true"` (a string) or a misspelt key makes that model Inconclusive on all four datasets, with no error.
- A wrong `true` is not caught either.

The source files exist and agree on what the file should say:
- A1 and A3 `vision_dependent` are true for all three models (`rotation_summary.json`; `face_input_validity_summary.json`, whose sha256 equals the A3 pin that A2 uses);
- G2 to G4 pass for Gemma 4 and Qwen3-VL and fail for Molmo2 (`vision_gates_*.json`, `verdict`).

**Handling.** A run precondition:
- build the gates file by a short script from those files, as `a8_a4_gates.py` does for A8;
- record its sha256 and contents in the dated run note before the run.

### A4-9 (R). The amendment carries no appended deviations; the known ceiling disclosure lives elsewhere

**Text.** Line 3: "Later changes are dated deviations, appended below." The file ends at line 272 with no Deviations section, and `git diff a4-freeze` on it is empty.

**Record.** The ceiling estimator point appears in three places, none of them the amendment:
- DECISIONS 2026-11-08 (`research/DECISIONS.md:383-385`);
- `supplementary_methods.md:78`;
- Supplementary Note 1, section 7, item 2.

A2's practice is a "## Deviations" section inside the frozen file (`research/PREREG_A2_DRAFT.md:330`).

**Is the known ceiling disclosure accurate and complete?** It is accurate.
- `:360` fits both halves with `pooled_do(..., 1.0)`.
- The target is the mean of per-subject fits at the selected ridge (`research/scripts/human_au_effects.py:585-590, 625`).

It is complete in substance. Two further differences in the same estimator could be named for completeness:
- the pooled fit weights stimuli by trial count;
- on basic WC it adds an unpenalised cohort covariate (`human_au_effects.py:346-350`).

How large is the difference? Rerunning the frozen procedure at the selected ridge of 0.1 (conversational datasets already use 1.0):

| Dataset | Raw ceiling, frozen | Raw ceiling, ridge 0.1 | Residual ceiling, frozen | Residual ceiling, ridge 0.1 |
|---|---|---|---|---|
| Basic WC | .9901 | .9900 | .9701 | .9704 |
| Basic EA | .9892 | .9891 | .9828 | .9828 |

The target estimator's own same-stimuli reliability, already stored in `human_au_effects.json` (`reliability.do_effect_family_25.overall`), gives ceilings of .994, .993, .981 and .980. So "approximate" is right, and the effect on the descriptive ratio is in the third decimal.

The amendment's table (lines 34-39) still puts "Ridge lambda" beside the ceiling column, which reads as though the ceilings used those values.

**Other known items, checked.**
- Exempt cells: pair-swapped within-limit shares of .85 on conversational WC and .62 on conversational EA, and 66 of 68 if counted, all match `a4_recovery.json`. `RECOVERY["pair_error_withdrawn"]` applies to both statistics (`scripts/a4_recovery.py:187`), as line 201 says.
- V1: 113 of 144, as above. The pilot has no ties, so a strict reading of "rising" also gives 113.
- The two RESULTS_REGISTRY wording points (`RESULTS_REGISTRY.md:247-248`) were not re-derived.

**Handling.** Append dated deviations to the amendment, following A2: the ceiling estimator, and whichever of A4-1, A4-2, A4-3, A4-7 and A4-8 are adopted.

### A4-10 (R). Molmo2's headline is written as Inconclusive, not Descriptive

**Text.** Line 17: "Molmo2 fails G2 and G4, so it is descriptive if run at all."

**Code.**
- Molmo2's per-dataset decisions are "Descriptive" (`:449`).
- `headline` (`:453-468`) maps anything other than Aligned and Not aligned to "Inconclusive".
- `:659-661` writes a headline for every model with cells.

**Scenario.**
- In the probe, `headline` on four "Descriptive" decisions returns `{'basic': 'Inconclusive', 'conversational': 'Inconclusive'}`.
- The synthetic full run writes the same for Molmo2.
- That run's Molmo2 `p_holm` is NaN on every dataset (`:447`). So `summary.json` holds four `NaN` tokens, which Python reads but a strict JSON parser rejects.

**Handling.** A reporting rule in the dated note: Molmo2's headline is reported as descriptive, and `models.molmo2.headline` is not quoted. STATE records scored A4 runs for Gemma 4 and Qwen3-VL only, so this may not arise.

### A4-11 (R). The ratio to the ceiling is about 1,000 times the CFA when R_S is zero or negative

**Text.**
- Line 101: the ratio is reported.
- Line 129: "A missing or non-positive R_S cannot support Not aligned."

**Code.** `:432` divides by `sqrt(max(r_s, 1e-6))`.

**Scenario.** With CFA .35 and ceiling .97:
- R_S of -.2 gives a ratio of 360.8;
- R_S of 0 gives the same;
- NaN gives NaN.

This is not hypothetical. In the synthetic full run, the prior-only model had R_S of -.078 on basic EA and -.352 on conversational EA (residual). `over_ceiling` came out as:
- 102.6 for residual and 179.0 for raw on basic EA;
- -18.1 for residual and 144.6 for raw on conversational EA.

A small positive R_S also inflates it: R_S of .031 gave -1.02.

**Handling.** A reporting rule: `over_ceiling` is reported as undefined when R_S is not positive.

### A4-12 (R). Three outputs differ from what the text and the results slots imply

**(a) V2's measured value is not written.** Only the boolean `v2_image_dependence` is written (`:295, 656`). The other gates' numbers do reach the output: V1's share in `detector`, V3's in `replicate`, V4's inputs in `P_main_mean`. The V2 RMS has no key.

**(b) The direct contrast is not reported "by dose".**
- Text, line 173, lists "the direct contrast, by dose" among sensitivities, "each on residual and raw CFA".
- The code gives raw and residual CFA at amplitude 1 only (`:650-653`).
- Dose is the mean absolute effect at each amplitude (`:229-230`), not CFA by dose.

**(c) Blind readout.**
- Text, line 160: "The blind result is reported only if these pass". It does not say raw or residual.
- The code writes `cfa_six_labels`, a raw cosine, whatever the gate (`:530-535`).
- Line 86 says "the human rows no longer sum to zero". In fact it is each cue's column over options that stops summing to zero once the "other" row is dropped.

**Handling.** Disclosed limits in the reporting note:
- V2's value is not in the output;
- the direct contrast's CFA is at amplitude 1, with dose read from magnitudes;
- the blind CFA is raw and is quoted only when `passed` is true.

### A4-13 (N). R_S is Spearman-Brown stepped up, which the frozen text does not say; and the bound it corrects comes from a bootstrap

**Text.**
- Line 124: "R_S is the model's split-half reliability across stimuli on that statistic".
- Line 101: "20 splits".

**Code.** `split_half` (`:140-150`) steps the mean half-against-half cosine up by Spearman-Brown (`:149`).
- Supplementary Note 1, section 4, states this; the amendment does not.
- The stepped value is larger, so Not aligned is easier to reach than under an unstepped reading.

There is a second effect. `ci[1]` comes from bootstrap refits, which add resampling noise to an already noisy estimate. Their cosines therefore centre below the point estimate, and dividing that bound by the square root of the full-sample R_S does not fully undo the attenuation.

Recovery ran this exact code. Its boundary check (Not aligned when the true CFA is at or above .30) stayed at or under .026 for residual CFA (`a4_recovery.json`), so the combined rule behaved there.

A synthetic check measured the size of this. It used the frozen `estimate` on recovery's human-like observer at 0.25x on conversational EA, with 200 draws and 20 splits, over three replicates:

| Replicate | True residual CFA | Point estimate | CI | CI midpoint | `ci[1] / sqrt(R_S)` |
|---|---|---|---|---|---|
| 1 | .864 | .846 | (.606, .835) | .720 | .966 |
| 2 | .906 | .754 | (.505, .787) | .646 | .897 |
| 3 | .754 | .538 | (.276, .701) | .488 | .857 |

- The interval's midpoint sat .10 to .13 below the point estimate.
- In replicate 1 the point estimate lay above the upper bound.
- The corrected bound stayed at or above the truth, or within .01 of it, so the rule was not liberal in these cases.

**What Supplementary Note 1 says.** During this audit, commit `9572a18` added a "Numerical check (2026-11-12)" bullet to Supplementary Note 1, section 4 (line 162), backed by `research/scripts/residual_cea_attenuation.py`.
- The bullet says that dividing by the model's reliability alone, "which is the form A4's Not aligned rule applies to its interval's upper bound", was within -.027 to +.010 of the truth.
- But that simulation divides the point estimate (`observed / sqrt(r_s)`, per the script's docstring), and the script has no bootstrap or interval step.
- The rule divides `ci[1]`, a bootstrap percentile whose replicates sit lower, as the table above shows. In replicate 2 the corrected bound fell .009 below the truth.

So the simulation supports the form of the correction, not its calibration as applied. For the rule as applied, the evidence is recovery's boundary rate (at most .026).

**Handling.**
- A one-line dated clarification that R_S is the Spearman-Brown stepped split-half value. No threshold change.
- In Supplementary Note 1, line 162: say that the simulation corrects the point estimate, and cite recovery's boundary check for the interval form the rule uses.

### A4-14 (N). The analysis needs single-threaded BLAS to finish in reasonable time

**Code.** `scripts/a4_recovery.py:25-27` sets OMP, OpenBLAS and vecLib threads to 1 before importing numpy. `a4_analyse.py` does not.

**Measured here** (analysis env, 8 cores). One `fit_tau` on basic WC takes:
- 24 to 27 s with default threads;
- 0.16 to 0.18 s with one thread.

The analysis makes about 2,040 fits per model and dataset, so with default threads it would take days on this machine. The thread count can also change floating-point results in the last bits.

**Handling.** Environment only, no code change:
- run with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1`, as recovery did;
- record this, and the numpy, scipy and pandas versions, in the run note. `environment.yml` does not pin them.

For scale: the synthetic run with every frozen constant (below) took 38 minutes of wall time for three models, single-threaded on a shared machine (1,117 s of user CPU).

### A4-15 (N). Smaller points

- **(a) Rule precedence.** When the Aligned and Not aligned conditions both hold, `decide` returns Aligned (`:165-168`). The probe used CFA .31, CI (.05, .25) and R_S .9. The text does not order the two. This needs the point estimate above the bootstrap upper bound, which did happen once in A4-13's three weak-effect replicates. So it is rare rather than impossible.
- **(b) V2 for the EA datasets uses Western faces.** All 100 image-free stimuli are basic WC or conversational WC renders (`scripts/a4_run.py:61-62`, with the manifest in dataset order). So V2 for both EA datasets is measured on Western faces. V3 and V4 are per dataset, and V1 is global. Line 141 says the gates run "per model and task", which this satisfies.
- **(c) Amplitude sets use rendered amplitudes.**
  - They are fitted on the rendered (scaled and clipped) amplitudes, which is what `au_json` holds for those rows (`research/scripts/a4_stimuli.py:79-83`; `:550`).
  - So do(AU=1) at 0.75x extrapolates beyond the design's top amplitude.
  - The 1x comparison uses the configurations kept in the 0.75x set (`:548, 554`).
  - Line 65 says only "repeat the fit".
- **(d) Three images are each shared by two configuration strings** within a dataset, because their 33-column coding is the same. Each such image enters the fit twice. This is negligible.
  - Conversational WC: `AU10L-50_AU10R-50` and `AU10-50`; `AU25-12-66_AU25-66` and `AU25-12-66`.
  - Conversational EA: `AU6-50` and `AU6L-50_AU6R-50`.
- **(e) Recovery differs from the analysis in more than line 233 says.**
  - Recovery's permutation counts (1,999 cue and 199 signal; `a4_design.py:91-92`) also differ from the analysis (9,999 and 999). The disclosure at line 233 names only bootstrap draws and splits.
  - `scripts/a4_recovery.py:4` still says "the frozen 1,200 configurations per dataset", although every configuration is used.
- **(f) Wording, line 101.** "Corrected values are not used for decisions" refers to the ceiling ratio. The Not aligned rule does use a corrected bound (lines 124, 131).
- **(g) Shared seed.** Every model and dataset uses the same seed (`:425`), so bootstrap indices are shared across models on a dataset. Not an error.

## Synthetic end-to-end runs

No model and no render was involved. Cells were built by `make_cells.py` for every cell the frozen `planned_cells` lists on the rebuilt manifest: 78,946 per model, 12 shards, in `a4_run.score()`'s format, each with a first-position bias of +1 logit.

Three synthetic systems stood in for the models:

| Code key | System |
|---|---|
| `gemma4` | Human-like: the mean of the WC and EA human log-odds weights for each task, with logit noise of SD .5 per stimulus |
| `qwen3vl` | Prior-only: no AU effect |
| `molmo2` | Human-like |

The gates file was written by hand: everything passed except Molmo2's `g2_g4`.

**1. Every frozen constant, through the real `main()`.** The command was `python -m scripts.a4_analyse` with the frozen defaults and `--detector research/reproduction/a4_v1_pilot.csv`, writing to scratch, single-threaded. Synthetic output sha256: `3623a48e...`.

- **Top level.** The statistic is residual. V1 is .7847, pass. `human_reference` equals the ceiling file's values (.877, .780, .767, .732).
- **Human-like Gemma.** Aligned on all four datasets, and the headline is Aligned for both tasks.

  | Dataset | Residual CFA | Residual CI | R_S |
  |---|---|---|---|
  | Basic WC | .886 | (.862, .902) | .994 |
  | Basic EA | .906 | (.879, .923) | .989 |
  | Conversational WC | .927 | (.914, .932) | .990 |
  | Conversational EA | .896 | (.883, .902) | .992 |

  p_cue is .0001 and p_holm .0002 (twice the cue p) on every dataset. All seven gates were true, and the blind gate passed.
- **Prior-only Qwen.** Inconclusive on all four datasets: signal p was .127, .806, .527 and .860, and V3 failed. Its blind gate failed on B4.
- **Molmo2.** Descriptive on all four datasets, with the headline "Inconclusive" (A4-10).
- **Output keys.** Every slot name in results.md sections 5 and 6 was present (listed under "Checked and consistent").
- **Rotation averaging.** The first-position bias did not stop the human-like model from reaching Aligned. Averaging over the complete rotation set spreads a pure position bias evenly over the named options, which `test_position_preference_cancels_over_rotations` checks. The bias's effect on the synthetic estimates was not measured separately.

**2. Only Gemma's shards present** (reduced counts, set through `estimate.__kwdefaults__`: 999 cue shuffles, 99 signal shuffles, 40 bootstrap draws, 4 splits).
- The output has no Qwen entry anywhere.
- Gemma's `p_holm` equals its `p_cue` on every dataset (A4-1).

**3. One NaN in one of Qwen's cells** (same reduced counts). The NaN was in one rotation-0 cell of a basic WC main stimulus.

| Dataset | Qwen residual CFA | Qwen p_cue | Qwen signal p | Qwen V4 | Qwen decision |
|---|---|---|---|---|---|
| Basic WC (NaN cell) | NaN | .001 | .01 | false | Inconclusive |
| Basic EA (untouched) | .101 | .116 | .81 | true | Inconclusive |

On basic WC, .001 and .01 are the floors at 999 and 99 shuffles; at the frozen counts they would be .0001 and .001 (A4-2).

**4. Shard 7 removed.** Lost shares were .0874, .0852, .0850 and .0826, so the model failed `clips_kept` on every dataset with no error (A4-3).

## Checked and consistent

**Estimand (text lines 90, 97, 103).**
- `fit_tau` (`:77-83`) fits a ridge logistic per option on an intercept, centred gender (plus or minus .5, as in `human_au_effects.py:183`) and the 33 AU amplitudes. Intercept and gender are unpenalised, and the ridge is the dataset's `ridge_lambda_selected` (0.1, 0.1, 1.0, 1.0).
- G-computation of do(AU=1) minus do(AU=0) uses the other AUs as observed and renormalises over options (`human_au_effects.py:149-170`).
- The basis is 25 families; bilateral composites set both sides together (`:29-31`; members listed and checked).
- The model counts as one subject with fractional responses, one unit weight per stimulus.
- The human matrix is the stored `do_effect_probability_au_full_vs_none` at the same 25 columns. Its `effect_names` equal the code's, and its labels match the option order. Column sums are within 2e-4 of zero after rounding.

**R and residual CFA.**
- `remove_common` (`:86-97`) removes each matrix's own mean profile across cues, one rank, as line 97 says. Model and human each use their own.
- The same R is used in the interval, the split-half, the cue null, the sensitivities, the amplitude sets, the direct contrast, the ceiling, the reference and the energy shares. For subsets the profile comes from the full matrix, then columns or rows are dropped (line 173).

**Responses (line 84).**
- Each rotation's log probabilities are renormalised, mapped to the canonical option order by name, and averaged over the complete cyclic set. A stimulus missing any rotation is dropped (`:48-63`).
- There are 6 rotations for basic and 4 for conversational, with abstain last (`a4_design.py:39, 143-146`).
- A lost share above .05 fails the dataset (`:421, 428`).

**Design matrix.**
- For every main configuration in all four datasets, the manifest's `au_json` equals the human coding exactly: maximum absolute difference 0.
- Each configuration string has one AU vector in the human trials.
- Gender is chosen by hash from the genders humans saw. Identities follow line 61. The renderer rejects unknown columns and amplitudes outside 0 to 1 (`render_au_faces.py:415-422`).

**Inference.**
- Signal (`:116-124`): 999 shuffles of stimuli against responses, on the summed squared OLS AU coefficients; p < .05.
- Cue null (`:100-113`): 9,999 shuffles of the model matrix's 25 columns against the fixed human matrix, one-sided, on each statistic. A flat matrix gives p = 1.
- Interval (`:127-137`): 2,000 stimulus bootstrap draws, each refitted, 2.5 and 97.5 percentiles.
- R_S (`:140-150`): 20 random half splits, each half refitted, per statistic.
- Holm (`analyse_rotation.holm`, standard step-down) runs within a dataset, over `gemma4` and `qwen3vl` only, on the confirmatory statistic. Molmo2 is "Descriptive" and outside Holm (`:440-450`).

**Decision rule (lines 121-133; `:153-169`).**
- Inconclusive unless every gate is True and signal p < .05.
- Aligned when Holm p < .05, CFA at least .30 and the CI lower bound is above 0.
- Not aligned when R_S > 0 and `ci[1] / sqrt(R_S)` < .30. A NaN or non-positive R_S cannot give Not aligned.
- Aligned uses the uncorrected value. The statistic is the one `a4_recovery.json` names, and recovery's 66 checks ran this same `decide`.

**Thresholds equal to the text.**

| Rule | Value in the code |
|---|---|
| SESOI | .30 |
| Alpha | .05 |
| Lost share | fails above .05 |
| R_S lower-bound flag | .85 |
| V1 | .70 |
| V2 | .25 nats on 50 stimuli |
| V3 | p < .01 with positive delta |
| V4 | .90 for abstain, .95 for any option |
| B1 to B4 | 1e-3; .05; .25 on 50 stimuli; p < .01 |

**Gates in the decision.** The decision uses `rq0`, `g2_g4`, V1 to V4 and `clips_kept` (`:448, 654-656`). `rq0` admits A3 Pass with A1 Pass or Fail, and rejects a missing A1. The headline (`:453-468`) follows line 137.

**V1, V3 and the blind gate.**
- V1 (`:265-287`) counts covered pairs only: 24 columns by 6 identities, 144 pairs. "Rising" means weakly monotone with an overall rise.
- V1's unconfirmed columns are AU11L and AU11R, AU13, 16, 22, 27, 38, 39 and 43.
- A missing `--detector` stops the run (`:622-623`).
- V3 (`:240-262, 401-412`) compares the same configuration on two faces against another configuration's replicate within the same gender and ordered face pair, by sign flip with 9,999 draws.
- The six blind families exist in A2's frozen lexicon, and Confused and Thinking do not.

**Behaviour (lines 166-167; `:302-314, 562-566`).**
- Pearson r is taken with option means removed. Modal agreement is reported with Cohen's kappa.
- Each configuration's human response is the mean over its trials.
- Every manifest configuration is in the human table, and every human label is in the label order.

**Freeze controls.**
- `a4_run.py` refuses to run without `--a4_frozen` (`:129-130`). The job passes that flag only when `A4_FROZEN=1` (`slurm_jobs/study3_a4.sh:26, 58-59`).
- Smoke files (`cells_<model>_smoke_<k>`) are excluded by `shard_files`.
- Cells carrying another design hash are refused.
- The runner and the analysis both check the manifest hash.
- The confirmatory statistic comes from `a4_recovery.json`, with its design hash checked. Recovery can only withhold `STATISTIC` (`a4_recovery.py:196-198`).

**Output keys for results.md sections 5 and 6.** Every slot name exists:
- `datasets.<dataset>.<model>`: `residual.{cfa, ci, r_s, over_ceiling, lower_bound_only}`, `raw.{cfa, ci}`, `p_holm`, `p_signal`, `gates`, `decision`, `sensitivities`, `direct_cfa_residual` and `behaviour.{output_r, modal_agreement, modal_kappa}`;
- `models.<model>`: `headline`, `blind` and `scale`;
- `human_reference`, which the human-level reading needs.

They were confirmed in the synthetic output above.

**Cost table counts.** `planned_cells` gives 78,946 cells per model: 77,996 MCQ (52,428 basic plus 25,568 conversational), 800 blind, 50 blind image-free and 100 image-free. This matches lines 258-262.

**Paper text.**
- Supplementary Note 1, sections 2 and 4: every code line it cites was checked against the frozen code, and its values match `a4_human_ceiling.json`.
- results.md section 5 rule text matches lines 121-137. The slots in sections 5 and 6 are covered above.
- methods.md "RQ1 on randomised cues (A4)" and supplementary_methods.md S6 match the text, and S6 carries the ceiling disclosure.

## Not checked, and why

- **Cluster copies** of the code, manifest, renders and shards. SSH is barred by the audit rules. The pins hold on this machine. STATE records "113 pins OK on the cluster" at the 2026-11-09 sync, which I did not verify.
- **Any property of real A4 cells**, such as finiteness, completeness or duplicates. That needs A4 output.
- **The renderer, the detector and the GPU scoring path** beyond their hashes, the column mapping and the calls the runner makes. The prompt and pixel stripping were read.
- **The two RESULTS_REGISTRY wording points**, which were not re-derived.

## Scratch files

All under `/private/tmp/claude-502/-Users-eb2007/cf56ec16-1588-406d-9db5-6872ba6bedc3/scratchpad/audit_a4/`.

| Purpose | Files |
|---|---|
| Rebuilt manifest | `manifest.csv` |
| Synthetic cell generator, using the frozen `planned_cells` | `make_cells.py` |
| Synthetic cells | `cells_full/`; `cells_gemma_only/` holds links to Gemma's cells; `cells_nan/` holds Gemma links plus Qwen copies with one NaN |
| Edge-case probes (A4-1, 2, 4, 5, 6, 10, 11, 15a) | `probe_edges.py` |
| Ceiling rerun | `ceiling_rerun.json`, `ceiling_rerun.log` |
| What-if ceiling at the selected ridge | `ceiling_at_selected_ridge.py` |
| BLAS timing | `time_fit.py` |
| Reduced-draw `main()` driver | `run_main_reduced.py` |
| Bootstrap position check (A4-13) | `bootstrap_attenuation.py` |
| Synthetic end-to-end outputs | `summary_full.json`, `summary_gemma_only.json`, `summary_nan.json` and their logs (`full_run.log`, `gemma_only.log`, `nan_run.log`) |
