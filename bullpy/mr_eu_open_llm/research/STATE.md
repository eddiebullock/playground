# STATE

Single source of truth for the paper-v2 agent. Update after every meaningful step.

**Updated:** 2026-11-09 Europe/London

## Current phase

**Rebuilding on a clean task.** Prompt confound found 2026-11-04 (DECISIONS). Branch B stays as a record, but accuracy and last-token probes are confounded. Amendment A1 is the active confirmatory step. Literature-led redesign under way in `research_notes/`.

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

## Decoder

All video jobs must pin cv2 4.11 and pass `research/scripts/decoder_preflight.py`. See amendment A1.1.

## Access confirmed

EU-Emotion: full access and licence, per user 2026-11-04. Full RMET access. CREMA-D form done. Nothing from these goes in the public repo.

## Branch / git

- Default branch: `main`.
- Integration branch: `paper-v2`.
- Task branch: `p1-pipeline`.
- Tags: `v2-phase0`, `prereg-v1`, `a2-freeze`, `a4-freeze`, `a5-freeze`, `a7-freeze`, `a8-freeze`.
- Repo visibility: **public**.

## Blockers (non-idle)

- Official H1 gate failed Holm (perm-resolution floor). Branch B locked; do not retune.
- MRMET option-count verification waits for Phase 6.
- Phase 4 cancelled under Branch B.
