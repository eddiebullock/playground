# STATE

Single source of truth for the paper-v2 agent. Update after every meaningful step.

**Updated:** 2026-11-07 Europe/London

## Current phase

**Rebuilding on a clean task.** Prompt confound found 2026-11-04 (DECISIONS). Branch B stays as a record, but accuracy and last-token probes are confounded. Amendment A1 is the active confirmatory step. Literature-led redesign under way in `research_notes/`.

**2026-11-08 evening:** ROADMAP Phase 0 is closed except 0.13. A6 local items closed (PR 87): recovery run 5 matches run 4, freeze prepared, cluster items only. A7 audited (PR 86): numeric complete-cut rule, blocks are not cuts, RQ3 "not testable" where no complete cut is shown; A7 code and freeze (0.13) under way and must precede any read of A2 cells. Literature (PR 85): Gold et al. wording fixed; CFA name clash with a WACV 2024 method is NEEDS_INPUT 9. Qwen A1 arrays still PENDING, estimate 2026-11-09 06:56 to 08:40; 20 of 30 rotation files.

## Next actions

**A4 frozen 2026-11-07:** tag `a4-freeze`, design hash `12c454da`, PR 73 merged as `fc4a327`, before any A1 result was read and before any model saw a render. Residual CFA is confirmatory, recovery 66 of 66 (66 of 68 counting the two withdrawn cells). Render finished 2026-11-07: arrays 40268765 and 40270481 (tasks 0 and 1 failed in 7 s on cpu-q-168 and were rerun on another node), 40 tars, 11,761 images as planned, 2.2 GB, in `results/a4/renders` on the cluster, no model. Scoring waits for RQ0, then a timing smoke per eligible model.

**A5 (RQ4) frozen 2026-11-08:** tag `a5-freeze`, design hash `fd20495b`, PR 80, before any A2 cell existed. After recovery run 6, balance verdicts (Face, Default, Even) are descriptive only, because `shared_family_gain0.5` fails at .094; the evidence classes are confirmatory, and none of their checks fail. A2 test checked under skewed gain nulls (PR 79): per-route FWER up to .14 in one skew direction; verdicts stay as frozen, with a fixed step-down p < .01 sensitivity. Both are in DECISIONS 2026-11-08.

Goals and what done means: `research/GOALS.md`. The user gave full autonomy on 2026-11-07 and accepted all recommendations, including the four A2 choices.

1. A1 rotation evals: Gemma and Molmo complete, 20 of 30 files on the cluster at 2026-11-07 10:02, not pulled, nothing read. Qwen arrays 39982137 and 39982138 still pending on priority, Slurm estimate 8 Nov about 19:55. The estimate has moved later since 6 Nov. Account fair share is 0, and a 59 min `cas_*` job started on 7 Nov while these 12 h tasks waited, which suggests backfill. About 8 h per task, up to 78 GPU-hours, 2,859 available. When all 30 exist, pull them, run `research/scripts/analyse_rotation.py`, then for each model run `research/scripts/same_clip_family_analysis.py --model-route --au-csv research/reproduction/eu_au_measures.csv --model M --model-evals 'results/baseline/eu_emotions/M/eval_v2_eu_emotions_M_video_only_seed42_rot*_real.json'`. `eu_au_measures.csv` is the right file: the route uses LibreFace and OpenFace 3 from it on `model_1fps` frames, passes no GraphAU file, and yields 243 clips by 4 families. Permutations default to 4,999. Do not analyse early: Holm covers all three models.
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
- Tags: `v2-phase0`, `prereg-v1`, `a2-freeze`, `a4-freeze`, `a5-freeze`.
- Repo visibility: **public**.

## Blockers (non-idle)

- Official H1 gate failed Holm (perm-resolution floor). Branch B locked; do not retune.
- MRMET option-count verification waits for Phase 6.
- Phase 4 cancelled under Branch B.
