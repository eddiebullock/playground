<!--
Draft, 2026-11-08 (ROADMAP step 0.9); brought up to date 2026-11-12 after the independent draft review (research/reviews/draft_review_2026-11-13.md, finding 2).
Every tag, commit, date and hash below was read from the repository: `git tag -l` gives a2-freeze, a4-freeze, a5-freeze, a7-freeze, a8-freeze, a9-freeze, prereg-v1, v2-phase0, and `git ls-remote --tags origin` shows each on GitHub (a7, a8 and a9 checked 2026-11-12). Design hashes are the `DESIGN_SHA256` constants in `scripts/a7_design.py`, `research/scripts/a8_design.py` and `scripts/a9_design.py`.
Commit dates are from `git log -1 --format=%ci` (UTC). `git cat-file -e` confirmed each linked file exists at its tag or commit.
A1, A1.1 and A3 have no tag. They are identified by commit, and each commit is on origin/main.
The journal requires "an active link to the preregistration in the Methods section" and "the date of preregistration" (preregistration policy, read with WebFetch on 2026-11-08). It does not say whether a tagged commit counts. [USER: ask the editors, or deposit on a registry.]
-->

# Preregistration and amendment links

Repository: https://github.com/eddiebullock/playground, project path `bullpy/mr_eu_open_llm/`.

| Amendment | Content | Frozen (date, UTC) | Tag or commit | `design_sha256` | Link |
|---|---|---|---|---|---|
| v1 | Original task and hypotheses (tests not used in this paper) | 2026-10-30 14:01 | tag `prereg-v1`, commit f2cd59d | none | https://github.com/eddiebullock/playground/tree/prereg-v1/bullpy/mr_eu_open_llm/research/PREREGISTRATION.md |
| A1 | Rotated task, real against mismatch | 2026-11-04 11:39 | commit 410da27 (no tag) | none; analysis code committed before data | https://github.com/eddiebullock/playground/tree/410da27e312efbe6d870e9f49317597a00ba3863/bullpy/mr_eu_open_llm/research/PREREGISTRATION.md |
| A1.1 | Decoder fix; A1 restarted before any result was read | 2026-11-04 15:39 | commit b3af2cc (no tag) | none | https://github.com/eddiebullock/playground/tree/b3af2cc73d80cba1fbc9db85e60396ec8c7f7587/bullpy/mr_eu_open_llm/research/PREREGISTRATION.md |
| A3 | A1's design on a single face crop | 2026-11-04 19:42 | commit b6c712d (no tag) | none; code sha256 `face_input_validity.py` 8a681a65..., slurm script 2bf3a9f7... (`PREREG_A3_DRAFT.md`, line 5) | https://github.com/eddiebullock/playground/tree/b6c712da204a25ab7b6281c7647cd3f45f39aca7/bullpy/mr_eu_open_llm/research/PREREG_A3_DRAFT.md |
| Natural-clip RQ1 rules | Unit, null, strength rule, clip rule; reading under each RQ0 outcome | 2026-11-04 19:41 (strength rule result, commit c6559cc); 2026-11-04 19:43 (all five rotations, commit 48cd38c); 2026-11-07 11:52 (reading under each RQ0 outcome, commit 9e410e3) | commits c6559cc, 48cd38c, 9e410e3 (no tag) | none; criteria hash 4b0a8c3c (`same_clip_family_criteria.json`) | https://github.com/eddiebullock/playground/tree/9e410e30f01537d6f63ec1325c82a64a7e6400dd/bullpy/mr_eu_open_llm/research/DECISIONS.md (holds all three entries) |
| A2 | Blind readout gate and stimulus interventions | 2026-11-07 11:22 | tag `a2-freeze`, commit ca38d89 | 0ae22118d3b2690c581595890b9bb8b904e1f7860f37be9ac909f925b7058de7 | https://github.com/eddiebullock/playground/tree/a2-freeze/bullpy/mr_eu_open_llm/research/PREREG_A2_DRAFT.md |
| A4 | Randomised cue test on rendered faces | 2026-11-07 17:16 | tag `a4-freeze`, commit 568b6f5 | 12c454da512bfe4c239e6c3ef3610d80df6fd7c2e9119a9ff8df913a3c1a3e94; manifest 2ef1161e895224d499b360144690a07c5440bb018105e9c2ffc762f68badd2d1 | https://github.com/eddiebullock/playground/tree/a4-freeze/bullpy/mr_eu_open_llm/research/PREREG_A4_DRAFT.md |
| A5 | Prior and evidence on the blind readout (RQ4) | 2026-11-08 13:59 | tag `a5-freeze`, commit 9b12720 | fd20495bf411b97bb876a32b0d93994b98bcc73a05aa104c6976f3fe2988efd4 | https://github.com/eddiebullock/playground/tree/a5-freeze/bullpy/mr_eu_open_llm/research/PREREG_A5_DRAFT.md |
| A6 | Shared errors on CREMA-D (RQ5) | Never frozen. Closed as a limit on 2026-11-10, when its own pre-freeze first-frame check failed; never run on a model | none | none | Not preregistered; the closing entry is DECISIONS 2026-11-10 |
| A7 | Vision-path cuts (RQ3): Stage 0 completeness, Stage 1 route arm | 2026-11-08 21:31 | tag `a7-freeze`, commit 62c6474 | 9e40cfcfbe8761703da22cf2400c3af6cc7f05019df8b186f91e421dce50bb09 | https://github.com/eddiebullock/playground/tree/a7-freeze/bullpy/mr_eu_open_llm/research/PREREG_A7_DRAFT.md |
| A8 | Larger models in the same families through the frozen pipelines | 2026-11-09 10:59 (design text; code pins follow as a dated pre-data step) | tag `a8-freeze`, commit cbf0826 | 95dd41b5d3c33b79ee9e54a20d34ea1e18b6c9156fcabd719ebfb084709b7b50 | https://github.com/eddiebullock/playground/tree/a8-freeze/bullpy/mr_eu_open_llm/research/PREREG_A8_DRAFT.md |
| A9 | The randomised cue test on a second human sample (Liu et al. 2022, Experiment II) | 2026-11-09 17:33 | tag `a9-freeze`, commit bc41374 | 21341eb47a4d5da81e7d672e21c56c83029cbf9ba2b928f98f10fcba35220192 | https://github.com/eddiebullock/playground/tree/a9-freeze/bullpy/mr_eu_open_llm/research/PREREG_A9_DRAFT.md |

The tag `v2-phase0` (commit d47c0be, 2026-10-30) marks a reproduction snapshot and is not a preregistration.

## Deviations after freezing, by amendment

The journal asks for every deviation with its rationale. Each is logged in the amendment and in DECISIONS.

| Amendment | Date | Deviation | Where |
|---|---|---|---|
| v1 | 2026-11-02 | Holm floor with 1,000 permutations; rule not retuned | `research/PREREGISTRATION.md`, Deviations |
| v1 | 2026-11-04 | Prompt confound; task withdrawn, replaced by A1 | same; DECISIONS 2026-11-04 |
| A1 | 2026-11-04 | Decoder fault; all runs discarded unread (A1.1) | same |
| A2 | 2026-11-07 | Source support corrected before any data; predictions unchanged | `research/PREREG_A2_DRAFT.md:332-353`; DECISIONS 2026-11-07 |
| A1 | 2026-11-08 | Errored trials: scored as the frozen pipeline scores them, through a wrapper; settled before any A1 file was read | DECISIONS 2026-11-08, "A1 errored trials" |
| A2 | 2026-11-08 | Test liberal under a skewed null; .01 sensitivity fixed before any cell | `research/PREREG_A2_DRAFT.md`, Deviations, 2026-11-08; DECISIONS 2026-11-08 |
| A2 | 2026-11-12 | Pre-data rulings after an independent code audit: the fill rule also governs Opposes; torn-line repair; model identity; equivalence intervals and shuffles from a declared script; gate inputs; pins at run time; failed cells not rerun. No code or hash changed | `research/PREREG_A2_DRAFT.md`, Deviations, 2026-11-12; DECISIONS 2026-11-12 |
| A4 | 2026-11-08 | Human ceiling uses a related estimator; disclosed, code unchanged | `research/PREREG_A4_DRAFT.md`, Deviations (recorded there 2026-11-12); DECISIONS 2026-11-08; Supplementary Note 1 |
| A4 | 2026-11-12 | Pre-data rulings after an independent code audit: Holm family fixed at the two named models; rows with non-finite log probabilities unusable; shards complete and clean; inputs at pinned bytes; gates file built by script; Aligned and Not aligned both met read as Inconclusive; single-threaded BLAS; reporting. No code or hash changed | `research/PREREG_A4_DRAFT.md`, Deviations, 2026-11-12; DECISIONS 2026-11-12 |
| A5 | 2026-11-08 | Balance verdicts descriptive, decided before the freeze | `research/PREREG_A5_DRAFT.md:570-588`. Before the freeze, so not a deviation from the frozen text |
| A5 | 2026-11-12 | Pre-data rulings after an independent code audit: input errors void a run; reporting additions; norms hash checked. No code or hash changed | `research/PREREG_A5_DRAFT.md`, Deviations, 2026-11-12; DECISIONS 2026-11-12 |
| A7 | 2026-11-10 | Stage 1 scope recorded after Stage 0 and before any A2 output: route arm run, token arm not (Stage 0 did not admit it). Made under A7's own rule, so not a deviation | DECISIONS 2026-11-10 |
| A7 | 2026-11-12 | Route arm submitted before A2's output exists; its preconditions are applied by A7's analysis code ("Not tested" where one fails), replacing the 2026-11-10 wording "is not run" for that case | DECISIONS 2026-11-12 |
| A8 | 2026-11-09 | A step with an incomplete output after two out-of-memory attempts counts as no result for that model (before any A8 output) | `research/PREREG_A8_DRAFT.md`, Dated notes |
| A9 | 2026-11-09 | Lead decision 8, taken before the freeze but after recovery run 2 was seen: decisions stay confirmatory, the weak-effect accuracy claim is withdrawn under face-correlated noise, and the replaced rule's reading is reported beside each decision. Before the freeze, so not a deviation from the frozen text; disclosed because of its timing | `research/PREREG_A9_DRAFT.md`, lead decisions; DECISIONS 2026-11-09 |
| A9 | 2026-11-12 | Status note (frozen 2026-11-09, tag governs) and the A4 pre-data rulings carried over to the A4 functions A9 imports. No code or hash changed | `research/PREREG_A9_DRAFT.md`, Deviations, 2026-11-12 |
| A9 | 2026-11-13 | Scored runs submitted before the timing smokes ran, against checklist step 4's order, to gain queue age; the smokes still run first and only lower the scored time limit. The pre-run check now enforces replication rule (ii). No code or hash changed | `research/PREREG_A9_DRAFT.md`, Deviations, 2026-11-13; DECISIONS 2026-11-13 |

## Notes

- The `/tree/<tag>/` form was requested. On 2026-11-08 WebFetch opened the A4 link and GitHub showed the file at tag `a4-freeze`. The other links were not opened; check each before submission.
- Links to commits use the full hash, since a short hash in a URL can become ambiguous.
- Methods asks whether to tag A1 and A3. Tagging is a repository change for the user to approve [USER].
- Logistics entries (scheduling, storage, file placement) change no design and are not listed here; they are in DECISIONS.
