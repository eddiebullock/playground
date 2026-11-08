<!--
Draft, 2026-11-08 (ROADMAP step 0.9). Every tag, commit, date and hash below was read from the repository on 2026-11-08:
`git tag -l` gives a2-freeze, a4-freeze, a5-freeze, prereg-v1, v2-phase0, and `git ls-remote --tags origin` shows all five on GitHub.
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
| A6 | Shared errors on CREMA-D (RQ5) | Not frozen | none | draft c1961048... (changes with any edit) | Not preregistered |
| A7 | Vision-path cuts (RQ3) | Not frozen | none | draft 9126e416... (changes with any edit) | Not preregistered |

The tag `v2-phase0` (commit d47c0be, 2026-10-30) marks a reproduction snapshot and is not a preregistration.

## Deviations after freezing, by amendment

The journal asks for every deviation with its rationale. Each is logged in the amendment and in DECISIONS.

| Amendment | Date | Deviation | Where |
|---|---|---|---|
| v1 | 2026-11-02 | Holm floor with 1,000 permutations; rule not retuned | `research/PREREGISTRATION.md`, Deviations |
| v1 | 2026-11-04 | Prompt confound; task withdrawn, replaced by A1 | same; DECISIONS 2026-11-04 |
| A1 | 2026-11-04 | Decoder fault; all runs discarded unread (A1.1) | same |
| A2 | 2026-11-07 | Source support corrected before any data; predictions unchanged | `research/PREREG_A2_DRAFT.md:332-353`; DECISIONS 2026-11-07 |
| A2 | 2026-11-08 | Test liberal under a skewed null; .01 sensitivity fixed before any cell | DECISIONS 2026-11-08. [PENDING: not yet appended to the Deviations section of `PREREG_A2_DRAFT.md`; see the report] |
| A4 | 2026-11-08 | Human ceiling uses a related estimator; disclosed, code unchanged | DECISIONS 2026-11-08; Supplementary Note 1 |
| A5 | 2026-11-08 | Balance verdicts descriptive, decided before the freeze | `research/PREREG_A5_DRAFT.md:570-588`. Before the freeze, so not a deviation from the frozen text |

## Notes

- The `/tree/<tag>/` form was requested. On 2026-11-08 WebFetch opened the A4 link and GitHub showed the file at tag `a4-freeze`. The other links were not opened; check each before submission.
- Links to commits use the full hash, since a short hash in a URL can become ambiguous.
- Methods line 26 asks whether to tag A1 and A3. Tagging is a repository change for the user to approve [USER].
