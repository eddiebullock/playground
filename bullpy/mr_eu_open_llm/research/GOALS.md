# GOALS

**Set:** 2026-11-07, on the user's instruction to work without input until these are met. Status lives in `STATE.md`; this file says what done means.

**Overarching question:** how similar are vision-language models to humans in how they read mental states from faces, not only in the answers they give? (`RESEARCH_QUESTIONS.md`)

**Target:** one integrated paper, written so that it stands whatever the results are, plus the public reproduction package. Primary venue Nature Human Behaviour, fallback TMLR, both to confirm at submission (`research/LIT_CHECK_2026-11-07.md`). The claim worth citing is a cue-level test of VLMs against randomised human cue effects, which we located no earlier example of, on top of a validity audit that the field needs regardless of outcome.

## Rules that do not bend

1. Nothing is read before its analysis is locked. Frozen thresholds are not changed to get a result. Deviations are dated and disclosed.
2. A null is a publishable result if the null rule is met. No goal below is defined by a result going one way.
3. No stimuli, frames or edited images in the public repo.
4. Free resources only. Do not touch `cas_*` jobs. No file read by pending rotation jobs changes on the cluster until they finish.
5. Never fabricate a citation or a number. Every paper number traces to a script, commit and output file (`RESULTS_REGISTRY.md`).
6. Submission, authorship and any email to other researchers are the user's. The agent gets everything ready for them.

## Goals

| ID | Goal | Done when | Depends on |
|---|---|---|---|
| G1 | RQ0: does the answer depend on the face? | A1 analysed, read against A3 per model, recorded | Qwen arrays |
| G2 | RQ1 on natural clips | Family pattern score per model, with clip rule, recorded | G1 |
| G3 | A2 frozen | Audit of code against text done, open items closed, hashes recorded, tag `a2-freeze`, before A1 is read | None |
| G4 | Gate B and RQ2 stimulus arm | A2 run and analysed on eligible models, each prediction classed by the decision rule | G1, G3 |
| G5 | RQ1 on randomised human cue effects, the novel core | Headless render of Snoek AU recipes validated; amendment A4 frozen before any model sees a render; models run; effect matrices compared with the human matrix against a noise ceiling | Render pipeline |
| G6 | RQ4: perception or word knowledge | Prior and evidence separated with image-free, grey and mismatch baselines on the blind readout, in a frozen amendment | G4 |
| G7 | RQ3: causal route | Vision-path cuts, only at sites shown to be complete cuts, in a frozen amendment. Reported as not testable where no complete cut exists | G1, G4 |
| G8 | RQ5: shared errors | CREMA-D run across its fixed option set, a cue-targeted test of one human confusion, in a frozen amendment | G1 |
| G9 | Cue Effect Alignment theory | Estimand defined, non-identifiability result proved in full, or the claim dropped | None |
| G10 | Literature current | Novelty rechecked, every cited paper read at the level the text claims, venues and deadlines verified | None |
| G11 | Paper | Full draft replacing `research/paper/*_v1.md`, figures from scripts, all numbers traced, supplement, submission-ready for the chosen venue | All above that completed |
| G12 | Reproduction package | One documented path from public inputs to every table and figure; clean checkout test passes | G11 |

## Order

Work runs in parallel wherever a goal is not waiting on the queue. Now: G3, G5 render, G9, G10. Once Qwen finishes: G1, G2, then G4 at once. G6 to G8 follow G4, and G11 is drafted throughout, with result sections written only from locked output.

## If time or compute runs short

G1, G2, G4 and G5 make the paper. G6 is largely a reanalysis of G4's cells. G7 and G8 may become scoped limits or a second paper. The decision and the reason go in `DECISIONS.md`.

## What only the user can do

- Log in to CSD3 when the SSH connection drops, which needs their TOTP.
- Submit the paper, approve authorship, and send any email, such as asking Jack's group for GFG renders.
