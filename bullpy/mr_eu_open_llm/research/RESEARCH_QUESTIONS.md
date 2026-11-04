# Research questions v2

**Written:** 2026-11-04. Replaces the Alignment, Mechanism and Causality questions as the study's framing.
**Not a preregistration.** Each confirmatory test still needs its own amendment before data. Sources: `research_notes/VLM versus human emotion mechanisms/`. Updated the same day with the human-mechanisms review.

## Overarching question

How similar are vision-language models to humans in *how* they read mental states from faces, not only in the answers they give?

Similarity is split into the three degrees of Schyns, Snoek and Daube (TiCS 2022): same behaviour, same features, same computation. Each question below answers one degree, and each is causal wherever it can be.

## Questions

**RQ0. Validity: does the answer depend on the face?**
With option position and prompt cues controlled, how much of a model's accuracy disappears when the prompt stays the same and the face is swapped?
Test: amendment A1, real versus mismatch across five rotations. A model that fails makes no mental-state claim. No later RQ runs on it.

**RQ1. Features: do models and humans use the same facial information?**
Which facial cues (action units, gaze, head angle, dynamics) change the judgement of a given mental state, and do they have the same direction and relative weight in models as in humans?
Test: compare the matrix of cue effects on each answer option in humans (randomised action units, Snoek data) with the same matrix in models, against a human noise ceiling.

**RQ2. Causal stimulus test: if you change the cue, does the judgement change the way a human's does?**
Holding identity and everything else fixed, is a human-diagnostic cue necessary and sufficient to move a model's judgement, for example from happy to sad, and does the dose-response follow the human curve?
Test: graded edits of one cue against a zero-delta edit through the same generator, with verified action units.

**RQ3. Causal route: does the cue reach the answer through an identifiable path, or can it merely be read out?**
Is the cue's effect carried from image tokens through the language model to the answer, and does blocking that path remove the effect?
Test: cut vision paths at model-specific sites, with positive controls, joint knockouts for redundancy, and mediation by the cue.

**RQ4. Computation: when models get it right, is that perception or word knowledge?**
How much of each judgement comes from face evidence and how much from label priors and semantics, and does the balance differ from humans?
Test: separate prior from evidence with image-free and mismatch baselines and a blind, option-free readout, fit per model and compared with humans. Extend to inversion as a test of holistic processing.

**RQ5. Shared errors: when humans and models make the same mistake, is it for the same reason?**
Does a shared confusion come from the same cue, or from shared label semantics or the options offered?
Test: on options crossed with images, change the cue that drives a human confusion and check whether it moves that pair and not others. This is the old Alignment, Mechanism and Causality set, rebuilt.

## Design constraints from the human-mechanisms review

- **Weak target labels.** Face-only humans chose the target on 23 to 26% of Kind, Unfriendly and Jealous items (chance 17%; `data/eu_emotions_human_entropy.json`). Scoring against the acted label treats these as errors, and high model accuracy there would suggest non-facial leakage. Report every model result against the human response distribution as well as the acted label, and flag these 26 items.
- **Bored-Unfriendly.** Unfriendly is one of those labels and Bored was offered on its items, so this pair may reflect clips that read as bored, not a shared confusion. RQ5 must rule that out.
- **Head angle.** No located study shows head pose alone flipping happy to sad. A downward tilt raises perceived dominance through apparent brow lowering (Witkower and Tracy 2019, abstract level). Keep head angle in RQ1 and RQ2 as a tested cue, with no prediction of a category flip.
- **Levers with human evidence for a category shift:** playback speed, morphing, adaptation, composites from different regions, gaze on ambiguous faces, and context.
- **Playback speed needs a frame-policy change.** Models saw frames at 1 fps, usually 4 to 11. Before a speed test, establish whether models use temporal information at all, for example by shuffling frame order against the original.
- **Complex states.** No located reverse-correlation models exist for Bored, Worried, Disappointed, Kind or Unfriendly; most cue evidence is correlational and from basic emotions. Eye aperture (Lee and Anderson 2017) is the cleanest causal test for a complex state.
- **RMET.** It loads on vocabulary, so language-backed models need a text-only control and a foil swap.
- **Actor and label** are only partly crossed: each face actor portrayed 10 emotions plus neutral (O'Reilly 2016).

## What changed from the old questions

| Old question | Problem | Now |
|---|---|---|
| Alignment: same confused pairs? | Partly answered (Strachan et al.); confusion depends on which options were offered | RQ5, with options crossed with images |
| Mechanism: pair-specific or generic difficulty? | Cannot be tested when options are fixed by label | Inside RQ5 |
| Causality: is structure used or just readable? | Tested at a site dominated by prompt text | RQ3, on vision paths |
| Not asked | Supervisor: what makes a human do this, and is it encoded? | RQ1 |
| Not asked | Supervisor: if you intervene, does it change? | RQ2 |
| Not asked | Supervisor: do models do it a very different way? | RQ4 |

## Novelty

| RQ | Status in the literature | Risk |
|---|---|---|
| RQ0 | Routine, but needed; a cross-encoding audit of VLM emotion work is a small contribution of its own | Low |
| RQ1 | No located VLM classification image or action-unit tuning curve | High scoop speed |
| RQ2 | No located intervention linking human-diagnostic cues to a VLM judgement | High scoop speed |
| RQ3 | Causal tracing through image tokens exists; none located on a facial cue | Medium |
| RQ4 | Templates exist in general (Barua 2026, Liu ECCV 2026); emotion version open | Medium |
| RQ5 | No located intervention shifts one specific human confusion pair | Low to medium |

Null searches ran for one day with some indexes unavailable. Check CCN, VSS and CogSci 2026 abstracts before claiming priority.

## General contribution for ML venues

Causal Feature Alignment: an estimand comparing randomised human cue effects with interventional model cue effects, with a noise ceiling, recovery checks, and a proof that behavioural alignment does not imply it. The theorem is a proposal, not yet proved.
