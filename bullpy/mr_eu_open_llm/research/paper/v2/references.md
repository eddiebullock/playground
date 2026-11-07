# References and reading status

Drafted 2026-11-07. Numbering follows first use in `introduction.md`, then `methods.md`. No reference was added from memory.

**Status key.** Taken from the repo's notes, not from any new reading.

- **read in full**: the notes record a full read.
- **partial**: full text obtained, but only methods or keyword passages were read (P in `LIT_CHECK_2026-11-07.md`). This label is added to the requested three because calling these "read in full" would be false. An asterisk (partial*) means a summarising fetch tool was used, so quotes need rechecking.
- **abstract**: abstract or official listing only.
- **snippet**: search summary or metadata only.

Titles appear only where the repo records them. Where it does not, take the title from the DOI when building the final list.

**Where a DOI or arXiv ID came from.** Repo notes, except as marked: `C` was resolved on 2026-11-07 through the Crossref API, and `Ax` through the arXiv API. `U` means derived from a URL in the notes and not resolved separately.

| # | Citation | Venue | DOI or arXiv ID | Status | Repo source | Used for |
|---|---|---|---|---|---|---|
| 1 | Strachan et al., "GPT-4o reads the mind in the eyes" | Computers in Human Behavior 181, 108992 (2026) | 10.1016/j.chb.2026.108992; arXiv 2410.22309 | read in full (arXiv version, 2026-11-07); published version blocked, not compared | LIT_CHECK s2 | RMET performance; fixed option order |
| 2 | Refoua et al. | Scientific Reports (2026) | 10.1038/s41598-026-39292-y | read in full (PMC copy; OSF materials not read) | LIT_CHECK s2 | RMET performance; fixed option order |
| 3 | Nelson et al. | npj Digital Medicine (2025) | 10.1038/s41746-025-01985-5 (U) | read in full (PMC copy; supplement not included) | `prior_ai_human_comparison.md` | Performance on posed faces |
| 4 | Schyns, Snoek and Daube, "Degrees of algorithmic equivalence between the brain and its DNN models" | Trends in Cognitive Sciences (2022) | 10.1016/j.tics.2022.09.003 (C) | read in full (Glasgow Enlighten, version of record) | `prior_ai_human_comparison.md` | Three degrees of similarity |
| 5 | Zheng et al. | ICLR 2024 Spotlight | arXiv 2309.03882 | read in full (arXiv v4) | LIT_CHECK s3 | Selection bias and option identifiers |
| 6 | Pezeshkpour and Hruschka | arXiv | arXiv 2308.11483 | read in full (arXiv v1) | LIT_CHECK s3 | Option order in text models |
| 7 | Li, Qiu and Xu | arXiv preprint (2026) | arXiv 2609.29278 | read in full (arXiv v1) | LIT_CHECK s2 | First-slot picks in a VLM |
| 8 | Bhattacharyya and Wang | NAACL Findings 2025 | arXiv 2502.05660 | read in full (arXiv v1) | LIT_CHECK s2 | Option order on evoked emotion |
| 9 | Zhao et al. | arXiv preprint (2026) | arXiv 2608.17102 | read in full (arXiv v1) | LIT_CHECK s2 | Randomised options as routine control |
| 10 | Xing et al., CMES | arXiv preprint (2026) | arXiv 2609.34742 | read in full (arXiv v1) | `prior_ai_human_comparison.md`; LIT_CHECK s2 | No option list in main readout; vector-level controls only, no image-free or mismatch baseline |
| 11 | Agarwal et al. | arXiv preprint (2026) | arXiv 2604.15280 | read in full (arXiv v2, with supplement) | LIT_CHECK s1, s2 | Option order not reported |
| 12 | Hallmen et al. | arXiv preprint (2026) | arXiv 2610.08162 | read in full for the main paper (arXiv v2); partial for the supplement (S5 to S7, S13 to S15, keyword scan of the rest) | LIT_CHECK s1, s2 | Yes/no per category, no option position; one fixed list in the text arm; no-image control in S14 |
| 13 | Zhang et al. | ICML 2026 (accepted per arXiv comment) | arXiv 2605.21980 | read in full (arXiv v2) | LIT_CHECK s1 to s3 | Image-level counterfactuals; partial analogue |
| 14 | O'Reilly et al. | Behavior Research Methods (2016) | 10.3758/s13428-015-0601-4 | read in full (publisher PDF, rechecked directly; supplement not obtained) | `human_mechanisms.md` | EU-Emotion stimuli and norms |
| 15 | Geirhos, Meding and Wichmann | NeurIPS 2020 | arXiv 2006.16736 | read in full (arXiv; figure pages by caption only) | `prior_ai_human_comparison.md` | Error consistency |
| 16 | Han, Poggio and Cheung, "System identification of neural systems: If we got it right, would we know?" | ICML 2023 (PMLR 202) | arXiv 2302.06677 (Ax) | read in full (arXiv) | Report; theory note, Open | Related limits. Relation unread |
| 17 | Teneggi and Sulam | NeurIPS 2024 | arXiv 2405.19146 | read in full for the main text (arXiv v2); appendices skimmed | Report; theory note, Open | Statistical tests of concept importance on natural data, by conditional independence; not an identifiability result |
| 18 | Jack et al. | PNAS (2012) | 10.1073/pnas.1200155109 | abstract (Europe PMC; full text blocked at PMC and Europe PMC) | `human_mechanisms.md` | Reverse correlation |
| 19 | Snoek et al. | Science Advances (2023) | 10.1126/sciadv.abq8421 | read in full (PMC copy, rechecked 2026-11-07; supplement not included) | `notes_au_render.md` | Randomised AU design and data |
| 20 | Huynh et al. | arXiv preprint (2026) | arXiv 2606.15779 | read in full (arXiv v1) | LIT_CHECK s1 | Latent AU interventions |
| 21 | Chen et al., FOCUS | ACL 2026 | arXiv 2601.06931 | read in full (arXiv v2) | LIT_CHECK s1 | Race and gender face counterfactuals |
| 22 | Kataria, Ghosh, Ye, Dhall et al., VideoFace-X-LLM, "When, Where, and Why: Hierarchical Causal Explainability for Video Face Understanding with MLLMs" | arXiv preprint (2026) | arXiv 2610.17791 (authors and title Ax) | read in full (arXiv v1) | LIT_CHECK s1 | Region necessity tests |
| 23 | Tsangko et al. | arXiv preprint | arXiv 2506.19079 | read in full (arXiv v1) | LIT_CHECK s1 | Teeth visibility and VLM performance |
| 24 | Barrett et al. | Psychological Science in the Public Interest (2019; journal to confirm from DOI) | 10.1177/1529100619832930 | partial (PMC author manuscript: opening, summaries and conclusions; correction notice unseen) | Report | Felt-state inference is weak |
| 25 | Snoek et al., AU and emotion data | University of Amsterdam data repository | 10.21942/uva.21261885.v1 | read in full (downloaded and inspected) | `human_au_effects.json` | Human data |
| 26 | Cao et al., "CREMA-D: Crowd-Sourced Emotional Multimodal Actors Dataset" | IEEE Transactions on Affective Computing (2014; journal to confirm from DOI) | 10.1109/taffc.2014.2336244 (C) | abstract (Europe PMC; full text blocked at PMC and Europe PMC) | `RESEARCH_QUESTIONS.md` | Calibration clips |
| 27 | LibreFace | arXiv | arXiv 2308.10713 | read in full (arXiv v2) | `causal_methods.md` | AU detector |
| 28 | OpenFace 3.0 | arXiv | arXiv 2506.02891 | read in full (arXiv v1) | `causal_methods.md` | AU detector |
| 29 | Gilbert et al., FACSHuman | Behavior Research Methods (2021) | 10.3758/s13428-021-01559-9 | read in full (publisher PDF) | LIT_CHECK s3; report | Render targets |
| 30 | Wegrzyn et al. | PLOS ONE (2017) | 10.1371/journal.pone.0177239 | read in full (PMC copy) | `human_mechanisms.md` | P1 to P4, P6 |
| 31 | Blais et al. | Neuropsychologia (2012) | 10.1016/j.neuropsychologia.2012.08.010 | abstract (Europe PMC; CORE copy blocked) | `human_mechanisms.md` | P1; eye-only views |
| 32 | Ponari et al. | Emotion (2012) | 10.1037/a0028588 | abstract (Europe PMC; no open copy found) | `human_mechanisms.md` | P5 |
| 33 | Carbon | Frontiers in Psychology (2020) | 10.3389/fpsyg.2020.566886 | read in full (PMC copy) | `human_mechanisms.md` | P7, P8 |
| 34 | Grundmann et al. | PLOS ONE (2021) | 10.1371/journal.pone.0249792 | read in full (PMC copy; OSF supplement not included) | `human_mechanisms.md` | Overall mask accuracy loss |
| 35 | Pazhoohi et al. | PLOS ONE (2021) | 10.1371/journal.pone.0257740 | read in full (PMC copy; S1 File not included) | `human_mechanisms.md` | Restates Carbon; own response frequencies in S1 Table unread |
| 36 | Kumar and Srinivasan | Emotion (2011) | 10.1037/a0025453 | abstract (Europe PMC; no open copy found) | `human_mechanisms.md` | P9, P10 |
| 37 | Smith and Merlusca | Emotion (2014) | 10.1037/a0035588 | abstract (Europe PMC; no open copy found) | `human_mechanisms.md` | Task dependence |
| 38 | Richoz et al. | Journal of Vision (2024) | 10.1167/jov.24.1.7 | read in full (PMC copy; Supplement 1 not included) | `human_mechanisms.md` | Dynamic against static advantage under phase noise; background to P12 only |
| 39 | Bould et al. | Cognition and Emotion (2008) | 10.1080/02699930801921156 | abstract (Solent repository record; no open copy found) | `human_mechanisms.md` | P12 |
| 40 | Ambadar et al. | Psychological Science (2005) | 10.1111/j.0956-7976.2005.01548.x | abstract (Europe PMC; no open copy found) | `human_mechanisms.md` | P13 |
| 41 | Bould and Morris | British Journal of Psychology (2008) | 10.1348/000712607x206702 | abstract (Europe PMC; no open copy found) | `human_mechanisms.md` | P13 |
| 42 | Jack et al. | Current Biology (2014) | 10.1016/j.cub.2013.11.064 | partial (main text reread 2026-11-07, Glasgow Enlighten; supplement unread) | A2, settled choice 4 | P14, P15 |
| 43 | Lee and Anderson | Psychological Science (2017) | 10.1177/0956797616687364 | abstract (Europe PMC; no open copy found) | `human_mechanisms.md` | P11 |
| 44 | Lee et al. | Psychological Science (2013) | 10.1177/0956797612464500 | abstract (Europe PMC; no open copy found) | `human_mechanisms.md` | P11 |
| 45 | Baron-Cohen et al. | Visual Cognition (1997) | 10.1080/713756761 | read in full (Autism Research Centre copy) | `human_mechanisms.md` | Eye-only views |
| 46 | Cunningham et al. | ACM (2005; journal to confirm from DOI) | 10.1145/1077399.1077404 | abstract (Crossref record; CORE copy blocked) | `human_mechanisms.md` | Eye-only views; complex masks |
| 47 | Nusseck et al. | Journal of Vision (2008) | 10.1167/8.8.1 | abstract (Europe PMC; journal page blocked) | `human_mechanisms.md` | Complex masks |
| 48 | Reinl and Bartels | Frontiers in Psychology (2015) | 10.3389/fpsyg.2015.01107 | read in full (PMC copy) | `human_mechanisms.md` | Reversal |
| 49 | Witkower and Tracy | Psychological Science (2019) | 10.1177/0956797619838762 | abstract (Europe PMC; no open copy found) | `human_mechanisms.md` | Head angle exclusion |
| 50 | Gold et al., "The efficiency of dynamic and static facial expression recognition" | Journal of Vision (2013) | 10.1167/13.5.23 | abstract (Europe PMC API; journal page HTTP 403, PMC page reCAPTCHA, both stopped) | Citation check | Human efficiency equal for dynamic, reversed, randomised and static expressions: contrary evidence for P12, P13 |

## Open citation tags in the drafts

- [CITE NEEDED: Westfall and Young]: step-down maxT, `methods.md`, RQ2 Inference. No source in the repo.
- [CITE NEEDED: Spearman]: attenuation correction, `methods.md`, Proposition 5. No source in the repo.
- Holm correction, sign-flip tests, G-computation and Jensen-Shannon divergence are named without citation. Add standard references if the journal expects them. None is in the repo.

## Must read before submission

**Update 2026-11-07.** `citation_check.md` supersedes the list below for status: still below read in full are #18, 24, 26, 31, 32, 36, 37, 39, 40, 41, 42 (supplement), 43, 44, 46, 47, 49 and 50. #50 Gold et al. is the highest priority.

This lists everything below a clean full read that the text leans on: 44 items.

**Priority 1: the novelty and audit claims (12).** The Introduction's counts of what each paper did or did not do rest on these. Check each for option handling and for any image-free or mismatch control, since a keyword search can miss a control.

| # | Why |
|---|---|
| 1, 2 | "Fixed option order" and no mismatch control. Check the published version of 1 |
| 8 | Option order tested; "none found" for mismatch came from a keyword search |
| 9, 11 | Summarising tool only. Recheck every statement and quote. #12 is cleared for its main text |
| 13 | The nearest work. Confirm the "partial analogue" reading, the stimulus mix, and the final sections |
| 20 | The direct scoop signal. Summarising tool only. Confirm "acts on a latent" and the pixel-level future work statement |
| 21, 22, 23 | Named as neighbours, so confirm that none uses randomised human cue effects |

**Priority 2: the framing, theory and option-order claims (12).**

| # | Why |
|---|---|
| 4 | The three degrees frame the whole paper, on a snippet |
| 5, 6, 7 | Option effects in the Introduction. Confirm what each measured, and note text models against VLMs |
| 15 | Definition of error consistency, which the identifiability claim addresses |
| 16, 17 | The theory note requires a full read before any novelty claim for these results |
| 24 | Scope statement on felt states |
| 27, 28 | Detector capability and training data lineage |
| 29 | Render targets; validation effect sizes unread |
| 14 | Read in full, but only through a summarising tool. Recheck every number used |

**Priority 3: the sources of A2's human predictions and exclusions (20: #30 to #49).** The confirmatory predictions lean on these, and all but #42 are at abstract level. Check direction, stimuli and task for each. #42's supplement is unread.

**Not on the list, with reason.** #3 and #18 are background, and no argument rests on them. #26 is named only as the source of calibration clips. #19 and #25 are read in full.

**Also unresolved in the repo's own notes.** Head-pose sources (Mignault and Chaudhuri 2003, Hess et al. 2007) are not cited, and must not be cited first-hand as sources for a pitch effect. Domes 2007 and Sowden 2021 are not cited: the first failed to replicate, and the second is on point-light faces.
