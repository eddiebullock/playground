<!--
Draft, 2026-11-08 (ROADMAP step 0.9). Answers for the Nature Portfolio reporting summary, from the repository.
Markers: [PENDING] = the repo will supply it once an analysis is locked; [USER] = only the user can answer.
No result that is not locked in RESULTS_REGISTRY is stated.
-->

# Reporting summary: draft answers

## How far the requirements were verified (2026-11-08)

| Requirement | Source | How checked | Status |
|---|---|---|---|
| Article: abstract up to 150 words, unreferenced; main text up to 5,000 words excluding abstract, Methods, references and figure legends; up to 8 display items; no reference limit; unheaded Introduction, Results and Methods with subheadings, Discussion without | https://www.nature.com/nathumbehav/content | WebFetch, 2026-11-08, after the site's cookie redirect | Verified through WebFetch's summary, which quoted these phrases. Matches outline.md (read 2026-11-07) |
| Reporting summary required where relevant, for behavioural and social sciences; published with accepted papers | https://www.nature.com/nathumbehav/editorial-policies/reporting-standards | WebFetch, 2026-11-08 | Verified through WebFetch's summary |
| Data availability statement for every research paper; access to the minimum dataset made clear; identifiers for public data | same page | WebFetch, 2026-11-08 | Verified through WebFetch's summary |
| "Code availability" statement for central custom code, after data availability and before references; DOI-minting repository (Zenodo, Code Ocean), cited in references; code shared with editors and reviewers, and peer reviewed when central | same page | WebFetch, 2026-11-08 | Verified through WebFetch's summary |
| Preregistration: strongly encouraged for non-clinical confirmatory research; active link in Methods and date of preregistration; all deviations disclosed with rationale; originally planned analyses reported for deviations other than flaw or feasibility; no acceptance until met | https://www.nature.com/nathumbehav/editorial-policies/preregistration-policy | WebFetch, 2026-11-08 | Verified; WebFetch returned these sentences as quotes |
| Whether a tagged commit in a public repository counts as a preregistration | same page | WebFetch, 2026-11-08 | **Not stated by the page.** Ask the editors, or deposit the amendments on a registry [USER] |
| Ethics statement wording | Research ethics policy (separate page) | Not fetched | **Unverified** |
| The item list of the reporting summary form itself | https://www.nature.com/documents/nr-reporting-summary-flat.pdf | WebFetch, 2026-11-08 | **Unverified.** The PDF downloaded, but its text could not be extracted here (no PDF text tool on this machine). The headings below follow the structure of the Nature Portfolio form as known before this session. Check every heading against the current form before filling it in |
| Page charges, preprint policy | not fetched | | **Unverified** (outline.md, "Journal limits") |

WebFetch passes each page through a summarising model. Quotes above came back as quotes, but re-read each page before submission.

## Statistics

For all statistical analyses, confirm the following are given in the figure legends, table legends, main text or Methods.

| Item | Answer |
|---|---|
| Exact sample size for each group or condition, as a discrete number and unit | Yes. EU-Emotion: 243 clips, 17 actors (Methods, Human data). Snoek: 60, 60, 20, 20 observers and trial counts per dataset (Methods; RESULTS_REGISTRY, A4 human target). A4: 15,225 rows, 11,761 unique images (Methods, A4 Stimuli). [PENDING: items lost per model, per analysis] |
| Whether measurements were from distinct samples or the same sample measured repeatedly | Yes. Models are measured repeatedly on the same items; real and mismatch are within item; A2 contrasts are within item and averaged per actor (Methods) |
| Statistical tests used, one- or two-sided | Yes. Actor-level sign-flip tests, one-sided where a direction was stated (A1, A3, Gate B, A2 predicted direction), two-sided for A2 Opposes; permutation tests for A4 signal and alignment and for the natural-clip score (Methods) |
| Description of all covariates tested | Yes. Category intercept and actor residualisation for natural clips; centred face gender for the Snoek and A4 fits (Methods) |
| Assumptions or corrections, such as tests of normality and adjustment for multiple comparisons | Yes. Holm across models for A1, A3 and A4; Westfall-Young step-down maxT within model and route for A2; A5 step-down within its families (Methods; `research/PREREG_A5_DRAFT.md`) |
| Full description of statistical parameters, including central tendency and variation | Yes. Means with actor-bootstrap or stimulus-bootstrap 95% intervals (Methods) |
| Null hypothesis tests: test statistic, confidence intervals, effect sizes, degrees of freedom and P values, with exact P values where suitable | [PENDING: per result. A3 values are locked in RESULTS_REGISTRY] |
| Bayesian analysis: priors and MCMC settings | Not applicable. No Bayesian analysis |
| Hierarchical and complex designs | Partly applicable. Actor is the unit for EU-Emotion tests; the A4 human target averages per-subject fits (Methods) |
| Effect size estimates, such as Cohen's d or Pearson's r, with how they were calculated | Yes. Delta in accuracy (A1, A3); CEA and residual CEA as cosines (A4); pattern score as Pearson r (natural clips); CLR change (A2) |

## Software and code

| Item | Answer |
|---|---|
| Data collection | No new human data. Model outputs were produced by the repository's scripts (`scripts/evaluate.py`, `scripts/face_input_validity.py`, `scripts/run_stimulus_interventions.py`, `scripts/a4_run.py`) on a GPU cluster, with OpenCV 4.11 pinned for video decoding (amendment A1.1). [PENDING: Python, PyTorch and transformers versions, from `environment.yml` and the cluster logs] |
| Data analysis | The repository's analysis scripts, named per analysis in Methods. LibreFace and OpenFace 3.0 for AU estimation; MediaPipe FaceMesh 0.10.14 for landmarks (Methods). [PENDING: full version list] |
| Custom code availability to editors and reviewers | Public at https://github.com/eddiebullock/playground, path `bullpy/mr_eu_open_llm/`, MIT License. [PENDING: Zenodo DOI, USER to create] |

## Data

| Item | Answer |
|---|---|
| Data availability statement | See `data_code_availability.md` in this folder |
| Accession codes, unique identifiers or web links for public datasets | Snoek AU data, https://doi.org/10.21942/uva.21261885.v1; EU-Emotion norms in the repository; CREMA-D from its authors' repository |
| Restrictions on availability | EU-Emotion videos through the dataset's authors; FACSHuman renders not published (LGPL-3.0 targets with no output clause); CREMA-D videos not redistributed by us; RMET not used in reported analyses |

## Research involving human participants, their data, or biological material

| Item | Answer |
|---|---|
| Reporting on sex and gender | No participants were recruited. Secondary human data: [USER: whether to report sex or gender of the original observers and actors, from the source papers] |
| Reporting on race, ethnicity or other socially relevant groupings | Snoek observers were Western (WC) and East Asian (EA) groups, each shown faces of their own ethnicity (Methods, Human data [19]). A4's EA avatar faces are approximate and not based on East Asian people (Methods) |
| Population characteristics | Not applicable for new data. For secondary data, see the source papers [14,19,26] |
| Recruitment | Not applicable. No recruitment |
| Ethics oversight | [USER: the ethics or exemption statement; methods.md line 46 is PENDING. Name the body that approved or exempted the secondary analysis, if any] |

## Field-specific reporting

Select: **Behavioural and social sciences**. [USER: confirm. The study could also be read as computational; the behavioural section fits the human data and the preregistered tests.]

## Behavioural and social sciences study design

| Item | Answer |
|---|---|
| Study description | Quantitative, preregistered, confirmatory tests of three open vision-language models against existing human data: a validity audit (A1, A3), cue-effect comparison on rendered faces (A4), stimulus interventions (A2), and prior against evidence (A5). Natural-clip pattern alignment under rules dated 2026-11-04 and 2026-11-07 in DECISIONS |
| Research sample | Models: Gemma 4 E4B, Qwen3-VL-8B, Molmo2-O-7B at pinned revisions. Stimuli: 243 EU-Emotion face clips (17 actors), Snoek AU configurations (4,659, 2,323, 2,318 and 2,325 per dataset), 24 CREMA-D calibration clips. Human data: EU-Emotion face-only norms; Snoek WC and EA observers |
| Sampling strategy | No power analysis set sample size, because stimulus sets and human samples are fixed (Methods, Statistics and sample size). A4 uses every available configuration, and recovery tested that count |
| Data collection | Model forwards on a GPU cluster with exact option probabilities, or 20 draws at temperature 1.0 (A1). No human data collected |
| Timing | A1 and A3 frozen 2026-11-04; A2 and A4 frozen 2026-11-07; A5 frozen 2026-11-08 (see `preregistration_links.md`). [PENDING: run dates of A1, A2 and A4 model runs] |
| Data exclusions | Items lost per the frozen missing-data rules: A3 needs all 5 real and all 5 mismatch rotations; natural clips need all 5 rotations, descriptive above 5% lost; A4 drops a stimulus without every rotation, and fails a dataset above 5% lost. A3 lost none (RESULTS_REGISTRY, A3). [PENDING: losses for A1, natural clips, A2, A4] |
| Non-participation | Not applicable |
| Randomization | Option position rotated cyclically through every position (A1, A3, A4). Mismatch partners fixed by seed 42. Human cue effects come from randomised AU presentation in the source study [19] |

## Reporting for specific materials, systems and methods

| Item | Answer |
|---|---|
| Materials and experimental systems (antibodies, cell lines, animals, clinical data, dual use) | Not applicable. [USER: confirm "dual use research of concern" is not applicable] |
| Methods (ChIP-seq, flow cytometry, MRI) | Not applicable |
| Plants | Not applicable |

## Open items

1. [USER] Ethics or exemption statement.
2. [USER] Whether tagged commits count as preregistration for the journal, or whether to deposit the amendments on a registry such as OSF.
3. [USER] Create the Zenodo DOI.
4. [PENDING] All result-dependent rows, after ROADMAP phases 1 to 3.
5. Check every heading above against the current reporting summary form, which could not be read as text on 2026-11-08.
