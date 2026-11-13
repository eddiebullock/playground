<!--
Draft, 2026-11-08 (ROADMAP step 0.9). Consistent with research/paper/v2/methods.md, sections "Data availability" and "Code availability", and with DECISIONS 2026-11-08, "EU-Emotion permissions settled".
Journal rules, as read with WebFetch on 2026-11-08 from "Reporting standards and availability of data, materials, code and protocols | Nature Human Behaviour": every research paper needs a data availability statement that makes access to the minimum dataset clear, with accession codes or identifiers for public data; papers with central custom code need a "Code availability" statement after the data statement and before the references; code should be deposited in a DOI-minting repository such as Zenodo or Code Ocean and cited in the references. That reading came through WebFetch's summarising model, so quote the page itself before submission.
Placeholders: [PENDING] for facts the repo will supply later; [USER] for what only the user can supply.
-->

# Data availability

No new human data were collected for this study.

Human cue effects come from public trial-level data from Snoek et al. [19], deposited at the University of Amsterdam data repository under CC BY 4.0 (https://doi.org/10.21942/uva.21261885.v1) [25].

The replication on a second human sample (A9) uses the public trial-level data of Liu et al. [60], Experiment II, deposited at Mendeley Data under CC BY 4.0 (https://doi.org/10.17632/dr853shk56.2). They are not copied into the repository; `research/scripts/a9_human_target.py` reads them from a local download, and its outputs are checked by sha256 (`scripts/reproduce.py`).

EU-Emotion [14] is a collaborator dataset. Its item-level face-only response norms, as used here, are public in the repository (`data/eu_emotions_human_entropy.json`). The clips, frames and edited images are not in the repository. Access to the EU-Emotion videos is through the dataset's authors [USER: contact route or address for access requests].

CREMA-D [26] clips were used for readout calibration (24 clips, labels and votes unread). CREMA-D is distributed by its authors (repository CheyneyComputerScience/CREMA-D) under the Open Database License, with contents under the Database Contents License (`research/CREMA_D_PLAN.md`, line 3). We do not redistribute its videos. The clip identifiers we used are in `research/reproduction/crema_calibration_clips.csv`.

Rendered faces for A4 were built from FACSHuman targets [29], which are licensed LGPL-3.0 with no clause on outputs. We therefore do not publish any rendered image. The AU configurations come from the public Snoek data. The render manifest (`results/a4/manifest.csv`) is not committed, but `scripts/reproduce.py --quick` regenerates it from public inputs and checks it against the sha256 pinned in `research/PREREG_A4_DRAFT.md` (line 5). The renders can then be rebuilt with the published code from the FACSHuman and MakeHuman assets.

Reading the Mind in the Eyes Test (RMET) materials are not used in the analyses reported here and are not in the repository. [USER: confirm that no RMET-based result appears in the final paper; if one does, add its access terms here.]

Derived outputs (analysis summaries, human effect matrices, ceilings, recovery results and audit files) are in `research/reproduction/` in the repository, archived at [PENDING: Zenodo DOI, USER to create].

# Code availability

All analysis code, preregistered design files and their frozen hashes are in the public repository https://github.com/eddiebullock/playground, under `bullpy/mr_eu_open_llm/`, released under the MIT License (`bullpy/mr_eu_open_llm/LICENSE`). The version used for this paper is archived at [PENDING: Zenodo DOI and release tag, USER to create the DOI]. The command `python scripts/reproduce.py --quick` rebuilds the derived outputs that need only public inputs. Its latest clean-checkout test, on 2026-11-13 at commit 6fffab1, rebuilt and matched 40 outputs and verified 112 more against pinned sha256; its one failure, a recorded input hash that changes when an upstream file is rebuilt in the same run, was fixed and the affected steps matched on rerun (`research/reproduction/REPRODUCE.md`, "Clean-checkout test, 2026-11-13"). [PENDING: rerun the clean-checkout test on the submission commit, covering every new output (ROADMAP 4.5).] Outputs that need model forwards ran on a GPU cluster and are marked "cluster" in `research/reproduction/REPRODUCE.md`. Model weights were the public revisions listed in Methods (Gemma 4 E4B commit 83df0a88, Qwen3-VL 8B 0c351dd0, Molmo2-O 7B 78441065).

# Notes for the user

- The journal asks for code in a DOI-minting repository, cited in the references. Creating the Zenodo record needs your login.
- Reference numbers [14], [19], [25], [26], [29] and [60] follow `research/paper/v2/references.md`, whose numbering is provisional.
- methods.md says EU-Emotion videos are withheld "for size". DECISIONS 2026-11-08 adds the reason of a simple rule across stimulus sets. Both reasons are compatible; the text above states neither, only the access route.
