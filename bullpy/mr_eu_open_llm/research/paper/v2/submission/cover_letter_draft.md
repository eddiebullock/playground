<!--
Draft, 2026-11-08 (ROADMAP step 0.9). Claims are limited to what is locked in RESULTS_REGISTRY or proved in Supplementary Note 1. Results are PENDING.
Submission and authorship belong to the user. Nothing here has been sent.
-->

[USER: date]

[USER: handling editor's name, if known], Editor
Nature Human Behaviour

Dear [USER: editor],

We submit "[PENDING: title]" for consideration as an Article in Nature Human Behaviour.

Some vision-language models are reported to match people at reading mental states from faces. We ask how similar these models are to humans in how they read faces, not only in the answers they give. The manuscript makes three contributions.

First, a validity audit. Our own earlier task placed the correct option first on all 243 items and gave each of 21 labels one option set, so its accuracy could not be told apart from position or prompt text. We withdrew it. In a spot check of eight recent papers, several of the claims we located cannot rule out option order or prompt text. We then measured how much accuracy each model loses when the face is swapped under a fixed prompt, with option position rotated. Swapping the face cost all three models accuracy, by 7 to 40 points, although one model stayed at chance on the right face.

Second, an identifiability result. We prove that agreement on natural items does not in general determine which facial cues a system uses: two systems can agree on every natural item and still differ in every cue effect. We define Cue Effect Alignment, a comparison of a model's cue effects with human effects estimated under randomised cue presentation.

Third, a cue-level test. We compare model cue effects on rendered faces with randomised human cue effects from public data, against a noise ceiling and a between-group reference. [PENDING: one sentence on A4.] We also test preregistered human predictions under stimulus interventions. [PENDING: one sentence on A2.]

Every confirmatory analysis was fixed in a dated amendment or decision-log entry before its data were analysed, and the rules of the cue-level and intervention tests can each return a match, a mismatch or an inconclusive result. Amendments, frozen hashes and every dated deviation are listed in the manuscript. The paper is written to stand whatever the results are, and nulls are reported as results. Code and derived outputs are public, and a reproduction script rebuilds every output that needs only public inputs and checks the rest against pinned hashes.

[USER: why this journal and its readership; one or two sentences.]

[USER: statement that the manuscript is not under consideration elsewhere, and any preprint.]

[USER: competing interests.]

[USER: suggested and excluded reviewers, if any.]

We would also ask the editors one question. The amendments are preregistered as tagged commits in a public repository. Please tell us whether this meets the journal's preregistration requirement, or whether we should deposit them on a registry.

Yours sincerely,

[USER: corresponding author name, affiliation and email]
[USER: co-author names and affiliations]
