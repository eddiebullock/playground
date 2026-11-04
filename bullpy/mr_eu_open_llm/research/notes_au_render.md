# Rendering the Snoek AU configurations for a model

**Checked:** 2026-11-04, from the dataset, Snoek's public code and the tool repositories. Nothing was installed or rendered.

**Short answer:** No exact reproduction is possible. The dataset gives enough to rebuild each AU configuration approximately on a different avatar. It does not give enough to reproduce the stimuli humans saw. Treat any free-tool render as a new stimulus set of the same AU recipes, and validate it before comparing it with human effects.

## What the dataset gives

- Which AUs were active on each stimulus, and their amplitudes on a 0 to 1 scale.
- Face gender and a face code for each stimulus.
- For East Asian and conversational data, the row label keeps the 42-AU raw coding, including compounds (`AU25-12-33`) and sixths-step amplitudes.
- For Western basic data, the label was rebuilt after recoding and rounding, so raw compounds and exact amplitudes are lost.

## What it does not give

- **Temporal parameters.** Stimuli were 1.25 s animations, and each AU had a randomised time course (thesis, citing Yu et al. 2012). None of this is in the files. Static frames or new dynamics would be an assumption.
- **GFG faces or renderer.** There are no videos, frames, meshes or textures. Snoek's `viz/generate_face_for_figures.py` imports a `GFG` library and a `quick_FACS_blendshapes_v2dense` file that are not in the repository. PyPI's `gfg` is an unrelated geotechnical package. I found no public release.
- **Identities.** The eight base faces are not available. Western subjects 01-15 saw blends, and the blend weights are in a raw file that is not public.
- **Amplitude calibration.** A GFG amplitude of 0.5 has no known match on another avatar's slider. A linear mapping would be a guess.

## Free tools

| Tool | Licence | Coverage of the 33 columns | Fit |
|---|---|---|---|
| FACSHuman (MakeHuman plugin) | LGPL-3.0 | Target files exist for 31 columns directly, including L and R where needed. AU27 is named in `au.json`, but no target file is in the repo. AU22 exists only as combined `22_25` lip targets. | Best coverage. GUI plugin with a Python 2.7 header, last pushed 2021. It loads `.facs` JSON and has image and video buttons, but I found no headless batch path. About 2,300 configurations per set would need scripting against MakeHuman, and this was not tested. It needs MakeHuman installed. |
| openFACS | MIT | 18 AUs and no laterality. The API lacks AU11, 13, 16, 22, 24, 27, 38, 39 and 43, and adds 28 and 45. | Scriptable over UDP on an Unreal Engine build, but too few AUs. Rejected for this use. |
| FaReT | GPL | Six emotion poses; no AU control found (earlier note, not rechecked). | Unsuitable. |

AU27 and AU22 matter here: AU27 is the top surprise cue in both groups, and AU22 is the top conversational "other" cue in both.

## Recommended route

1. Email Jack's group (Glasgow) for GFG renders of the shared configurations, or for render access. This is the only route to the human stimuli, and it should come first.
2. As a fallback, or a domain-shift check, render through FACSHuman, reporting AU22 and AU27 as approximated or dropped.
3. Check each avatar render with an AU detector, and check that models behave sensibly on avatar faces. Otherwise a domain shift could look like a strategy difference.
4. Test one-at-a-time AU interventions alongside the human configurations. RQ1 compares effect matrices, so a model need not see the identical configurations. In that case use the `subjects_and_stimuli` ceiling in `research/reproduction/human_au_effects.json`, not the same-stimuli ceiling.
5. Decide static versus dynamic explicitly. Humans saw movement; these effects are averaged over time course.

## Unverified

Whether Glasgow will share renders. Whether FACSHuman runs on this Mac or on current MakeHuman. Whether MakeHuman's own expression targets could supply AU27. The disk impact of MakeHuman.

## Checked against the paper, 2026-11-04

Snoek et al. 2023 (read in full from the PMC copy) deposits data and analysis code only. The availability statement gives no route to stimuli, so renders must be requested. Snoek is marked as corresponding author. Investigation is credited to Chen, Garrod and Jack.

Stimulus facts for any re-render: 1.25 s, 30 frames at 24 fps, frontal view, flat lighting, black background, one of eight base faces of the participant's own ethnicity. Each AU had six time-course parameters (onset, offset and peak latency, peak amplitude, acceleration, deceleration). The paper analysed peak amplitude only, as we do. Tasks were 7AFC (basic) and 5AFC (conversational) including "other". An example movie is in the supplement of Chen et al. 2021, J Vis.

Trial count: the paper gives 119,382 Western basic trials excluding "other"; the public files give 119,387. The 5-trial difference is unexplained. It also gives 128,400 East Asian and 83,540 conversational, which match.

The paper's "AU ablation" removes AUs from hypothesis models, not from stimuli shown to people. Cite it accurately: the human causal claim rests on the randomised stimuli.

