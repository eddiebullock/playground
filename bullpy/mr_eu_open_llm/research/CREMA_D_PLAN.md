# CREMA-D plan (prereg Phase 6)

Face-only human ratings for RQ1 and RQ5. Source: Cao et al. 2014, doi:10.1109/TAFFC.2014.2336244; CheyneyComputerScience/CREMA-D commit 1658cd34. Licence: ODbL (database) and DbCL (contents). Access: registration form submitted by the user.

## Face-only coding

`queryType == 2` in the raw responses; row names in `tabulatedVotes.csv` are `100000 * queryType + clipNum`, so face-only rows start with 2. Checked directly, not only from the README:

- With upstream's 10 s filter, raw `queryType` 2 reproduces all 7,442 prefix-2 rows exactly (1.0000). It matches prefix 1 and 3 on at most 0.039 of clips.
- `summaryTable.csv` FaceVote equals the prefix-2 modal vote on every clip.
- The behaviour fits faces: Happy recognition is 0.877 here, 0.290 for type 1 and 0.855 for type 3. Overall order is voice 0.400, face 0.574, audio-visual 0.631.

Raters saw these clips silently. Model input must be frames only, with no audio track.

## Option set

Six options on every clip: Anger, Disgust, Fear, Happy, Sad, and **No Emotion**. There is no None of the above. Cao et al. describe raters choosing an emotion from the five targets or selecting "No Emotion" if they perceive no specific emotion. The data code that choice N and filenames call the condition NEU, but raters did not see "Neutral". Model prompts use "No Emotion" (`OPTION_LABELS`); `category` keeps "Neutral" as the actor's intended condition.

All responses fall in these six codes, every option was chosen for every intended emotion, and each label was used across all 91 actors. So the set is fixed and human confusion does not depend on which foils were offered. Chance is 1/6, and a target is the actor's intended emotion, not the human mode.

Not available: the on-screen order of the options, and whether it varied. Human position effects are therefore unknown, as for EU-Emotion (A1). Raters also always did voice, then face, then audio-visual.

## Selection rule

Fixed before any model is run on CREMA-D. `research/scripts/crema_select.py`, seed 42, standard library only. Output `research/reproduction/crema_d_selection.csv`.

1. Eligible: at least 8 face-only ratings after upstream's filter, and not one of the 4 clips the README lists as faulty (1076_MTI_NEU_XX, 1076_MTI_SAD_XX, 1064_TIE_SAD_XX, 1064_IEO_DIS_MD).
2. Each of the 91 actors gets 9 clips: one per emotion plus 3 extra, the extra emotions cycling across actors ordered by sex then id. Total 819.
3. Within an actor-emotion cell, take the clip with the least-used sentence for that emotion, then for that actor, then overall, then the least-used intensity level. Remaining ties go to SHA-256 of seed and clip id.
4. An empty cell stops the script; balance is never relaxed silently.

No filter on agreement, entropy or accuracy, and no model output. `select_clips` receives only file names, rating counts and actor sex; a test confirms that scrambling votes leaves the selection unchanged.

**Why 8.** The 10 s filter drops slow trials, and slow trials are more common on ambiguous clips, so rating count correlates with ambiguity (Spearman -0.09 with entropy). A floor of 9 would remove 545 clips and lean against ambiguity; 8 removes 85 (1.1%). This relationship was seen before the floor was fixed, so the floor is not blind to it, but it was chosen to remove less ambiguity, not more. Selected clips match all clips: mean entropy 0.700 v 0.699 nats, unanimous 0.132 v 0.135.

Emotion quota is equal, so Neutral (1,087 of 7,442 upstream) is slightly over-represented. Intensity labels exist only on sentence IEO, so 58 selected clips carry them; do not analyse level from this subset.

## Option position

The target must not sit in a fixed position (A1, defect 1). `target_position` assigns each clip one of six positions, balanced within emotion in hash order, so it tracks neither label nor actor. `load_crema_d_items` cyclically rotates the one option set to put the target there, with `human_counts` reordered to match. `rotation=0..5` puts every target in every position once. Cyclic rotation keeps neighbouring labels adjacent.

## Reliability

`research/scripts/crema_reliability.py` (2,000 splits of raters within clip, seed 42) writes `crema_d_reliability.json` and per-clip `crema_d_clip_reliability.csv`. Clip-level means use about 10 raters, so entropy is a weak target: use within-emotion ceilings, and do not compare a model with entropy without its ceiling.

## Videos

On CSD3 at `~/rds/hpc-work/study3/data/crema_d/`, flat `<clip_id>.flv`, with `crema_d_lfs_manifest.csv`. Fetch with `research/scripts/crema_fetch_videos.py`: LFS pointers from GitHub, bytes from the GitLab mirror, every file checked against its pointer sha256. Never on the Mac. Raw ratings are in gitignored `data/external/crema_d/`; re-download from the repository's master branch and regenerate with the scripts above.

Checked by `research/scripts/crema_verify_videos.py` (`crema_d_video_check.csv`): 819 of 819 are FLV with matching sha256, 480x360 at 29.97 fps, 1.40 to 4.87 s, 272 MB. All carry an audio track that must stay unused. On a 40-clip sample, cv2 seeks landed on the exact frame. Three h264 clips (1039_WSI_SAD_XX, 1061_TIE_ANG_XX, 1072_WSI_ANG_XX) report 2 or 3 more frames than decode, so a seek to the last index fails; the verifier flags them by design, and they lose one sampled frame.

`.flv` is not yet accepted by `load_stimulus_as_images` in `scripts/frame_sampling.py` or by `MEDIA_EXTS` in `scripts/evaluate.py`. Extending the first is needed before any model run.
