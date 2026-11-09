# Notes for draft amendment A9 (Liu et al. 2022, Experiment II)

Working notes, 2026-11-09. The draft is `research/PREREG_A9_DRAFT.md`. Nothing here is a decision.

## Data handling

- Liu files: `data/external/liu2022/` (gitignored), sha256 pinned in `scripts/a9_design.py`. Read with `scipy.io.loadmat`, never copied.
- Chen et al. 2024: only `data/external/chen2024/WE_resp_data.mat` (604,773 bytes, sha256 `1ef37151f6749f472ba2d53f9e184708bad3c5857b22a04e3da8c0b9dca7bf52`) was fetched, by a ranged read of the member from OSF `data.zip` (file id `mjwt2`, zip sha256 `57e4d097...e030` per OSF), and its CRC32 checked against the zip's central directory. The East Asian file and the 20.9 MB permutation file were not fetched. Can be deleted once the overlap result is accepted; `a9_chen_overlap.json` keeps the counts.
- Manifests: `results/a9/manifest_designA.csv` and `manifest_designB.csv` (gitignored by `*.csv`). Configurations only, no images.

## Chen 2024 against Snoek basic WC

Method: within a Snoek cohort all subjects share stimuli and row order; within a Chen stimulus group likewise. So a stimulus is identified by the vector of all subjects' responses. Chen participant j pairs with Snoek subject j + 1 (by L1 distance of the seven response counts, Snoek subject j + 1 is nearest for all 60 Chen participants, at exactly 2,400 minus Snoek's rows for 53 of 60 and at most 206; the second nearest is never closer than 172).

- Cohort B (Snoek 16 to 60, 45 subjects): 2,342 of 2,342 rows match a unique Chen trial exactly; agreement 1.000 for every subject. AU family sets agree on 1,912 rows; every disagreement is a Snoek AU12 (one-sided) or bilateral AU7 that Chen's 41-AU file does not code.
- Cohort A (Snoek 01 to 15, blended faces): 2,211 of 2,368 exact; per-subject agreement .964 to .9996. Chen codes AU41 and AU42 here, which Snoek merged into AU43.
- Verdict: same data. Chen holds all 2,400 trials per subject; Snoek's public files drop rows (see `human_au_effects.TRIAL_COUNT_NOTE`). The 3 percent disagreement for subject 07 in cohort A is unexplained.

## AU map check

`a9_human_target.snoek_token_check()` rebuilds Snoek's 33 columns from the raw tokens in Snoek's own row labels (basic EA, conversational EA and WC; 6,966 configurations): zero mismatches, 55 column collisions all consistent with the maximum. Snoek's 42 tokens equal Liu's 42 labels with "Open" and "i" removed.

## Things noticed, not acted on

- Liu stimuli carry 1 to 6 raw AUs (median 4), against 1 to 4 (median 3) in the article's procedure text.
- One raw amplitude (2.45e-5) rounds to zero at 4 decimals and is dropped; that stimulus keeps its other AUs.
- Anger's top families in the Liu fit are AU09 and AU10, not AU04; Snoek showed a related AU4 pattern (`anger_au4_diagnostic` in A4's human file). Not investigated.
- Recovery thread oversubscription: run with `OMP_NUM_THREADS=1` and friends set before numpy loads (as `-m scripts.a9_recovery` does); in a shell that has already imported numpy, a replicate took 51 s instead of 1.1 s.
