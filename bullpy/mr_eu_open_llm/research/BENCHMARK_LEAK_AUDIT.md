# Benchmark leak audit: do public affect MCQ benchmarks share our 6AFC confound?

Status: exploratory and post hoc. Annotation text only; no model was run. Date: 2026-11-09.
Script: `research/scripts/benchmark_leak_audit.py`. Output: `research/reproduction/benchmark_leak_audit.json`.
Tests: `tests/test_benchmark_leak_audit.py` (19 tests, synthetic inputs).

## Why

Our EU-Emotion 6AFC prompt put the target at option 1 on 243/243 items and used one fixed option set per
target label (21 sets for 21 labels). An option-set-only predictor matched the activation probes
(`research/reproduction/prompt_confound_audit.json`; DECISIONS 2026-11-04). This audit asks whether public
multiple-choice benchmarks of facial emotion or affect for VLMs and MLLMs have the same two leaks.

## What was measured, per subset

1. **Answer position.** Counts of the correct answer's 0-based position *in the order the model sees*. Chi-square
   goodness of fit against "each item's answer is uniform over its own k options". The Monte Carlo p uses 20,000
   draws and handles mixed k. Counts are also split by k.
2. **Option set leak.** Option set = unordered set of normalised option strings. Reported: number of distinct sets
   and labels, and a leave-one-out (LOO) option-set predictor. The predictor guesses the most common correct label
   among the *other* items with the same set, or 1/k if there is no such item.
3. **Text-only baselines.** Always-same-position (in-sample best, LOO) and same-position-given-k (LOO), plus the
   option-set predictor. Two supplementary predictors: answer prior (LOO, pick the offered option with the highest
   P(correct | offered) elsewhere) and longest option.

**Calibration.** LOO majority rules fall below 1/k when classes are balanced, and in-sample best position sits
above it. So each LOO and in-sample accuracy is compared with its own null: 1,000 simulations in which every
answer is redrawn uniformly from its item's options and the predictor is recomputed. The JSON reports the null
mean, the null 97.5% quantile and p_null. The p_null floor is 1/1001.

**Multiplicity.** Holm correction over 216 p values (54 subsets by 4 tests). With 1,000 simulations, no p_null can
fall below Holm 0.196. Only the position test (Monte Carlo floor 5e-5) can pass Holm, so option-set and prior
results below are judged on effect size against the null 97.5% quantile, with raw p reported. Several Face-Human-Bench
rows are views of the same 50 items, which makes the Holm family conservative.

**Provenance.** Each source records its URL at a pinned commit or revision, its byte count and its sha256:
whole-file sha256 for every file used, and member sha256 for zip members. Two exceptions: the MMT-Bench TSV
(631 MB) was hashed while streaming, and its image column was never stored. EmoBench-M JSONs were extracted from a
12 GB zip by HTTP range requests, so the zip's sha256 is the LFS value reported by the HF API, not one this script
computed. Parquets were downloaded to scratch, hashed, stripped of image columns and deleted. No image or video was
downloaded. Scratch: `/private/tmp/claude-502/-Users-eb2007/cf56ec16-1588-406d-9db5-6872ba6bedc3/scratchpad/bench_audit/`
(49 MB of cached annotation text).

## Benchmarks checked (11)

| Benchmark | Source (revision) | Subset audited | How the official code presents options |
|---|---|---|---|
| MMBench EN dev | HF lmms-lab/MMBench_EN (ecd23e0) | category `image_emotion`: 200 rows = 50 questions x 4 CircularEval rotations; base copy kept | CircularEval (rotations) |
| MMStar | HF Lin-Chen/MMStar (bc98d66) | l2 `image emotion`, 31 | as stored |
| MMT-Bench VAL | HF OpenGVLab/MMT-Bench (2255d3a), `MMT-Bench_VAL.tsv` | 6 tasks with l2 `emotion`, 20 each | as stored |
| SEED-Bench v1 | HF AILab-CVC/SEED-Bench (81cb157) | image questions matching an emotion keyword regex, 92 (post hoc; v1 has no emotion dimension) | as stored |
| FaceXBench | GitHub Kartik-3004/facexbench (5202125) | affectnet/rafdb single- and multi-image expression MCQ, 100 each | as stored |
| Face-Human-Bench | GitHub lxq1000/Face-Human-Bench (622f1f3) and HF mirror InQ2025 (4d5218c) | face_expression_basic and _compound, dev and test, 50 each | `eval.py` rotates options with `random.seed(100)`; replayed exactly |
| FaceBench | HF wxqlab/FaceBench (a4f72f7) | `raf_db_q0` emotion SCQ, 3,068 | as stored (`inference.py`) |
| EEmo-Bench (single) | HF Workerred/EEmo-Bench_single (9795f7c) | all MCQ items, 8 subsets; evoked emotion, not faces | not checked |
| HumanVBench | HF datajuicer/HumanVBench (9aea56e) | Emotion_Recognition, Emotion_Temporal_Analysis, Emotion_Intensity_Compare, Attitude_Recognition | official eval code not found (README branch returns 404); UNVERIFIED |
| MMAFFBen | HF lzw1008/MMAFFBen (26d21c2) | FER2013_test (7,178), CFAPS_EC_test (174); fixed numbered list of 7 labels | fixed list |
| EmoBench-M | HF GMLHUHE/Emobench-M (5659068), JSONs inside the zip | 12 classification sets; fixed label list per set | fixed list |

## Not checked

- **MME.** Has no emotion subtask and uses yes/no answers. Its categories were looked up on the
  lmms-lab-encoder/MME mirror (d6c9023) during scoping. The script does not do this.
- **EmotionBench (CUHK-ARISE) and EmoBench (Sabour et al.).** Text-only LLM benchmarks. EmotionBench is also gated,
  and was not accessed.
- **FABA-Bench, FERBench (arXiv 2511.00389), FEABench (FEALLM).** No public annotation file located. UNVERIFIED.
- **Other items skipped.** MMAFFBen EMOTIC and text sets (context or text, not faces), EmoBench-M `smile.json`
  (open-ended) and FaceBench `raf_au_q0` (multi-select).
- **Answers not public.** MMBench test and MMT-Bench test have no public answers, so only dev and VAL were used.

## Results

Columns: chance = mean 1/k; max pos = largest share of answers at one presented position; posK = LOO
same-position-given-k accuracy; opt-set = LOO option-set accuracy (null 97.5% quantile in brackets);
prior = LOO answer-prior accuracy (null 97.5% quantile in brackets). Rows marked with an asterisk have a fixed
option list, so "position" there is the label's place in the list, and every text-only predictor reduces to
"answer the majority class".

| Benchmark / subset | n | chance | max pos (p_mc) | posK | opt-set (q975) | prior (q975) |
|---|---|---|---|---|---|---|
| MMBench image_emotion, unique, base order | 50 | .25 | .36 (.037) | .18 | .288 (.415) | .13 (.43) |
| MMStar image emotion | 31 | .25 | .35 (.28) | .18 | .452 (.419) | .45 (.47) |
| MMT-Bench facial_expression_recognition | 20 | .25 | .45 (.24) | .45 | .300 (.400) | .42 (.50) |
| MMT-Bench micro_expression_recognition | 20 | .25 | .30 (.88) | .00 | .575 (.525) | .42 (.50) |
| MMT-Bench facial expression change | 20 | .25 | .40 (.18) | .40 | .258 (.258) | .32 (.47) |
| MMT-Bench scene / artwork / body emotion | 20 each | .25 | .35-.40 (>.2) | 0-.40 | .388 / .550 / .129 (.44 / .45 / .46) | .35 / .70 / .05 (.55 / .55 / .50) |
| SEED-Bench emotion-keyword subset | 92 | .25 | .29 (.79) | .29 | .264 (.296) | .57 (.35) |
| FaceXBench affectnet single | 100 | .25 | .29 (.77) | .29 | .289 (.364) | .34 (.37) |
| FaceXBench rafdb single | 100 | .25 | .29 (.36) | .00 | .428 (.365) | .48 (.38) |
| FaceXBench affectnet multi-image | 100 | .253 | .32 (.30) | .32 | .645 (.377) | .62 (.39) |
| FaceXBench rafdb multi-image | 100 | .253 | .30 (.31) | .31 | .333 (.362) | .54 (.38) |
| Face-Human-Bench, raw JSON (all 4 files, both sources) | 50 each | .25 | **1.00 (5e-5, Holm .011)** | 1.00 | as below | as below |
| Face-Human-Bench dev basic, as presented by eval.py | 50 | .25 | .34 / .30 (.26 / .66) | .34 / .15 | .565 (.39) | .52 (.42-.44) |
| Face-Human-Bench dev compound, as presented | 50 | .25 | .36 (.12) | .36 | .730 (.370) | .54 (.42) |
| Face-Human-Bench test basic, as presented | 50 | .25 | .34 / .30 | .34 / .15 | .425 (.39) | .26 (.42) |
| Face-Human-Bench test compound, as presented | 50 | .25 | .36 (.12) | .36 | .375 (.355) | .27 (.42) |
| FaceBench RAF-DB emotion * | 3068 | .143 | .39 (5e-5) | .39 | .386 (.160) | .39 |
| EEmo-Bench yes/no subsets (4) | 395-722 | .50 | .50-.52 (>.37) | <=.52 | .51-.55 (.54-.56) | same |
| EEmo-Bench emotion What/How | 1366 | .333 | .34 (.47) | .34 | .379 (.370) | .36 (.37) |
| EEmo-Bench arousal What/How | 194 | .333 | .39 (.30) | .39 | .417 (.402) | .63 (.44) |
| EEmo-Bench valence / dominance What/How | 205 / 195 | .333 | .42 / .42 (.028 / .040) | .42 / .42 | .422 / .332 (.415 / .347) | .39 / .38 |
| HumanVBench Emotion_Recognition | 263 | .247 | .26 (**5e-5, Holm .011**) | .33 (.32) | .247 (.247) | .32 (.30) |
| HumanVBench Emotion_Temporal_Analysis | 133 | .247 | .26 (**2e-4, Holm .039**) | .18 | .248 (.248) | .27 (.28) |
| HumanVBench Emotion_Intensity_Compare | 112 | .234 | .32 (**5e-5, Holm .011**) | .51 (.36) | .275 (.275) | .47 (.34) |
| HumanVBench Attitude_Recognition | 164 | .25 | .30 (.028) | .30 | .494 (.341) | .49 (.34) |
| MMAFFBen FER2013 * | 7178 | .143 | .25 (5e-5) | .25 | .247 (.154) | .25 |
| MMAFFBen CFAPS * | 174 | .143 | .29 (5e-5) | .29 | .287 (.218) | .29 |
| EmoBench-M MOSI / SIMS / ch-simsv2s * | 457-500 | .333 | .61 / .54 / .47 (5e-5) | same | same as max pos (.39) | same |
| EmoBench-M FGMSA * | 250 | .20 | .52 (5e-5) | .52 | .516 (.268) | .52 |
| EmoBench-M mer2023 * | 411 | .167 | .26 (5e-5) | .26 | .256 (.217) | .26 |
| EmoBench-M MELD / MC-EIU emotion * | 500 | .143 | .20 / .19 (.011 / 5e-5) | | .196 / .192 (.186) | |
| EmoBench-M MOSEI, RAVDESS song/speech, MUSTARD, funny * | 448-500 | 1/k | at 1/k (p >= .53) | | at or below null | |

### Reading, benchmark by benchmark

**Face-Human-Bench is the closest match to our confound.**
- **Position, raw files.** In both the GitHub data and the HF mirror, the raw JSON stores the correct answer at
  index 0 for 50/50 items in each expression file (dev and test, basic and compound). That is the same pattern as
  our 243/243.
- **Position, official evaluation.** The official `eval.py` rotates each item's options by a random offset
  (`random.seed(100)`). Replaying that sequence gives balanced presented positions: max share .30-.36, p .12-.66.
  The paper says the options are shuffled so that answers are spread evenly. Results from the official pipeline are
  therefore not position-confounded. A harness that reads the JSON without that step would be. Whether any
  third-party harness does this is UNVERIFIED.
- **Answer pool, dev split.** In dev basic, only 4 of the 7 offered labels are ever correct. Disgust, surprise and
  fear are offered 23-29 times each and are never the answer. The option-set predictor reaches .565 in dev basic
  and .730 in dev compound, against null 97.5% quantiles of .39 and .37. Rotation preserves the option set, so this
  leak survives the official evaluation.
- **Answer pool, test split.** The effect is much weaker: .425 and .375, against quantiles of .39 and .355
  (raw p .010-.014). It does not survive Holm.
- **Overall.** Partial overlap with our case: a raw position leak that the official code corrects, plus an
  answer-pool imbalance, strong in dev and marginal in test. It is not one fixed option set per label: dev basic has
  30 sets for 4 labels.

**HumanVBench has a structural position leak.**
- **The pattern.** Most items have 4 options. Whenever an item has 5, the correct answer is the 5th: 18/18 in
  Emotion_Recognition, 7/7 in Emotion_Temporal_Analysis and 36/37 in Emotion_Intensity_Compare. The position test
  passes Holm in all three (Holm p .011-.039).
- **What a text-only rule gains.** "Answer E if E exists, else guess" scores .327 against chance .247 in
  Emotion_Recognition, and .512 against .234 in Emotion_Intensity_Compare.
- **Free-text options.** These options are unique per item, so the option-set predictor sits exactly at its null.
  The answer prior gains a little in Emotion_Recognition and Emotion_Intensity_Compare.
- **Caveat.** The official evaluation code was not found, so whether options are shuffled at run time is UNVERIFIED.
  As released, a text-only shortcut is in the data.

**FaceXBench: balanced positions, but answer priors in some files.**
- **Positions.** Balanced (max share .29-.32, all p > .3), consistent with the authors' reported letter distribution
  (25.12 / 25.32 / 24.92 / 24.64%).
- **rafdb single.** "Happy" is correct in 37 of the 67 items that offer it, so the option-set predictor reaches .428
  against a quantile of .365 (raw p .003).
- **affectnet multi-image.** Counting questions where "1" is correct in 44 of the 58 items that offer it give .645
  against a quantile of .377.
- **What this means.** These are content and label priors from the source datasets, not a construction artefact
  like ours. Neither passes Holm, which floors at .196 here, but both effects are well beyond the null quantiles.
- **What the authors reported.** Random-choice and frequent-choice baselines (25.10% and 26.68% overall). They report
  no text-only baseline.

**MMBench image_emotion: no leak once rotation copies are removed.**
- **The trap.** The lmms-lab dev file stores each question four times, with options rotated for CircularEval.
  A first pass that left the copies in reached .54 on the option-set predictor, purely because each item's own
  copies sat in the same option set. After keeping one copy per question (n = 50), the predictor scores .288 against
  a null mean of .248 (p .37).
- **Positions.** Base-order positions are mildly uneven ([18, 9, 17, 6], p .037, Holm 1.0). The official
  CircularEval removes this anyway.
- **Lesson for readers.** Anyone auditing MMBench-derived files must deduplicate CircularEval copies first.

**MMStar and MMT-Bench: small subsets, no clear leak.**
- **MMStar** (n = 31). Positions are balanced (p .28). The option-set predictor reaches .452 against a quantile of
  .419 (raw p .013). That comes from repeated templated sets: "happiness, sadness, anger, loneliness" appears on
  3 items, and the answer is sadness on all 3.
- **MMT-Bench** (n = 20 per task). No position effect. Artwork emotion (option set .55, prior .70) and
  micro-expression (option set .575) exceed their null quantiles at raw p .001-.011. With 20 items each, and none
  passing Holm, these are flags, not findings.

**Fixed-list benchmarks (FaceBench, MMAFFBen, EmoBench-M): no option-set leak by construction.**
- **No option-set leak.** Every item shows the same label list, so the list carries no item-specific information.
- **Class imbalance instead.** "Position" here is the label, and the skew is class imbalance in the source data. A
  text-only majority-class answer beats the nominal 1/k chance by a wide margin: FaceBench RAF-DB .386 vs .143,
  FER2013 .247 vs .143, CFAPS .287 vs .143, EmoBench-M MOSI .608 vs .333, FGMSA .516 vs .200, SIMS .543 vs .333,
  ch-simsv2s .474 vs .333, mer2023 .256 vs .167.
- **Balanced sets.** RAVDESS song and speech, MOSEI, MUSTARD and funny are balanced. MELD and MC-EIU are close to
  balanced.
- **Why it matters.** The imbalance matters only where a paper uses 1/k as its "random" floor. What EmoBench-M uses
  as its random baseline is UNVERIFIED here.
- **A text channel not tested.** EmoBench-M prompts also contain the spoken transcript, a separate text channel
  this audit does not test.

**EEmo-Bench (evoked affect, not faces): small priors only.**
- **Positions.** Yes/no items are balanced. The What/How valence and dominance items show mild position skew
  (p .028 and .040, Holm 1.0).
- **Arousal What/How.** "High arousal" is correct in 52 of the 77 items that offer it, so the answer prior reaches
  .63 against a quantile of .44.
- **Emotion What/How.** The option-set predictor is barely above its quantile (.379 vs .370).

**SEED-Bench keyword subset: positions balanced.**
- **The prior.** The answer prior reaches .57 against a quantile of .35. "Smiling" is correct in 12 of the 16 items
  that offer it and "happy" in 9 of 15, while "angry" is correct in only 2 of 26.
- **Caveat.** This is a valence prior in image content, on a heterogeneous post hoc keyword subset.

## Prior reports

- **MMBench.** Its authors introduced CircularEval because answers can depend on option order (Liu et al., 2023,
  arXiv 2307.06281), so position bias there is known and handled.
- **MMStar.** Built because many items in earlier benchmarks could be answered without the image (Chen et al.,
  2024, arXiv 2403.20330). That is a general text-answerability critique, not specific to affect items.
- **Face-Human-Bench.** The paper states that options are shuffled to balance answer letters, consistent with
  `eval.py`. The paper does not describe the raw-JSON ordering or the dev answer-pool imbalance.
- **FaceXBench.** The paper reports balanced answer letters and random and frequent-choice baselines.
- **Not found.** Within a limited search (one web search plus the two papers above) I found no report of the
  HumanVBench 5th-option pattern, the Face-Human-Bench dev label pool, or the option-set analyses here. That is
  "not found", not "never reported".

## Verdict

Our specific confound is not a field-level pattern. It combines a constant answer position with one option set per
target label. None of the 11 public benchmarks audited reproduces both as their official code presents items.
Face-Human-Bench stores answers first, but its official pipeline rotates them.

Weaker text-only shortcuts are common, though:
- a structural position leak in HumanVBench (5-option items always answer E);
- answer-pool and label priors in Face-Human-Bench dev, FaceXBench, SEED, EEmo and small MMStar/MMT subsets;
- class imbalance in fixed-list benchmarks, which lets a text-only majority guess beat the nominal chance rate by
  9-32 points.

The transferable lesson is methodological. Report text-only baselines (position, option set, answer prior,
majority class) next to 1/k. Deduplicate rotation copies before any leave-one-out check. It is not a claim that
affect benchmarks are broadly invalid. Balanced results exist and are reportable here: FaceXBench positions,
MMBench after deduplication, Face-Human-Bench positions as presented, RAVDESS, MOSEI and the EEmo yes/no items.

## Limitations

- **Small subsets.** Many subsets are small (20-50 items), and the Face-Human-Bench rows are not independent.
- **p-value floor.** Option-set and prior p values are floored at 1/1001 and cannot pass Holm over 216 tests, so
  those conclusions rest on effect size against the null quantile.
- **Simple predictors.** The predictors are simple. Question stems, transcripts and captions were not used, so
  text-only accuracy here is a lower bound on what a language model could exploit.
- **Mirrors.** The MMBench data came from the lmms-lab mirror, not the OpenCompass TSV. The MME category check used a
  mirror.
- **Run-time order.** Official run-time option order was verified for Face-Human-Bench, FaceBench and MMBench only.
- **Licences.** SEED-Bench is CC BY-NC 4.0. MMBench_EN, MMStar and EEmo-Bench carry no licence tag on HF. Only
  annotation text was read, and only counts and short label strings are kept in the output.

## Reproduce

```
/opt/anaconda3/envs/mr_eu_open_llm/bin/python research/scripts/benchmark_leak_audit.py
/opt/anaconda3/bin/pytest tests/test_benchmark_leak_audit.py
```

The first run downloads roughly 0.8 GB over the network (mostly the streamed MMT-Bench TSV, never written to disk).
It keeps about 50 MB of annotation text in scratch, and later runs reuse the cache. The null simulations dominate
the run time.
