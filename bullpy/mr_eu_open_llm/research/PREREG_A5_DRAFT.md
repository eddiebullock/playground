# DRAFT amendment A5: prior and evidence on the blind readout (RQ4)

**Status: DRAFT. Not frozen, not tagged, not preregistered.** Nothing here binds until it is audited, committed and tagged. It must freeze before any full A2 cell is read.

Draft `design_sha256` of the block below: `bd9003d8b536468c1e7a0abfd2e672f1db7af749623d0b8ba294bc960faaaf95`. It changes with any edit to the block and is restated at the freeze.

**Written:** 2026-11-07, before any A1 result was read, before any full A2 cell exists, and before any A5 quantity was computed on model output.
**Goal:** G6 in `research/GOALS.md`.
**Depends on:** A2's run (G4). RQ0 is read through the frozen A1 and A3 table by A2's `rq0_status`, per model and route. Gates G2 to G4 as in A2.
**Models.** Gemma 4 and Qwen3-VL, each on `single_face` and `video_face`. Molmo2 is out under A2, so it has no cells here.
**Code, not yet written:** `scripts/a5_design.py`, `scripts/a5_analyse.py`, `scripts/a5_recovery.py`, `tests/test_a5.py`. `a5_design.py` parses the design block from this file and refuses a hash mismatch. The analysis imports A2's frozen `load_shards`, `gate_b`, `rq0_status`, `human_comparable`, `js_divergence`, `signflip_maxt` and bootstrap functions, and edits none of them.

**What was known when this was written.**

- No A1 result, and no full A2 cell.
- A3's result as stated in `STATE.md`: all three models pass on `single_face`.
- The smoke_v2 mechanics in A2's appendix on 3 items, including lexicon mass on image-free and grey inputs. No contrast was computed from them, and no family scores were opened for this draft.
- The EU-Emotion human norms.
- CREMA-D votes, for H3 only, with no CREMA-D model output.

A scratch run on synthetic observers, using the real 243 labels and actors and no model output, checked the identity below to 1e-9, and showed that Even and Absent are reachable at 17 actors. It is a code check, not recovery.

## Question

RQ4. When a model judges a face on the blind readout, how much of that judgement is fixed in advance by words, by the prompt or by seeing any face, and how much comes from this face? Does the face part carry the expression, and does the balance differ from humans?

**Scope.** Causal language applies to the model only. Evidence here means information that arrives through the pixels. It does not mean perception as a human has it, since a face can trigger stored word associations. RQ1 and RQ2 test which cues are used.

**Sources.** Separating evidence from bias on VLMs has been done outside emotion: Barua et al. (arXiv 2610.19407, partial read to char 100,000) fit evidence and bias on conflict items, and Liu et al. (ECCV 2026, abstract) define a vision against language sensitivity on depth cues. A grey-image control on faces exists in Hallmen et al. (arXiv 2610.08162, supplement Sec. S14, partial read of the supplement). No emotion version was located (`prior_ai_human_comparison.md`, searched 2026-11-04). Unlike a fitted model, the split below is an identity, so it needs no functional form.

## Why a reanalysis, and what the data are

A5 needs no new forward. All inputs are A2 outputs, and three facts about them shape the design:

- **The blind prompt names no item.** So a mismatch needs no cell of its own. A face's score for any family is already in that face's `original` cell. Gate B's mismatch is exactly this reading.
- **`image_free` and `grey` exist once per route.** They are rows in each shard's run report, not per-item cells, and they hold family scores but not word scores. Analysis checks that they agree across shards within the tolerance in the design block, or the route fails its baseline check.
- **`image_free` keeps its placeholder tokens** (`strip_pixels` drops pixel tensors only). It is a no-pixels input, not a text-only input. A2 lists this as a limit, and Qwen's mass is route-dependent on it.

Only `single_face` and `video_face` are used. `single_full` and `legacy_full` have no valid baseline or Gate B.

## Design block

All fixed constants are here and nowhere else. The hash is sha256 of the bytes between the fence lines, excluding the fences, with a final newline.

```json
{
  "amendment": "A5",
  "a2_design_sha256": "0ae22118d3b2690c581595890b9bb8b904e1f7860f37be9ac909f925b7058de7",
  "a2_code_sha256": {
    "scripts/stimulus_design.py": "89d256c1072e0768fc747294553c5710adaf42184a175866d7bd75bcc6104978",
    "scripts/stimulus_interventions.py": "93c601eb42d6700d4f0fc8ba72317ed5c2e354105d57fcac611c59660125deb3",
    "scripts/run_stimulus_interventions.py": "f7880c94e9671cc294cd4eaeece2108d7cd1d7a4c7e6f9e30a1c495964e6cd82",
    "scripts/analyse_stimulus_interventions.py": "aba4f533daad6210e87f7bb4e68fc4627805cb4dfd9d9308df734fada4edc8b1"
  },
  "a3_summary_sha256": "c1c64e6975b32356a3314268c054f0dee483ddf74b11cd66d03b2faf972e64a4",
  "models": ["gemma4", "qwen3vl"],
  "routes": ["single_face", "video_face"],
  "n_items": 243,
  "n_actors": 17,
  "n_families": 21,
  "seed": 42,
  "alpha": 0.05,
  "ci_decision": 0.95,
  "ci_equivalence": 0.90,
  "bootstrap_draws": 10000,
  "label_shuffles": 1000,
  "margin_fraction": 0.10,
  "lambda_margin": 0.10,
  "lexicon_words": 106,
  "token_floor": 8.0,
  "min_total_nats": 0.05,
  "baseline_agreement_nats": 0.001,
  "grey_mass_floor": 0.05,
  "min_correct_items": 10,
  "min_correct_actors": 5,
  "word_share_cut": 0.5,
  "survival_cut": 0.5,
  "temperature_sensitivity": [0.5, 2.0],
  "weak_labels": ["Kind", "Unfriendly", "Jealous"],
  "recovery": {
    "replicates": 500,
    "expected_verdict_min": 0.80,
    "other_named_verdict_max": 0.10,
    "inconclusive_max": 0.20,
    "phi_grid": [0.1, 0.3, 0.5, 0.7, 0.9],
    "seed": 5042
  }
}
```

`margin_fraction`, `lambda_margin`, `min_total_nats`, `word_share_cut` and `survival_cut` are **[proposed]**, with the audit to confirm them.

## Prior and evidence, defined

Everything below is per model m and route r, and is never pooled across models or routes.

**Judgement.** For clip i on `original`, q_i is the 21-family distribution: each family's joint probability, renormalised over the lexicon. So q_i is conditional on the answer being in the lexicon. Lexicon mass is reported beside every quantity.

**Baselines.** w is the same distribution on the `image_free` row, and g on the `grey` row. Each is one fixed vector for that m and r.

**Default.** For clip i of actor a, q_def is the mean of actor means over all actors except a. It is the model's judgement of a face in general, with no contribution from this actor.

**Template.** q_tpl is the mean over clips with i's acted label from actors other than a. It is what other people's portrayals of this label produce. Label needs no mapping to a word here.

**Identity.** Writing log for the natural log, sums for sums over the 21 families, and u = 1/21:

    T_i = sum q_i log(q_i / u)                 information in the judgement
    W_i = sum q_i log(w / u)                   word prior
    G_i = sum q_i log(g / w)                   any image
    F_i = sum q_i log(q_def / g)               any face
    E_i = sum q_i log(q_tpl / q_def)           expression-linked evidence
    R_i = sum q_i log(q_i / q_tpl)             residual: this clip, this actor

    T_i = W_i + G_i + F_i + E_i + R_i          exactly, for every clip

The sum holds by cancellation, so nothing is fitted. Two groups matter:

    P_i = W_i + G_i + F_i = sum q_i log(q_def / u)      the default part
    D_i = E_i + R_i       = sum q_i log(q_i / q_def)    the face part, a KL divergence, so at least 0

**What this split can and cannot show.**

- A fixed prior cannot raise mean accuracy across balanced labels on a blind readout, because a fixed vector is right only for the labels it favours. So the prior acts on which faces get which words, and on how little the answer varies. Accuracy is therefore not the estimand.
- P is not all word knowledge. W is what no pixels produce. G and F are what an uninformative image and a generic face add. The word share below separates them.
- D is not all perception. R holds identity, clip and pixel effects. E holds what generalises across actors for a portrayed label, and it is held out, so identity cannot reach it.
- Both are on the model's own probability scale, so a sharper softmax changes them. Temperature is a stated sensitivity.

## Estimands

Clip values are averaged within actor, then across actors, so each actor counts once.

**Primary 1: balance.** B = mean over actors of (D minus P). Equivalent reading: Phi = D / T, the face share of the judgement, and B / T = 2 Phi - 1. B above 0 means the face moves the judgement further than the default does.

**Primary 2: expression-linked evidence.** E* = mean over actors of E minus the mean of E over `label_shuffles` shuffles of labels within actor. Under no label signal, E has a small negative expectation, so it must be centred. Shuffling within actor keeps each actor's label set and the partial crossing.

Neither is B4 or B5. Those test whether the acted family's word rises. E* uses no word and no family named for the label, and B uses no label at all. A test must show a synthetic observer that passes E* and fails B4, and one that passes B4 with B below 0, so that A2's audit fault of a check duplicating another cannot recur.

**Riders, always reported, with actor-bootstrap 95% CI.**

| Rider | Definition | Use |
|---|---|---|
| Word share | omega = W / P, when P > 0 | Above `word_share_cut` with CI excluding it: the default is word prior. Below with CI excluding it: the default comes from seeing an image or a face. Otherwise mixed |
| Chain | Means of W, G, F, E, R, and lexicon mass on real, grey and image-free | When grey mass is under `grey_mass_floor`, G and F are reported only as their sum, since g then rests on very little mass. B and E* do not use w or g |
| Right-answer survival | C is the clips whose top family is the acted one. S = mean of q_d(i)(t_i) over C, divided by the mean of q_i(t_i) over C. d(i) is the Gate B partner, seed 42 | The share of probability on right answers that remains when the face is wrong. This is the literal RQ4 wording. Not estimated with under `min_correct_items` clips or `min_correct_actors` actors. S at or above `survival_cut` beside an Evidence-led verdict is reported as a conflict |

The mismatch baseline enters through B4, through S, and through the positive control. A swap has no place in the identity, since a donor's face is not a prior.

## Inference

- **Unit.** The actor, 17 actors. Models answer with no sampling noise, so stimulus sampling across actors is the only source of error.
- **Test.** One matrix per analysis, 17 rows, with one column for B (two-sided) and one for E* (one-sided, above zero) for each eligible model and route. Signs flip by whole rows. All 2^17 flips are enumerated by `signflip_maxt`, with Westfall-Young step-down maxT at familywise alpha .05. Maximum 8 columns. Models are not pooled; they share a correction only.
- **Floors.** With 17 actors the one-sided floor is about 8e-6, so the actor count does not limit these tests.
- **Interval.** Actor-bootstrap, `bootstrap_draws` draws, seed 42. 95% for direction, 90% for equivalence.
- **Dependence.** q_def and q_tpl use the other actors, so actor rows are slightly dependent. Recovery measures the effect on error rates.
- **Partial crossing.** E uses templates from the same label's other actors, so it stays inside one actor set (A2, Inference). Any contrast across the two actor sets is descriptive.
- **Margin [proposed].** 10% of that model and route's mean T, so B is within margin when Phi is in .45 to .55. It is fixed by T, which involves no label and no direction. A2's disclosed fault of a margin taken from another outcome does not arise.

## Classes and verdict

**Evidence class.**

| Class | Requires |
|---|---|
| Shown | E* passes (step-down p < .05, 95% CI lies above 0), and Gate B passed |
| Unmapped | E* passes, and B4 or B5 failed. Portrayals move the judgement consistently, but not to the acted word |
| Absent | E* not significant, its 90% CI inside the margin, B4 failed, and the positive control holds |
| Unresolved | Anything else |

The positive control for Absent is that face information can reach this model's output: RQ0 passed on this route and G4 passed. The control is not a magnitude taken from noise, which was A2's dose fault.

**Balance class.**

| Class | Requires |
|---|---|
| Face | B above 0, step-down p < .05, 95% CI lies above 0 |
| Default | B below 0, step-down p < .05, 95% CI lies below 0 |
| Even | Neither significant, and the 90% CI inside the margin. This is the null on balance |
| Unresolved | Anything else |

**Decision table.** Applied per model and route. All verdicts require the preconditions in the next section.

| Evidence | Balance | Verdict | Reading |
|---|---|---|---|
| Shown | Face | **Evidence-led** | The judgement depends on this face more than on any default, and carries the expression |
| Shown | Default | **Default-led with evidence** | Expression gets through, but the fixed part dominates. The word share names its source |
| Shown | Even | **Even** | Evidence present, and neither part dominates |
| Unmapped | Face | **Face-led, unmapped** | Portrayals are read consistently, with the wrong words |
| Unmapped | Default or Even | **Default-led, unmapped** | As above, with the fixed part at least equal |
| Absent | Default or Even | **No face evidence** | Null on evidence. With a word share above the cut, this is the word-knowledge result |
| Absent | Face | **Face-dependent, no expression** | Null on evidence. The answer varies with the face, but not with what it portrays: identity or clip properties |
| Unresolved on either axis, in any combination | | **Inconclusive** | With a reason code, from the list below |

Every result also carries its word share and survival rider, and is read against the human comparison where one exists.

## Preconditions and Gate B

A5 follows A2's rule that a missing gate has not passed. But A2 stops RQ1 and RQ2 at a failed Gate B, and it does not stop RQ4, because a failure there can be the RQ4 answer. So Gate B is used by part.

| State | A5 result |
|---|---|
| RQ0, G2 to G4 or delivery not passed for this model and route | **Not tested**. Quantities computed for the record and carry no verdict. This covers Molmo2 and any model with no A2 cells |
| B1 or B2 fails | **Not tested**. The readout is not measured on this route: deterministic scoring or coverage failed |
| Baseline rows missing or disagreeing across shards | B and E* stand, since neither needs w or g. Word share and chain are not reported, and the verdict carries `no_baseline` |
| B3 fails | Decision table applies. Images barely move the readout, so a Default result or Absent is expected. `b3_failed` is attached |
| B3 passes, B4 or B5 fails | Decision table applies. Evidence can be Unmapped or Absent, never Shown |
| All of B1 to B5 pass | Decision table applies. Evidence can be Shown |

**So a failed Gate B has a defined result.** It cannot produce Evidence-led or Default-led with evidence. It can produce Face-led unmapped, Default-led unmapped, No face evidence, or Face-dependent no expression. If every eligible model fails Gate B on both routes, RQ4 is answered on the blind readout by those verdicts, and any reading from the MCQ stays exploratory, as in A2.

A failed Gate B beside an RQ0 pass means the face reaches the MCQ answer and does not reach the blind readout in this form. That is reported as readout dependence, and it is not called proof that the model lacks perception.

## Null and Inconclusive

There are two distinct nulls, and neither is a default.

**Null on balance (Even).** All of: preconditions met, neither B test significant, 90% CI inside the margin, and T at or above `min_total_nats`.

**Null on evidence (No face evidence, or Face-dependent no expression).** All of: preconditions met, E* not significant, its 90% CI inside the margin, B4 failed, the positive control holds, and the same floor on T.

**Inconclusive.** Exactly these reason codes: `evidence_unresolved` (E* not significant with its CI outside the margin, or B4 and E* disagreeing in a way no row covers), `balance_unresolved` (B not significant with its CI outside the margin), `no_structure` (T under the floor).

**Inconclusive cannot be the default outcome here, by construction.**

1. Every verdict row is reachable with a model's real inputs. All of the Null rows can be reached on both routes and on both models, so no quantity is capped in the way 8 of A2's 16 contrasts are.
2. A verdict needs only A2 outputs, which exist for every eligible cell, plus RQ0 and the gates. The gate loader must return True on a fixture built from the real pinned paths, with a test that runs the verdict function end to end. This guards against A2's second audit finding, in which the A3 path made every verdict Inconclusive.
3. Recovery must reach every named verdict and hold Inconclusive under `inconclusive_max` for each observer not placed on a boundary. If this fails, the margin or the rule is revised before freezing and dated, not after.
4. A failed gate is never by itself a cause of Inconclusive. It gives Not tested or leads to a row above. An output row with no verdict or reason code is a code fault and stops the analysis.

## Comparison with humans

No human saw a blank, grey or swapped face in any dataset here, so W, G, F and any human "prior" do not exist. **A5 never states that a model relies more on word knowledge than humans do.** It compares only these.

| ID | Quantity | Human data | Status |
|---|---|---|---|
| H1 | Delta J: Jensen-Shannon divergence to the human distribution for the default, minus the same for the full judgement, on the item's named options, None of the above removed from both (`human_comparable`) | EU-Emotion norms, 57 to 299 respondents per item | Valid. Actor-level sign-flip with own step-down family, two-sided. Above 0: this face moves the judgement towards people, beyond the default. The template form is reported beside it |
| H2 | Within-label correlation of the model's target share with human p_target, as in A1 | EU-Emotion norms | Valid. Actor-bootstrap CI, descriptive. No noise ceiling, since no rater-level data are held |
| H3 | Lambda = E / (E + P) on a fixed option set: humans from votes, models on the same 819 clips, on MCQ and on the blind readout restricted to six families | CREMA-D votes, from A6's cells | Valid formula. Only if A6 reaches Stage 1. Paired actor-bootstrap 95% CI of the difference, 91 actors, with `lambda_margin` as its equivalence bound. Else reported as not tested |

**H3 details.** E and P are linear in q_i, so clip-level rating noise from about 10 raters does not bias them, whereas D would be biased by it, and so H3 does not use D or R. A human "default" is response bias and clip mix, not word knowledge, so H3 says only that the same formula gives the same or a different balance. The human side is fixed before any A6 model cell exists.

**Flagged items and scoring.** H1 and H2 are repeated without the 26 flagged items and against the human modal label, per A2. Model accuracy well above human on those items is read as possible leakage.

## Sensitivities

All are coded and reported beside the primary, and none changes a verdict.

- Excluding the 3 smoke items.
- Excluding the 26 flagged items.
- E* with templates grouped by the human modal label.
- Clips with at least `token_floor` tokens across the face.
- Canonical word only, for T, P, D, E and R. Not for W and G, because baseline rows hold no word scores.
- Flat reference by family size, u_f = words in family / `lexicon_words`, to bound family-size bias.
- Temperature: q raised to each `temperature_sensitivity` value and renormalised, repeating B and E*.
- The default computed with all actors, to show the effect of leaving one out.
- Basic labels only, and complex labels only, labelled descriptive. Family-level tables of W, E and R by label, with partial crossing noted.

## Recovery

Before any confirmatory read, synthetic observers go through the same code, including A2's `gate_b`, on the real 243 items and 17 actors. Replicates and pass rule are in the design block. The noise has a shared actor part and correlated families, not only independent cells, because A2 disclosed that its recovery used independent noise and large effects, and so showed mechanics, not power.

| Observer | Construction | Expected verdict |
|---|---|---|
| Prior-only, word | Fixed vector, image-free equal to it, small noise | No face evidence, word share high |
| Prior-only, face default | Fixed vector on faces, flat image-free | No face evidence, word share low |
| Evidence-only | Flat default, label signal to the acted family | Evidence-led |
| Mixed, face heavy | Phi near .7 and .9, signal to the acted family | Evidence-led |
| Mixed, prior heavy | Phi near .1 and .3, signal to the acted family | Default-led with evidence |
| Even | Phi near .5, signal to the acted family | Even |
| Identity | Default plus actor shift, no label signal | Face-dependent, no expression |
| Noise | Default plus large item noise | Face-dependent, no expression |
| Unmapped, face | Consistent label shift to the wrong families, Phi near .8 | Face-led, unmapped |
| Unmapped, default | The same at Phi near .2 | Default-led, unmapped |
| Spread | Label signal with fixed spill to neighbouring families | Evidence-led, showing spill is not separated from evidence (a limit) |
| Boundary | Phi near .45 and .55, E* near the margin | Reported, not judged |

An observer on `phi_grid` is scaled by Phi, and the signal strength is varied, so the report gives power at realistic sizes as well as at large ones. Pass rule: for each non-boundary observer, the expected verdict in at least `expected_verdict_min` of replicates, any other named verdict in at most `other_named_verdict_max`, and Inconclusive in at most `inconclusive_max`. Every named verdict must be reached at least once.

If an observer fails, the failure is reported. Any change comes before freezing, is dated, and is not made after any A2 cell is read.

## Limits

- Lexicon conditioning hides out-of-lexicon answers. Grey mass was .02 and .08 on the 3 smoke items, so g can rest on little. B and E* do not need it, but the chain does.
- `image_free` keeps placeholders, and what a model does with them differs by model. W is the best no-pixels estimate available, not a pure word prior.
- Family sizes differ, so W holds some lexicon structure. The flat-by-size sensitivity bounds this.
- The 106-word lexicon sums have no word-boundary check (A2, disclosed).
- E cannot separate perceiving an expression from a pixel pattern that happens to go with a label in this stimulus set, such as how a given label was filmed. It holds across actors, not across stimulus sets.
- Spill from a perceived category to neighbouring words sits inside E. A text-only control would address this, but none is in A2's outputs. A6's text primes give the only such measure, for six labels.
- Phi depends on softmax scale, so it is not comparable across models as a magnitude. Verdicts within a model are the claim.
- One clip gives one judgement, and 17 actors bound everything.
- No human prior exists. "Differs from humans" is limited to H1 to H3.

## Inversion: not included, costed

RQ4's text proposes inversion as an extension. It is left out of A5, for three reasons.

- It is not a reanalysis. A2's frozen code has no inverted condition, so it needs a new runner.
- It does not bear on this split. Inversion leaves w and g unchanged and keeps every pixel, so it tests holistic processing, not prior against evidence.
- The human target is thin. The located direction for basic labels is in McKelvie 1995 and Psalta and Andrews 2014 (both abstract level, `human_mechanisms.md`), with no located inversion study on complex states or EU-Emotion. Strachan et al. (read in full, arXiv version) concerns eye-region items.

**Cost if it is added later.** One `inverted` cell on `single_face`, 243 items: Gemma 0.22 and Qwen 0.36 GPU-hours from A2's table, so 0.59 at both readouts, or 0.31 blind only. It would need its own amendment before data, with predictions marked BASIC ONLY.

## Cost

| Item | GPU-hours |
|---|---|
| A5 analysis, all models and routes | 0 |
| Recovery | 0, CPU only |
| H3 | 0 beyond A6's Stage 1 |

## Fixed before freezing

1. Audit of this text, then code, then recovery, with outputs under `research/reproduction/a5_recovery.json`.
2. Confirm the **[proposed]** values, and record the resulting `design_sha256`.
3. Freeze before any full A2 cell is read. If A2's cells have been read for any purpose first, A5 becomes exploratory and says so.

## Faults from A2's audit, and how A5 avoids them

| A2 fault | Here |
|---|---|
| B5 duplicated B4 | B and E* are shown by test not to be B4 or B5 |
| Null rule uncoded or unexecutable | Two null rules, each from named outputs, coded and reachable |
| Verdicts ignored gates | Gates are inputs; missing means not passed |
| Paths that made every verdict Inconclusive | End-to-end test on the real pinned paths |
| Dose passed on noise | The positive control is RQ0 and G4, not a magnitude |
| Most sensitivities not run | Listed, all coded |
| Constants outside the hash | One hashed block, parsed by the code |
| Recovery showed mechanics only | Graded effects, actor noise, 500 replicates |
| Margin from another outcome | Margin from T, the outcome's own scale |

## Read status of sources

F full text read, P partial, A abstract only, per `research/paper/v2/references.md`, `citation_check.md` and the notes named.

| Source | Status | Used for |
|---|---|---|
| Barua et al., arXiv 2610.19407 | P, to char 100,000 | Fitted evidence and bias, not emotion |
| Liu et al., ECCV 2026, arXiv 2607.01503 | A | Vision against language sensitivity, depth cues |
| Hallmen et al., arXiv 2610.08162 | F main, P supplement | Grey-image control |
| Strachan et al., CHB 2026 | F, arXiv version | Inversion on eye items |
| O'Reilly et al. 2016 | F, supplement not obtained | EU-Emotion norms |
| Cao et al. 2014 | A; data read directly | CREMA-D |
| McKelvie 1995; Psalta and Andrews 2014 | A, in notes only | Inversion direction |
