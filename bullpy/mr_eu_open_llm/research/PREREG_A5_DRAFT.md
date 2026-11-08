# DRAFT amendment A5: prior and evidence on the blind readout (RQ4)

**Status: DRAFT. Not frozen, not tagged, not preregistered.** Nothing here binds until it is audited, committed and tagged. It must freeze before any full A2 cell is read.

Draft `design_sha256` of the block below: `fd20495bf411b97bb876a32b0d93994b98bcc73a05aa104c6976f3fe2988efd4`. It changes with any edit to the block and is restated at the freeze.

**Written:** 2026-11-07, before any A1 result was read, before any full A2 cell exists, and before any A5 quantity was computed on model output.
**Goal:** G6 in `research/GOALS.md`.
**Depends on:** A2's run (G4). RQ0 is read through the frozen A1 and A3 table by A2's `rq0_status`, per model and route. Gates G2 to G4 as in A2.
**Models.** Gemma 4 and Qwen3-VL, each on `single_face` and `video_face`. Molmo2 is out under A2, so it has no cells here.
**Code:** `scripts/a5_design.py`, `scripts/a5_analyse.py`, `scripts/a5_recovery.py`, `scripts/a5_family_diagnostic.py`, `scripts/a5_checks.py`, `scripts/a5_dependence_diagnostic.py`, `scripts/a5_skew_diagnostic.py`, `scripts/a5_gain_guard_diagnostic.py`, `tests/test_a5.py`. `a5_design.py` parses the design block from this file and refuses a block whose hash differs from the value pinned in the code or stated above. The analysis imports A2's frozen `load_shards`, `gate_b`, `rq0_status`, `human_comparable`, `js_divergence`, `signflip_maxt` and bootstrap functions, and edits none of them.

**What was known when this was written.**

- No A1 result, and no full A2 cell.
- A3's result as stated in `STATE.md`: all three models pass on `single_face`.
- The smoke_v2 mechanics in A2's appendix on 3 items, including lexicon mass on image-free and grey inputs. No contrast was computed from them, and no family scores were opened for this draft.
- The EU-Emotion human norms.
- CREMA-D votes, for H3 only, with no CREMA-D model output.

A scratch run on synthetic observers, using the real 243 labels and actors and no model output, checked the identity below to 1e-9, and showed that Even and Absent are reachable at 17 actors. It is a code check, not recovery.

A 40-replicate pilot of `a5_recovery.py` ran on 2026-11-07 and finished at 17:14, on all 39 observers, with synthetic data only. It was run to check mechanics and runtime. Its block differed from the current one only by an unused H3 pseudocount, since removed. Its summary printed the names of three failed checks (`false_positive|all_null_family|any_B_column`, `observer|mixed_phi0.5_signal2.0`, `observer|mixed_phi0.7_signal2.0`), and these were seen. Nothing in the block, a threshold or an observer was changed after it. The recovery parameters were written before it ran, and were accepted by the study lead as written before any rate was seen.

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
  "dominance_ratio": 2.0,
  "min_balance_cells": 4,
  "gain_heterogeneity_min_z": 2.0,
  "gain_heterogeneity_floor": 1e-9,
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
    "false_positive_max": 0.07,
    "phi_grid": [0.1, 0.3, 0.5, 0.7, 0.9],
    "signal_grid": [0.25, 0.5, 1.0, 2.0],
    "pass_signal": 2.0,
    "unmapped_signals": [1.0, 2.0],
    "seed": 5042,
    "companion_cells": 3,
    "companion_sweep": [0, 1],
    "companion_shuffles": 100,
    "actor_sd": 0.3,
    "item_sd": 0.5,
    "portrayal_gain_log_sd": 0.3,
    "gain_sweep": [0.0, 0.5],
    "gain_sweep_phi": [0.1, 0.5, 0.9],
    "family_factors": 3,
    "family_factor_share": 0.4,
    "prior_only_scale": 1.5,
    "prior_only_noise": 0.1,
    "label_free_prior_scale": 0.3,
    "identity_shift": 1.5,
    "noise_item_sd": 1.8,
    "spill_share": 0.5,
    "spill_neighbours": 2,
    "boundary_phi": [0.3334, 0.6666],
    "false_even_phi": [0.32, 0.68],
    "false_even_signals": [1.0, 2.0],
    "inside_even_phi": [0.4, 0.6],
    "all_null_replicates": 2000,
    "all_null_shuffles": 1000,
    "boundary_e_share": 0.10,
    "baseline_noise": 0.05,
    "calibration_draws": 200,
    "balance_zero_calibration_draws": 10000,
    "shared_family_gains": [0.3, 0.5]
  }
}
```

`margin_fraction`, `dominance_ratio`, `lambda_margin`, `min_total_nats`, `word_share_cut` and `survival_cut` are **[proposed]**, with the audit to confirm them. `dominance_ratio` and the balance test were set after recovery runs 1 and 2 (see the two sections on changes after recovery).

## Code hashes

A5's own code and the helpers it imports, sha256, checked by `a5_analyse.py` and `a5_recovery.py` before they run. They are pinned here, outside the block, because `a5_design.py` holds the block's hash.

```text
scripts/a5_design.py 44ebd027a8dcf41cbe553896564d35d3eee82ffa7605ec1fc64667829b015b58
scripts/a5_analyse.py e726c03adb5e6110d26e92e2578f9eb0fddb7240479033c34b0e4a69d5702e0e
scripts/a5_recovery.py 56d9d195238d50195b9f3ed5919ad6c7e904bde39633e74c34f93f5e34c3f7e8
scripts/a5_family_diagnostic.py 06c0a372a3876f749b5baddacdb6f06bed19fcb3d1f9b46e3eeb5e5103989da3
scripts/a5_checks.py b38e05fdb26ddc171623eda7f999463e0856938cfe37cd8827fc89cb02fb6b5e
research/scripts/analyse_rotation.py 98890c5bff65f13abeb0549b61fe414dc9fa30082b84bcb564671683ec12e24f
src/paper_v2/option_score.py ae949e4c1c3abc6a775a535c4e0a5042bce59ba14f7ac1afeefd476f032a5a2b
src/paper_v2/schema.py 4acc626669439384115ae0efcb52cec4edee02c07662a242212d5f7b9c359b47
src/paper_v2/metrics.py 1fc15676cae999e5185cfd016e3dd9db616c71ec0bb4c5d38cd49a4e7b886568
scripts/a5_dependence_diagnostic.py 6f8ea78a57c46e0604c567cc1aadba805a0a1a52d79170e7c73c9c2496ba8d41
scripts/a5_skew_diagnostic.py f7493fd4acb978fa988babd64ba066b16d0fa6de9ca8c2b47b53dad095b9ed90
scripts/a5_gain_guard_diagnostic.py 7b6393a4f404ec7afdbe3c98d4813aeeb27e4184cfc9fb1c40f00fa6d59ae31f
```

## Prior and evidence, defined

Everything below is per model m and route r, and is never pooled across models or routes.

**Judgement.** For clip i on `original`, q_i is the 21-family distribution: each family's joint probability, renormalised over the lexicon. So q_i is conditional on the answer being in the lexicon. Lexicon mass is reported beside every quantity. A failed `original` cell is absent from every mean here and is counted.

**Baselines.** w is the same distribution on the `image_free` row, and g on the `grey` row. Each is one fixed vector for that m and r. The first shard's row is used once the agreement check passes, as in A2's own Gate B call, which also feeds B3 from the first shard.

**Default.** For clip i of actor a, q_def is the mean of actor means over all actors except a. It is the model's judgement of a face in general, with no contribution from this actor.

**Template.** q_tpl is the mean over clips with i's acted label from actors other than a. It is what other people's portrayals of this label produce. Label needs no mapping to a word here. On the 243 items every label has at least 7 actors, so every clip has a template. Under `E* with templates grouped by the human modal label` a clip may not, and such a clip leaves E and R, stays in T, P and D, and is counted.

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
- Only D and T are bounded below by 0. P and W can be negative, so Phi is a ratio of signed means, not a bounded share. Leaving out the clip's actor is what does this. With a flat default, q_def is the overall mean minus a share of this actor's own deviation, so it leans away from the clip. On a synthetic observer with no prior (signal 2, 200 draws; `a5_checks.json`, section `phi_above_one`): leaving the actor out gives P = -.006 and Phi = 1.012, with Phi above 1 in every draw. Using all actors gives P = +.006 and Phi = .988, never above 1. Partial crossing is not the cause: labels reassigned at random across all clips gave the same result (P = -.007, Phi = 1.014, and +.006, .988). The effect is about 1% of T here.
- Both are on the model's own probability scale, so a sharper softmax changes them. Temperature is a stated sensitivity.

## Estimands

Clip values are averaged within actor, then across actors, so each actor counts once.

**Primary 1: balance.** B = mean over actors of (D minus P). Equivalent reading: Phi = D / T, the face share of the judgement, and B / T = 2 Phi - 1. B above 0 means the face moves the judgement further than the default does.

**Primary 2: expression-linked evidence.** E* = mean over actors of E minus the mean of E over `label_shuffles` shuffles of labels within actor. Under no label signal, E has a small negative expectation, so it must be centred. Shuffling within actor keeps each actor's label set and the partial crossing.

Neither is B4 or B5. Those test whether the acted family's word rises. E* uses no word and no family named for the label, and B uses no label at all. A test must show a synthetic observer that passes E* and fails B4, and one that passes B4 with B below 0, so that A2's audit fault of a check duplicating another cannot recur.

**Riders, always reported, with actor-bootstrap 95% CI.**

| Rider | Definition | Use |
|---|---|---|
| Word share | omega = W / P, a ratio of means, when P > 0 | Above `word_share_cut` with CI excluding it: the default is word prior. Below with CI excluding it: the default comes from seeing an image or a face. Otherwise mixed. A bootstrap draw with P at or below 0 has no finite omega and counts in the upper tail. If more than 2.5% of draws do, or P itself is not above 0, omega is not stated and the reading is mixed |
| Chain | Means of W, G, F, E, R, and lexicon mass on real, grey and image-free | When grey mass is under `grey_mass_floor`, G and F are reported only as their sum, since g then rests on very little mass. B and E* do not use w or g |
| Right-answer survival | C is the clips whose top family is the acted one and whose partner's `original` ran. S = mean of q_d(i)(t_i) over C, divided by the mean of q_i(t_i) over C, each mean taken within actor and then across actors, as above, and the CI resamples actors on the ratio. d(i) is the Gate B partner, seed 42 | The share of probability on right answers that remains when the face is wrong. This is the literal RQ4 wording. Not estimated with under `min_correct_items` clips or `min_correct_actors` actors. S at or above `survival_cut` beside an Evidence-led verdict is reported as a conflict |

The mismatch baseline enters through B4, through S, and through the positive control. A swap has no place in the identity, since a donor's face is not a prior.

## Inference

- **Unit.** The actor, 17 actors. Models answer with no sampling noise, so stimulus sampling across actors is the only source of error.
- **Tests.** Two step-down families, each over the eligible model and route cells, at most 4 columns each, at familywise alpha .05. They answer different questions and play different parts in the decision table, so each is corrected only within itself. Models are not pooled; they share a correction only.
  - **E\*.** One-sided, above zero, by A2's `signflip_maxt`: all 2^17 sign flips of whole actor rows, Westfall-Young step-down maxT.
  - **B.** Two-sided. B must pass two step-down maxT tests over the B columns, and its p-value is the larger of the two:
    - the exact sign-flip (`signflip_maxt`), which is valid if actor rows are symmetric about zero under the null;
    - the actor bootstrap-t (`bootstrap_t_maxt`, `bootstrap_draws` draws, seed 42), which resamples whole actor rows, centres each draw's t on the observed mean, and so estimates the null distribution of t without assuming symmetry.

    Rows of B are skewed when actors differ in how strongly they portray. Recovery run 2 showed that the sign-flip alone then rejects about twice as often as nominal (see Changes after recovery run 2). The bootstrap-t is only approximately valid with 17 actors. Taking the larger p-value is valid wherever either test is, and it costs some power. Recovery reports the combined test's error rate down to a one-cell family. B's 95% and 90% intervals are bootstrap-t intervals.
- **Floors.** With 17 actors the one-sided floor is about 8e-6, so the actor count does not limit these tests.
- **Interval.** Actor bootstrap, `bootstrap_draws` draws, seed 42: percentile for E\* and the riders, bootstrap-t for B. 95% for direction, 90% for equivalence. Phi is reported with a 90% percentile interval beside every verdict, as a description.
- **Dependence.** q_def and q_tpl use the other actors, so actor rows are slightly dependent. Recovery measures the effect on error rates.
- **Partial crossing.** E uses templates from the same label's other actors, so it stays inside one actor set (A2, Inference). Any contrast across the two actor sets is descriptive.
- **Margin and dominance [proposed].** E\*'s margin is `margin_fraction`, 10% of that model and route's mean T. T involves no label and no direction, so A2's disclosed fault of a margin taken from another outcome does not arise. B has no plug-in margin. Neither part dominating is tested directly on the actor rows: `dominance_ratio` times P minus D, and `dominance_ratio` times D minus P, must each have a one-sided 95% bootstrap-t lower bound above 0 (two one-sided tests). With the ratio at 2, this is D / P between 1/2 and 2, which is true Phi in .333 to .667, without carrying the error of an estimated T. A wide margin on E\* would make Absent easy, so it stays at 10%.

## Classes and verdict

**Evidence class.**

| Class | Requires |
|---|---|
| Shown | E* passes (step-down p < .05, 95% CI lies above 0), and Gate B passed |
| Unmapped | E* passes, and any of B3, B4 or B5 failed. Portrayals move the judgement consistently, but not to the acted word. If B3 is the only failure, B4 and B5 hold and the shift is too small on the image scale to be called mapped, and `b3_failed` says so |
| Absent | E* not significant, its 90% CI inside the margin, B4 failed, and the positive control holds |
| Unresolved | Anything else |

The positive control for Absent is that face information can reach this model's output: RQ0 passed on this route and G4 passed. The control is not a magnitude taken from noise, which was A2's dose fault.

**Balance class.**

| Class | Requires |
|---|---|
| Face | B above 0, step-down p < .05, 95% CI lies above 0 |
| Default | B below 0, step-down p < .05, 95% CI lies below 0 |
| Even | B not significant, its 95% CI includes 0, and both one-sided tests show that neither part is `dominance_ratio` times the other. This is the null on balance. It means the balance could not be told from zero and neither part dominates. It does not mean Phi is .5 |
| Unresolved | Anything else |

**Minimum family size for a balance class (`min_balance_cells`, 4).** Face, Default and Even are named only when B's step-down family has at least 4 eligible cells, which is every model and route A5 can have. In a smaller family the balance class is withheld: it reads Unresolved with reason `balance_small_family`, the verdict is Inconclusive, and the class it would have had is printed beside it as `balance_withheld`, which is not a result. The evidence class is still reported. This rule was added after recovery run 3 failed (see Changes after recovery run 3). In practice, if RQ0 or a gate removes any cell, no model gets a balance verdict from A5.

**Decision table.** Applied per model and route. All verdicts require the preconditions in the next section.

| Evidence | Balance | Verdict | Reading |
|---|---|---|---|
| Shown | Face | **Evidence-led** | The judgement depends on this face more than on any default, and carries the expression |
| Shown | Default | **Default-led with evidence** | Expression gets through, but the fixed part dominates. The word share names its source |
| Shown | Even | **Evidence, neither part dominant** | Evidence present, and neither part is twice the other |
| Unmapped | Face | **Face-led, unmapped** | Portrayals are read consistently, with the wrong words |
| Unmapped | Default | **Default-led, unmapped** | As above, with the fixed part dominant |
| Unmapped | Even | **Unmapped, neither part dominant** | As above, and neither part is twice the other |
| Absent | Default | **No face evidence** | Null on evidence, with the fixed part dominant. Only this row, with a word share above the cut, is the word-knowledge result |
| Absent | Even | **No face evidence, neither part dominant** | Null on evidence. The face moves the judgement about as much as the default does, but not by what it portrays. Never read as the word-knowledge result |
| Absent | Face | **Face-dependent, no expression** | Null on evidence. The answer varies with the face, but not with what it portrays: identity or clip properties |
| Unresolved on either axis, in any combination | | **Inconclusive** | With a reason code, from the list below |

Every result also carries the 90% CI of Phi, its word share and survival rider, and E\* / D beside Evidence-led, the share of the face part that is expression-linked. Each is read against the human comparison where one exists.

## Preconditions and Gate B

A5 follows A2's rule that a missing gate has not passed. But A2 stops RQ1 and RQ2 at a failed Gate B, and it does not stop RQ4, because a failure there can be the RQ4 answer. So Gate B is used by part.

| State | A5 result |
|---|---|
| RQ0, G2 to G4 or delivery not passed for this model and route | **Not tested**. Quantities computed for the record and carry no verdict. This covers Molmo2 and any model with no A2 cells. A2's sham is reported and is not a precondition here, since it checks the mask pipeline and A5 reads only `original` cells |
| B1 or B2 fails | **Not tested**. The readout is not measured on this route: deterministic scoring or coverage failed |
| Baseline rows missing or disagreeing across shards | B and E* stand, since neither needs w or g. Word share and chain are not reported, and the verdict carries `no_baseline`. If the first shard has no `image_free` row, B3 has no input, so B3 has not passed and that row of this table applies too |
| B3 fails | Decision table applies. Images barely move the readout, so a Default result or Absent is expected. `b3_failed` is attached. Evidence can be Unmapped or Absent, never Shown |
| B3 passes, B4 or B5 fails | Decision table applies. Evidence can be Unmapped or Absent, never Shown. `readout_dependence` is attached here and to the B3 row |
| All of B1 to B5 pass | Decision table applies. Evidence can be Shown |

**So a failed Gate B has a defined result.** It cannot produce Evidence-led, Default-led with evidence, or Evidence, neither part dominant. It can produce Face-led unmapped, Default-led unmapped, Unmapped neither part dominant, either No face evidence row, or Face-dependent no expression. If every eligible model fails Gate B on both routes, RQ4 is answered on the blind readout by those verdicts, and any reading from the MCQ stays exploratory, as in A2.

A failed Gate B beside an RQ0 pass means the face reaches the MCQ answer and does not reach the blind readout in this form. That is reported as readout dependence, and it is not called proof that the model lacks perception.

## Null and Inconclusive

There are two distinct nulls, and neither is a default.

**Null on balance (Even).** All of: preconditions met, B not significant, its 95% CI including 0, both one-sided tests of neither part dominating passed, and T at or above `min_total_nats`.

**Null on evidence (No face evidence, or Face-dependent no expression).** All of: preconditions met, E* not significant, its 90% CI inside the margin, B4 failed, the positive control holds, and the same floor on T.

**Inconclusive.** Exactly these reason codes: `evidence_unresolved` (E* not significant with its CI outside the margin, E* significant with a 95% CI that does not lie above 0, or B4 and E* disagreeing in a way no row covers, which means B4 passed and E* did not), `balance_unresolved` (B not significant and dominance not excluded by the one-sided tests, or significant with a 95% CI that includes 0, or not significant with a 95% CI that excludes 0), `balance_small_family` (a balance class would be named, but B's family has fewer than `min_balance_cells` cells; added after run 3), `no_structure` (mean T under the floor). `no_structure` applies to every row, and is checked after the Not tested conditions and before the classes.

**Inconclusive is not a default by construction, but it is common in some regions.** Every verdict row is reachable, and no quantity is capped in the way 8 of A2's 16 contrasts are. Whether a row is reached depends on the data, though. Recovery run 2 showed that at true Phi between the tested points, Inconclusive can be the most likely outcome: at Phi .6 and signal 2 it was .492. So the earlier claim that Inconclusive cannot be the default outcome was too strong, and it is withdrawn. What holds is the following.

1. Every verdict row is reachable with a model's real inputs, as recovery shows (`best_rate_of_each_named_verdict`).
2. A verdict needs only A2 outputs, which exist for every eligible cell, plus RQ0 and the gates. The gate loader must return True on a fixture built from the real pinned paths, with a test that runs the verdict function end to end. This guards against A2's second audit finding, in which the A3 path made every verdict Inconclusive.
3. Recovery must reach every named verdict, each by an observer whose truth is that row. It is also judged on holding Inconclusive under `inconclusive_max` for each observer not placed on a boundary. **Changed 2026-11-08, after run 4:** this point first said that a failure on `inconclusive_max` meant the margin or the rule would be revised before freezing. Run 4 failed it at Phi .5 and .7 with signal 2, and the rule was kept, so the text now says what was done. A failure on power is reported as a failure and accepted, not revised away, provided the observer fails only into Inconclusive, with any other named verdict at most `other_named_verdict_max`. Inconclusive is the safe failure: it withholds a claim and makes no wrong one. A failure into a wrong named verdict, or on any false-positive check, still blocks the freeze.
4. A failed gate is never by itself a cause of Inconclusive. It gives Not tested or leads to a row above. An output row with no verdict or reason code is a code fault and stops the analysis.

## Comparison with humans

No human saw a blank, grey or swapped face in any dataset here, so W, G, F and any human "prior" do not exist. **A5 never states that a model relies more on word knowledge than humans do.** It compares only these.

| ID | Quantity | Human data | Status |
|---|---|---|---|
| H1 | Delta J: Jensen-Shannon divergence to the human distribution for the default, minus the same for the full judgement, on the item's named options, None of the above removed from both (`human_comparable`) | EU-Emotion norms, 57 to 299 respondents per item | Valid. Actor-level sign-flip with own step-down family over the eligible cells, two-sided. Above 0: this face moves the judgement towards people, beyond the default. The template form is reported beside it |
| H2 | Within-label correlation of the model's target share with human p_target, as in A1: both demeaned within label, then pooled. Target share is on the item's named options | EU-Emotion norms | Valid. Actor-bootstrap CI, descriptive. No noise ceiling, since no rater-level data are held |
| H3 | Lambda = E / (E + P) on a fixed option set: humans from votes, models on the same 819 clips, on MCQ and on the blind readout restricted to six families | CREMA-D votes, from A6's cells | Valid formula. Only if A6 reaches Stage 1. Paired actor-bootstrap 95% CI of the difference, 91 actors, with `lambda_margin` as its equivalence bound. Else reported as not tested |

**H3 details.** E and P are linear in q_i, so clip-level rating noise from about 10 raters does not bias them, whereas D would be biased by it, and so H3 does not use D or R. A human "default" is response bias and clip mix, not word knowledge, so H3 says only that the same formula gives the same or a different balance. The human side is fixed before any A6 model cell exists. On the 819 clips' raw vote shares every template used is above 0 wherever a clip has votes, so no smoothing is needed, and any smoothing moved E by almost half when tried, so none is applied. Lambda is computed from the vote shares by `lambda_share` and is .950 (E .600, P .031; `a5_checks.json`, section `h3_human_lambda`, which also holds the pseudocount comparison). Since P is small, Lambda sits near its ceiling for people. H3's model side needs A6's cell format, which does not yet exist: `h3_difference` is coded and tested on arrays, and a loader is a dated addition before any A6 cell is read.

**Flagged items and scoring.** H1 and H2 are repeated without the 26 flagged items and against the human modal label, per A2. For H1, the modal repeat replaces the human distribution with a point mass on the modal label. For H2, it replaces the target. Model accuracy on named options well above mean human p_target on those items is read as possible leakage.

## Sensitivities

All are coded and reported beside the primary, and none changes a verdict. Each sensitivity row runs B and E\* each alone, by the same tests as the primary, with no family correction, so its p-values are not comparable with the primary's step-down p-values.

- Excluding the 3 smoke items.
- Excluding the 26 flagged items.
- E* with templates grouped by the human modal label.
- Clips with at least `token_floor` tokens across the face, by A2's `below_token_floor`. If no clip remains, the row is empty and says so.
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
| Even | Phi near .5, signal to the acted family | Evidence, neither part dominant |
| Identity | Default plus actor shift, no label signal | Face-dependent, no expression |
| Noise | Default plus large item noise | Face-dependent, no expression |
| Unmapped, face | Consistent label shift to the wrong families, Phi near .8 | Face-led, unmapped |
| Unmapped, default | The same at Phi near .2 | Default-led, unmapped |
| Unmapped, even | The same with B calibrated to zero (added after the run 4 audit) | Unmapped, neither part dominant |
| Spread | Label signal with fixed spill to neighbouring families | Evidence-led, showing spill is not separated from evidence (a limit) |
| Boundary | Phi at the edges of the Even range, .3334 and .6666, E* near its margin | Reported, not judged |
| False Even | True Phi just outside the Even range, .32 and .68, at `false_even_signals` | Balance class Even at most `false_positive_max` |
| Inside Even | True Phi .4 and .6, inside the range but not at .5 | Reported, not judged. Shows how often Even is reached away from .5 |
| Shared-gain families | 4 null columns at signal 2, B calibrated to zero, one draw of actor gains shared by all 4, at gain .3 and .5 (added after the run 4 audit) | Any B column at most `false_positive_max` |
| All-null family | No label signal and a true balance of zero in all 4 cells, `all_null_replicates` replicates, `all_null_shuffles` in every cell | Any B column, and any E* column, at most `false_positive_max` |

An observer on `phi_grid` is scaled by Phi, and the signal strength is varied, so the report gives power at realistic sizes as well as at large ones. Pass rule: for each non-boundary observer, the expected verdict in at least `expected_verdict_min` of replicates, any other named verdict in at most `other_named_verdict_max`, and Inconclusive in at most `inconclusive_max`. Every named verdict must be reached at least once.

**How each replicate runs.** A replicate is a full family, not one cell. It holds the observer's cell and `companion_cells` (3) cells with no truth on either test: no label signal, and Phi calibrated to .5, so a true balance of approximately zero. Each of A5's two step-down families therefore runs over 4 columns, the most it can have, against columns that give it nothing. Real columns from one model share clips, so they will correlate, and these do not, so this is the worst case for power. Every observer whose truth is null on a test (no label signal, or a true balance of approximately zero) also runs with 0 and 1 companions (`companion_sweep`). With 0 companions the family is one cell, the smallest A5 can have, as when A1 fails and `rq0_status` admits only `single_face` on one model. A liberal test shows most there. The prior scale that gives a stated Phi is found by bisection on `calibration_draws` fixed draws, which are separate from the replicates. Observers whose truth is a balance of zero (the null, Phi .5 on the grid, and the gain sweep at .5) are instead calibrated on B itself, mean B over mean T across `balance_zero_calibration_draws` (10,000) separate fixed draws, so that B is zero to about .002 of T on those draws. The mean over replicates of each replicate's B / T is a different quantity, a mean of ratios, and can be further from zero: in run 4 it was -.0103 at `mixed_phi0.5_signal2.0`. True Phi is defined by A5's own estimator, the ratio of mean D to mean T over actors, averaged over those draws. The measured Phi in the recovery table is a mean over replicates of that ratio, so it can differ slightly from the true value.

**What is judged.** The pass rule binds at `pass_signal` (2.0 nats, the size in A2's recovery) for each non-boundary observer. Smaller signals in `signal_grid` and `unmapped_signals`, and the `gain_sweep` observers, report power and are not judged. Each named verdict must be reached at least once. The two prior-only observers must also give their stated word-share reading in at least `expected_verdict_min` of replicates.

**False positives.** On observers with no label signal, the rates at which E* passes, B4 passes and any evidence verdict appears must each be at most `false_positive_max`. So must the rate of a named Face or Default balance class (`B_dominance_named`) and the rate of a significant B, on observers with a true balance of approximately zero, in families of at least `min_balance_cells` cells; the rate of any B column being significant in each shared-gain family (`shared_family_gains`), where all 4 columns are null and share the actors' portrayal gains; the rate of any B column or any E* column being significant in the all-null family; and the rate of the Even balance class on every observer whose true Phi is outside the Even range, where true Phi is the stated value or, for an observer placed otherwise, A5's estimator over `calibration_draws` draws. In families of 1 and 2 cells, E* and B4 are judged as above. Checks that are 0 there by construction, any evidence verdict and a named balance class, are printed as vacuous and not counted, and the raw B rate is printed and not judged (`unjudged_below_min_balance_cells`). The limit is .05 plus about two standard errors at 500 replicates, and it was not raised after runs 1 or 2.

**Two departures from plain A2 code, in recovery only.** Neither touches `a5_analyse.py`'s real path. Recovery workers replace A1's `actor_sign_flip_p` and `actor_bootstrap_ci` in memory with versions that cache their seeded draws, and `tests/test_a5.py` checks that they return the same values. Companion cells use `companion_shuffles` (100) label shuffles, not `label_shuffles`, since a companion's only role is a null column. The observer's own cell uses the full count, and the all-null family uses the full count in every cell.

If an observer fails, the failure is reported. Any change comes before freezing, is dated, and is not made after any A2 cell is read.

## Changes after recovery run 1

Everything in this section was decided **after** reading recovery rates, on 2026-11-07, on synthetic observers only, before any A2 cell was read. Run 1 is kept as `research/reproduction/a5_recovery_run1.json` (block hash `db41a6f8...`, 500 replicates, overall fail).

**What failed in run 1.** `all_null_family|any_B_column` at .076 against the limit of .07. `mixed_phi0.5_signal2.0` with Even .002, Inconclusive .958, all `balance_unresolved`, and other named verdicts .04. The best Even rate on any observer was .474. Phi .7 and Phi .3 at signal 2 passed at .804 and .88, and lost their remaining replicates to `balance_unresolved`.

**Cause of the Even failure, tested.** Mean half-width of B's 90% CI, as a share of T, rose with signal: .077 at signal .25 to .156 at signal 2. A direct test at Phi .5 (`scripts/a5_checks.py`, output `research/reproduction/a5_checks.json`, section `cause_of_even`, 300 draws per setting, separate from recovery) showed that portrayal gain drives the rise. With gain 0, the expected half-width is flat or falling, .080 at signal .25 and .069 at signal 2. With gain .3 it rises from .080 to .167, and the actor's log gain explains .64 of the variance in actor rows at signal 2. At signal .25 it explains .002. Differences fixed by an actor's label set are small (SD .012 to .049). The old margin was also short of reach without any gain: at half-width .07 to .08 on T, and allowing for the estimate's own spread, 80% power needs a margin near .14. So 10% of T could not give Even at 80% with 17 actors at any signal.

**Balance margin, changed from 10% of T to one third of T (`balance_margin_fraction`). Superseded after run 2 by the direct test below, which keeps the same meaning.** The argument is about meaning. Even is the claim that neither part dominates. A fifty per cent excess is a difference, but one part being at least twice the other is the plain sense of dominating, so Even is the claim that D / P lies between 1/2 and 2, which is Phi in .333 to .667. That is wider than the expected minimum needed to pass, about .30 by the usual equivalence arithmetic at signal 2 and gain .3, and it was not found by searching for a passing value. It weakens the claim, and the verdict wording and Limits now say so: Even means not shown to be twice, and does not mean Phi is near .5. `margin_fraction` stays at 10% and now applies to E* only, since a wider margin there would make Absent easier. It was not changed.

**New checks, because a wider margin makes a false Even easier.** Observers at true Phi .32 and .68 at signals 1 and 2, with Even limited to `false_positive_max`. The same limit on every observer whose true Phi is outside the range. Boundary observers moved to .3334 and .6666. Inside-range observers at Phi .4 and .6, reported.

**Calibration, changed from 12 to 200 draws (`calibration_draws`).** Run 1's stated Phi was off for some observers: the no-evidence null had measured Phi .514 and B / T +.028, so its "true balance of zero" was not zero, and this affected the B columns of the all-null family and the companions. This corrects the observers' truth and no threshold.

**All-null family.** It now has its own observer, with `all_null_replicates` (2000) replicates and the real analysis's 1000 shuffles in every cell. `false_positive_max` was not raised, and alpha and `signflip_maxt` were not changed.

## Changes after recovery run 2

Decided after reading recovery rates and an independent audit, on 2026-11-07, on synthetic observers only, before any A2 cell was read.

**Run 2** is kept as `research/reproduction/a5_recovery_run2.json`: block hash `2103ffef...`, 500 replicates, overall fail, with one failed check.
- The failure was `observer|mixed_phi0.7_signal2.0`, Evidence-led at .78 against .80, with Inconclusive .20, all `balance_unresolved`. B's raw two-sided rate there was .994, and the 8-column step-down cut it to .78: columns that answer a different question (E\*) took part in B's correction.
- Even passed at Phi .5 with .854.
- The all-null family passed after recalibration: any B column .0355 over 2000 replicates.
- At Phi .6 and signal 2, Inconclusive was .492.

**The run 1 null-family failure was miscalibration, not the shuffle shortcut.** `scripts/a5_family_diagnostic.py`, seed 6042, 2000 replicates per arm, output `research/reproduction/a5_family_diagnostic.json`. Any B column significant, Wilson 95% intervals:
- Run 1's null (prior .473, mean B / T +.027): .073 (.062 to .085) with 1000 shuffles, and .0725 (.062 to .085) with 100.
- Recalibrated null (prior .4843, mean B / T -.001): .0335 (.026 to .042) with both shuffle counts.

The shortcut changed nothing. The excess came from a null that was not null.

**The sign-flip test of B was liberal (audit finding, reproduced).** At a true balance of approximately zero, run 2's raw two-sided rate for B was .108 at `mixed_phi0.5_signal2.0` (portrayal gain .3) and .128 at `gain0.5_phi0.5`, against .056 at `gain0.0_phi0.5`. So it is driven by heterogeneity, which skews the actor rows. The 8-column family hid this. A one-cell family, which A5 can have if A1 fails and only `single_face` is admitted, would not. Changes:
1. B now needs both the sign-flip and an actor bootstrap-t (Inference).
2. Every null observer also runs in families of 1 and 2 cells, and every family size is judged. Since run 3, the raw B test is judged only in families of `min_balance_cells`, and the named claim at every size (Changes after recovery run 3).
3. `true_balance_zero` is set on `gain0.0_phi0.5` and `gain0.5_phi0.5`.

**B and E\* are now separate step-down families.** They answer different questions and play separate parts in the decision table. Correcting B for E\* columns, as run 2 did, is what took Phi .7 from .994 raw to .78, with no gain in validity for either test. The false-positive checks at one cell bound the cost of a smaller family.

**Even is now tested directly and needs B not told from zero.**
- The plug-in margin of one third of T is replaced by two one-sided bootstrap-t tests on the actor rows, 2P - D > 0 and 2D - P > 0 (`dominance_ratio` 2). This is the same claim, D / P between 1/2 and 2, without the error in an estimated T, which the plug-in ignored.
- Even also needs B's 95% CI to include 0, so that it means what the text says: the balance could not be told from zero.

**Names now match the rules.**
- Even with evidence is "Evidence, neither part dominant".
- Unmapped with Even is a new row, "Unmapped, neither part dominant", split from "Default-led, unmapped".
- Absent with Even is "No face evidence, neither part dominant", and is never read as the word-knowledge result. Only Absent with Default can be.
- Every verdict prints the 90% CI of Phi, and Evidence-led prints E\* / D.

**Other changes.**
- A5's code and the helpers it imports are pinned in Code hashes and checked before analysis and recovery.
- Default paths resolve against the repository.
- A missing `rotation_summary.json`, `vision_gates_*` or `input_delivery_*` stops the analysis rather than reading as a failed gate.
- The first shard's `image_free` row is used whatever its status, as A2 does.
- Numbers in this draft now trace to `a5_checks.py`, `a5_family_diagnostic.py` and the recovery outputs.
- The null is described as approximately zero, true Phi is defined by A5's own estimator, and measured Phi is a mean of ratios.
- The claim that Inconclusive cannot be the default outcome is withdrawn (Null and Inconclusive).

## Changes after recovery run 3

Decided after reading recovery rates, on 2026-11-08, on synthetic observers only, before any A2 cell was read. No limit was changed: `false_positive_max` stays .07 and the pass rule is as before.

**Run 3** is kept as `research/reproduction/a5_recovery_run3.json`: block hash `228bdf48...`, 500 replicates, overall fail, with five failed checks.
- B significant at a true balance of zero, against .07: `gain0.5_phi0.5` with 0 companions .12, with 1 companion .092, and `mixed_phi0.5_signal2.0` with 0 companions .084. With 3 companions they were .054 and .034.
- `observer|mixed_phi0.5_signal2.0`: "Evidence, neither part dominant" .752, Inconclusive .214, all `balance_unresolved`.
- `observer|mixed_phi0.7_signal2.0`: Evidence-led .598, Inconclusive .398, all `balance_unresolved`.

**First cause checked: calibration.** Observers with a true balance of zero were calibrated on mean Phi over 200 draws. Over 3000 fresh draws their B / T was -.017 at portrayal gain .5 and -.009 at signal 2, against .004 for the null and -.0005 at signal 1 (`a5_run3_calibration_check.json`). They are now calibrated on B itself over 10,000 draws (Recovery). This was right to fix but explained little. With 400 replicates on the recalibrated observers, B significant was .12 (gain .5, 0 companions), .075 (1 companion), .0525 (3 companions) and .0725 (signal 2, 0 companions) (`a5_run3_recalibrated_diagnostic.json`).

**Second cause checked: dependence through the leave-one-out default.** Each actor's default is built from the other actors, so actor rows are not independent, which both of B's tests assume. `scripts/a5_dependence_diagnostic.py` replaced that default with a fixed population default, with the same replicate seeds, 400 replicates per observer (`a5_run3_dependence_oracle.json`, `a5_run3_dependence_loo.json`):

| Observer | Companions | Leave-one-out default | Fixed default |
|---|---|---|---|
| gain .5, Phi .5 | 0 | .085 | .08 |
| gain .5, Phi .5 | 1 | .0825 | .0825 |
| signal 2 (gain .3), Phi .5 | 0 | .0475 | .0475 |
| gain 0, Phi .5 | 0 | .055 | .0525 |
| gain 0, Phi .5 | 1 | .0125 | .01 |

So dependence is not the cause. Rates are near nominal with no gain heterogeneity and stay near .08 with strong heterogeneity, whatever the default. Across the three samples, gain .5 gave .12, .12 and .085 in a one-cell family, and .092, .075 and .0825 in a two-cell family. With 17 actors and skewed rows, neither the sign-flip nor the bootstrap-t holds its level in a family that small. Resampling the whole pipeline inside each draw would remove the dependence, which is not the cause, so it was not adopted.

**Change: a minimum family size for a balance class (`min_balance_cells`, 4).** Set from the rates above. At 4 cells every true-null B rate seen was at or under .054. At 1 and 2 cells, rates at gain .5 exceeded .07 in every sample. Families of 3 cells were not tested, so 3 is not allowed. Below 4, the balance class is withheld and the verdict is Inconclusive with `balance_small_family` (Classes and verdict). The B false-positive checks, a named Face or Default class (`B_dominance_named`) and the raw test, are judged only at 4 cells. In smaller families the named rate is 0 by construction and is printed as vacuous after the run 4 audit, and the raw rates are printed, unjudged. This narrows what A5 can say, and Limits says so.

**How a reviewer could read this rule.** It was set after run 3 failed, from the same rates. It takes the failing checks, B at 1 and 2 cells, out of what is judged, by making the claim they tested impossible there. A reviewer could fairly read that as making failing checks unjudgeable rather than passing them. The defence is that the rule withholds the claim, and does not keep the claim while dropping the test. Its cost is that A5 names no balance unless all 4 cells are eligible. The run 4 audit then found that the 4-cell result rested on companion columns with no portrayal gain, so the shared-gain families below were added to test 4 cells as the real study would have them.

**Tests.** `observers()` now runs the 10,000-draw calibration, so the test that lists observer names uses 100 draws. The recovery uses the full value.

**Run 4** is kept as `research/reproduction/a5_recovery_run4.json`: block hash `df4583f0...`, 500 replicates, overall fail, with two failed checks, both Inconclusive above `inconclusive_max` (.20). No other named verdict exceeded .10 in either.
- `observer|mixed_phi0.5_signal2.0`: "Evidence, neither part dominant" .744, Inconclusive .226, all `balance_unresolved`, Default-led with evidence .03.
- `observer|mixed_phi0.7_signal2.0`: Evidence-led .598, Inconclusive .398, all `balance_unresolved`.

These are a power limit at the largest signal, and Inconclusive is the safe failure: B's interval widens with signal under portrayal gain, so balance goes unresolved more often at signal 2 than at signal 1. They are reported as failures, and the pass rule was not changed. Power for the expected verdict, by true Phi (rows) and signal in nats (columns):

| Phi | .25 | .5 | 1 | 2 |
|---|---|---|---|---|
| .1 | .03 | .282 | .994 | .998 |
| .3 | .03 | .704 | .994 | .846 |
| .5 | .048 | .758 | .924 | .744 |
| .7 | .082 | .852 | .986 | .598 |
| .9 | .076 | .874 | 1.0 | .966 |

Every judged false-positive check passed, but 24 of them, the named balance class and any evidence verdict on the 1- and 2-cell copies, were 0 by construction, so they showed nothing (corrected after the audit below). A named Face or Default class at a true balance of zero: at most .042 in families of 4 (`gain0.5_phi0.5`). The raw B test in families of 4: at most .048. Any B column in the all-null family: .031. The raw rates in smaller families, printed and not judged, reproduce the problem: `gain0.5_phi0.5` .108 (1 cell) and .08 (2 cells), `mixed_phi0.5_signal2.0` .074 (1 cell). Every named verdict was reached, but "Unmapped, neither part dominant" only as a false positive, at .0045 in the all-null family, since no observer had that row as its truth (corrected after the audit below).

**Report rebuilt once, no replicate re-run.** Run 4's report built its power table by observer phi and signal, and the companion-sweep copies, which share both, overwrote each grid row. The Phi .5 row read 0 because the copies' balance is withheld. The key now skips sweep copies, and the report was rebuilt from run 4's stored per-observer table (`scripts/a5_recovery.py` build_report). Every other field is identical. The report as first written is kept as `a5_recovery_run4_as_written.json`. Run 4's replicates came from `scripts/a5_recovery.py` `a68903cd...`, and the rebuild used `6833655d...`. Both are recorded in the file's `rebuilt_from`.

## Changes after the run 4 audit

Decided on 2026-11-08, after an independent audit of run 4 and the code at `b0f16d9`, on synthetic observers only, before any A2 cell was read. No limit was changed.

**Four cells had not been shown valid for a realistic family (audit blocker).** Every companion column and every all-null column had no label signal, so it carried no portrayal-gain skew. Run 4's 4-cell rate (.048 at gain .5) was one skewed column among three calm ones. In the study, the 4 columns are 2 models x 2 routes on the same 17 actors, so the gain is shared and skewed in all 4. The recovery now has `shared_family_gain0.3` and `shared_family_gain0.5`: 4 null columns at signal 2, calibrated to B = 0, with one draw of actor gains shared by all 4. Any B column significant is judged against `false_positive_max`. Gain .3 is the default (`portrayal_gain_log_sd`), and .5 is the stress level. These are judged checks, so a failure at .5 fails the recovery.

**A skew-targeted test was tried and not adopted.** `scripts/a5_skew_diagnostic.py`, output `a5_run4_skew_diagnostic.json`, 1000 replicates per null arm and 400 per power arm, compared tests on the same data. The table gives familywise rates (any of the columns significant) in the null arms, and the rate for the observer's column in the power arms:

| Arm | Sign-flip | Bootstrap-t | Hall bootstrap-t | A5 now (sign-flip and bootstrap-t) | Sign-flip and Hall |
|---|---|---|---|---|---|
| 4 shared null columns, gain .3, signal 2 | .083 | .056 | .053 | .056 | .043 |
| 4 shared null columns, gain .5 | .132 | .086 | .08 | .086 | .071 |
| 1 null column, gain .3, signal 2 | .069 | .057 | .062 | .057 | .054 |
| 1 null column, gain .5 | .147 | .114 | .102 | .114 | .091 |
| Phi .3, signal 2 (power) | .915 | .853 | .61 | .853 | .61 |
| Phi .7, signal 2 (power) | .855 | .615 | .945 | .615 | .853 |
| Phi .3, signal 1 (power) | .998 | .998 | .975 | .998 | .975 |
| Phi .7, signal 1 (power) | 1.0 | .998 | .998 | .998 | .998 |

The Hall version is a bootstrap-t on Hall's (1992) monotone cubic transformation of t, the successor to Johnson's (1978) skewness-corrected t, so the estimand is still the actor mean of B. It brings gain .5 only to .071 and still fails there. It moves power from one tail to the other: Phi .3 at signal 2 falls from .853 to .61, which would lose the Default-led recovery (run 4: .846). So it fails the condition for adoption, and A5 keeps its test. At the default gain, A5's test holds the familywise rate, .056 with shared gains. At gain .5 it does not, .086, and that is stated as a limit and judged in the recovery.

**Other changes from the audit.**
- Checks that are 0 by construction, a named balance class and any evidence verdict in 1- and 2-cell families, are printed as vacuous and not counted (Recovery, False positives).
- "Unmapped, neither part dominant" now has its own observer, `unmapped_even_signal2.0` (and at signal 1, not judged): wrong-family signal, calibrated to B = 0. A named row counts as reached only through an observer whose truth is that row. The no-evidence null now states its truth, "No face evidence, neither part dominant", and is still not judged.
- The false-Even check covers every observer whose true Phi lies outside the Even range. For observers placed by something other than a stated Phi, true Phi is A5's estimator over `calibration_draws` draws.
- `decide` now requires the family size, and a test runs all 9 named rows at family sizes 1 to 5.
- `src/paper_v2/schema.py`, `src/paper_v2/metrics.py` and both diagnostics are pinned. A5 reads A3 only through A2's `a3_summary`, which accepts only the bytes A2 pins, and `a5_analyse.py` now stops if A2's pin differs from `a3_summary_sha256` in this block.
- Point 3 of the list under Null and Inconclusive is rewritten as a dated change, and the reason code `balance_small_family` is added to the list of codes there.
- The descriptions of the family size and of calibration precision are corrected.

**Run 5** is `research/reproduction/a5_recovery.json`: block hash `a916f08d...`, 500 replicates, overall fail, with five failed checks. Unchanged observers kept their seeds, so their numbers equal run 4's.

Failed, false positives at gain .5 with shared gains, as stated beforehand:
- `shared_family_gain0.5|any_B_column`: .094 against .07.
- `shared_family_gain0.5|B_significant` (the observer's own column): .072 against .07. A named Face or Default class there was .062.

Failed, power, all into Inconclusive with reason `balance_unresolved`, other named verdicts at most .03:
- `observer|mixed_phi0.5_signal2.0`: "Evidence, neither part dominant" .744, Inconclusive .226.
- `observer|mixed_phi0.7_signal2.0`: Evidence-led .598, Inconclusive .398.
- `observer|unmapped_even_signal2.0` (new): "Unmapped, neither part dominant" .756, Inconclusive .222, Default-led unmapped .022. At signal 1, not judged, it was .934.

Passed:
- `shared_family_gain0.3|any_B_column`, at the default gain: .054. Its own column: .022.
- All 112 judged false-positive checks except the two above. 52 checks in 1- and 2-cell families are printed and not counted: the vacuous ones, and the raw B rates, which reach .098 (`gain0.5_phi0.5`, 2 cells) and .086 (`unmapped_even_signal2.0`, 1 cell).
- Every named row was reached by an observer whose truth is that row. The lowest best rate was .934, for "Unmapped, neither part dominant" and for "No face evidence, neither part dominant".
- The power table is as in run 4.

**Status for freezing.** Under point 3 as rewritten above, the three power failures are accepted, because they fail only into Inconclusive. The gain .5 shared-family failure is a false-positive failure, so it is not accepted as power. It is recorded as a failure and stated as a limit: when all 4 columns share strong portrayal-gain skew (log-SD .5), the B family's error rate is about .09 rather than .05. Whether the actors in the real data are that heterogeneous is not known before the data are read. Freezing with this failure on record is a decision for the coordinator and is not made here.

## Changes after recovery run 5

Decided on 2026-11-08, after run 5, on synthetic observers only, before any A2 cell was read. No limit was changed.

**The question.** Run 5 failed the shared-gain family at gain .5 (.094). The coordinator asked for a guard written before data: estimate actors' portrayal-gain heterogeneity from quantities A5 already has, other than B. If the estimate exceeded .4, halfway between the validated .3 and the failing .5, balance verdicts would be withheld. The guard was to be dropped if its estimator was too noisy to hold the gain .5 family without firing often at gain .3.

**The estimator** (`gain_heterogeneity` in `a5_analyse.py`; block settings `gain_heterogeneity_min_z` and `gain_heterogeneity_floor`).
- Per clip, d is the log-odds of the acted family under the judgement minus under the leave-one-out default. A gain that multiplies an additive push on the acted family's logit moves d by exactly that push.
- Per actor, m is the mean of d, and v is its sampling variance.
- Across actors, the between-actor variance is the variance of m less the mean of v (method of moments, floored at 0). The log-SD is sqrt(log(1 + variance / mean^2)), from a lognormal gain's coefficient of variation.
- It is estimable only where the mean push is above 2 standard errors.
- A family's value is the mean over its eligible cells, since the actors are shared.

**Diagnostic.** `scripts/a5_gain_guard_diagnostic.py`, output `a5_run5_gain_guard_diagnostic.json`, 400 replicates per arm. The estimator was estimable in every replicate of every arm with a push.

| True gain log-SD | Signal .5 | Signal 1 | Signal 2 |
|---|---|---|---|
| 0 | .12 (SD .15) | .05 (.06) | .03 (.03) |
| .3 | .40 (.17) | .34 (.08) | .33 (.07) |
| .5 | .64 (.16) | .55 (.11) | .56 (.11) |

Mean estimate, with its SD across replicates, at Phi .5. It tracks the true spread, with a small upward bias that grows as the signal falls.

How often a guard at .4 would fire, and what it would do:
- 4 shared null columns at gain .5: familywise rate .077 without the guard and .022 with it. It fired in 93% of replicates.
- 4 shared null columns at gain .3: .07 without and .07 with. It fired in 17%.
- Gain .3 single columns: it fired in 12% (signal 2), 19% (signal 1) and 53% (signal .5).
- Grid points at gain .3: 14% to 28% at signal 1 and 2, and 53% to 66% at signal .5. At Phi .3 and signal 2 it fired in 28%, which would take the Default-led recovery from .846 to about .61.

**Decision: the guard is dropped.** It holds the gain .5 family (.022), but it fails the other condition. At the validated gain it fires in a fifth to a quarter of replicates at realistic signals, and in over half at weak ones. It would turn passing named-verdict recoveries into failures. The noise is not only estimator error. With 17 actors drawn at log-SD .3, the sample SD itself varies by about .05, so even a perfect estimator would cross .4 regularly, and 17 actors cannot reliably tell .3 from .4. The gain .5 shared-family failure stays recorded, and is stated as a limit. The estimate is now reported beside every cell (`gain_heterogeneity`) and family (`family_gain_log_sd`) at analysis, as a description that changes no verdict. A family estimate near or above .5 then says the balance tests ran in the region where recovery showed them liberal, about .09 familywise rather than .05. The recovery reports the estimate for each observer with a push (`gain_heterogeneity`), with the rate at which a guard at .4 would have fired.

**Run 6** is `research/reproduction/a5_recovery.json`: block hash `fd20495b...`, 500 replicates, overall fail. Run 5 is kept as `a5_recovery_run5.json`. With the guard dropped, run 6 differs from run 5 only in the descriptive gain report. Every verdict rate and every check is identical, with the same five failures:
- `shared_family_gain0.5|any_B_column` .094, and its own column .072, against .07.
- `observer|mixed_phi0.5_signal2.0`, `observer|mixed_phi0.7_signal2.0` and `observer|unmapped_even_signal2.0`, each into Inconclusive.

The recovery's gain report (`gain_heterogeneity`) gives the estimator's mean (SD) and the rate at which a guard at .4 would have fired:

| Observer | True gain | Signal | Estimate | Would fire |
|---|---|---|---|---|
| `gain0.0_phi0.5` | 0 | 2 | .03 (.03) | 0 |
| `mixed_phi0.5_signal0.5` | .3 | .5 | .38 (.18) | .49 |
| `mixed_phi0.5_signal1.0` | .3 | 1 | .35 (.09) | .24 |
| `mixed_phi0.5_signal2.0` | .3 | 2 | .34 (.07) | .18 |
| `mixed_phi0.3_signal2.0` | .3 | 2 | .35 (.07) | .22 |
| `shared_family_gain0.3` | .3 | 2 | .34 (.07) | .16 |
| `gain0.5_phi0.5` | .5 | 2 | .55 (.11) | .94 |
| `shared_family_gain0.5` | .5 | 2 | .55 (.11) | .93 |

**Freezing on the rules as written.** Point 3 accepts the three power failures. It does not accept the shared-family failure at gain .5, because that is a false-positive check, and point 3 says a false-positive failure still blocks the freeze. So on the rules as written, A5 is not freezable. Reclassifying that observer as unjudged now, after it failed, would be the kind of change this draft has refused elsewhere.

The choices left are for the coordinator, and each must be dated:
- (a) Freeze with the failure on record, under a written exception to point 3, with the limit stated.
- (b) Remove balance verdicts from A5's confirmatory claims, and report B, Phi and the gain estimate descriptively. The evidence classes, whose checks all pass, then carry the confirmatory claims.
- (c) Change the design so that the test holds under shared skew, for example by testing on more units than 17 actors. That is not available in A2's data.

## Limits

- Even is a coarse claim. With 17 actors it can only say neither part is twice the other, true Phi in .333 to .667, not that Phi is near .5. It weakens as actors differ in how strongly they portray (see the sections on changes after recovery). Away from .5 inside that range, Inconclusive is common.
- A balance class needs all 4 cells (`min_balance_cells`). B's test rejects too often at a true balance of zero in a family of 1 or 2 cells when actors differ strongly in how they portray (Changes after recovery run 3), so in a smaller family A5 says nothing about balance. Validity at 4 cells rests on recovery with 3 null companion columns. Real companions share clips and may carry a real B, and their effect on the step-down at that size is not tested.
- When all 4 columns share strong portrayal-gain skew, B's family is liberal. With shared gains at gain .5, the familywise rate was .086 in the skew diagnostic and .094 in recovery run 5 (limit .07), which is a recorded failure. At the default gain .3 it was .056. The estimated gain log-SD (`family_gain_log_sd`) is reported at analysis so a reader can see which region the data are in; it changes no verdict, since a guard on it fired too often at the validated gain (Changes after recovery run 5). A skew-corrected bootstrap was tried and not adopted, because it lost power at Phi .3 (Changes after the run 4 audit).
- Face and Default are tests, not sizes. A significant B with D / P near 1 is still Face or Default, and Phi with its interval beside the verdict gives the size.
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
| Recovery showed mechanics only | Graded effects, actor noise, full 4-column families for B and for E*, shared-gain null families, 500 replicates |
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
