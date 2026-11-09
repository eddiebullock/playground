# Supplementary Note 1: estimand, residual CEA and identifiability

**Status:** draft, 2026-11-08. Written from `research/theory/causal_feature_alignment.md` (the theory note), `research/PREREG_A4_DRAFT.md` (frozen, tag `a4-freeze`) and the frozen A4 code. The sha256 of `scripts/a4_analyse.py`, `scripts/a4_recovery.py`, `scripts/a4_design.py` and `research/scripts/human_au_effects.py` in the working tree was checked against the frozen list in the amendment (line 7) on 2026-11-08, and all four match. So the line numbers below refer to the frozen code. No model output was used. Every number cites its file. Section 6 (relation to prior work) was added on 2026-11-08 from a reading of `references.md` #16 and #17.

Plain summary. CEA compares two systems' cue effect matrices by their cosine. Residual CEA, the confirmatory statistic of A4, first removes from each matrix the part of every cue's effect that lies along that matrix's own average cue effect. It then takes the cosine. This note defines residual CEA as the code computes it, proves the properties the amendment relies on, and states which theory-note results carry over to it.

## 1. Estimand: CEA in set-to form

**Orientation.** The theory note writes tau as p by K, cues by options. The A4 code holds it as K by p, options by cues. This note uses the code's orientation. Nothing else changes.

**Effect matrix, set-to form.** For cue (AU family) j and option k,

    tau(k, j) = E_{z~Q}[ f^k(z with z_j set to 1) - f^k(z with z_j set to 0) ]

with the other cues at their observed values and Q the design (theory note, line 17). A4 estimates this by G-computation from a fitted model, not from paired stimuli:

1. Responses per stimulus are the option probabilities averaged over a complete rotation set (`scripts/a4_analyse.py:48-63`).
2. One ridge logistic per option, one-vs-rest, on an intercept, centred face gender and 33 AU amplitudes. Intercept and gender are unpenalised. Ridge strength is the human value for the dataset (`scripts/a4_analyse.py:77-82`; values in `research/reproduction/human_au_effects.json`, field `ridge_lambda_selected`: 0.1 for basic, 1.0 for conversational).
3. For each AU family, the linear predictor is recomputed with all members of the family set to 1, and with them set to 0. Bilateral families set both sides together. One-vs-rest probabilities are renormalised over options on each stimulus, and the difference is averaged over stimuli (`research/scripts/human_au_effects.py:149-170`, called at `scripts/a4_analyse.py:83`).
4. The basis is 25 families (`scripts/a4_analyse.py:29-32`, `research/scripts/human_au_effects.py:138-146`). The human matrix is the stored `do_effect_probability_au_full_vs_none`, restricted to the same 25 columns (`scripts/a4_analyse.py:317-319`). It is the mean of per-subject do effects (`research/scripts/human_au_effects.py:586-590, 625`).

Because probabilities are renormalised, each column of tau sums to zero over options (`research/scripts/human_au_effects.py:154-155`). A4 has K = 7 options for basic and K = 5 for conversational (`research/PREREG_A4_DRAFT.md:77-78`), and p = 25.

**CEA.** CEA(S, T) = <tau_S, tau_T> / (||tau_S||_F ||tau_T||_F), undefined if either matrix is zero (theory note, lines 19-23; `scripts/a4_analyse.py:36-39`, which returns NaN in that case). With zero column sums the entries have mean zero, so CEA equals the Pearson correlation over entries. In A4 this is "raw CEA" and is secondary (`research/PREREG_A4_DRAFT.md:94, 99`).

## 2. Residual CEA

### 2.1 Definition, as implemented

Let tau be K by p with zero column sums. Define

    m = tau 1_p / p                          (the mean of the columns: each option's average effect over cues)
    u = m / ||m||                            (if m is not zero)
    R(tau) = tau - u (u' tau) = (I_K - u u') tau
    R(tau) = tau                             (if m = 0, by convention)

    residual CEA(S, H) = cos(R(tau_S), R(tau_H))

Step by step, with the code:

| Step | What | Code |
|---|---|---|
| Mean profile | `mean = tau.mean(axis=1)`: mean over the 25 family columns, one value per option | `scripts/a4_analyse.py:92` |
| Zero case | if `norm(mean) == 0`, return tau unchanged | `scripts/a4_analyse.py:93-95` |
| Unit direction | `unit = mean / size` | `scripts/a4_analyse.py:96` |
| Projection | `tau - outer(unit, unit @ tau)`: removes from each column its component along `unit` | `scripts/a4_analyse.py:97` |
| Statistic | `cosine(remove_common(tau), remove_common(human))` | `scripts/a4_analyse.py:179-181` |

What is removed, and at what level:

- It is a matrix-level operation, per system and per dataset, on the fitted 25-family effect matrix. It is not applied per stimulus, per trial or to the responses.
- The removed direction lies in option space. It is the system's own average cue effect, a K-vector. Each cue keeps its effect minus its projection on that direction.
- Each matrix uses its own direction: the model's R uses the model's mean profile, the human's R uses the human's. There is no shared projection (`research/PREREG_A4_DRAFT.md:97`: "that matrix's own mean profile across cues").
- It is not an option prior. An option prior that adds to the probabilities cancels in tau already, for both statistics (Section 2.3).
- It is not the same as subtracting each option's mean over cues, tau - m 1'. Both leave zero row sums (Proposition S1), but R removes a rank-one term u (u' tau), whose cue weights are u' tau, not the constant ||m||.

**Where else R is used, with the same function.** The model's bootstrap interval (`scripts/a4_analyse.py:127-137`, human residual computed once at line 130), the model's split-half reliability (`:140-150`), the cue-label null (`:179-180` passes the residual matrices to `cue_null_p`, `:100-113`), the human ceiling (`:362-367`), the between-group reference (`:327`), the human energy shares (`:393, 396`), the sensitivities (`:193-203`), the amplitude sets (`:553, 558`), the direct contrast (`:653`) and the recovery truth (`scripts/a4_recovery.py:145-146`).

**Subsets.** For the family and row sensitivities, R is applied to the full 25-column matrix first and the columns or rows are dropped afterwards (`scripts/a4_analyse.py:194-197`). This matches the amendment, "Subsets take the common profile from the full matrix" (`research/PREREG_A4_DRAFT.md:173`). So a subset value is not R applied to the subset.

**Which statistic decides.** `STATISTIC = "residual"` is fixed in the hashed design (`scripts/a4_design.py:45`). Recovery can only withhold it: `choose_statistic` returns it if it passed, otherwise None (`scripts/a4_recovery.py:196-198`). The analysis reads the choice from `a4_recovery.json` and stops if none passed (`scripts/a4_analyse.py:593-601, 628`). The file records `"confirmatory_statistic": "residual"` with the frozen `design_sha256` (`research/reproduction/a4_recovery.json`).

### 2.2 Properties

**Proposition S1 (algebra of R).** Let tau be K by p with 1_K' tau = 0 and m = tau 1_p / p not zero. Then:

(a) R(tau) has zero column sums and zero row sums. So its entries have mean zero, and residual CEA equals the Pearson correlation over entries.

(b) R(c tau) = c R(tau) for any real c not zero, and R(tau Pi) = R(tau) Pi for any p by p permutation Pi. So residual CEA is scale free, and shuffling the columns of R(tau) is the same as applying R to tau with shuffled columns.

(c) For any v in R^p with ||m|| + mean(v) not zero, R(tau + u v') = R(tau). In words: adding to each cue any multiple of the system's own mean profile does not change R. In particular, if tau = a b' for a K-vector a and a p-vector b with mean(b) not zero (one profile for every cue, scaled per cue), then R(tau) = 0 and residual CEA is undefined.

(d) If K = 2, R(tau) = 0 for every tau with m not zero.

*Proof.* (a) 1' m = 1' tau 1 / p = 0, so 1' u = 0. Then 1' R(tau) = 1' tau - (1' u) u' tau = 0. For the rows, R(tau) 1 = tau 1 - u u' tau 1 = p m - p u (u' m) = p m - p u ||m|| = 0. Mean-zero entries make the cosine equal to the Pearson correlation.

(b) The mean of c tau is c m, so its unit direction is sign(c) u and u u' is unchanged. Hence R(c tau) = (I - u u') c tau = c R(tau). For Pi, Pi 1 = 1, so tau Pi has the same m and the same u, and R(tau Pi) = (I - u u') tau Pi = R(tau) Pi.

(c) The mean of tau + u v' is m + u mean(v) = (||m|| + mean(v)) u. This is not zero by assumption, so its unit direction is plus or minus u and the projector I - u u' is unchanged. Then R(tau + u v') = (I - u u') tau + (I - u u') u v' = R(tau), since (I - u u') u = 0. For tau = a b', m = a mean(b), so u = plus or minus a / ||a|| and (I - u u') a = 0, giving R(tau) = 0. The cosine of a zero matrix is undefined, and the code returns NaN (`scripts/a4_analyse.py:39`).

(d) For K = 2 the zero-sum vectors in R^2 form a line, spanned by u. Every column of tau lies on it, so (I - u u') tau = 0. QED

Notes on S1.

- (b) justifies the cue-label null on the residual statistic: `cue_null_p` shuffles the columns of R(tau_model) against a fixed R(tau_human) (`scripts/a4_analyse.py:111-112`). By (b) this is the null of applying R after shuffling AU labels.
- (c) is the exact form of the amendment's sentence "A system that answers every cue with one profile, scaled by salience ... has nothing left after R" (`research/PREREG_A4_DRAFT.md:97`; docstring at `scripts/a4_analyse.py:89-90`). It holds when the single profile is exact in the effect matrix and the cue scalings do not average to zero. At mean(b) = 0 exactly, m = 0 and the code returns tau unchanged (`:93-95`).
- The recovery salience-profile observer uses one profile in logit weights, not in tau (`scripts/a4_recovery.py:87-91`). The logistic link makes its probability-scale tau only close to rank one. Its true residual CEA ranges from -.129 to .059 across the twelve dataset and strength cells, against .043 to .383 for raw CEA (`research/reproduction/a4_recovery.json`, `tables`, field `truth_mean`). So "nothing left" is approximate in practice.
- (d) means residual CEA needs at least three options. A4 has five or seven.

**Proposition S2 (relation to raw CEA, when mean profiles are parallel).** Let tau_S and tau_H have zero column sums, non-zero mean profiles, and parallel mean directions, u_S = plus or minus u_H. Write u = u_H, b_X = tau_X' u (a p-vector of cue weights on the common direction) and c_X = ||b_X||^2 / ||tau_X||_F^2 (the common-profile share). Then

    CEA(S, H) = sqrt(c_S c_H) cos(b_S, b_H) + sqrt((1 - c_S)(1 - c_H)) residual CEA(S, H)

where a cosine involving a zero vector is taken as 0; its weight is then 0 too.

*Proof.* Because u u' is the same for both, tau_X = u b_X' + R(tau_X) with u' R(tau_X) = 0. The cross terms vanish: <u b_S', R(tau_H)> = trace(b_S u' R(tau_H)) = 0, and likewise the other. Also <u b_S', u b_H'> = (u' u)(b_S' b_H) = b_S' b_H. So <tau_S, tau_H> = b_S' b_H + <R(tau_S), R(tau_H)>. The same split gives ||tau_X||^2 = ||b_X||^2 + ||R(tau_X)||^2, so ||b_X|| = sqrt(c_X) ||tau_X|| and ||R(tau_X)|| = sqrt(1 - c_X) ||tau_X||. Dividing by ||tau_S|| ||tau_H|| gives the identity. QED

Reading. The common-profile share c_H is what `human_energy` reports as `common_profile_share` (`scripts/a4_analyse.py:396`). It is .2764, .2634, .3242 and .3535 on basic WC, basic EA, conversational WC and conversational EA (`research/reproduction/a4_human_ceiling.json`). A system whose matrix is exactly one profile along u_H, with c_S = 1, has raw CEA equal to sqrt(c_H) cos(b_S, b_H). That can reach sqrt(c_H): .526, .513, .569 and .595, computed here from those four shares. These exceed the .30 floor (`scripts/a4_design.py:73`) with no cue-specific agreement at all. Its residual CEA is undefined (S1 c). This is the mechanism behind raw CEA's false Aligned rate of up to .40 on conversational WC in recovery, against .03 for residual CEA (`research/reproduction/a4_recovery_run3.json` and `a4_recovery.json`, `tables`, highest `aligned` over the different-cues, prior-only, generic and salience-profile observers). When mean profiles are not parallel the cross terms do not vanish, and no such identity is claimed.

### 2.3 What residual CEA does and does not say

**What it says.** Residual CEA is the cosine between the cue-specific parts of two effect matrices, where "cue-specific" means what remains after removing, separately for each system, the component of every cue's effect along that system's own average cue effect. It is unchanged by rescaling either matrix, and by adding to any cue any multiple of the system's own mean profile (S1 b, c). So a system whose cues differ only in how strongly they push one shared response pattern cannot score on it.

**What it does not say.**

- It discards the common component. Two systems with different, or opposite, mean profiles can have residual CEA of 1. That component holds .26 to .35 of human squared effects (`research/reproduction/a4_human_ceiling.json`, `common_profile_share`). Raw CEA, reported as secondary, carries it.
- It is invariant to adding only the system's own mean profile. Adding a common response pattern in any other direction changes u, and so changes R. A model that answers "None of the above" less whenever anything moves is removed exactly only if that pattern is along its own mean profile. The amendment's protection against such a model comes from the cue-label null, which keeps any effect common to all cues (`research/PREREG_A4_DRAFT.md:113`; `scripts/a4_analyse.py:103-105`).
- The two systems are projected along different directions. Residual CEA is not a cosine within one shared subspace.
- It does not remove option priors, and does not need to. A cue-independent bias added to the probabilities cancels in tau's difference, for raw and residual CEA alike. A bias on the logit scale does not cancel, because the effects are on the probability scale through the logistic link and renormalisation (`research/scripts/human_au_effects.py:163-169`). It rescales tau in a way R does not undo. A prior-only system is caught by the signal test, not by R (`research/PREREG_A4_DRAFT.md:111`).
- It is not linear in tau, because u is estimated from the same matrix. Noise in tau moves u. When ||m|| is small against that noise, u is poorly determined, and R is discontinuous at m = 0. For the human matrices m is far from zero, since the common share is .26 or more. For a model it is unknown in advance.
- It is undefined when R(tau) = 0: for K = 2, or for an exactly rank-one matrix along its own mean profile (S1 c, d). In the code a NaN cosine cannot meet the Aligned or Not aligned conditions, so `decide` returns Inconclusive (`scripts/a4_analyse.py:165-169`). A fitted matrix is never exactly rank one, so this is a limit case.
- Like CEA, a high value means a shared pattern, not the same strategy (`research/PREREG_A4_DRAFT.md:135`). It does not say the system uses the same cues by the same route.

## 3. Identifiability results, and whether they carry to residual CEA

The theory note's statements and proofs are not repeated here. This section says what each result states and whether it carries to residual CEA.

**Theorem 1 and Corollary 1.1** (theory note, lines 29-41). With finitely many natural items and the intervened points off the natural support, any zero-sum effect matrix is reachable by a system that matches a given system on every natural item. So for any r in [-1, 1] there is a system behaviourally identical to H on every behavioural metric with CEA(S, H) = r. Theorem 1 is about tau itself, so it carries to any function of tau. The corollary carries to residual CEA as follows.

**Corollary S3.** Under the conditions of Theorem 1 with T = H, suppose fbar = E_P[f_H] is in the interior of the simplex, tau_H has a non-zero mean profile m_H with direction u_H, R(tau_H) is not zero, and (K - 2)(p - 1) >= 2. Then for any r in [-1, 1] there is a system S with f_S = f_H on every natural item and residual CEA(S, H) = r.

*Proof.* Let W be the set of K by p matrices N with 1' N = 0, N 1 = 0 and u_H' N = 0. Its columns lie in the (K - 2)-dimensional subspace orthogonal to 1 and u_H, and the row constraint removes one column's freedom, so W has dimension (K - 2)(p - 1) >= 2. R(tau_H) is in W, by S1 (a) and u_H' R(tau_H) = 0. Let U = R(tau_H) / ||R(tau_H)||, take a unit V in W orthogonal to U, and set N = r U + sqrt(1 - r^2) V, which is in W and has cosine r with R(tau_H). Put M = N + u_H 1_p'. Its columns sum to zero, its mean profile is N 1 / p + u_H = u_H, which is not zero, so R(M) = (I - u_H u_H') M = N - u_H (u_H' N) + u_H 1' - u_H 1' = N. Interior fbar gives a > 0 small enough that fbar + a M(:, j) is on the simplex for all j. Theorem 1 with tau* = a M gives S with f_S = f_H on the natural items and tau_S = a M. By S1 (b), R(tau_S) = a N, so residual CEA(S, H) = cos(N, R(tau_H)) = r. QED

So behaviour on natural items does not constrain residual CEA either. In A4, (K - 2)(p - 1) is 120 for basic and 72 for conversational. The theory note's caveats apply unchanged: the construction is not smooth, it shows a logical gap and not how trained networks behave, and the set-to form is covered only when neither intervened point is a natural item (theory note, lines 17, 41).

**Proposition 2** (theory note, lines 43-53). Two linear observers on correlated cues: behaviour converges as rho goes to 1 while CEA stays at 0. **It does not carry as stated.** The pair has two options and one cue each, so each matrix is one profile and R removes all of it (S1 c, d). Residual CEA is undefined for that pair. A three-option analogue keeps the message:

*Example S4.* Let cues 1 and 2 have correlation rho and cue 3 be independent of both, all standardised. Let a and c be linearly independent zero-sum K-vectors, K >= 3, and let f_S(z) = f0 + a z_1 + c z_3 and f_T(z) = f0 + a z_2 + c z_3, with Q = P, unit doses and values small enough to stay on the simplex. Then E||f_S - f_T||^2 = 2 (1 - rho) ||a||^2, which goes to 0 as rho goes to 1, while residual CEA(S, T) = 1/2 for every rho.

*Proof.* f_S - f_T = a (z_1 - z_2), and E(z_1 - z_2)^2 = 2 (1 - rho). For linear observers tau is the weight matrix at unit dose: tau_S = [a, 0, c], tau_T = [0, a, c]. Both have mean profile (a + c) / 3, which is not zero by independence, so both use the same projector P = I - u u' with u parallel to a + c. Let alpha = P a. Then P c = P(a + c) - P a = -alpha, and alpha is not zero, since a is not parallel to a + c. So R(tau_S) = [alpha, 0, -alpha] and R(tau_T) = [0, alpha, -alpha]. Their inner product is ||alpha||^2 and each norm is sqrt(2) ||alpha||, so the cosine is 1/2. QED

The shared cue 3 sets the 1/2. The two systems' use of the correlated pair contributes nothing to residual CEA, while their behaviour becomes identical.

**Proposition 3** (theory note, lines 55-80). For linear observers, ||tau_S - tau_T||_F^2 lies between delta^2 / lambda_max and delta^2 / lambda_min times the behavioural mean squared difference. **The upper bound carries when the two systems share a mean-profile direction; the lower bound does not.** If u_S = plus or minus u_T, then R(tau_S) - R(tau_T) = P (tau_S - tau_T) with one projector P, and ||P X|| <= ||X||, so the upper bound holds for the residual matrices. The lower bound fails: take D = B_S - B_T = u w' with w not zero. Then tau_S = tau_T + delta u w', which keeps the mean direction (S1 c), so R(tau_S) = R(tau_T), while the behavioural difference is w' Sigma w > 0. When the directions differ, R is applied with different projectors and no bound is stated here. The measured spectra in the theory note's table (lambda_min .054 on EU-Emotion LibreFace cues, against .44 to .47 on Snoek's 25 merged cues; lines 71, 78) describe the cue designs and do not depend on the statistic.

**Proposition 4** (theory note, lines 82-88). Regression of a response on measured cues can show alignment produced by an unmeasured cue. This concerns the same-clip analysis on natural items, not A4, and it is about the estimator, not the statistic. In A4 the cues are set at random and then rendered. Anything the renderer changes along with an AU is part of that AU's effect, so A4's tau is the total effect of setting the AU in these renders (`research/PREREG_A4_DRAFT.md:241, 246`).

**Proposition 5** (theory note, lines 90-97). See Section 4.

## 4. Noise ceiling

Proposition 5 shows that for independent, zero-mean noise the observed cosine approximates CEA times sqrt(R_S R_H), as a ratio of expectations. As implemented:

- **Human side.** Subjects and stimuli are both split in half, 200 times. Each half is fitted by a pooled ridge logistic, converted to family do effects, and the halves are compared by cosine. The mean is stepped up by Spearman-Brown [58, 59], and the stored "ceiling" is its square root, sqrt(R_H) (`scripts/a4_analyse.py:331-368`; `scripts/a4_design.py:50`). The residual ceiling repeats this with R applied to each half (`:362, 367`).
- **Model side.** The stimuli are split in half, 20 times, the matrix is refitted on each half, and R_S is the Spearman-Brown corrected mean cosine, separately for raw and residual (`scripts/a4_analyse.py:140-150`; `scripts/a4_design.py:48`).
- **Corrected value, descriptive.** `cfa / (ceiling * sqrt(r_s))`, for each statistic with its own ceiling and R_S (`scripts/a4_analyse.py:430-432`).
- **Decision.** Not aligned uses the interval's upper bound over sqrt(R_S), the model's reliability only (`scripts/a4_analyse.py:167`; `research/PREREG_A4_DRAFT.md:124`). Below R_S of .85 the uncorrected value is read as a lower bound (`scripts/a4_design.py:51`; `scripts/a4_analyse.py:433`).

| Dataset | sqrt(R_H), raw | sqrt(R_H), residual | Common-profile share | WC to EA reference, raw | WC to EA reference, residual |
|---|---|---|---|---|---|
| basic WC | .990 | .970 | .276 | .877 | .780 |
| basic EA | .989 | .983 | .263 | as above | as above |
| conversational WC | .978 | .974 | .324 | .767 | .732 |
| conversational EA | .974 | .960 | .354 | as above | as above |

Source: `research/reproduction/a4_human_ceiling.json` (fields `ceiling`, `ceiling_residual`, `common_profile_share`, `wc_to_ea_cfa`), rounded to three places. Each statistic is read against its own reference.

**Does Proposition 5 carry to residual CEA?** Only in part.

- If the projectors were fixed and known, R(tau-hat) = P tau + P e with P e zero mean and independent across systems. Proposition 5 then applies to the residual matrices unchanged, with s replaced by E||P e||^2, which is at most s.
- In A4 each projector is estimated from the noisy matrix. Then R(tau-hat) is not linear in the noise and Proposition 5 does not hold exactly. The split-half residual reliability measures the combined effect empirically, but no identity links it to the attenuation of residual CEA. The disattenuated residual value is therefore an approximation without a proof here.

**Spearman's correction.** Proposition 5 is Spearman's correction for attenuation [51] (`references.md` #51), written for matrices. Spearman's form is r_pq = r_p'q' / sqrt(r_p'p' r_q'q'), where r_p'p' and r_q'q' are correlations between independent repeated series of observations of the same quantity (Part 2, Section 2, "'Attenuation' by Errors", read in the 2010 International Journal of Epidemiology reprint). The split-half reliabilities here play that role, after the Spearman-Brown step-up [58, 59].

## 5. What the study claims and does not claim

**Claims, if the decision rule gives them.**

- Aligned, on a dataset: on these renders, the model's cue-specific effect pattern, after removing its own mean profile, has residual CEA with the human pattern of at least .30, with Holm-corrected cue-label p < .05, a bootstrap lower bound above 0, a significant AU signal and all gates passed (`research/PREREG_A4_DRAFT.md:121-125`; `scripts/a4_analyse.py:153-169`).
- Not aligned on these renders: the upper bound over sqrt(R_S) is below .30, with signal and gates passed (`research/PREREG_A4_DRAFT.md:124, 127`).
- That behaviour on natural items cannot settle either claim (Theorem 1, Corollary 1.1, Corollary S3).

**Does not claim.**

- That a model uses the same cues, or the same strategy, as humans. Recovery's pair-swapped observer, with 16 of 25 families swapped in 8 pairs, can still clear .30 (`research/PREREG_A4_DRAFT.md:135`).
- Anything about the common component. Residual CEA removes it by construction, and raw CEA is secondary.
- Human-level alignment, unless the interval reaches the residual between-group reference, .780 basic and .732 conversational (`research/reproduction/a4_human_ceiling.json`), and that comparison is descriptive (`research/PREREG_A4_DRAFT.md:135`).
- An accuracy claim for residual CEA on weak conversational effects. Pair-swapped accuracy at conversational 0.25 times is withdrawn, and residual CEA reads low there: .74 against a true .87 for the human-like observer on conversational EA (`research/PREREG_A4_DRAFT.md:199-201, 216`).
- That the noise-corrected residual value is exact (Section 4).
- Anything beyond these avatars, a static peak frame, these 25 families and these observer groups (`research/PREREG_A4_DRAFT.md:239-250`).

**Literature.** The human target and randomised AU design are Snoek et al. 2023 (`references.md` #19) with their data (#25). Error consistency, one of the behavioural metrics Theorem 1 covers, is Geirhos et al. (#15). The theory note's Open list names Teneggi and Sulam (#17) and Han et al. (#16) as needing a full read before any novelty claim. Section 6 compares them with these results. Neither paper contains or implies any of them, but this note still makes no claim of novelty for the propositions. As the theory note says, they are elementary, and the contribution is the estimand and its use.

## 6. Relation to prior work

Read for this section on 2026-11-08. Han, Poggio and Cheung (`references.md` #16): arXiv 2302.06677 v2, full text, figures by caption only. Teneggi and Sulam (#17): arXiv 2405.19146 v2, Sections 1 to 3, 4.1, the Section 5 overview with 5.1 and the 5.3 remarks, Section 6 and Appendix D.1.2. The rest of Sections 4 and 5 and Appendices A to C were not reread here. An earlier pass (`citation_check.md`) read the main text in full and skimmed the appendices. The comparison is ours. Neither paper mentions CEA or anything like it.

### 6.1 Han, Poggio and Cheung (#16)

**What it does.** It is an empirical simulation and has no theorem. Known trained networks stand in for brain recordings. Source models are scored against a target network's internal layer activations by ridge regression (median Pearson r over target units) and by linear CKA. The sources are AlexNet, VGG11, ResNet18, CORnet-S, ViT-B/32 and MLP-Mixer-B/16, and in a second test 12 CNNs and 14 ViTs. The question is whether the best-scoring source has the target's architecture. Under these ideal conditions it usually does. But for several layers of VGG11, ResNet18 and CORnet-S a different architecture scores highest, or does not score significantly lower. Identifiability depends on the stimulus set, where texture and noise images do worst, and also on target coverage and depth. CNN and ViT classes overlap on many layers. Its "identifiability index" is a normalised score difference, not identifiability in the statistical sense. It recommends more natural stimuli and more recorded units. It also proposes "specially designed, critical stimuli to distinguish between different architectures" (Discussion).

**Relation to CEA and residual CEA.**

- *Object and data.* #16 asks about architecture and uses internal activations. CEA asks about cue effects, a property of the input-output map under intervention, and uses output distributions. #16 tests a finite set of trained networks. Theorem 1 covers all systems.
- *Theorem 1 and Corollary 1.1.* Neither is in #16, in statement or in substance. #16 has no existence or impossibility result, and it never evaluates a system off its stimulus set. The shared idea is only qualitative: a similarity score over a stimulus set is a many-to-one summary ("Such ambiguity is due to the many-to-one mapping of a model onto a leaderboard score", section 1.1). Theorem 1 makes one version of that exact, for output behaviour and cue effects. It is not a special case of anything in #16, and #16 is not a special case of it.
- *Propositions S1 and S2, Corollary S3, Example S4.* Nothing in #16 corresponds. Regression scores and CKA are not effect matrices, and no projection like R appears.
- *Direction of the evidence.* #16 is evidence about trained networks, which Theorem 1 does not address. With natural images under ideal conditions, activation similarity found the right architecture most of the time. That does not contradict Theorem 1, which is a logical limit. It does show that the gap Theorem 1 allows need not be large in trained networks, so the theorem should be presented as a limit and not as an expectation (theory note, "What this does not say"). A reviewer may set #16's call for natural stimuli against our claim that natural items cannot settle cue use. The targets and the metrics differ, and #16's critical-stimuli proposal is close to the designed intervention A4 uses.

### 6.2 Teneggi and Sulam (#17)

**What it does.** It sets out a testing framework for one fixed classifier. Concepts are scores z_j = <c_j, h>: the model's own embedding h projected onto concept vectors c_j from a text encoder. So "both Ŷ and Z_j are fixed functions of the same random variable H" (Section 3.1). There are three null hypotheses:

- global importance, Ŷ independent of Z_j (Definition 1);
- global conditional importance, Ŷ independent of Z_j given Z_-j (Definition 2);
- local conditional importance for one input. This compares predictions on embeddings sampled given the observed concepts with and without j (Definition 3).

Each null is tested sequentially by betting, with kernel MMD payoffs (SKIT, c-SKIT, x-SKIT). The tests control Type I error (Propositions 1 and 2) and, through a greedy post-processor, false discovery rate. Concepts are ranked by rejection time. The experiments use Gaussian synthetic data, counting MNIST digits, and zero-shot classification on AwA2, CUB and Imagenette with eight CLIP-style encoders. Ranks are compared across encoders by a weighted Kendall's tau, and against the datasets' concept annotations by F1. The formal results are the validity of the tests and Lemma 1: for a linear classifier with d of at least 3, independence of Ŷ and Z_j neither implies nor follows from orthogonality of w and c_j.

**Relation to CEA and residual CEA.**

- *Estimand.* CEA's tau is interventional. It is the change in each option's probability when cue j is set to 1 rather than 0 in the stimulus, other cues as observed, averaged over the design Q. That gives a signed K-vector per cue, and two systems are compared by the cosine of their matrices (Section 1). #17 tests whether statistical dependence, marginal or conditional, exists under the data distribution, and it ranks concepts by rejection time. It estimates no effect size and no sign. c-SKIT draws Z_j from P(Z_j | Z_-j) in the test data and does not query the model again (its eq. 13). x-SKIT picks embeddings by nearest neighbour so that "samples are coming from real images" (Appendix D.1.2). Both stay on the natural support. In #17 the concepts are defined in the model's embedding, not in the stimulus, so two models are probed on different concept variables.
- *Comparing systems.* #17's nearest analogue to CEA is its rank agreement across eight encoders. That compares importance ranks between models on natural data. It does not compare effect patterns, and it has no human reference.
- *Theorem 1 and Corollary 1.1.* Neither is in #17, in statement or in substance. Where the concepts are defined by the stimulus and the tests run on natural data, Theorem 1 applies to #17's tests. Two systems that agree on every natural item give the same answers to the global and global conditional nulls, whatever their effects off the natural support. This is our inference, and #17 does not discuss it. Where the concepts are embedding projections, two systems are not even probed on the same variables.
- *Proposition 2 and Example S4.* The nearest material in #17 is its remark that marginal importance "ignores the information contained in the rest of them, and concepts may be correlated" (Section 3.1), with a Gaussian example in which Z_3 depends on Z_1 (Section 4.1). This motivates its conditional null, and it is the standard distinction between marginal and conditional dependence. It is not our result: it has no pair of observers, no effect matrix and no limit as the correlation goes to 1.
- *Proposition 3, a point against a broad reading of our framing.* c-SKIT finds dependence on a concept given the others from residual natural variation, without breaking correlations. Our own Proposition 3 says the same for linear observers with lambda_min above 0: natural data do bound their effects. So "to learn which cues a system uses, one has to break the natural correlations between cues" (`introduction.md`) holds in general only because of the off-support freedom in Theorem 1, unmeasured cues (Proposition 4) and misspecified or nonlinear observers. For a well-specified smooth observer with a well-conditioned cue covariance and no relevant unmeasured cue, natural data carry information about cue use.
- *Propositions S1 and S2, Corollary S3, Proposition 5.* Nothing in #17 corresponds. It has no mean-profile removal, no cosine between systems and no reliability correction. Lemma 1 is the nearest technical item, and it concerns one classifier and one concept vector.
- *Overlap with the study's question.* #17 lists as future work "user-studies to quantitatively evaluate the alignment of the produced ranks with human intuition" (Section 5.3). Those would be intuition ratings, not randomised human cue effects.

### 6.3 What this means for the claims

- Neither paper states or proves any result of ours, in this note or in the theory note, and none of ours is a special case of a result in either. Neither has an identifiability theorem: #16 is empirical, and #17's formal results are the validity of its tests and Lemma 1.
- The reading supports no further novelty claim for the proofs. Theorem 1 rests on the elementary fact that a function's values on a finite support do not fix its values off it. S1 is linear algebra. Proposition 5 is Spearman's correction [51]. Other literatures were not searched for this section, including positivity and extrapolation in causal inference and identifiability of learned representations. So no claim that the results are new beyond these two papers is supported.
- What neither paper has is the estimand and its use: an interventional, per-option cue-effect matrix, compared by cosine between two systems one of which is a human group, its residual form, and a randomised human design to estimate it. That matches the theory note's framing and is the contribution this note claims.
- Overlap that limits claims. First, Proposition 3 and #17 both show that natural data constrain cue use for some systems. The Introduction's sentence on breaking correlations needs a qualifier such as "or assume that no relevant cue is unmeasured and the observer is well specified", as `citation_check.md` also notes. Second, #16's success in the ideal case means Theorem 1 should keep its stated limit: it shows a logical gap, not how trained networks behave.

### 6.4 Policy capturing, the lens model, and Weller and Barkett (#62, #65, #66)

Added 2026-11-09. #62 read in full (`references.md`). #65 and #66 are cited for the existence and aim of a literature only, at the reading level stated in `references.md`; no specific result of theirs is used.

**The older literature.** Judgement research has compared people's cue weights for decades. In the lens model tradition, a judge's responses and the environment are each fitted by linear regression on the same cues, and the fits are compared [65]. Policy-capturing studies fit each judge's weights from profiles in which cues are varied by design, and guidance for that method treats the correlation among cues in the profiles as a design issue [66]. Comparing two judges by their cue weights, with the cues varied by the experimenter, is therefore not new, and we do not claim it. What this study adds to that tradition is narrower: the comparison is between a model and a human group on faces, the human effects come from randomised facial action units, the effects are per response option, and the residual form removes a shared mean profile (Section 2).

**Weller and Barkett [62].** Process alignment there is the cosine between two ridge-logistic coefficient vectors, one fitted to an organisation's historical decisions and one to an LLM's decisions on the same case profiles, drawn from real data. No cue is varied independently. In our terms it is an observational, regression-based cosine of the form Proposition 4 addresses: where cues are correlated or a relevant cue is unmeasured, two decision-makers can get similar coefficients without sharing effects, and Theorem 1 says agreement on natural cases cannot settle the question in general. Their setting (text decisions, no faces, no randomised human design) does not overlap with ours, and their paper does not claim cue use in the interventional sense. It is the nearest current use of the same functional form in model alignment that we located.

**Consequence for the claims.** The contribution claimed in Section 6.3 stands, with the qualifier that comparing cue weights between judges is old, and that CEA's form, a cosine of cue-effect profiles, is shared with recent observational work. The novelty is in applying an interventional comparison to a vision-language model against randomised human facial cue effects, not in the idea of comparing cue weights.

## 7. Open gaps

1. **Proposition 5 for residual CEA.** No exact attenuation result with an estimated projector. A delta-method or simulation bound is needed. The recovery runs show behaviour on synthetic observers but do not prove it.
2. **Human ceiling estimator.** The ceiling halves are pooled fits at ridge 1.0 for all four datasets (`scripts/a4_analyse.py:360`). The human target is the mean of per-subject fits at the selected ridge, which is 0.1 for basic (`research/scripts/human_au_effects.py:585-590`; `research/reproduction/human_au_effects.json`). So R_H is the reliability of a related estimator, not of the target itself. The amendment does not state this. The earlier log-odds ceiling used the same default (`research/scripts/human_au_effects.py:353, 401-402`).
3. **Proposition 3 with different projectors.** No bound is given when u_S and u_T differ. R is discontinuous at m = 0, so any bound needs ||m|| bounded away from zero.
4. **Proposition S2 with non-parallel profiles.** The cross terms are not bounded here.
5. **Set-to form of Theorem 1.** Unchanged from the theory note: the case where an intervened point is a natural item is not covered.
6. **Finite-sample bound on CEA**, raw or residual, remains open (theory note, Open).
7. **Citations.** Spearman is now `references.md` #51. The Spearman-Brown step-up (Section 4) is now `references.md` #58 and #59. The Introduction's sentence on breaking natural correlations needs the qualifier set out in Section 6.3.
8. **Literature beyond #16 and #17.** Positivity and extrapolation in causal inference, and identifiability of learned representations, were not searched (Section 6.3).
