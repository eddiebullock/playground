# Causal Feature Alignment: estimand and identifiability

**Status:** working note, 2026-11-07. Proofs are complete for the statements as written. Numbers come from `research/scripts/cfa_identifiability.py` (output `research/reproduction/cfa_identifiability.json`, tests in `tests/test_cfa_identifiability.py`). No model output was used.

Plain summary: two systems can give the same answers on every natural item and still use cues in unrelated ways. Only an intervention that breaks natural cue correlations can tell them apart. Proposition 3 says how much behaviour can reveal: it depends on the smallest eigenvalue of the natural cue covariance.

## Setup

A stimulus has cues z in R^p. A system S maps z to a response distribution f_S(z) on K options. Natural stimuli have distribution P. A design Q gives the stimuli used for measuring effects, and delta_j > 0 is a chosen dose for cue j.

**Effect matrix.** For cue j and option k,

    tau_S(j, k) = E_{z~Q}[ f_S^k(z + delta_j e_j) - f_S^k(z) ]

Each row sums to zero, because f_S lies on the simplex.

**Causal Feature Alignment.** For systems S and T,

    CFA(S, T) = cos(tau_S, tau_T) = <tau_S, tau_T> / (||tau_S||_F ||tau_T||_F)

Row sums of zero make the mean of the entries zero, so this equals the Pearson correlation over entries. It is undefined if either matrix is zero.

**Behavioural metric.** Any functional of the joint distribution of items and responses on P. This covers accuracy, confusion matrices, error consistency (Geirhos et al.), and representational similarity computed on outputs. If f_S = f_T P-almost everywhere, S and T have the same value on every such metric.

The human effect matrix tau_H comes from a randomised design. Nothing below assumes human responses are linear or that the cue list is complete, except where stated.

## Theorem 1: behaviour on natural items does not constrain alignment

Let P have finite support Z, take Q = P, and suppose the map (z, j) -> z + delta_j e_j is injective on Z x {1..p} with image disjoint from Z. Let T be any system, with fbar = E_P[f_T], and let tau* be any p by K matrix with zero row sums and fbar + tau*(j, .) on the simplex for all j. Then there is a system S with f_S = f_T on Z and tau_S = tau*.

**Proof.** Set f_S = f_T on Z, and f_S(z + delta_j e_j) = fbar + tau*(j, .) for each z in Z and each j. Injectivity and disjointness make this well defined. It is a distribution by assumption. Off these points, take any value. Then

    tau_S(j, k) = E_P[fbar^k + tau*(j, k) - f_T^k(z)] = fbar^k + tau*(j, k) - fbar^k = tau*(j, k). QED

**Corollary 1.1.** Let H be a reference system with tau_H non-zero, let fbar be in the interior of the simplex, and let p(K - 1) >= 2. For any r in [-1, 1], there is an S matching T on every natural item with CFA(S, H) = r. In particular, take T = H. Then S is behaviourally indistinguishable from H on Z, on every behavioural metric, however many trials are collected, and yet CFA(S, H) can be any value, including -1.

**Proof.** Let U be the unit vector tau_H / ||tau_H||_F. The space of zero-row-sum matrices has dimension p(K - 1) >= 2, so there is a unit vector V in it orthogonal to U. Put M = r U + sqrt(1 - r^2) V, which has cosine r with tau_H. Interior fbar gives a > 0 small enough that fbar + a M(j, .) is on the simplex for all j. Apply theorem 1 with tau* = a M. Cosine is scale free. QED

**What this does not say.** The construction changes f off the natural support and is not smooth, so it shows that behaviour imposes no logical constraint. It does not show that trained networks behave this way. Propositions 2 and 3 give the smooth version, and the empirical question is what the results of this study address.

## Proposition 2: correlated cues, smooth systems

Let z have two standardised cues with correlation rho, and let S and T be linear observers on one output: f_S(z) = a + b z_1 and f_T(z) = a + b z_2, each with Q = P and small enough b that outputs stay in range when mapped to a two-option distribution. Then:

    corr(f_S, f_T) = rho
    E[(f_S - f_T)^2] = 2 b^2 (1 - rho)
    CFA(S, T) = 0, for equal doses

The first two lines are direct computation. For the third, take tau_S = delta (b, 0) and tau_T = delta (0, b) on that output. They are orthogonal.

So as rho -> 1, behaviour converges and alignment stays at 0.

## Proposition 3: what behaviour can and cannot bound

Let f_S(z) = a_S + B_S z and f_T(z) = a_T + B_T z be linear observers, with B a K by p matrix, and let P have covariance Sigma with smallest eigenvalue lambda_min. Let D = B_S - B_T, with equal doses delta on all cues and means matched so that a_S + B_S mu = a_T + B_T mu. Then

    (delta^2 / lambda_max) E_P ||f_S - f_T||^2 <= ||tau_S - tau_T||_F^2 <= (delta^2 / lambda_min) E_P ||f_S - f_T||^2

Both bounds are attained. If lambda_min = 0, behavioural equivalence places no constraint on the upper side.

**Proof.** With means matched, f_S - f_T = D(z - mu), so E_P ||f_S - f_T||^2 = tr(D Sigma D') . With s_i the rows of D, tr(D Sigma D') = sum_i s_i' Sigma s_i, which lies between lambda_min ||D||_F^2 and lambda_max ||D||_F^2 by the Rayleigh quotient. For linear observers tau_S - tau_T = delta D', up to transposition, so ||tau_S - tau_T||_F^2 = delta^2 ||D||_F^2. Substituting gives both bounds. Equality holds when every row of D is a multiple of the eigenvector for lambda_min, or for lambda_max. QED

**Reading.** 1 / lambda_min is an invisibility factor: the most that a unit of behavioural disagreement can understate a squared difference in cue effects, per unit dose. A randomised design makes Sigma close to the identity, which is why it identifies cue effects. For a softmax observer the same statement holds locally, on the Jacobian of the log-odds at a reference point.

**Measured values, correlation matrices of the cue tables.**

| Cue table | Rows | Cues | lambda_min | Condition number | 1 / lambda_min | Mean abs r |
|---|---|---|---|---|---|---|
| EU-Emotion, LibreFace | 243 clips | 15 | .054 | 76.1 | 18.4 | .23 |
| EU-Emotion, OpenFace 3 | 243 clips | 10 | .125 | 22.4 | 8.0 | .21 |
| Snoek design, basic WC | 4,659 configurations | 33 | .178 | 13.7 | 5.6 | .036 |
| Snoek design, basic EA | 2,322 | 33 | .199 | 11.3 | 5.0 | .038 |
| Snoek design, conversational WC | 2,314 | 33 | .209 | 11.0 | 4.8 | .038 |
| Snoek design, conversational EA | 2,322 | 33 | .199 | 11.3 | 5.0 | .038 |

Limits on these numbers. Cue counts differ. Most of the Snoek lambda_min comes from left and right versions of one AU, which are correlated by design (largest r about .8), so the design is much better conditioned for the family-level cues used in the study than the table shows. The EU-Emotion cues are detector estimates, so measurement noise changes the spectrum, and clip-level means ignore within-clip variation. Whether correlations differ by label or actor was not examined here.

## Proposition 4: measured-cue regression can show alignment without shared cues

Let T respond linearly to an unmeasured cue U, f_T = a + b U, and let U = G z + e on P, with e uncorrelated with z and a matrix G of regression slopes. Then the population regression of f_T on z has slopes b G, whereas the true effect of intervening on z alone is zero unless the edit also changes U. Two systems, one using z directly with weights bG and one using U, have the same regression slopes on P.

**Proof.** The slope is Cov(f_T, z) Var(z)^-1 = b Cov(U, z) Var(z)^-1 = b G. An intervention on z that leaves U fixed leaves f_T fixed. QED

This is why the confirmatory claim for same-clip analysis is "aligned with the human pattern", not "uses the same cues" (DECISIONS 2026-11-04). The pattern score is a regression result. Only interventions (RQ2) and randomised human effects (G5) can reach the causal claim.

## Proposition 5: noise ceiling

Let tau-hat_S = tau_S + e_S and tau-hat_H = tau_H + e_H, with e_S and e_H zero mean, independent of each other, and with sum of variances s_S and s_H over entries. Then

    E<tau-hat_S, tau-hat_H> = <tau_S, tau_H>
    E||tau-hat_S||^2 = ||tau_S||^2 + s_S, and likewise for H

So the observed cosine approximates CFA times sqrt(R_S R_H), where R_S = ||tau_S||^2 / (||tau_S||^2 + s_S), as a ratio of expectations. This is Spearman's attenuation correction, on a matrix. The study estimates R_H by split-half across subjects (`human_au_effects.py`), and R_S from replicate renders or identities. When the model sees other stimuli than humans did, stimulus sampling enters s_H and s_S, so the `subjects_and_stimuli` ceiling applies, not the same-stimuli one. Corrected values above 1 can occur from sampling and are reported uncorrected beside them.

## What the study needs from this

1. Report tau for the model under a design that breaks natural correlations, and do not treat the same-clip pattern score as causal.
2. Compare CFA with error consistency and output similarity on the same system. The claim needs a demonstration in which they disagree, and whether one exists here is an empirical result, not an assumption.
3. Recovery: synthetic observers with known tau through the same pipeline, including the Proposition 2 pair.

## Open

- A finite-sample version: a confidence bound on CFA given trials per stimulus, stimuli per cue and subjects.
- Whether the bound in Proposition 3 can be tightened for softmax observers with bounded Jacobians.
- Relation to Teneggi and Sulam's concept tests, and to Han et al.'s system-identification limits, needs a full read before any claim of novelty for these results. They are elementary, so the contribution is the estimand and its use, not the proofs.
