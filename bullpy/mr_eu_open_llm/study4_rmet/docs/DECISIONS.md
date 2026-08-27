# DECISIONS — PROTOCOL V2 CARD mech redesign

## Checkpoint
Pre-redesign results copied to `checkpoints/pre_v2_card_mech_20260806/` (restore by copying `results/` back).

## Framing
- Flagship = profile-diagnostic vs generic ambiguity, with causal reuse/steer as the mech contribution.
- Legacy H1/H2 demoted to Result 0 (boundary condition).
- No EU-Emotion migration; no ToM-circuit claims.

## Construct choices
- Primary diagnosticity = EQ logistic slope (`trait_sensitivity_coef`).
- Secondary = ASC−control accuracy gap (alexithymia unavailable in CARD).
- Human entropy = Shannon over overall 4AFC choices (timeouts excluded).
- Item classes = median splits on human data only (pre-registered in `item_classes_preregistered.json`).
- No global label RDM (item-specific foils).

## Analysis stack
- B1/B2: `behavioural_profile_alignment.py` on existing evals (k=10 lower bound).
- M1: LOO Ridge probes + RSA vs feature/entropy/diagnosticity RDMs (`rsa_probe_card_axes.py`).
- C1 geometry: activation-geometry axes + class reuse (`causal_rmet_axes.py`).
- C1 causal: `steer_rmet_axes.py` wraps parent hook patterns (ADD α·axis; `last_token` + `all_tokens`); Slurm `rmet_causal_steer_smoke.sh`. Entropy + random controls; ΔJS to low-EQ/ASC CARD soft targets; item-class reuse (not EU cross-emotion).
- Contamination: `contamination_option_order_smoke.py` (+ offline limitation JSON).

## Isolation
All new code under `study4_rmet/`. Parent scripts read-only.

## C1 decision (2026-08-12 job 33536702)
- qwen3vl L4 full steer: **near-null / non-specific** (~84% zero ΔJS; largest mean shift on **random** axis).
- Paper lane locked: behavioural negatives (Result 0 + B1) + controlled causal null; geometry/probes demoted to exploratory.
- Do not run gemma/molmo steer unless null-robustness is explicitly desired.
- Optional only: k≥20 re-eval for B1/B2 soft labels; contamination option-order GPU smoke; MRMET if requested later.
