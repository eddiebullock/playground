# PROTOCOL V2 — CARD-grounded psychometric × mech-interp on RMET

Working title: *Do VLMs encode CARD profile-diagnostic structure on RMET, or only a generic ambiguity route?*

Status: redesign of study4_rmet (Aug 2026). Supersedes the Phase-1 H1/H2 headline framing; legacy results retained as **Result 0**.

Checkpoint of pre-redesign outputs: `study4_rmet/checkpoints/pre_v2_card_mech_20260806/`.

---

## 1. Contribution bar (non-negotiable)

This paper is an **important contribution only if** it delivers a clear answer to:

> Do VLMs encode CARD **profile-diagnostic** structure on RMET, or merely a **generic ambiguity/difficulty** route — and is that distinction **causal**?

### Acceptable high-value outcomes
- **Causal dissociation:** diagnosticity axis ≠ entropy/ambiguity axis under patch/steer/reuse.
- **Strong negative with controls:** surface competence / above-chance accuracy without profile-aligned mechanism.
- **Positive alignment** that survives causal tests (less likely given Result 0 nulls; fine if true).

### Unacceptable as the headline alone
- Accuracy tables vs humans (crowded field: Strachan et al. 2024; cross-ethnic MLLM RMET).
- Uncorrected multi-layer RSA fishing.
- EQ-slope ↔ model accuracy Spearman on n=36 as the main claim (demoted to Result 0).

### Design filter
If an analysis would still be the same paper after deleting CARD traits/ASC, demote or delete it.  
If it would still be the same paper after deleting causal reuse/steer, it is not yet the mech contribution.

---

## 2. What this study is / is not

**Is:** A test of whether VLMs that perform RMET encode the individual-differences / profile-conditioned structure those items capture in humans (EQ continuum, SQ/D, ASC), or only a generic route to labels (ambiguity / entropy / difficulty).

**Is not:**
- Another multimodal RMET leaderboard.
- A claim that RMET = theory of mind (prefer: advanced eye-region mental-state / complex emotion recognition).
- A duplicate of the EU-Emotion ambiguity / confused-pair mech paper.
- “We ran probes and RSA” as the contribution (methods are tools; novelty is the CARD ID target + reuse dissociation).

**Stay on RMET + CARD.** Do not migrate flagship analyses to EU-Emotion.

---

## 3. Literature positioning

| Anchor | How we position |
|--------|-----------------|
| Accuracy-saturated RMET–VLM (Strachan et al. 2024 GPT-4o; cross-ethnic MLLM RMET) | Beyond mean accuracy → psychometric / profile structure |
| Error-space (Strachan concentration / structure) | Add **trait-/ASC-conditioned** CARD targets, not only mean human confusion |
| Human–AI difficulty misalignment | Capability ≠ human struggle / trait-linked structure |
| RMET construct caveats (emotion recognition vs ToM; verbal IQ; alexithymia comorbidity; weak factors) | Constrain claims; no “ToM circuits” |
| Emerging VLM emotion mech-interp (probe/patch/steer) | Tools; novelty = CARD individual-differences target + class reuse |

---

## 4. Unique data

- CARD EyesTest item-level: `data/processed/card_rmet_item_level.csv` (N≈2907)
- Traits: `eq_total`, `sq_total`, `d_score`, `aq_total`, `spq_total`, `asc_diagnosis` (~991 ASC)
- Answer key + eyes-only stimuli (no 4AFC labels in pixels)

**Critical constraint:** RMET options are **item-specific**. There is no shared 20-state confusion matrix. Do **not** fake EU-style cross-emotion reuse. Reuse is defined over **item classes** (high vs low trait-diagnosticity; high vs low entropy).

---

## 5. Research questions

### Primary
Do VLMs encode the item-level / choice structure that differentiates human cognitive profiles on RMET (EQ, SQ/D, ASC), and is that structure causally used — or do they solve RMET via a generic ambiguity/difficulty mechanism that ignores trait-diagnostic geometry?

### Secondary
1. **Behavioural signature.** Model response entropy ↔ human choice entropy; model soft labels ↔ profile-conditioned human distributions (low-EQ vs high-EQ; ASC vs non-ASC). Compare to legacy EQ-sensitivity ↔ accuracy.
2. **Geometry.** Activation RDMs vs human targets from entropy / diagnosticity / stratified divergence — not primarily |EQ_i − EQ_j|.
3. **Probing.** Is trait-diagnosticity linearly decodable? Is human entropy decodable as a control (generic ambiguity)?
4. **Causal localisation / steering / patching.** Does intervening on a diagnosticity-derived axis shift predictions toward low-EQ / ASC-like CARD patterns?
5. **Reuse (key discriminator).** Does the diagnosticity site/direction also drive high-entropy low-diagnostic items (and vice versa)? Shared → generic; dissociable → profile-relevant.

---

## 6. Pre-registered primary tests

| ID | Test | Primary? |
|----|------|----------|
| **B1** | Human ↔ model Shannon entropy correlation (n=36 items; Spearman + permutation; bootstrap CI) | Yes |
| **B2** | Profile-conditioned distribution alignment: mean JS/KL of model soft labels to low-EQ / high-EQ / ASC human choice distributions (vs overall); compare to legacy H1 | Yes |
| **M1** | Probe peak: decode trait-diagnosticity vs human entropy from activations (same layers); report both; multiplicity-honest | Yes |
| **C1** | Reuse / steer dissociation: diagnosticity axis vs entropy axis; transfer across pre-registered item classes; random-direction + shuffled-class controls | Yes (causal headline) |
| Result 0 | Legacy H1/H2 (EQ-sensitivity ↔ accuracy; trait-sensitivity RDM RSA) | Demoted boundary |

### Pre-registered item classes (from human data only)
- **High vs low trait-diagnosticity:** median split on EQ logistic slope (`trait_sensitivity_coef`); secondary: ASC−control accuracy gap.
- **High vs low human entropy:** median split on Shannon entropy of overall choice distribution.

---

## 7. Constructs

| Name | Definition |
|------|------------|
| `human_entropy` | Shannon entropy of overall 4-option choice distribution (valid choices 1–4) |
| `trait_diagnosticity` | Per-item EQ logistic slope (z_EQ → correct); also report ASC gap and low−high EQ accuracy gap |
| `eq_stratified_confusion` | Choice distributions by EQ tertile; JS/KL(low-EQ ‖ high-EQ) |
| `asc_stratified_confusion` | Choice distributions ASC vs non-ASC; JS/KL |
| `model_entropy` | Shannon entropy of model sample soft labels (k samples) |
| `diagnosticity_axis` | Mean activation difference (high − low diagnosticity items) at a layer |
| `entropy_axis` | Mean activation difference (high − low entropy items) |

---

## 8. Model scope honesty

| Arm | Models | Role |
|-----|--------|------|
| Open-weight mech | qwen3vl, gemma4, molmo2 | Probe / RSA / patch / steer |
| Commercial behavioural | GPT-5, Claude Opus, Gemini Flash | Behavioural B1/B2 only |

Open models currently ~25–47% accuracy (qwen best). Treat mech results as **mechanism-under-moderate-competence**, not SOTA social cognition. Caveat near-chance models (molmo, gpt5) harshly.

For new soft-label tests, prefer k≥20–50 samples/item when re-running; current k=10 is a lower bound.

---

## 9. Robustness / limitations (must report)

1. **Contamination:** Classic RMET is web-exposed. Eyes-only crops reduce label leakage; option-order / paraphrase smoke where feasible; interpret upright accuracy cautiously. MRMET = future out-of-sample (not blocking).
2. **Construct:** Prefer “mental-state / complex emotion recognition from eye region.” No ToM-circuit claims. Alexithymia not in CARD — document as limitation for ASC contrasts.
3. **Stats / power:** n=36 detects |r|≳0.45 at 80% power. Do not headline scalar Spearman as confirmatory. Permutation/bootstrap; multiplicity honesty for layer×model.
4. **VLM intervention:** Last-token patching may fail for vision-dependent behaviour. Start with last-token (compatible with extracts); document / try multi-position when hooks allow; always include random-direction and shuffled-class controls; entropy axis as control vs diagnosticity.
5. **Isolation:** All code under `study4_rmet/`; read-only parent helpers; no study3 edits.

---

## 10. Non-claims

- Not discovering theory-of-mind circuits.
- Not EU-Emotion confusion geometry / shared-label reuse.
- Not a claim that above-chance RMET implies human-like trait structure (Result 0 already falsifies that for EQ-difficulty maps).
- Not diagnostic framing of ASC as the gold-standard “deficit” for machine errors (dimensional EQ/SQ/D primary; ASC secondary covariate).

---

## 11. Result 0 (legacy — preserve)

From `results/SUMMARY.md` and `results/robustness/`:

- H1 EQ trait-sensitivity ↔ model accuracy: **null** (6 models).
- H2 trait-sensitivity RDM ↔ activations: **mostly null**.
- A3 KL: exploratory.
- Power: n=36 detects |r|≳0.45 only.

**Boundary condition:** above-chance RMET ≠ human EQ-linked item-difficulty map.

---

## 12. Engineering

- Seed=42; argparse/JSON/CSV style matching study4.
- Fail loudly; resume-safe for expensive evals.
- Secrets via `study4_rmet/.env` only.
- Sync: `./study4_rmet/sync.sh` only.

### Script map (V2)

| Phase | Script |
|-------|--------|
| 1 | `scripts/build_card_rmet_structure.py` → `results/card_structure/` |
| 2 | `scripts/behavioural_profile_alignment.py` → `results/behavioural_v2/` |
| 3 | `scripts/rsa_probe_card_axes.py` → `results/mech/` |
| 4a | `scripts/causal_rmet_axes.py` → axes + geometry reuse |
| 4b | `scripts/steer_rmet_axes.py` (+ `slurm_jobs/rmet_causal_steer_smoke.sh`) |
| Robustness | `scripts/contamination_option_order_smoke.py` |
| Runner | `scripts/run_protocol_v2_phases.sh` |
