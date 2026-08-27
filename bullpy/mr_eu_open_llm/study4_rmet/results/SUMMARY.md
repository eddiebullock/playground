# study4_rmet — SUMMARY (PROTOCOL V2)

Checkpoint of pre-redesign outputs: `checkpoints/pre_v2_card_mech_20260806/`.  
Protocol: `docs/PROTOCOL_V2_CARD_MECH.md`.

Language: eye-region mental-state / complex emotion recognition — **not** ToM circuits.

---

## Result 0 (legacy; demoted boundary condition)

Above-chance RMET ≠ human EQ-linked item-difficulty map.

| Claim | Outcome |
|-------|---------|
| H1 EQ-slope ↔ model accuracy (6 models) | Null (perm p ≫ .05; meta Pearson r≈−0.02) |
| H2 trait-sensitivity RDM ↔ activations | Mostly null; exploratory early-qwen L4 only |
| Power (n=36) | Detects \|r\|≳0.45 at 80% power |

See also `results/robustness/robustness_report.md` (archived framing).

---

## Phase 1 — CARD structure

Artifacts: `results/card_structure/`.

- N=2907; ASC n=991; alexithymia **not in CARD** (limitation).
- Per-item: `human_entropy`, `trait_diagnosticity_eq_slope`, ASC/EQ gaps, JS(low-EQ vs high-EQ), etc.
- Pre-registered classes: high/low diagnosticity; high/low entropy (median splits).
- Feature RDM documented as substitute for invalid global label RDM.

---

## Phase 2 — Behavioural (B1 / B2)

Artifacts: `results/behavioural_v2/`.

### B1 — human ↔ model entropy (primary)

All six models: Spearman n.s. (perm p > .25). Boot CIs include 0 (except claude slightly negative CI).

**Interpretation:** models do **not** track human choice-entropy structure either — strengthens “not just EQ-difficulty null; also not generic-ambiguity matching” at the behavioural soft-label level (k=10).

### B2 — profile-conditioned JS (model soft ↔ CARD strata)

Mean JS to overall human distribution (lower = closer): molmo 0.28 < gemma/gemini ~0.30 < gpt5/qwen/claude ~0.34–0.35.

`mean_js_eq_low − mean_js_eq_high` slightly negative for most (soft labels a bit closer to low-EQ than high-EQ) but small; not a confirmatory profile match. Molmo caveat: near-chance accuracy.

Legacy H1 reconfirmed null in the same tables.

---

## Phase 3 — Geometry / probes (M1)

Artifacts: `results/mech/*_rsa_probe_layers.csv`, `mech_rsa_probe_summary.json`.

Open models only (existing activations).

| Model | Peak probe entropy ρ (LOO Ridge) | Peak probe diagnosticity ρ | Notes |
|-------|----------------------------------|----------------------------|-------|
| qwen3vl | ~0.50 (L4) | ~0.35 (L4) | Exploratory; LOO n=36; multiplicity across layers |
| gemma4 | ~−0.09 | ~0.29 (L15) | Weak |
| molmo2 | ~−0.13 | ~0.38 (L12) | Near-chance model — interpret harshly |

RSA vs diagnosticity RDM: qwen L4 still ρ≈0.18, p_perm≈0.04 (same caveat as Result 0). Feature-composite and entropy RDMs mostly n.s.

**M1 takeaway:** some linear decodability of diagnosticity/entropy in open VLMs is possible (esp. early qwen), but this is **not** yet causal evidence and must not be sold as ToM.

---

## Phase 4 — Causal reuse (C1)

Artifacts: `results/mech/causal_axis_geometry_all.csv`, axis `*.npy`, `steer_protocol_*.json`.

### 4a Geometry (CPU; complete)

Mean high−low class activations:

- **Diagnosticity vs entropy axis alignment:** consistently **negative** (≈ −0.24 to −0.66) across models/layers → axes are **not** the same direction.
- **Reuse:** projection of entropy classes onto the diagnosticity axis is typically **opposite sign / smaller** than the own-class effect → geometry suggests **dissociable** axes (generic-mechanism-not-forced).
- Controls: random axes rarely match own gaps; shuffled-class own-gaps still large (axis defined from classes — report honestly).

### 4b Steer/patch (qwen3vl L4 full; complete)

HPC job `33536702` (`MODE=full`, 36 items, α ∈ {±1,±2}, `last_token` + `all_tokens`, k=10).  
Artifacts: `steer_summary_qwen3vl_layer4.{csv,json}`, `steer_trials_qwen3vl_layer4.csv` (900 rows).

**C1 outcome: controlled near-null (no profile-specific causal effect).**

- **~84%** of item×condition rows have |ΔJS_eq_low| ≈ 0 (soft labels unchanged by steer).
- Mean |ΔJS| is tiny (typically ≲ 0.01); largest mean shift is **random +2 all_tokens** (~+0.013, *away* from low-EQ/ASC).
- Diagnosticity interventions do **not** systematically move soft labels toward CARD `p_eq_low` / `p_asc` more than entropy or random.
- Occasional small negative ΔJS (toward profile) appear for mixed axes/modes (e.g. entropy +1 last_token ≈ −0.010) without a diagnosticity-specific pattern → not a reuse dissociation headline.
- Geometry anti-alignment (4a) therefore does **not** imply a causally used profile axis under these hooks/α.

Interpretation (mechanism-under-moderate-competence): qwen can answer RMET above chance and show dissociable *activation geometry*, but steering along diagnosticity vs entropy vs random at L4 does not produce CARD profile-conditioned behavioural shifts. Treat as **causal null for C1**, not as discovery of EQ circuits.

Do **not** escalate to gemma/molmo steer unless seeking null-robustness; not required for the contribution bar once the qwen full null is reported with controls.

---

## Robustness / limitations (reported)

- Contamination: classic RMET web exposure; eyes-only; option-order smoke script + offline limitation JSON under `results/robustness/contamination/`
- Construct: mental-state / complex emotion from eye region — not ToM; alexithymia absent from CARD
- Stats: n=36; permutation/bootstrap; do not headline uncorrected layer×model peaks
- VLM patch: last-token may fail for vision-dependent behaviour — `all_tokens` included; both modes near-null here
- Scope: open models moderate competence; molmo/gpt5 near-chance — interpret harshly
- Steer: α≤2, single layer (L4), k=10; stronger α / other layers optional but not needed to claim “no clear profile-specific effect at tested settings”

---

## Contribution status vs bar

| Bar element | Status |
|-------------|--------|
| Beyond accuracy leaderboard | Yes (Result 0 + B1/B2) |
| CARD profile targets | Yes (Phase 1–2) |
| Diagnosticity vs entropy | Yes (M1 exploratory; C1 geometry anti-alignment) |
| Causal steer/patch | **Yes — qwen L4 full; near-null vs random/entropy controls** |
| Contamination / construct caveats | Documented |

**Scientific bottom line (write-up):**  
VLMs show limited–moderate RMET competence without matching human EQ-difficulty maps (**Result 0**) or human entropy maps (**B1 null**); profile soft-label alignment is weak (**B2**). Activation geometry can look dissociable for diagnosticity vs entropy (**4a**), and probes can weakly decode those scalars (**M1**, exploratory) — but **causal steering on qwen3vl L4 does not produce diagnosticity-specific shifts toward CARD low-EQ/ASC patterns (**C1 null**)**. High-value outcome delivered: surface competence / correlational geometry **without** a causally used profile-aligned mechanism under tested interventions.

---

## How to reproduce

```bash
./study4_rmet/scripts/run_protocol_v2_phases.sh
# GPU C1 (HPC):
MODEL=qwen3vl LAYER=4 sbatch study4_rmet/slurm_jobs/rmet_causal_steer_smoke.sh
MODE=full MODEL=qwen3vl LAYER=4 sbatch study4_rmet/slurm_jobs/rmet_causal_steer_smoke.sh
./study4_rmet/sync.sh pull
```
