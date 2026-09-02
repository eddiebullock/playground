# Study 3 v2 — Implementation Plan (mechanistic arm redesign)

**Scope this pass:** Qwen3-VL, layer 4, last-token only.  
**Authoritative RQ/method spec:** `study3_protocol_v2.md` (must be present in repo before code merge; **not found in workspace at plan time — add alongside this file**).  
**Pilot preserved:** behaviour + probe/RSA + causal axes + medium steer remain valid; causal *interpretation* is superseded.

---

## 0. Why v2 (internalized, not relitigated)

The pilot collapsed **pairwise confusability** and **generic item difficulty** in the causal-axes stage (confusability ↔ entropy axis alignment **r = 0.95**). That matches generic entropy-neuron / calibration literature and predicts **fixed additive steering nulls** when the direction is entangled with task computation (Liu 2026 precedent). v2 keeps probing/RSA for representational claims and replaces steering-as-primary-causal-test with **necessity ablation** (mean-projection replacement) and a **double-dissociation** readout (pair-specific vs generic degradation, vs random-axis baseline).

**Terminology (protocol-wide):**
- **Activation patching / ablation** = substitute or remove a site's content along a direction; test necessity.
- **Activation steering** = additive nudge; secondary / contingent alignment test only.

Do not let steering results justify ablation design or vice versa in code comments.

---

## 1. Script-by-script plan

### 1.1 `scripts/build_human_confusion.py` — confirm only

| | |
|---|---|
| **Change** | None expected |
| **Reuse** | `human_entropy`, `confusability_1_minus_p_target`, `top_foil_label`, `human_target_label` per item in `human_confusion_meta.json` |
| **Verify** | Already exposes per-item difficulty for difficulty-matched RSA (confirmed) |

Pair-level mean entropies (243 trials): Bored–Unfriendly 1.15, Interested–Kind 1.24, Disappointed–Worried 0.85 (pool SD ≈ 0.34).

---

### 1.2 `scripts/causal_eu_confusion_axes.py` — extend

| | |
|---|---|
| **Reuse** | Axis construction (confusability, entropy, random, pair_*), `save_axes`, pilot JSON/CSV outputs |
| **New (i) Entanglement diagnostic** | For each pair axis `p` and generic axes `g ∈ {entropy, confusability}`: `|cos(p,g)|`, specificity ratio `‖p_⊥g‖ / ‖p‖` (or 1 − cos²), report in `pair_axes[]` and top-level `entanglement` block in JSON |
| **New (ii) Difficulty-matched non-confused pairs** | For each top confused pair: sample ≥3 label-pairs (target, top-foil) **not** in confused set with mean item entropy within ±0.5 SD of confused pair mean (widen to ±1 SD + log if <3 matches; pilot data: 5–19 matches at ±0.5 SD). Store matched pair list in JSON for RSA script consumption |
| **Runtime** | CPU only, <1 min (same as pilot) |

**Output:** extend `qwen3vl_eu_causal_axes_layer4.json`; axes `.npy` unchanged.

---

### 1.3 `scripts/confusability_probe_rsa.py` — extend

| | |
|---|---|
| **Reuse** | LOO Ridge probes, raw RSA vs human RDM, per-layer CSV/JSON |
| **New** | Per layer, add `rsa_difficulty_matched`: for each confused pair, build subset RDM restricted to (confused-pair items + matched non-confused-pair items); RSA vs human RDM on that subset; aggregate (mean ρ across pairs). Keep raw RSA as `rsa_human_confusion_rho` labeled **not difficulty-controlled** in JSON metadata |
| **Field location** | Extend existing `results/mech/{model}_confusability_probe_rsa.json` → each layer row gets `rsa_difficulty_matched` object with `rho`, `n_pairs`, `matching_tolerance_sd`, `per_pair` |
| **Runtime** | CPU, ~2–3 min (pilot 33851632 was 2 min); + negligible for matched subsets |

Gemma/Molmo JSONs get the same field when re-run (optional; pilot 3-model raw RSA already done).

---

### 1.4 `scripts/steer_eu_confusion_axes.py` — retain, demote

| | |
|---|---|
| **Change** | Docstring + log strings: **secondary / exploratory steering**; reference v2 protocol; do not describe as primary causal test |
| **Reuse** | Full pilot implementation, `steer_*` CSV schema, slurm `study3_causal_steer.sh` |
| **Runtime** | Unchanged (medium ~3 h job 33776847) |

Contingent alignment test (steer toward human JS) runs only if ablation shows selective pair-specific necessity.

---

### 1.5 `scripts/ablate_eu_confusion_axes.py` — **new (primary causal)**

| | |
|---|---|
| **Reuse from pilot** | `load_trial_table`, axis `.npy` loading, `find_layer_module`, generation path, pair tagging, smoke/medium/full item selection pattern |
| **Intervention** | **Mean ablation (default):** on forward pass at L4 last-token, replace projection onto unit axis `a` with dataset-mean projection (computed offline from pilot activations for same trial set). Orthogonal component unchanged. **Fallback:** zero projection if mean-ablation shows no effect even on random control (log `ablation_method: zero_projection`) |
| **Conditions per trial** | `baseline` (no hook); `ablate_entropy`; `ablate_<pair>` × top_pairs; `ablate_random` — always include random control |
| **NOT steering** | No `alpha`; hook modifies `h[:, -1, :] -= (proj - mean_proj) * a` (equivalent to mean replacement along `a`) |
| **Metrics** | Per trial: `correct`, `pred`, `confusion_rate` (chose `top_foil_label` or either label of ablated pair as appropriate), `pair_membership`. Summary: Δaccuracy and Δconfusion_rate vs baseline, split **own pair** vs **other confused pairs** (mirror `steer_summary_*.csv` columns) |
| **Sampling** | **1 greedy/low-T sample per condition** (deterministic readout); optional `--n_samples` for robustness later — default 1 to fit 12h wall |
| **Outputs** | `results/mech/qwen3vl_eu_ablation_layer4.json`; `ablate_summary_qwen3vl_layer4.csv`; `ablate_trials_qwen3vl_layer4.csv` |
| **Comments / names** | File prefix `ablate_*`; class `AxisMeanAblator`; log `"Running EU ablation (patching)"` not steer |

**Slurm:** new `slurm_jobs/study3_causal_ablate.sh` — mirror steer modes (`smoke`, `medium`, `full`), log `ABLATE_ARGS` at start (lesson from MODE=medium bug).

#### Runtime estimate vs pilot medium steer (33776847, ~3 h)

| Mode | Trials | Conditions | Gens/trial | Total gens | Est. time @ ~25 s/gen |
|---|---|---|---|---|---|
| smoke | 3 | 6 | 6 | 18 | ~8 min |
| medium | 36 | 6 | 6 | 216 | **~1.5 h** |
| full (243) | 243 | 6 | 6 | 1458 | **~10 h** |

Medium ablation is **~2× faster** than medium steer (6×1 vs 13×5 gens). Full ablation **fits 12 h QOS**; full steer (~18 h) does not.

If `--n_samples 5` added later: multiply by 5 → full ~50 h → **split into 2×12 h jobs** with `--trial_offset` / `--trial_limit` (same pattern needed for steer full; not yet implemented).

---

### 1.6 `scripts/plot_study3_mech_figures.py` — extend

| | |
|---|---|
| **New** | `fig5_ablation_dissociation`: grouped bars — own-pair vs other-pair Δaccuracy (and/or Δconfusion rate) for pair ablations vs entropy ablation vs random, with 0-line and random baseline marked |
| **Manuscript order** | Fig1 behaviour → Fig2 geometry → **Fig4 axis geometry** → Fig3 steer (pilot, supplementary framing) → **Fig5 ablation** → FigS1 entropy |

---

## 2. Double-dissociation predictions (v2)

| Outcome | Interpretation |
|---|---|
| Pair ablation ↓ own-pair accuracy/confusion **>>** other pairs; entropy ablation broad | **Causal pair-specific** site (despite entanglement in axis construction) |
| Entropy ablation broad; pair ablation ≈ random | **Generic difficulty** necessity only |
| Both ≈ random | **Null necessity** at L4 (consistent with pilot steer null) |
| Both selective on same items | Ambiguous — inspect per-item overlap with difficulty tertiles |

---

## 3. Power / scope honesty (§4)

**Pilot medium scale:** 36 trials (pair-enriched), 3 pairs with approximate n in subset: **8 / 7 / 13** own-pair items (Bored–Unfriendly / Interested–Kind / Disappointed–Worried).

**Pilot steer baseline:** global ΔJS ≈ 0; ~30–40% of trials show any JS movement under **any** axis including random → hook is live but not selective.

**New test is harder:** requires **differential** Δ(own pair) − Δ(other pairs) on accuracy or foil-choice rate, not global JS shift.

### Rough detectability (accuracy delta, single sample, medium n)

Assuming baseline accuracy ≈ 0.38 on subset:

| Subset | n (own) | SE(Δacc) | ~80% MDE (two-group diff) |
|---|---|---|---|
| Bored–Unfriendly | 8 | 0.17 | **~0.48** (≈4/8 items) |
| Interested–Kind | 7 | 0.18 | **~0.51** |
| Disappointed–Worried | 13 | 0.14 | **~0.38** (≈5/13) |
| Other confused (pooled) | ~28 | 0.09 | — |
| **Own vs other contrast** | 8 vs 28 | 0.19 | **~0.55** on accuracy difference |

**Implication:** Medium-scale ablation can detect **large** pair-specific collapses (e.g. own-pair accuracy −30 pp with minimal other-pair change). It **cannot** reliably detect subtle dissociations (~10 pp). Pilot steer effects were ~0 on JS and small on subsets → **medium run is a hook + sanity check**, not powered for weak dissociation.

**Recommendation:**
1. **Smoke (3 trials, 1 pair)** — confirm ablation hook live, CSV schema, random baseline.
2. **Medium (36 trials)** — exploratory double-dissociation; report with wide CIs / exact binomial.
3. **Full (243 trials, ~10 h)** — primary inferential run for necessity claims; still weak for Interested–Kind (n=7) per-pair — aggregate via mixed model across pairs or bootstrap.

Confusion-rate delta (foil choice) is **sparser** than accuracy → lower power; report both but prioritize accuracy for power planning.

---

## 4. Execution order (post-review)

1. Add `study3_protocol_v2.md` to repo (if not already elsewhere).
2. Implement 1.2 → 1.3 → 1.5 → 1.6 + slurm ablate script.
3. CPU: re-run `causal_eu_confusion_axes` + `confusability_probe_rsa` (Qwen).
4. GPU smoke ablation → verify logs + 1 trial delta ≠ 0 for entropy vs random.
5. GPU medium ablation (~1.5 h).
6. If hook live and any pair shows own > other + random: GPU full ablation (~10 h).
7. Pull results; regenerate figures including fig5.
8. Steering re-run **only if** protocol contingent alignment clause triggered.

---

## 5. Flags / ambiguities for protocol owner

1. **`study3_protocol_v2.md` missing from repo** — implement against copy provided in redesign prompt until file is checked in.
2. **Confusion-rate definition:** use `top_foil_label` from meta, or either member of ablated pair when target is one of them — confirm in protocol §Causal.
3. **Dataset mean projection:** compute on all 243 trials or only evaluated subset — recommend **243** for stable mean, apply on eval subset.
4. **`confusability` axis in ablation:** pilot entangled with entropy — include as exploratory fourth generic ablation or omit to avoid redundant conditions? Recommend **entropy only** as generic + pair + random (saves 243 gens on full run).
5. **Future extension (out of scope this pass):** Gemma/Molmo ablation; other layers; `all_tokens` patch mode — after Qwen L4 double-dissociation validated.

---

## 6. Smoke test checklist (before medium/full)

```bash
# HPC
MODE=smoke MODEL=qwen3vl LAYER=4 sbatch slurm_jobs/study3_causal_ablate.sh
grep ABLATE_ARGS logs/study3_ablate_*_${JOB}.out
grep -c '^ablate ' logs/study3_ablate_*_${JOB}.out   # expect 3
head results/mech/ablate_summary_qwen3vl_layer4.csv
```

Expect: non-empty CSV; random ablation effect ≤ pair ablation on own-pair items (direction TBD); no `Traceback` in err log.
