# Study 3 agent brief — Mental-State Confusability in VLMs (v2)

**Use this document** to onboard a Claude agent on Study 3. The **pilot is complete**; the **mechanistic causal arm is being redesigned** (v2). Do not discard pilot data.

**Repo:** `mr_eu_open_llm` · **HPC:** `~/rds/hpc-work/study3` · **Do not touch study2** · **study4_rmet (RMET): convergent-check only, max one sentence in EU paper**

**Protocol authority:** `docs/study3_protocol_v2.md` (when present) + `docs/study3_v2_implementation_plan.md`  
**GPU walltime cap:** 12 h (`gpu2` QOS). Log `STEER_ARGS` / `ABLATE_ARGS` at job start (MODE=medium/full bug burned pilot steer time once).

---

## 1. Research questions (v2 framing)

### Definitions (keep separate everywhere)

| Term | Meaning |
|---|---|
| **Pairwise confusability** | Representational/behavioural overlap between two *specific* concepts (e.g. Worried vs Disappointed), beyond item difficulty |
| **Generic item difficulty / uncertainty** | How hard/ambiguous an item is, independent of *which* alternative competes |

Pilot mistake: causal “confusability axis” (target vs top foil) aligned with entropy axis at **r = 0.95** → generic calibration/difficulty signal, not pair-specific geometry.

### Motivating RQ (behaviour) — **pilot answered, no new compute**

Does per-item model 6AFC entropy correlate with human response entropy?

### Primary RQ (mechanistic v2)

Is human-like **pair-specific** confusion structure **causally necessary** at a localised site, or does necessity implicate only a **generic difficulty** direction (and/or fail entirely under entanglement)?

### Secondary RQs (v2)

| RQ | Method |
|---|---|
| Representational specificity | Probe (decodable?) + RSA raw + **difficulty-matched RSA** |
| Causal specificity | **Ablation** of pair axis → own-pair degradation >> other pairs |
| Causal generality | **Ablation** of entropy/generic axis → broad degradation |
| Reuse double-dissociation | Pair ablation selective vs entropy ablation broad (vs **random ablation baseline**) |
| Entanglement diagnostic | cos(pair, entropy), specificity ratio in axis JSON |
| Alignment (contingent) | Additive **steering** toward human JS — **only if** ablation shows selective necessity; pilot steer null stays reported as exploratory |

### Terminology (enforced document-wide)

- **Activation patching / ablation** = necessity (replace/remove projection along axis)
- **Activation steering** = additive α × axis (exploratory; pilot done)

---

## 2. Pilot results (complete — superseded causal *framing* only)

### Behaviour (243 trials, video_only, 6AFC, 3 models)

| Model | Accuracy | H_fc vs H_human ρ |
|---|---|---|
| Qwen3-VL | 37.9% | ≈ 0 |
| Gemma 4 | 31.3% | +0.12 |
| Molmo2 | 19.8% | −0.03 |
| Chance | 16.7% | — |

Accuracy tracks human consensus terciles; entropy does not.

### Geometry (3 models, 3 depths, last-token activations)

- **Probe** confusability ρ ≈ **0.47–0.61**; entropy ρ ≈ **0.44–0.53**
- **RSA** vs human confusion RDM ≈ **0** (Qwen weak negative ~−0.08)

### Causal axes (Qwen L4, activation space, CPU)

- Confusability ↔ entropy alignment **0.95**
- Pair axes: own-vs-rest 0.53–0.80 in activations; weak cross-reuse

### Causal steer (Qwen L4, medium — job 33776847, pilot)

- 36 trials, 5 samples, ±1, last-token, 3 pair axes
- Global ΔJS ≈ 0; random ≈ named axes; ~30–40% trials move under any axis
- **Interpretation (v2):** expected null for entangled generic direction + fixed steering (Liu 2026 precedent); **not** evidence that no causal site exists anywhere

---

## 3. v2 mechanistic plan (not yet run at scale)

**Site:** Qwen3-VL L4 last-token only.

**Primary method:** `scripts/ablate_eu_confusion_axes.py` — mean-projection ablation along existing `.npy` axes; random control; double-dissociation metrics (Δaccuracy, Δconfusion rate: own pair vs other confused pairs).

**Extensions to pilot scripts:** entanglement metrics + difficulty-matched pairs in `causal_eu_confusion_axes.py`; `rsa_difficulty_matched` in `confusability_probe_rsa.json`.

**Runtime:** medium ablation ~**1.5 h**; full ~**10 h** (fits 12 h). Medium steer was ~3 h.

See `docs/study3_v2_implementation_plan.md` for power analysis (medium n underpowered for subtle dissociation).

---

## 4. Status table

| Stage | Status | Notes |
|---|---|---|
| Behaviour 3×243 | **Done — pilot** | No rerun |
| Human confusion RDM / meta | **Done — pilot** | `build_human_confusion.py` |
| Activation extract Qwen/Gemma/Molmo | **Done — pilot** | 243 trials, 4 frames |
| Probe + RSA (raw) 3 models | **Done — pilot** | |
| Difficulty-matched RSA | **Not run** | Extend JSON in place |
| Causal axes + entanglement metrics | **Partial — pilot** | Re-run CPU after code extend |
| Medium steer Qwen | **Done — pilot / exploratory** | Superseded as primary causal test |
| **Ablation double-dissociation** | **Code ready — not run** | `ablate_eu_confusion_axes.py` + `study3_causal_ablate.sh` |
| Contingent steer rerun | **Not run** | Only if ablation selective |
| Fig5 ablation | **Not generated** | After ablation CSV |
| Write-up (behaviour + geometry) | **Ready** | Causal section awaits v2 |

---

## 5. Key artifacts

### Pilot (pulled / local)

```
results/baseline/eu_emotions/{qwen3vl,gemma4,molmo2}/eval_v2_*_video_only_seed42.json
results/stats/rq1_1b_entropy_alignment.json
results/stats/human_calibration.json
results/mech/{qwen3vl,gemma4,molmo2}_confusability_probe_rsa.json
results/mech/qwen3vl_eu_causal_axes_layer4.json
results/mech/steer_summary_qwen3vl_layer4.csv
results/mech/steer_trials_qwen3vl_layer4.csv
results/figures/study3/fig1_behaviour.*
results/figures/study3/fig2_geometry.*
results/figures/study3/fig3_steer.*          # pilot exploratory steering
results/figures/study3/fig4_axis_geometry.*
results/figures/study3/figS1_entropy_scatter_all.*
```

### v2 (planned)

```
results/mech/qwen3vl_eu_ablation_layer4.json
results/mech/ablate_summary_qwen3vl_layer4.csv
results/mech/ablate_trials_qwen3vl_layer4.csv
results/mech/qwen3vl_confusability_probe_rsa.json  # + rsa_difficulty_matched per layer
results/figures/study3/fig5_ablation_dissociation.{png,pdf}
```

### Scripts

| Script | Role |
|---|---|
| `build_human_confusion.py` | Human RDM + per-item difficulty |
| `confusability_probe_rsa.py` | Probe + RSA (+ v2 matched RSA) |
| `causal_eu_confusion_axes.py` | Axes + entanglement (+ v2) |
| `ablate_eu_confusion_axes.py` | **v2 primary causal (patching)** |
| `steer_eu_confusion_axes.py` | Pilot / contingent steering only |
| `plot_study3_mech_figures.py` | All figures |
| `slurm_jobs/study3_causal_ablate.sh` | **v2 GPU ablation** |
| `slurm_jobs/study3_causal_steer.sh` | Pilot steering |

---

## 6. Suggested manuscript figure order

1. **Fig1** — Behaviour (accuracy, entropy null, consensus)
2. **Fig2** — Probe high, RSA null (3 models)
3. **Fig4** — Generic axis geometry (conf≈entropy; pair own-vs-rest in activation space)
4. **Fig3** — Pilot steering null (exploratory; caption: additive steering of entangled axis)
5. **Fig5** — Ablation double-dissociation (v2 primary causal)
6. **FigS1** — Entropy scatter all models

---

## 7. Guardrails

1. **No RMET in EU tables** — sister study; one convergent sentence max.
2. **No study2 overwrite.**
3. **12 h GPU walltime max** — split jobs with trial offsets if full + n_samples > 1.
4. **Verify logged args** before trusting completed GPU jobs.
5. **CSV written at end** of long jobs — timeouts lose partial outputs.
6. **Patching ≠ steering** in every script name, log, caption.
7. **Pilot nulls are real results** — v2 reframes, does not erase them.
8. **Scope honesty:** causal necessity claim is **Qwen L4** until extended explicitly.

---

## 8. One-sentence story (current)

VLMs show above-chance mental-state recognition and linearly readable generic difficulty in activations, but not human pairwise confusion structure in RSA, not human-like choice entropy, and (pilot) no selective control via fixed steering of an entangled axis — v2 tests whether **necessity ablation** reveals pair-specific vs generic causal use at L4.
