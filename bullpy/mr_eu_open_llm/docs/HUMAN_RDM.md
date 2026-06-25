# Human representational dissimilarity matrix (RDM) for Study 2 RSA

## Purpose

Representational Similarity Analysis (RSA) compares **model** stimulus-pair dissimilarities to **human** stimulus-pair dissimilarities. The human RDM is a symmetric `118 × 118` matrix (one row/column per EU-Emotions trial in manifest order) used by `scripts/rsa.py`:

```python
rho = spearmanr(model_rdm_upper_triangle, human_rdm_upper_triangle)
```

Model RDMs are built from mean-pooled hidden activations per trial (cosine distance). The human RDM must use the **same trial ordering** as `layer{L}_trial_ids.json` from activation extraction.

---

## What `build_human_rdm.py` does today

| Mode | Command | Output |
|------|---------|--------|
| Plan only | `python -m scripts.build_human_rdm` | Writes `data/human_rdm_source.json` with `"status": "pending"`; exits with error |
| Dev placeholder | `python -m scripts.build_human_rdm --allow_placeholder` | Random `118×118` matrix — **not for publication** |

The script does **not** yet ingest real human behavioural data. That builder needs to be implemented (see below).

---

## Three ways to obtain a human RDM (protocol order of preference)

### Option A — Targeted human 4AFC study (recommended if no published matrix exists)

Run the **same** 118 EU-Emotions stimuli with human participants using the **Study 1 Stage 2** procedure (4AFC, same label set).

**Collect per participant per trial:**
- `trial_id` (must match manifest)
- `chosen_label` (one of four options shown)
- `correct_label` (ground truth)
- optional: reaction time, confidence

**Minimum N:** pre-register (e.g. 20–30 participants for stable pairwise estimates); Golan-scale n=17 is a lower bound for group accuracy, not necessarily for 118×118 RDM stability.

**Build RDM from confusion profiles:**

1. For each stimulus `i`, compute a **response profile** vector over 27 mental states:  
   `p_i(l) = (# times any participant chose label l when viewing stimulus i) / (total choices for i)`  
   (Only four labels appear per trial in 4AFC; accumulate marginal choice frequencies across runs where each label appears as an option, or use full 27-dim vector from free-label coding if you re-run with full label set.)

2. Simpler **confusion-count RDM** (common in mental state work):  
   For each pair of stimuli `(i, j)`,  
   `D_ij = 1 - cosine_similarity(profile_i, profile_j)`  
   or `D_ij = ‖profile_i - profile_j‖₂` normalized to [0, 1].

3. Alternative **category-level dissimilarity** (if trial-level data sparse):  
   Use human confusion between **mental state categories** from aggregated 4AFC, then map to stimuli — weaker, document as limitation.

**Save:**
- `data/human_behaviour/eu_emotions_4afc_responses.csv`
- `data/human_rdm.npy` (float32, symmetric, zero diagonal)
- `data/human_rdm_source.json` with method, N participants, date, trial list hash

### Option B — Import published human RDM

If a published study used the same or closely matched EU-Emotions stimuli:

1. Obtain published pairwise dissimilarity or confusion data
2. Align rows/columns to `eu_emotions_118_manifest.json` trial order (reindex or subset)
3. Document mismatch in `human_rdm_source.json` (`"alignment": "subset"`, `"n_matched": ...`)

RSA is only valid where stimulus identity matches.

### Option C — Proxy from Golan / Mindreading (fallback, exploratory only)

Golan et al. (2006) reports **group accuracy**, not a full stimulus RDM. You cannot derive a 118×118 matrix from accuracy alone. Mark RSA with such a proxy as **exploratory** or wait for Option A.

---

## Target file format

### `data/human_rdm.npy`
- Shape: `(118, 118)` or `(n_trials, n_trials)` matching manifest
- Dtype: `float32`
- Symmetric: `D[i,j] == D[j,i]`
- Diagonal: `0`
- Values: dissimilarity in [0, 2] for cosine distance (typically [0, 1])

### `data/human_rdm_source.json`
```json
{
  "status": "complete",
  "method": "4afc_response_profile_cosine_distance",
  "n_stimuli": 118,
  "n_participants": 24,
  "trial_order_file": "data/eu_emotions_118_manifest.json",
  "response_file": "data/human_behaviour/eu_emotions_4afc_responses.csv",
  "created_at": "2026-...",
  "notes": "..."
}
```

### `data/human_behaviour/eu_emotions_4afc_responses.csv` (example columns)
```csv
participant_id,trial_id,chosen_label,correct_label,is_correct,option_set
P001,emotions_2/clip001.mov,Surprised,Surprised,true,"[...]"
```

---

## Planned builder extension (`build_human_rdm.py`)

Add subcommand or flags:

```bash
python -m scripts.build_human_rdm \
  --from_responses data/human_behaviour/eu_emotions_4afc_responses.csv \
  --manifest data/eu_emotions_118_manifest.json \
  --method response_profile_cosine \
  --output data/human_rdm.npy
```

**Algorithm (`response_profile_cosine`):**
1. Load manifest trial_ids in order → index map
2. Aggregate choices into `profiles[trial_id, 27]` (count matrix → row-normalize)
3. For each pair `(i,j)`: `D_ij = 1 - dot(profile_i, profile_j) / (‖profile_i‖ ‖profile_j‖)`
4. Assert symmetry; save `.npy` + update `human_rdm_source.json`

---

## Validation checks

```python
import numpy as np
D = np.load("data/human_rdm.npy")
assert D.shape[0] == D.shape[1] == 118
assert np.allclose(D, D.T)
assert np.allclose(np.diag(D), 0)
assert np.isfinite(D).all()
```

Cross-check trial order:

```python
import json
trial_ids = json.load(open("results/activations/.../layer7_trial_ids.json"))
manifest = json.load(open("data/eu_emotions_118_manifest.json"))
# trial_ids[i] must correspond to row i of human_rdm and activations
```

---

## RSA workflow once human RDM exists

```bash
python -m scripts.rsa \
  --activations results/activations/baseline_qwen2vl/qwen2vl/layer14_eu_emotions_seed42.npy \
  --human_rdm data/human_rdm.npy \
  --output results/rsa/baseline_qwen2vl/qwen2vl/rsa_layer14.json
```

Compare Spearman ρ across layers (baseline vs finetuned after Study 1 complete).

---

## Ethics / preregistration note

If collecting new human data (Option A), follow faculty ethics approval, informed consent, and pre-register N, exclusion criteria, and RDM construction **before** comparing to model RDMs to avoid double-dipping.
