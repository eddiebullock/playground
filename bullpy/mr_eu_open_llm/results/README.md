## Results directory layout (protocol v2)

v1 results (`baseline_*_frames4_*.json`, single-stage) are **not comparable** to v2. Filter on `protocol_version: "v2-study1-study2"`.

### Subdirectories

- `baseline/` — two-stage EU-Emotions / Mindreading eval JSONs
- `ablation/` — 30-trial 4 vs 16 frame comparisons
- `finetune/` — `hparam_sweep/`, `full_runs/`
- `activations/` — `{condition}/{model}/layer{L}_eu_emotions_seed42.npy`
- `probes/`, `rsa/`, `patching/`
- `stats/` — `best_model.json`, `confused_pairs_{model}.json`, summaries
- `test_runs/` — smoke tests

### v2 baseline naming

```
baseline/eu_emotions/{model}/eval_v2_{dataset}_{model}_fps1_cap16_two_stage_seed42.json
```

Required top-level fields: `protocol_version`, `frame_policy`, `primary_outcomes`, `mean_semantic_entropy`, `accuracy`, `trials[]` with `stage1` and `stage2` objects.

### RMET

The `rmet/` subfolder is legacy; not part of Study 1 protocol v2.
