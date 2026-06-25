## Frame policy ablation (gemma4, multimodal)

| Metric | composite_grid | native_video | Δ (native − composite) |
|--------|----------------|--------------|-------------------------|
| Strict 4AFC | 36.7% | 26.7% | -10.0 pp |
| Tolerant 4AFC | — | — | — |
| Free-response judge (primary) | — | — | — |
| ECE | — | — | — |
| AUROC (entropy conf.) | — | — | — |
| Mean semantic entropy | 2.489 | 2.695 | 0.206 |

- composite_grid strategy: `composite_grid` (enforce_multi_frame=True)
- native_video strategy: `single_first_frame` (enforce_multi_frame=False)
