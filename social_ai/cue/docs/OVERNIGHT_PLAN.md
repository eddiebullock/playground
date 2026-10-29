# Overnight plan, 2026-10-25 to 26

Eddie asked for all next jobs to run overnight without input, organised
around publishable studies (`docs/PUBLICATION_STRATEGY.md`). Commit and push
to `eddiebullock/playground` (branch `es-vlm/setup`) at each milestone; Eddie
authorised pushing for this session.

## GPU queue (`experiments/queue_overnight.sh`, sequential, logs to `experiments/overnight.log`)
| # | Job | Serves | Decision it informs |
|---|---|---|---|
| 0 | Fix-2 replies audit, seeds 1-2 | Paper 2, product | Keep fix 2, revert, or go structural |
| 1 | Readings audit v2 (48 scenarios x 6 disclosures incl. placebo), shipped scorer vs "words not reader" | Paper 2, product | Adopt the scorer line? |
| 2 | 39 new minimal pairs on Qwen 4B/8B, direct and stance | Paper 1 | Does the criterion result generalise beyond irony? |
| 3 | Write 6 / show 4, E4 odd (development) half | Product, Paper 1 4.5 | Does recall rise without hurting top-1? |
| 4 | iSarcasmEval held-out (160 real tweets), shipped vs wide | Product, Paper 1 4.5 | External validity; confirm or reject wide |
| 5 | llama3.1:8b, gemma3:12b, gemma3:4b: Hu, all pairs, MUStARD, direct and stance | Paper 1 | Breadth across families |
| 6 | Readings audit v2 across 4 more models | Paper 2 | Is disclosure drift model-general? |

## Labelling (no GPU)
Blind labels for jobs 3 and 4 (`experiments/e4_product/blind.py` style),
labelled by Claude, with a second independent Claude labeller on a sample
as a consistency check. Human validation is for Eddie (see the strategy).

## Writing
Paper 1 draft (`docs/papers/paper1_draft.md`), Paper 2 draft, DECISIONS
entries per result, morning summary in `docs/HANDOVER.md`.

## Rules kept
No money; disk >= 5 GB free; nothing participant-related leaves the
machine; tweets are not committed (ids and a checksum only); no contacting
anyone; no cluster.
