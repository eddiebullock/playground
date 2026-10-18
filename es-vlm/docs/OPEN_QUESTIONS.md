# Open questions

Conservative defaults were taken where the brief was ambiguous. Resolve
these before Phase 1 baselines lock.

## Identity and remotes (2026-10-18)

1. `git remote -v` on this clone is
   `https://github.com/eddiebullock/playground.git`, not
   `git@github-personal:eddiebullock/playground.git`. Proposed fix
   (not run until confirmed):

   ```
   git remote set-url origin git@github-personal:eddiebullock/playground.git
   ```

2. `git config --local user.name` / `user.email` were empty. Global
   values are `Edward Bullock` / `eddiebullock1@hotmail.co.uk`. Local
   was set to those for commits on `es-vlm/setup`. Confirm or replace
   with `--local` only.

3. `gh auth status` showed work account `eddieminimal` as active, and
   both tokens invalid. Switch to `eddiebullock` before any `gh`
   command. Repo visibility is unconfirmed (see
   `docs/DATA_GOVERNANCE.md`).

## Mac hardware

4. This Mac has 16 GB unified memory (`sysctl hw.memsize`). Milestone B
   peak allocated memory for Qwen3-VL-2B on MPS bf16 was 4073 MB. Still
   ask before assuming other machines match.

## Text-only control

5. `Qwen/Qwen3-8B-Instruct` was HTTP 401. Pinned `Qwen/Qwen3-8B`
   instead. Prefer Instruct if access is available?

## Stimuli paths

6. Exact on-disk filenames for RMET `manifest.json` and
   `answer_key/rmet_adult_answer_key.json`, EU-Emotions full manifest,
   and Mind Reading item dirs should be confirmed against `$ESVLM_*_DIR`
   before the first cluster smoke.

## S-domain tasks (Phase 1)

Accepted and fetched 2026-10-18. See `docs/DECISIONS.md`.

Kept: PhysBench 3681 text-option dynamics/relationships items;
CLEVRER validation 19804 explanatory/predictive/counterfactual MCQ;
PIQA 1838 validation items. Image-option PhysBench items are out
until we have multi-image scoring.

## huggingface_hub

8. Brief asked for `huggingface_hub>=0.34,<1.0`. Transformers 5.x now
   requires hub 1.x. Relaxed to `<2.0`. Confirm.

## InternVL3.5

7. Optional. Try only in a throwaway env. Drop after two load failures.

## Molmo2

9. Excluded from the primary panel on 2026-10-18 (processor / text-only
   residual path). Reopen only if we need a third architectural family
   and the adapter is rewritten. Do not treat a later Molmo run as
   confirmatory unless it is added back in the OSF prereg.
