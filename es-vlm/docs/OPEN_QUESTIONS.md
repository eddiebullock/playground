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

## S-domain tasks (for the user to choose; Phase 1)

Public physics / mechanical VQA candidates (licences and counts to be
confirmed before download; none fetched yet):

1. **ScienceQA** (image subset, physical/science items) — CC-BY-NC.
   ~21k QA pairs; filter to physical/mechanical diagrams.
2. **AI2 Science Questions / ARC** (text; some illustrated releases) —
   CC-BY. Challenge set ~2.6k. Text-only control more than visual S.
3. **Physion / Physion++** (physical scene understanding, video) —
   research licence. Hundreds of scenarios; not classic MCQ.
4. **CLEVR** (synthetic compositional visual reasoning) — CC-BY.
   ~100k questions; systematic but not mechanical-real-world.
5. **Something-Something V2** (action/mechanics video) — research
   licence; large. Would need an MCQ wrapper.

Do not download any of these until the user picks a set and a licence
that can sit next to the licensed clinical stimuli without mixing
trees.

## huggingface_hub

8. Brief asked for `huggingface_hub>=0.34,<1.0`. Transformers 5.x now
   requires hub 1.x. Relaxed to `<2.0`. Confirm.

## InternVL3.5

7. Optional. Try only in a throwaway env. Drop after two load failures.
