# Study A1 deviations

Hypothesis statements in `A1.md` are unchanged.

## 2026-10-29, before any run

Eddie: the study runs on open weights on the cluster only. No hosted API and no paid call, at any point.

The closed-model rows (`claude-opus-5-5`, `gpt-6.1-sol`, `gemini-4-argon`) are removed from the draft. H1, H2, and H5 average over the open checkpoint list only. The exploratory run is `Qwen/Qwen3-8B` alone.

Project storage has about 58 GB free (quota check, 29 Oct 2026). The thirteen checkpoints cannot be stored together. They run one at a time, starting with `Qwen/Qwen3-8B`, and a download is refused when the free space cannot hold that checkpoint.

## 2026-10-29, empty demographic profiles
The exploratory manifest build stopped on a drawn profile with no age, sex, education, occupation, or region. Condition 2 would have been an empty person block. Those profiles are not drawn. Age missing is still included when another demographic field is present, and the persona still omits the missing age. Hypothesis statements are unchanged.

## 2026-10-29, same quota as Study A2
A later quota check showed about 34 GB free. Study A2's model files and the shared environment account for the drop. The exploratory job still downloads only `Qwen/Qwen3-8B`. It starts when the remainder can hold that checkpoint and the A1 environment, and it does not remove A2 files.
