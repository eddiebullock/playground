# social_ai agent rules

Later phases run in new chat sessions and will not see the setup prompt. Follow the two sections below exactly.

The GitHub repo `eddiebullock/playground` is public. `social_ai/reports/` is git-ignored. Do not commit unpublished results, data files, row-level outputs, model weights, `local/`, or anything that names the cluster site.

## Non-negotiable rules

1. **Cluster access.** Read the local HPC context file in `~/minimal`; it describes the cluster, accounts, partitions, storage, limits and patterns. For **this project only**, Eddie grants a standing approval that replaces that file's section 0 rule 1 for read-only commands:
   - **Allowed without asking (read-only discovery):** `ls`, `find` (only under Eddie's RDS project storage, `-maxdepth` ≤ 6), `stat`, `du -sh`, `file`, `quota -s`, `wc -l`, `squeue -u`, `sacct`, `mybalance`, `sshare`, and reading the **first line only** (the header) of a data file.
   - **Still ask first:** anything that writes, moves or deletes; `rsync`/sync pushes; `sbatch`, `scancel`, `scontrol update`; installing environments; any command printing more than a header line of a participant file.
   - **How to connect:** reuse Eddie's existing master socket, with `-o ControlPath=<the path in the HPC context file> -o BatchMode=yes`. Never open a new interactive login (it needs Eddie's password and 2FA). If the socket is closed, stop and ask Eddie to reopen it.
   - **Login-node policy:** discovery commands only; seconds of compute; `nice -19`; no heavy `find` across the whole filesystem.
   - Keep a log of every cluster command you run in `local/cluster_log.md` (git-ignored).
2. **Terminal output goes to the AI model provider.** Everything printed in the terminal is sent to whichever model is running this session. So **never print participant rows**: no `cat`, `head -n 2+`, `less`, or DataFrame `.head()` of participant data. Inspect data only through scripts that print aggregates (counts, means, cells n ≥ 10). This applies to the Channel 4, CARD and YourBrainType files. The EU-Emotion workbook holds stimulus-level aggregates and may be printed.
3. **No site names in tracked files.** Never write the site, host, user, or account names into any tracked file, commit, branch or PR. Put them in a git-ignored `local/site.sh` (add `local/` to `.gitignore` before the first commit) and source it from job scripts.
4. **Participant data stays on the cluster.** Never copy row-level participant data to the laptop, to git, or to any API. Only aggregates (counts, per-item distributions with every cell n ≥ 10) may leave the cluster. Questionnaire item wording and published test stimuli are not participant data.
5. **Cluster hygiene** (from the HPC context file): never write to `$HOME`; redirect every cache to RDS; watch the inode quota as well as bytes; keep GPU jobs short (≤ 1 h where possible) so they backfill; checkpoint and resume via JSONL keyed by item; set `HF_HUB_OFFLINE=1` in eval jobs; system Python is 3.6, so install Miniforge under the project's RDS root; no ffmpeg on compute nodes (use `imageio-ffmpeg`).
6. **Fail loudly.** Missing data, a wrong path or zero rows must stop the job with a clear message, never "succeed" silently.
7. **Debugging rule.** After two failed fixes on the same problem, stop and report options.
8. **Never invent facts.** Scoring keys, item wording, codebook values and published answer keys must come from the codebook, a published source you can cite, or Eddie. If you can't find one, list it as unconfirmed. Never fill it in from memory.
9. **Git workflow.** See "Git workflow" below. Commit early and often; never commit data.

## Git workflow (applies to every phase)

- **Location.** The project lives at `/Users/eb2007/playground/social_ai`, a subfolder of the existing git repo `eddiebullock/playground` (remote `https://github.com/eddiebullock/playground`, default branch `main`). Do not create a new repo. Touch only files under `social_ai/`.
- **Check visibility first.** Run `gh repo view eddiebullock/playground --json visibility`.
  - If the repo is **public**, add `social_ai/reports/` to `.gitignore` and commit only code, configs, tests and prereg drafts. Unpublished results must not go public.
  - If it is **private**, aggregate reports may be committed.
  - Either way, never commit data files, row-level outputs, model weights, `local/` or anything naming the cluster site.
- **Branches.** One branch per phase: `social_ai/setup`, `social_ai/a1-audit`, `social_ai/a2-repair`, `social_ai/b1-vlm`, `social_ai/b2-mechanism`. Branch from an up-to-date `main`.
- **Commit often.** Commit after each working unit: a loader plus its test, a metric, a job script. Use small, descriptive messages prefixed with the phase, for example `setup: add Channel 4 loader and exclusion tests`. Run `pytest` before each commit; don't commit a failing suite.
- **Push often.** Push the branch after every few commits and at the end of every session.
- **Merge.** At each GATE, once tests pass, open a pull request into `main` with `gh pr create`, summarising what changed and the sanity values. Merge it with `gh pr merge --squash` **after Eddie approves the gate**. Then pull `main`.
- **Before every commit**, run a check (add it as a pre-commit hook) that refuses to commit:
  - files over 5 MB;
  - files matching `*.csv`, `*.xlsx`, `*.parquet`, `*.pkl`, `*.npy`, `*.safetensors`, `*.pt`;
  - any file containing the cluster host name or account names held in `local/site.sh`.
- **Parallel agents.** The study agents (A and B) each work in their own git worktree so they never edit the same files:
  - `git worktree add ../playground-a social_ai/a1-audit`
  - `git worktree add ../playground-b social_ai/b1-vlm`
  
  Shared code under `src/scog/` should be changed by only one study at a time. If a study needs a shared change, make it in a small separate PR and merge it first.
