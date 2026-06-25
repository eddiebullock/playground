"""
Inspect AI task entrypoint for EU-Emotions mental-state recognition.

Wraps existing trial foil logic, prompts, and scorers. Does not replace scripts/evaluate.py.

Usage (requires inspect-ai):
  pip install inspect-ai
  inspect eval inspect_eu/eu_emotions_task.py@eu_emotions_eval --model openai/gpt-4o

Local HF models use the evaluate.py pipeline via the bridge solver when configured.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from config import SEED
from inspect_eu.dataset import load_eu_emotions_samples, to_inspect_samples
from scripts.prompts import build_free_response_prompt, build_4afc_prompt


def _build_solver(stage: str = "free_response", condition: str = "video_only"):
    try:
        from inspect_ai.solver import Solver, TaskState, solver
    except ImportError as e:
        raise ImportError("inspect-ai required: pip install inspect-ai") from e

    @solver
    def eu_prompt_solver() -> Solver:
        async def solve(state: TaskState, generate) -> TaskState:
            meta = state.metadata or {}
            if stage in ("free_response", "stage1"):
                state.user_prompt.text = build_free_response_prompt(condition=condition)
            else:
                opts = meta.get("candidate_labels") or []
                state.user_prompt.text = build_4afc_prompt(opts, condition=condition)
            return await generate(state)

        return solve

    return eu_prompt_solver()


def eu_emotions_task(
    *,
    manifest: Optional[Path] = None,
    data_root: Optional[Path] = None,
    max_trials: Optional[int] = None,
    stage: str = "free_response",
    condition: str = "video_only",
    seed: int = SEED,
):
    """Build Inspect Task (call from @task wrapper when inspect-ai installed)."""
    from inspect_ai import Task
    from inspect_eu.scorers import build_inspect_scorers

    raw = load_eu_emotions_samples(
        manifest=manifest, data_root=data_root, max_trials=max_trials, seed=seed
    )
    dataset = to_inspect_samples(raw)
    primary = "free_response" if stage in ("free_response", "stage1") else "4afc"
    return Task(
        dataset=dataset,
        solver=_build_solver(stage=stage, condition=condition),
        scorer=build_inspect_scorers(primary=primary),
    )


try:
    from inspect_ai import task

    @task
    def eu_emotions_eval():
        """Inspect CLI entry: inspect eval inspect_eu/eu_emotions_task.py@eu_emotions_eval"""
        return eu_emotions_task(stage="free_response", condition="video_only")

    @task
    def eu_emotions_4afc():
        """Secondary 4AFC comparability task."""
        return eu_emotions_task(stage="stage2_4afc", condition="video_only")

except ImportError:
    def eu_emotions_eval():  # type: ignore
        raise ImportError("pip install inspect-ai to run Inspect tasks")

    def eu_emotions_4afc():  # type: ignore
        raise ImportError("pip install inspect-ai to run Inspect tasks")
