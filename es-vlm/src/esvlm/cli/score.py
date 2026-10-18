"""Baseline scoring CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from esvlm.utils.envcheck import check_job_env
from esvlm.utils.logging import echo_run_args, setup_logging


def main(argv: list[str] | None = None) -> int:
    check_job_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trial_offset", type=int, default=0)
    parser.add_argument("--trial_limit", type=int, default=0)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    log = setup_logging()
    echo_run_args(vars(args), log)
    if args.task == "synthetic":
        from esvlm.cli.smoke import main as smoke

        return smoke(
            [
                "--model",
                args.model,
                "--out",
                str(args.out),
                "--seed",
                str(args.seed),
            ]
        )
    raise SystemExit(
        f"task {args.task} needs a licensed manifest path from local/site.sh; "
        "not run in Milestone A/B"
    )


if __name__ == "__main__":
    raise SystemExit(main())
