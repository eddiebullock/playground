"""Steering sweep CLI. OSF is not a gate."""

from __future__ import annotations

import argparse

from esvlm.utils.envcheck import check_job_env
from esvlm.utils.logging import echo_run_args, setup_logging


def main(argv: list[str] | None = None) -> int:
    check_job_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--alphas", required=True)
    parser.add_argument(
        "--allow-unpreregistered",
        action="store_true",
        help="ignored; OSF is not required",
    )
    args = parser.parse_args(argv)
    echo_run_args(vars(args), setup_logging())
    raise SystemExit("confirmatory sweep runner is Milestone F")


if __name__ == "__main__":
    raise SystemExit(main())
