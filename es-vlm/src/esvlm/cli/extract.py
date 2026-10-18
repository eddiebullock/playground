"""Vector extraction CLI."""

from __future__ import annotations

import argparse

from esvlm.utils.envcheck import check_job_env
from esvlm.utils.logging import echo_run_args, setup_logging


def main(argv: list[str] | None = None) -> int:
    check_job_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    echo_run_args(vars(args), setup_logging())
    raise SystemExit("full-layer extraction is Milestone E; use esvlm-smoke for the Mac path")


if __name__ == "__main__":
    raise SystemExit(main())
