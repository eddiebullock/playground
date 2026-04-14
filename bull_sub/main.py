"""
CLI entrypoint: run the content pipeline once or on a daily schedule (9:00 default).
"""

from __future__ import annotations

import argparse
import logging

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from config import DEFAULT_SCHEDULE_CRON
from src.pipeline import run_full_pipeline


def _schedule() -> None:
    """Run ``run_full_pipeline`` every day at the configured local time."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sched = BlockingScheduler()
    cron = DEFAULT_SCHEDULE_CRON
    sched.add_job(
        run_full_pipeline,
        CronTrigger(hour=cron["hour"], minute=cron["minute"]),
        id="bull_sub_daily_pipeline",
        replace_existing=True,
    )
    sched.start()


def main() -> None:
    """Parse CLI arguments and dispatch."""
    parser = argparse.ArgumentParser(description="Grey Matters (bull_sub) content engine")
    parser.add_argument(
        "command",
        nargs="?",
        default="run-once",
        choices=["run-once", "schedule"],
        help="run-once (default): single pipeline run; schedule: daily APScheduler loop",
    )
    args = parser.parse_args()
    if args.command == "run-once":
        msg = run_full_pipeline()
        print(msg)
    else:
        _schedule()


if __name__ == "__main__":
    main()
