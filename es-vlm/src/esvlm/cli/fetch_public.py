"""Fetch public Paper 1/2 annotations into $ESVLM_PUBLIC_DIR.

NExT-QA CSVs and MMToM-QA questions only. No NExT-QA videos. No
257GB MMToM video zip.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from esvlm.tasks.mmtom import fetch_mmtom_text
from esvlm.tasks.nextqa import fetch_nextqa_annotations, fetch_nextqa_videos
from esvlm.utils.logging import echo_run_args, setup_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch public Paper 1/2 annotations")
    parser.add_argument("--dataset", default=os.environ.get("DATASET", "all"))
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ["ESVLM_PUBLIC_DIR"]) if os.environ.get("ESVLM_PUBLIC_DIR") else None,
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=Path(os.environ["ESVLM_PROJECT_DATA"]) if os.environ.get("ESVLM_PROJECT_DATA") else None,
    )
    args = parser.parse_args(argv)
    if args.root is None:
        raise SystemExit("ESVLM_PUBLIC_DIR is unset")
    log = setup_logging()
    echo_run_args(vars(args), log)
    wanted = args.dataset.lower()
    if wanted in {"nextqa", "all"}:
        fetch_nextqa_annotations(args.root, log)
    if wanted in {"mmtom", "all"}:
        fetch_mmtom_text(args.root, log)
    if wanted == "nextqa_videos":
        video_root = args.video_root or (args.root / "nextqa")
        fetch_nextqa_videos(Path(video_root) / "nextqa", log)
        return 0
    if wanted not in {"nextqa", "mmtom", "all"}:
        raise SystemExit(f"unknown dataset {args.dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
