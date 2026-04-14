import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def flatten_metrics(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return {
        "file": str(path),
        "model": obj.get("model"),
        "dataset": obj.get("dataset"),
        "n_frames": obj.get("n_frames"),
        "max_new_tokens": obj.get("max_new_tokens"),
        "temperature": obj.get("temperature"),
        "accuracy": obj.get("accuracy"),
        "n_trials": obj.get("n_trials"),
        "n_scored": obj.get("n_scored"),
        "n_correct": obj.get("n_correct"),
        "created_at_utc": obj.get("created_at_utc"),
        "transformers_version": (obj.get("run_metadata") or {}).get("transformers_version"),
        "torch_version": (obj.get("run_metadata") or {}).get("torch_version"),
        "git_commit": (obj.get("run_metadata") or {}).get("git_commit"),
        "slurm_job_id": (obj.get("run_metadata") or {}).get("slurm_job_id"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_glob", type=str, required=True, help='e.g. "results/baseline/eu_emotions/*/*.json"')
    ap.add_argument("--output_csv", type=Path, required=True)
    args = ap.parse_args()

    paths = [Path(p) for p in sorted(Path().glob(args.input_glob))]
    rows: List[Dict[str, Any]] = []
    for p in paths:
        try:
            rows.append(flatten_metrics(p))
        except Exception:
            continue

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        args.output_csv.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    main()

