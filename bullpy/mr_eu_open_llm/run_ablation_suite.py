#!/usr/bin/env python3
from __future__ import annotations

"""
Run modality ablation matrix for open VLMs on EU-Emotions and Mindreading.

EU and Mindreading use separate data roots. Audio conditions run only for models
listed in config.MODEL_AUDIO_CAPABILITIES with value True (currently gemma4).
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from config import MODEL_AUDIO_CAPABILITIES, MODALITY_CONDITIONS, SEED, STUDY_MODELS


def _run(cmd: List[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def _eval_cmd(
    *,
    model: str,
    dataset: str,
    condition: str,
    data_root: str,
    manifest: str,
    results_dir: str,
    seed: int,
    max_trials: Optional[int],
) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "scripts.evaluate",
        "--model",
        model,
        "--dataset",
        dataset,
        "--condition",
        condition,
        "--data_root",
        data_root,
        "--manifest",
        manifest,
        "--seed",
        str(seed),
        "--output",
        str(Path(results_dir) / f"eval_v2_{dataset}_{model}_{condition}_seed{seed}.json"),
    ]
    if max_trials is not None:
        cmd.extend(["--max_trials", str(max_trials)])
    return cmd


def main() -> None:
    p = argparse.ArgumentParser(description="Modality ablation suite for open VLMs.")
    p.add_argument("--data-dir-eu", required=True, help="EU-Emotions root (faces + UK Voices tree).")
    p.add_argument("--data-dir-mr", required=True, help="Mindreading root (item folders; not Emotions/Audio/).")
    p.add_argument("--manifest-eu", required=True, help="EU 118-trial manifest JSON.")
    p.add_argument("--manifest-mr", required=True, help="Mindreading trial manifest JSON.")
    p.add_argument("--results-dir", default="results/ablation")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--max-trials", type=int, default=None)
    args = p.parse_args()

    models = list(STUDY_MODELS)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    for model in models:
        _run(
            _eval_cmd(
                model=model,
                dataset="eu_emotions",
                condition="video_only",
                data_root=args.data_dir_eu,
                manifest=args.manifest_eu,
                results_dir=args.results_dir,
                seed=args.seed,
                max_trials=args.max_trials,
            )
        )
        _run(
            _eval_cmd(
                model=model,
                dataset="mindreading",
                condition="video_only",
                data_root=args.data_dir_mr,
                manifest=args.manifest_mr,
                results_dir=args.results_dir,
                seed=args.seed,
                max_trials=args.max_trials,
            )
        )

    audio_models = [m for m in models if MODEL_AUDIO_CAPABILITIES.get(m)]
    if not audio_models:
        print(
            "\nSkipping audio_only/multimodal: no MODEL_AUDIO_CAPABILITIES=True "
            f"(models: {models})."
        )
        return

    print(f"\nAudio ablations for: {audio_models}")
    for model in audio_models:
        for condition in ("audio_only", "multimodal"):
            if condition not in MODALITY_CONDITIONS:
                continue
            _run(
                _eval_cmd(
                    model=model,
                    dataset="eu_emotions",
                    condition=condition,
                    data_root=args.data_dir_eu,
                    manifest=args.manifest_eu,
                    results_dir=args.results_dir,
                    seed=args.seed,
                    max_trials=args.max_trials,
                )
            )
            _run(
                _eval_cmd(
                    model=model,
                    dataset="mindreading",
                    condition=condition,
                    data_root=args.data_dir_mr,
                    manifest=args.manifest_mr,
                    results_dir=args.results_dir,
                    seed=args.seed,
                    max_trials=args.max_trials,
                )
            )


if __name__ == "__main__":
    main()
