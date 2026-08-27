"""
Step 3 — Open-weight VLM RMET 4AFC evaluation (study4 only).

Reuses study3 loaders/generation via read-only imports from the HPC project root
(mirrored by study4_rmet/sync.sh push-repo-readonly). Does not modify study3 files.

Usage (on HPC, from ~/rds/hpc-work/study4_rmet):
  python -m study4_rmet.scripts.evaluate_rmet --model qwen3vl
  # or:
  python study4_rmet/scripts/evaluate_rmet.py --model qwen3vl

One deterministic pass (temperature 0) plus optional sampled repeats for entropy.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

STUDY4_ROOT = Path(__file__).resolve().parents[1]
# HPC layout: study4_rmet/ sits beside scripts/ and config.py after push-repo-readonly.
# Local layout: study4_rmet/ sits inside the mr_eu_open_llm repo.
_CANDIDATE_ROOTS = [
    STUDY4_ROOT.parent,  # local: repo root; HPC: study4_rmet's parent = project root
    STUDY4_ROOT,         # if scripts were copied beside this package
]
for root in _CANDIDATE_ROOTS:
    if (root / "scripts" / "evaluate.py").exists() and (root / "config.py").exists():
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        break

from PIL import Image  # noqa: E402

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from rmet_prompts import build_rmet_4afc_prompt  # noqa: E402


def _entropy(counts: Counter, options: Sequence[str]) -> float:
    total = sum(counts[o] for o in options)
    if total <= 0:
        return float("nan")
    h = 0.0
    for o in options:
        p = counts[o] / total
        if p > 0:
            h -= p * math.log(p)
    return float(h)


def load_manifest(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_eval(
    *,
    model_key: str,
    manifest_path: Path,
    stim_root: Path,
    output: Path,
    seed: int = 42,
    temperature_det: float = 0.0,
    n_samples: int = 10,
    sample_temperature: float = 0.7,
    max_items: Optional[int] = None,
    max_new_tokens: int = 128,
) -> Dict[str, Any]:
    import torch
    from transformers import AutoProcessor

    from scripts.emotion_parse import parse_emotion
    from scripts.evaluate import load_hf_model_for_key, resolve_model_path
    from scripts.model_inference import generate_model_response, generate_model_response_batch, seed_generation
    from scripts.multi_frame import prepare_images_for_model

    manifest = load_manifest(manifest_path)
    trials = list(manifest["trials"])
    if max_items is not None:
        trials = trials[: int(max_items)]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model_path = resolve_model_path(model_key)

    if model_key != "gemma4":
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    else:
        try:
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        except Exception:
            from transformers import Gemma4Processor  # type: ignore

            processor = Gemma4Processor.from_pretrained(model_path)

    model = load_hf_model_for_key(model_key, model_path, device, dtype)
    if device != "cuda":
        model = model.to(device)
    model.eval()

    pipe_cache: Dict[str, Any] = {}
    tokenizer = getattr(processor, "tokenizer", None)
    results_trials: List[Dict[str, Any]] = []

    for trial in trials:
        item = int(trial["item"])
        options = list(trial["options"])
        correct_label = trial["correct_label"]
        img_rel = trial["image"]
        img_path = stim_root / Path(img_rel).name if (stim_root / Path(img_rel).name).exists() else STUDY4_ROOT / img_rel
        if not img_path.exists():
            # Prefer stimuli dir next to manifest
            alt = manifest_path.parent / Path(img_rel).name
            img_path = alt if alt.exists() else img_path
        if not img_path.exists():
            raise FileNotFoundError(f"Missing stimulus for item {item}: tried {img_path}")

        image = Image.open(img_path).convert("RGB")
        images_for_proc, mf_meta = prepare_images_for_model(
            model_key, [image], enforce_multi_frame=False
        )
        prompt = build_rmet_4afc_prompt(options)

        # Deterministic pass
        seed_generation(seed, model_key, f"rmet_{item:02d}", "det")
        det_text = generate_model_response(
            model_key=model_key,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            model_path=model_path,
            prompt=prompt,
            images=[image],
            images_for_processor=images_for_proc,
            device=device,
            dtype=dtype,
            temperature=float(temperature_det),
            max_new_tokens=max_new_tokens,
            pipe_cache=pipe_cache,
            condition="video_only",
            prefer_loaded_model=True,
        )
        det_pred, det_reason = parse_emotion(det_text, options)
        det_correct = bool(det_pred is not None and det_pred.lower() == correct_label.lower())

        sample_preds: List[Optional[str]] = []
        sample_texts: List[str] = []
        if n_samples > 0:
            seed_generation(seed, model_key, f"rmet_{item:02d}", "sample")
            try:
                sample_texts = generate_model_response_batch(
                    model_key=model_key,
                    model=model,
                    processor=processor,
                    tokenizer=tokenizer,
                    model_path=model_path,
                    prompt=prompt,
                    images=[image],
                    images_for_processor=images_for_proc,
                    device=device,
                    dtype=dtype,
                    temperature=float(sample_temperature),
                    max_new_tokens=max_new_tokens,
                    pipe_cache=pipe_cache,
                    num_return_sequences=int(n_samples),
                    condition="video_only",
                    prefer_loaded_model=True,
                )
            except Exception:
                sample_texts = []
                for s_i in range(int(n_samples)):
                    seed_generation(seed, model_key, f"rmet_{item:02d}", "sample", s_i)
                    sample_texts.append(
                        generate_model_response(
                            model_key=model_key,
                            model=model,
                            processor=processor,
                            tokenizer=tokenizer,
                            model_path=model_path,
                            prompt=prompt,
                            images=[image],
                            images_for_processor=images_for_proc,
                            device=device,
                            dtype=dtype,
                            temperature=float(sample_temperature),
                            max_new_tokens=max_new_tokens,
                            pipe_cache=pipe_cache,
                            condition="video_only",
                            prefer_loaded_model=True,
                        )
                    )
            for t in sample_texts:
                pred, _ = parse_emotion(t, options)
                sample_preds.append(pred)

        counts = Counter(p for p in sample_preds if p is not None)
        parse_fail = sum(1 for p in sample_preds if p is None)
        ent = _entropy(counts, options) if sample_preds else float("nan")
        sample_acc = (
            sum(1 for p in sample_preds if p is not None and p.lower() == correct_label.lower())
            / max(1, sum(1 for p in sample_preds if p is not None))
        )

        results_trials.append(
            {
                "trial_id": trial["trial_id"],
                "item": item,
                "image": str(img_path),
                "options": options,
                "correct_label": correct_label,
                "multi_frame_strategy": mf_meta.get("multi_frame_strategy"),
                "prompt": prompt,
                "deterministic": {
                    "temperature": float(temperature_det),
                    "raw_output": det_text,
                    "prediction": det_pred,
                    "reasoning": det_reason,
                    "correct": det_correct,
                    "parse_fail": det_pred is None,
                },
                "samples": {
                    "n_samples": int(n_samples),
                    "temperature": float(sample_temperature),
                    "predictions": sample_preds,
                    "raw_outputs": sample_texts,
                    "distribution": {k: int(v) for k, v in counts.items()},
                    "parse_failures": int(parse_fail),
                    "accuracy": float(sample_acc) if sample_preds else None,
                    "entropy": ent,
                },
            }
        )

    n_scored = sum(1 for t in results_trials if not t["deterministic"]["parse_fail"])
    n_correct = sum(1 for t in results_trials if t["deterministic"]["correct"])
    payload = {
        "study": "study4_rmet",
        "model": model_key,
        "condition": "video_only",
        "seed": seed,
        "n_items": len(results_trials),
        "n_scored_deterministic": n_scored,
        "accuracy_deterministic": (n_correct / n_scored) if n_scored else None,
        "chance_level": 0.25,
        "n_samples": int(n_samples),
        "sample_temperature": float(sample_temperature),
        "manifest": str(manifest_path),
        "trials": results_trials,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # Clean per-item table
    import csv

    table_path = output.with_suffix(".csv")
    with table_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "item",
                "correct_label",
                "det_prediction",
                "det_correct",
                "det_parse_fail",
                "sample_accuracy",
                "sample_entropy",
                "sample_parse_failures",
            ],
        )
        w.writeheader()
        for t in results_trials:
            w.writerow(
                {
                    "item": t["item"],
                    "correct_label": t["correct_label"],
                    "det_prediction": t["deterministic"]["prediction"],
                    "det_correct": int(t["deterministic"]["correct"]),
                    "det_parse_fail": int(t["deterministic"]["parse_fail"]),
                    "sample_accuracy": t["samples"]["accuracy"],
                    "sample_entropy": t["samples"]["entropy"],
                    "sample_parse_failures": t["samples"]["parse_failures"],
                }
            )
    payload["table_csv"] = str(table_path)
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=("qwen3vl", "gemma4", "molmo2"))
    ap.add_argument(
        "--manifest",
        type=Path,
        default=STUDY4_ROOT / "data" / "rmet" / "stimuli" / "manifest.json",
    )
    ap.add_argument(
        "--stim_root",
        type=Path,
        default=STUDY4_ROOT / "data" / "rmet" / "stimuli",
    )
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_samples", type=int, default=10)
    ap.add_argument("--sample_temperature", type=float, default=0.7)
    ap.add_argument("--max_items", type=int, default=None, help="Smoke: limit items")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    args = ap.parse_args(argv)

    out = args.output
    if out is None:
        tag = f"max{args.max_items}" if args.max_items else "full"
        out = (
            STUDY4_ROOT
            / "results"
            / "model"
            / args.model
            / f"rmet_eval_{args.model}_{tag}_seed{args.seed}.json"
        )

    payload = run_eval(
        model_key=args.model,
        manifest_path=args.manifest,
        stim_root=args.stim_root,
        output=out,
        seed=args.seed,
        n_samples=args.n_samples,
        sample_temperature=args.sample_temperature,
        max_items=args.max_items,
        max_new_tokens=args.max_new_tokens,
    )
    print(
        f"model={payload['model']} acc_det={payload['accuracy_deterministic']} "
        f"n={payload['n_items']} -> {out}"
    )
    return 0


if __name__ == "__main__":
    # Allow `python study4_rmet/scripts/evaluate_rmet.py` without package install.
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    raise SystemExit(main())
