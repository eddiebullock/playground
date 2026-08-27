"""
Contamination / prompt robustness smoke: option-order sensitivity on RMET.

Classic RMET is web-exposed; upright accuracy may partly reflect memorization.
This smoke re-runs a few items with shuffled option order (same labels) and
reports whether deterministic predictions flip.

Does not prove absence of contamination; documents sensitivity.
Prefer open-weight local eval; commercial optional via existing API script.

Usage:
  python study4_rmet/scripts/contamination_option_order_smoke.py --model qwen3vl --max_items 6
  python study4_rmet/scripts/contamination_option_order_smoke.py --offline_report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

STUDY4_ROOT = Path(__file__).resolve().parents[1]
_CANDIDATE_ROOTS = [STUDY4_ROOT.parent, STUDY4_ROOT]
for root in _CANDIDATE_ROOTS:
    if (root / "scripts" / "evaluate.py").exists() and (root / "config.py").exists():
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        break

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

DEFAULT_MANIFEST = STUDY4_ROOT / "data" / "rmet" / "stimuli" / "manifest.json"
DEFAULT_STIM = STUDY4_ROOT / "data" / "rmet" / "stimuli"
DEFAULT_OUT = STUDY4_ROOT / "results" / "robustness" / "contamination"


def offline_limitation_report(outdir: Path) -> Dict[str, Any]:
    report = {
        "status": "limitation_documented",
        "classic_rmet_web_exposure": True,
        "eyes_only_stimuli": True,
        "optional_future": "MRMET out-of-sample check (not blocking)",
        "construct": (
            "Prefer mental-state / complex emotion recognition from eye region; "
            "do not claim ToM circuits; verbal knowledge / alexithymia comorbidity "
            "limit ASC contrasts (alexithymia absent from CARD)."
        ),
        "option_order_smoke": (
            "Run contamination_option_order_smoke.py with GPU/API to quantify "
            "prediction flips under option permutation."
        ),
        "paraphrase": "Not automated here; note as stretch / qualitative limitation.",
    }
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "contamination_limitations.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def run_order_smoke(
    *,
    model_key: str,
    manifest_path: Path,
    stim_root: Path,
    outdir: Path,
    max_items: int,
    seed: int,
    n_shuffles: int,
) -> Dict[str, Any]:
    import torch
    from PIL import Image
    from transformers import AutoProcessor

    from rmet_prompts import build_rmet_4afc_prompt
    from scripts.emotion_parse import parse_emotion
    from scripts.evaluate import load_hf_model_for_key, resolve_model_path
    from scripts.model_inference import generate_model_response, seed_generation
    from scripts.multi_frame import prepare_images_for_model

    if not torch.cuda.is_available():
        report = offline_limitation_report(outdir)
        report["status"] = "planned_only_no_cuda"
        (outdir / "option_order_smoke_summary.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        return report

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    trials = list(manifest["trials"])[:max_items]
    device = "cuda"
    dtype = torch.bfloat16
    model_path = resolve_model_path(model_key)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = load_hf_model_for_key(model_key, model_path, device, dtype)
    model.eval()
    tokenizer = getattr(processor, "tokenizer", None)
    pipe_cache: Dict[str, Any] = {}
    rng = np.random.default_rng(seed)

    rows = []
    for trial in trials:
        item = int(trial["item"])
        base_opts = list(trial["options"])
        correct = trial["correct_label"]
        img_path = stim_root / Path(trial["image"]).name
        image = Image.open(img_path).convert("RGB")
        images_for_proc, _ = prepare_images_for_model(
            model_key, [image], enforce_multi_frame=False
        )

        def _pred(opts: List[str], tag: str) -> Optional[str]:
            prompt = build_rmet_4afc_prompt(opts)
            seed_generation(seed, model_key, f"rmet_order_{item:02d}_{tag}", 0)
            text = generate_model_response(
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
                temperature=0.0,
                max_new_tokens=128,
                pipe_cache=pipe_cache,
                condition="video_only",
                prefer_loaded_model=True,
            )
            pred, _ = parse_emotion(text, opts)
            return pred

        base_pred = _pred(base_opts, "base")
        flips = 0
        for s in range(n_shuffles):
            perm = rng.permutation(len(base_opts))
            shuf = [base_opts[i] for i in perm]
            pred = _pred(shuf, f"shuf{s}")
            if pred != base_pred:
                flips += 1
            rows.append(
                {
                    "item": item,
                    "shuffle": s,
                    "base_pred": base_pred,
                    "shuf_pred": pred,
                    "flipped": pred != base_pred,
                    "correct_label": correct,
                    "base_correct": base_pred is not None
                    and base_pred.lower() == correct.lower(),
                }
            )

    import pandas as pd

    df = pd.DataFrame(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / f"option_order_smoke_{model_key}.csv", index=False)
    summary = {
        "status": "ok",
        "model": model_key,
        "n_items": max_items,
        "n_shuffles": n_shuffles,
        "flip_rate": float(df["flipped"].mean()) if len(df) else float("nan"),
        "limitation": (
            "Nonzero flip_rate implies order sensitivity; zero does not prove "
            "no memorization of famous RMET items."
        ),
    }
    (outdir / f"option_order_smoke_{model_key}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen3vl")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--stim_root", type=Path, default=DEFAULT_STIM)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max_items", type=int, default=6)
    ap.add_argument("--n_shuffles", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--offline_report",
        action="store_true",
        help="Write contamination limitation JSON without model load",
    )
    args = ap.parse_args(argv)

    if args.offline_report:
        report = offline_limitation_report(args.outdir)
        print(json.dumps(report, indent=2))
        return 0

    result = run_order_smoke(
        model_key=args.model,
        manifest_path=args.manifest,
        stim_root=args.stim_root,
        outdir=args.outdir,
        max_items=args.max_items,
        seed=args.seed,
        n_shuffles=args.n_shuffles,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
