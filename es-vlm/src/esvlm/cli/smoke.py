"""Mac smoke: smallest Qwen3-VL on synthetic fixtures."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from esvlm.analysis.plots import plot_alpha_accuracy
from esvlm.scoring.logprob import score_item
from esvlm.steering.hooks import SteeringHook
from esvlm.tasks.items import load_synthetic_items
from esvlm.tasks.synthetic import write_fixtures
from esvlm.utils.device import detect_device, torch_device
from esvlm.utils.envcheck import check_job_env
from esvlm.utils.io import append_score_rows, parquet_shard_path
from esvlm.utils.logging import echo_run_args, setup_logging
from esvlm.utils.seeding import seed_everything
from esvlm.vectors.contrasts import default_contrasts
from esvlm.vectors.extract import ResidualCatcher, difference_of_means


def main(argv: list[str] | None = None) -> int:
    check_job_env()
    parser = argparse.ArgumentParser(description="Mac smoke test for es-vlm")
    parser.add_argument("--model", default="qwen3vl_2b")
    parser.add_argument("--fixtures", type=Path, default=Path("tests/fixtures/synthetic"))
    parser.add_argument("--out", type=Path, default=Path("outputs/smoke"))
    parser.add_argument("--layers", default="0,1")
    parser.add_argument("--alphas", default="-1,0,1", metavar="LIST")
    parser.add_argument("--max-items", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(os.environ["ESVLM_MODELS_DIR"]) if os.environ.get("ESVLM_MODELS_DIR") else None,
    )
    args = parser.parse_args(argv)

    log = setup_logging()
    echo_run_args(vars(args), log)
    seed_everything(args.seed)
    write_fixtures(args.fixtures)
    items = load_synthetic_items(args.fixtures)[: args.max_items]
    if len(items) < 2:
        raise SystemExit("need synthetic fixtures")

    from PIL import Image

    from esvlm.models.loader import load_model, residual_modules

    t0 = time.perf_counter()
    loaded = load_model(args.model, models_dir=args.models_dir, offline=args.offline)
    load_s = time.perf_counter() - t0
    device = torch_device(loaded.device)
    layers = residual_modules(loaded)
    layer_ids = [int(x) for x in str(args.layers).split(",") if x != ""]
    alphas = [float(x) for x in str(args.alphas).split(",") if x != ""]

    contrasts = default_contrasts()
    pair = contrasts["E"]
    # extract directions from a few narratives at the requested layers
    directions: dict[int, object] = {}
    texts_pos = pair.positive[:8]
    texts_neg = pair.negative[:8]
    for layer_i in layer_ids:
        if layer_i < 0 or layer_i >= len(layers):
            raise SystemExit(f"layer {layer_i} out of range 0..{len(layers) - 1}")
        pos = _text_residuals(loaded.model, loaded.processor, layers[layer_i], texts_pos, device)
        neg = _text_residuals(loaded.model, loaded.processor, layers[layer_i], texts_neg, device)
        directions[layer_i] = difference_of_means(pos, neg)

    rows = []
    peak = 0.0
    tokens = 0
    t1 = time.perf_counter()
    for item in items:
        image = Image.open(item.media).convert("RGB")
        for layer_i in layer_ids:
            for alpha in alphas:
                hook_mod = layers[layer_i]
                with SteeringHook(hook_mod, directions[layer_i], alpha=alpha, mode="add"):
                    row = score_item(
                        loaded.model,
                        loaded.processor,
                        item,
                        device=device,
                        images=[image],
                        model_key=loaded.spec.key,
                        revision=loaded.spec.revision,
                        template_id=str(item.meta.get("template_id", "color")),
                        condition="smoke",
                        vector="E",
                        layer=layer_i,
                        alpha=alpha,
                        mode="add",
                        frames=1,
                        seed=args.seed,
                    )
                rows.append(row)
                tokens += row.tokens
                peak = max(peak, _peak_mem_mb())
    score_s = time.perf_counter() - t1

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    parquet = parquet_shard_path(out, "synthetic", loaded.spec.key, "smoke")
    append_score_rows(parquet, rows)

    # one plot: accuracy vs alpha at first layer
    first = layer_ids[0]
    by_alpha: dict[float, list[bool]] = {a: [] for a in alphas}
    for row in rows:
        if row.layer == first:
            by_alpha[row.alpha].append(row.correct)
    plot_alpha_accuracy(
        alphas,
        {f"layer {first}": [sum(by_alpha[a]) / max(1, len(by_alpha[a])) for a in alphas]},
        out / "smoke_alpha.png",
        title=f"{loaded.spec.key} synthetic accuracy",
    )
    summary = {
        "model": loaded.spec.key,
        "revision": loaded.spec.revision,
        "device": loaded.device.kind,
        "dtype": loaded.device.dtype_name,
        "n_items": len(items),
        "n_rows": len(rows),
        "load_seconds": round(load_s, 3),
        "score_seconds": round(score_s, 3),
        "seconds_per_item": round(score_s / max(1, len(items)), 3),
        "peak_mem_mb": round(peak, 1),
        "tokens_per_item": round(tokens / max(1, len(items)), 2),
        "parquet": str(parquet),
    }
    (out / "smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("smoke_summary %s", json.dumps(summary))
    detect_device()  # keep import used
    return 0


def _text_residuals(model, processor, module, texts: list[str], device):
    import numpy as np
    import torch

    acts = []
    for text in texts:
        inputs = processor(text=text, return_tensors="pt")
        items = dict(inputs)
        # Molmo2 text-only encode still emits a dummy image grid
        # (AssertionError: Expected 0 image grids, but got 1).
        for key in list(items):
            low = key.lower()
            if any(part in low for part in ("image", "pixel", "grid")):
                items.pop(key)
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in items.items()}
        catcher = ResidualCatcher([module])
        with torch.no_grad(), catcher:
            model(**inputs)
        acts.append(catcher.stacked()[0])
    return np.stack(acts, axis=0)


def _peak_mem_mb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated()) / (1024 * 1024)
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return float(torch.mps.current_allocated_memory()) / (1024 * 1024)
    except Exception:
        return 0.0
    return 0.0


if __name__ == "__main__":
    raise SystemExit(main())
