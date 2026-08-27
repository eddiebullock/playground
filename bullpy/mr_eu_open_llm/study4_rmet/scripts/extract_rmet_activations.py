"""
Step 5 — Extract hidden-state activations for each of the 36 RMET items (study4 only).

Reuses study3 loaders / forward helpers read-only (mirrored under study4 HPC root).
Does not modify study3 code. Molmo token_type_ids stripping lives in
scripts.activation_forward.build_forward_inputs.

Layers: config.LAYER_DEPTH_FRACTIONS via scripts.layer_map (0.125 / 0.375 / 0.75).
Pooling: last_token (matches study3 4AFC extraction convention).

Usage (HPC):
  python study4_rmet/scripts/extract_rmet_activations.py --model qwen3vl
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

from PIL import Image  # noqa: E402

from rmet_prompts import build_rmet_4afc_prompt  # noqa: E402


def _pool_last_token(h: Any) -> np.ndarray:
    import torch

    hf = h.detach().float()
    return hf[:, -1, :].cpu().numpy()


def run_extract(
    *,
    model_key: str,
    manifest_path: Path,
    stim_root: Path,
    output_dir: Path,
    seed: int = 42,
    max_items: Optional[int] = None,
) -> Dict[str, Any]:
    import torch
    from transformers import AutoProcessor

    from config import LAYER_DEPTH_FRACTIONS
    from scripts.activation_forward import build_forward_inputs, find_layer_module, run_forward
    from scripts.evaluate import load_hf_model_for_key, resolve_model_path
    from scripts.layer_map import get_layer_indices, n_layers
    from scripts.multi_frame import prepare_images_for_model

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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

    layer_map = get_layer_indices(model_key, fractions=list(LAYER_DEPTH_FRACTIONS), model=model)
    nl = n_layers(model_key, model)
    storage: Dict[str, List[np.ndarray]] = {k: [] for k in layer_map}
    hooks = []

    def _make_hook(bucket: List[np.ndarray]):
        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if hasattr(h, "detach"):
                bucket.append(_pool_last_token(h))

        return hook

    for frac_key, layer_idx in layer_map.items():
        mod = find_layer_module(model, layer_idx)
        if mod is None:
            raise RuntimeError(f"Could not find layer module {layer_idx} for {model_key}")
        hooks.append(mod.register_forward_hook(_make_hook(storage[frac_key])))

    trial_ids: List[str] = []
    errors: List[Dict[str, str]] = []

    try:
        for trial in trials:
            item = int(trial["item"])
            trial_id = str(trial.get("trial_id", f"rmet_{item:02d}"))
            trial_ids.append(trial_id)
            options = list(trial["options"])
            prompt = build_rmet_4afc_prompt(options)

            img_name = Path(trial["image"]).name
            img_path = stim_root / img_name
            if not img_path.exists():
                img_path = STUDY4_ROOT / trial["image"]
            if not img_path.exists():
                raise FileNotFoundError(f"Missing stimulus {img_path}")

            try:
                image = Image.open(img_path).convert("RGB")
                images_for_proc, _meta = prepare_images_for_model(
                    model_key, [image], enforce_multi_frame=False
                )
                inputs = build_forward_inputs(
                    model_key,
                    model,
                    processor,
                    prompt=prompt,
                    images=[image],
                    images_for_processor=images_for_proc,
                    device=device,
                    dtype=dtype,
                    condition="video_only",
                )
                before = {k: len(storage[k]) for k in storage}
                run_forward(model, inputs, model_key=model_key)
                for k in storage:
                    if len(storage[k]) == before[k]:
                        # Hook did not fire — pad zeros later
                        raise RuntimeError(f"No activation captured for {k} on {trial_id}")
            except Exception as e:
                errors.append({"trial_id": trial_id, "error": f"{type(e).__name__}: {e}"})
                for k in storage:
                    if storage[k]:
                        dim = storage[k][0].shape[-1]
                    else:
                        dim = 1
                    storage[k].append(np.zeros((1, dim), dtype=np.float32))
    finally:
        for h in hooks:
            h.remove()

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for frac_key, arrs in storage.items():
        acts = np.concatenate(arrs, axis=0)
        layer_idx = layer_map[frac_key]
        out_path = output_dir / f"layer{layer_idx}_rmet_seed{seed}.npy"
        np.save(out_path, acts)
        (output_dir / f"layer{layer_idx}_trial_ids.json").write_text(
            json.dumps(trial_ids, indent=2) + "\n", encoding="utf-8"
        )
        saved.append({"frac_key": frac_key, "layer_index": layer_idx, "path": str(out_path), "shape": list(acts.shape)})

    meta = {
        "study": "study4_rmet",
        "model": model_key,
        "condition": "video_only",
        "prompt_mode": "4afc",
        "pooling": "last_token",
        "n_layers": nl,
        "layer_map": layer_map,
        "layer_depth_fractions": list(LAYER_DEPTH_FRACTIONS),
        "n_trials": len(trials),
        "seed": seed,
        "errors": errors,
        "layers_saved": saved,
        "manifest": str(manifest_path),
    }
    (output_dir / "extract_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


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
    ap.add_argument("--output_dir", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_items", type=int, default=None)
    args = ap.parse_args(argv)

    out = args.output_dir
    if out is None:
        tag = f"smoke{args.max_items}" if args.max_items else "full"
        out = STUDY4_ROOT / "results" / "activations" / args.model / tag

    meta = run_extract(
        model_key=args.model,
        manifest_path=args.manifest,
        stim_root=args.stim_root,
        output_dir=out,
        seed=args.seed,
        max_items=args.max_items,
    )
    print(
        f"extract {meta['model']}: n={meta['n_trials']} layers={meta['layer_map']} "
        f"errors={len(meta['errors'])} -> {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
