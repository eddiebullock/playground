"""
Phase 4 C1 — Steer / patch along CARD diagnosticity vs entropy axes (study4 only).

Wraps parent hook utilities read-only (ActivationPatcher patterns + find_layer_module).
Does not edit parent scripts.

Intervention: ADD alpha * unit_axis to hooked layer hidden states
  - patch_mode=last_token (default; matches current activation extracts)
  - patch_mode=all_tokens (distributed-token control; VLM last-token often weak)

Axes (precomputed by causal_rmet_axes.py):
  - diagnosticity: mean(high_diag) - mean(low_diag)
  - entropy: mean(high_ent) - mean(low_ent)
  - random: unit Gaussian (seeded)

Primary readout: change in JS(model soft, CARD p_eq_low / p_asc).
Reuse readout: effect of diag-axis steer on high-entropy vs high-diag item classes.

Usage (HPC GPU):
  python study4_rmet/scripts/steer_rmet_axes.py --model qwen3vl --layer 4 --smoke
  python study4_rmet/scripts/steer_rmet_axes.py --model qwen3vl --layer 4 \\
      --patch_modes last_token,all_tokens --alphas -2,-1,1,2

CPU / no-GPU: writes intervention protocol JSON and exits 0 with status planned_only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

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

from rmet_prompts import build_rmet_4afc_prompt  # noqa: E402

PatchMode = Literal["last_token", "all_tokens"]
DEFAULT_MECH = STUDY4_ROOT / "results" / "mech"
DEFAULT_STRUCT = STUDY4_ROOT / "results" / "card_structure"
DEFAULT_MANIFEST = STUDY4_ROOT / "data" / "rmet" / "stimuli" / "manifest.json"
DEFAULT_STIM = STUDY4_ROOT / "data" / "rmet" / "stimuli"


def _js(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, float) + eps
    q = np.asarray(q, float) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))))


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


class AxisSteerer:
    """Add alpha * direction to hooked activations (steer, not replace)."""

    def __init__(
        self,
        direction: np.ndarray,
        *,
        alpha: float,
        patch_mode: PatchMode = "last_token",
    ) -> None:
        self.direction = _unit(np.asarray(direction, dtype=np.float32).reshape(-1))
        self.alpha = float(alpha)
        self.patch_mode = patch_mode
        self.hook_calls = 0
        self._handle = None
        self._dir_t = None

    def attach(self, module: Any, device: Any, dtype: Any) -> None:
        import torch

        steerer = self
        steerer._dir_t = torch.tensor(steerer.direction, device=device, dtype=dtype)

        def hook(_mod, _inp, out):
            steerer.hook_calls += 1
            h = out[0] if isinstance(out, tuple) else out
            if not hasattr(h, "shape") or h.ndim < 2:
                return out
            d = steerer._dir_t
            if d is None or d.shape[-1] != h.shape[-1]:
                return out
            delta = steerer.alpha * d
            h = h.clone()
            if steerer.patch_mode == "last_token":
                h[:, -1, :] = h[:, -1, :] + delta
            else:
                h = h + delta.view(1, 1, -1)
            if isinstance(out, tuple):
                return (h,) + out[1:]
            return h

        self._handle = module.register_forward_hook(hook)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def load_axis(mech_dir: Path, kind: str, model: str, layer: int) -> np.ndarray:
    path = mech_dir / f"axis_{kind}_{model}_layer{layer}.npy"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing axis {path}. Run causal_rmet_axes.py first."
        )
    return _unit(np.load(path))


def soft_from_preds(preds: Sequence[Optional[str]], options: Sequence[str]) -> np.ndarray:
    counts = Counter(p for p in preds if p is not None)
    dist = np.array([counts.get(o, 0) for o in options], dtype=float)
    if dist.sum() <= 0:
        return np.ones(len(options), dtype=float) / len(options)
    return dist / dist.sum()


def write_protocol(
    out_path: Path,
    *,
    model: str,
    layer: int,
    alphas: List[float],
    patch_modes: List[str],
    n_samples: int,
    seed: int,
) -> Dict[str, Any]:
    proto = {
        "status": "planned_only",
        "C1_primary": "reuse_steer_dissociation_diag_vs_entropy",
        "model": model,
        "layer": layer,
        "alphas": alphas,
        "patch_modes": patch_modes,
        "axes": ["diagnosticity", "entropy", "random"],
        "controls": [
            "random_unit_direction",
            "shuffled_item_class_axis (precomputed in causal_rmet_axes)",
            "entropy_axis_as_ambiguity_control",
            "all_tokens_vs_last_token (VLM visual-token caveat)",
        ],
        "readouts": {
            "profile_shift": "delta_JS to CARD p_eq_low and p_asc (negative delta = closer)",
            "reuse": (
                "mean |delta_JS| under diagnosticity steer on high_entropy items "
                "vs high_diagnosticity items; shared → generic; dissociable → profile-relevant"
            ),
        },
        "prediction": (
            "If diagnosticity axis is causally profile-relevant, ±alpha should move soft "
            "labels toward low-EQ/ASC CARD targets more than entropy/random axes, and "
            "effects should not fully transfer to high-entropy low-diagnostic items."
        ),
        "n_samples": n_samples,
        "seed": seed,
        "limitations": [
            "Classic RMET web contamination / possible memorization",
            "n=36 items; soft labels need k>=20 preferred",
            "last_token patching may fail for vision-dependent behaviour",
            "open-weight competence moderate; near-chance models caveat harshly",
            "not ToM; alexithymia absent from CARD",
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(proto, indent=2) + "\n", encoding="utf-8")
    return proto


def run_steer(
    *,
    model_key: str,
    layer: int,
    mech_dir: Path,
    structure_dir: Path,
    manifest_path: Path,
    stim_root: Path,
    outdir: Path,
    alphas: List[float],
    patch_modes: List[str],
    n_samples: int,
    sample_temperature: float,
    max_items: Optional[int],
    seed: int,
    smoke: bool,
) -> Dict[str, Any]:
    import torch
    from PIL import Image
    from transformers import AutoProcessor

    from scripts.activation_forward import find_layer_module, find_layer_module_name
    from scripts.emotion_parse import parse_emotion
    from scripts.evaluate import load_hf_model_for_key, resolve_model_path
    from scripts.model_inference import generate_model_response, seed_generation
    from scripts.multi_frame import prepare_images_for_model

    choice = json.loads(
        (structure_dir / "choice_distributions.json").read_text(encoding="utf-8")
    )
    classes = json.loads(
        (structure_dir / "item_classes_preregistered.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    trials = list(manifest["trials"])
    if smoke:
        max_items = max_items or 3
    if max_items is not None:
        trials = trials[: int(max_items)]

    device_s = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device_s == "cuda" else torch.float32
    device = torch.device(device_s)

    protocol_path = outdir / f"steer_protocol_{model_key}_layer{layer}.json"
    write_protocol(
        protocol_path,
        model=model_key,
        layer=layer,
        alphas=alphas,
        patch_modes=patch_modes,
        n_samples=n_samples,
        seed=seed,
    )

    if device_s != "cuda":
        result = {
            "status": "planned_only_no_cuda",
            "protocol": str(protocol_path),
            "note": "GPU required for steer/patch execution; protocol written.",
        }
        (outdir / f"steer_summary_{model_key}_layer{layer}.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        return result

    axis_diag = load_axis(mech_dir, "diagnosticity", model_key, layer)
    axis_ent = load_axis(mech_dir, "entropy", model_key, layer)
    rng = np.random.default_rng(seed)
    axis_rand = _unit(rng.normal(size=axis_diag.shape[0]).astype(np.float32))

    axes = {
        "diagnosticity": axis_diag,
        "entropy": axis_ent,
        "random": axis_rand,
    }

    model_path = resolve_model_path(model_key)
    if model_key != "gemma4":
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    else:
        try:
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        except Exception:
            from transformers import Gemma4Processor  # type: ignore

            processor = Gemma4Processor.from_pretrained(model_path)

    model = load_hf_model_for_key(model_key, model_path, device_s, dtype)
    model.eval()

    module = find_layer_module(model, layer)
    if module is None:
        raise RuntimeError(f"Could not resolve layer module for {model_key} L{layer}")
    module_name = find_layer_module_name(model, layer)
    tokenizer = getattr(processor, "tokenizer", None)
    pipe_cache: Dict[str, Any] = {}

    high_diag = set(classes["trait_diagnosticity"]["high"])
    high_ent = set(classes["human_entropy"]["high"])

    rows: List[Dict[str, Any]] = []

    def generate_once(
        trial: Dict[str, Any],
        *,
        steerer: Optional[AxisSteerer],
        tag: str,
        sample_i: int,
    ) -> Optional[str]:
        item = int(trial["item"])
        options = list(trial["options"])
        img_rel = trial["image"]
        img_path = stim_root / Path(img_rel).name
        if not img_path.exists():
            img_path = STUDY4_ROOT / img_rel
        image = Image.open(img_path).convert("RGB")
        images_for_proc, _ = prepare_images_for_model(
            model_key, [image], enforce_multi_frame=False
        )
        prompt = build_rmet_4afc_prompt(options)
        seed_generation(seed, model_key, f"rmet_steer_{item:02d}_{tag}", sample_i)
        if steerer is not None:
            steerer.hook_calls = 0
            steerer.attach(module, device, dtype)
        try:
            text = generate_model_response(
                model_key=model_key,
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                model_path=model_path,
                prompt=prompt,
                images=[image],
                images_for_processor=images_for_proc,
                device=device_s,
                dtype=dtype,
                temperature=float(sample_temperature),
                max_new_tokens=128,
                pipe_cache=pipe_cache,
                condition="video_only",
                prefer_loaded_model=True,
            )
        finally:
            if steerer is not None:
                steerer.remove()
        pred, _ = parse_emotion(text, options)
        return pred

    # (condition_name, axis_label, axis_vec|None, alpha, patch_mode)
    conditions: List[Tuple[str, str, Optional[np.ndarray], float, str]] = [
        ("baseline", "baseline", None, 0.0, "none"),
    ]
    for pm in patch_modes:
        for axis_name, axis_vec in axes.items():
            for a in alphas:
                if abs(a) < 1e-12:
                    continue
                conditions.append(
                    (f"{axis_name}_a{a:+g}_{pm}", axis_name, axis_vec, a, pm)
                )

    for trial in trials:
        item = int(trial["item"])
        options = list(trial["options"])
        human = choice["items"][str(item)]
        p_low = np.asarray(human["p_eq_low"], float)
        p_asc = np.asarray(human["p_asc"], float)
        item_class = {
            "high_diagnosticity": item in high_diag,
            "high_entropy": item in high_ent,
        }
        print(f"steer item {item}", flush=True)

        for cond_name, axis_label, axis_vec, alpha, pm in conditions:
            steerer = None
            if axis_vec is not None:
                steerer = AxisSteerer(axis_vec, alpha=alpha, patch_mode=pm)  # type: ignore[arg-type]
            preds: List[Optional[str]] = []
            for s_i in range(int(n_samples)):
                preds.append(
                    generate_once(trial, steerer=steerer, tag=cond_name, sample_i=s_i)
                )
            soft = soft_from_preds(preds, options)
            rows.append(
                {
                    "item": item,
                    "condition": cond_name,
                    "axis": axis_label,
                    "alpha": alpha,
                    "patch_mode": pm,
                    "js_eq_low": _js(soft, p_low),
                    "js_asc": _js(soft, p_asc),
                    "soft_dist": soft.tolist(),
                    "n_parsed": int(sum(p is not None for p in preds)),
                    "hook_module": module_name,
                    **item_class,
                }
            )

    import pandas as pd

    df = pd.DataFrame(rows)
    csv_path = outdir / f"steer_trials_{model_key}_layer{layer}.csv"
    df.to_csv(csv_path, index=False)

    # Aggregate delta JS vs baseline
    base = df[df["condition"] == "baseline"][["item", "js_eq_low", "js_asc"]].rename(
        columns={"js_eq_low": "js_eq_low_base", "js_asc": "js_asc_base"}
    )
    merged = df[df["condition"] != "baseline"].merge(base, on="item", how="left")
    merged["delta_js_eq_low"] = merged["js_eq_low"] - merged["js_eq_low_base"]
    merged["delta_js_asc"] = merged["js_asc"] - merged["js_asc_base"]

    summary_rows = []
    for (axis, alpha, pm), g in merged.groupby(["axis", "alpha", "patch_mode"]):
        # strip axis name pollution from condition parsing
        summary_rows.append(
            {
                "axis": axis,
                "alpha": float(alpha),
                "patch_mode": pm,
                "mean_delta_js_eq_low": float(g["delta_js_eq_low"].mean()),
                "mean_delta_js_asc": float(g["delta_js_asc"].mean()),
                "reuse_mean_delta_js_eq_low_high_entropy": float(
                    g.loc[g["high_entropy"], "delta_js_eq_low"].mean()
                )
                if g["high_entropy"].any()
                else float("nan"),
                "own_mean_delta_js_eq_low_high_diag": float(
                    g.loc[g["high_diagnosticity"], "delta_js_eq_low"].mean()
                )
                if g["high_diagnosticity"].any()
                else float("nan"),
                "n_items": int(g["item"].nunique()),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = outdir / f"steer_summary_{model_key}_layer{layer}.csv"
    summary_df.to_csv(summary_csv, index=False)

    result = {
        "status": "ok",
        "model": model_key,
        "layer": layer,
        "hook_module": module_name,
        "n_items": len(trials),
        "n_samples": n_samples,
        "patch_modes": patch_modes,
        "alphas": alphas,
        "trials_csv": str(csv_path),
        "summary_csv": str(summary_csv),
        "protocol": str(protocol_path),
        "aggregates": summary_rows,
        "interpretation_note": (
            "Negative mean_delta_js_* = closer to CARD profile target after steer. "
            "Compare diagnosticity vs entropy vs random; compare reuse on high_entropy "
            "vs own high_diag. last_token vs all_tokens documents VLM patch validity."
        ),
    }
    (outdir / f"steer_summary_{model_key}_layer{layer}.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen3vl")
    ap.add_argument("--layer", type=int, default=4, help="Peak layer (qwen M1 default 4)")
    ap.add_argument("--mech_dir", type=Path, default=DEFAULT_MECH)
    ap.add_argument("--structure_dir", type=Path, default=DEFAULT_STRUCT)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--stim_root", type=Path, default=DEFAULT_STIM)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_MECH)
    ap.add_argument(
        "--alphas",
        default="-1,1",
        help="Comma-separated steer strengths. Prefer --alphas=-1,1 (equals form) so negatives are not parsed as flags.",
    )
    ap.add_argument("--patch_modes", default="last_token,all_tokens")
    ap.add_argument("--n_samples", type=int, default=5)
    ap.add_argument("--sample_temperature", type=float, default=0.7)
    ap.add_argument("--max_items", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="3 items, fewer samples")
    ap.add_argument(
        "--protocol_only",
        action="store_true",
        help="Write intervention protocol JSON without loading the model",
    )
    args = ap.parse_args(argv)

    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    patch_modes = [x.strip() for x in args.patch_modes.split(",") if x.strip()]
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.protocol_only:
        path = args.outdir / f"steer_protocol_{args.model}_layer{args.layer}.json"
        write_protocol(
            path,
            model=args.model,
            layer=args.layer,
            alphas=alphas,
            patch_modes=patch_modes,
            n_samples=args.n_samples,
            seed=args.seed,
        )
        print(json.dumps({"status": "protocol_only", "path": str(path)}, indent=2))
        return 0

    n_samples = 3 if args.smoke and args.n_samples == 5 else args.n_samples
    result = run_steer(
        model_key=args.model,
        layer=args.layer,
        mech_dir=args.mech_dir,
        structure_dir=args.structure_dir,
        manifest_path=args.manifest,
        stim_root=args.stim_root,
        outdir=args.outdir,
        alphas=alphas,
        patch_modes=patch_modes,
        n_samples=n_samples,
        sample_temperature=args.sample_temperature,
        max_items=args.max_items,
        seed=args.seed,
        smoke=bool(args.smoke),
    )
    print(json.dumps({k: result[k] for k in ("status", "model", "layer", "protocol") if k in result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
