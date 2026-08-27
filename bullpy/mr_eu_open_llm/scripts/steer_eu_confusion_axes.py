#!/usr/bin/env python3
"""Causal steer along EU confusability / entropy / pair axes (study3).

Intervention: add alpha * unit_axis to layer hidden states during 6AFC sampling.
Primary readout: delta JS(model soft, human response distribution).
Reuse readout: pair-axis steer effect on own pair trials vs other confused pairs.

Requires causal_eu_confusion_axes.py outputs in results/mech/.

Usage (HPC GPU):
  python -m scripts.causal_eu_confusion_axes --model qwen3vl --layer 4
  python -m scripts.steer_eu_confusion_axes --model qwen3vl --layer 4 --smoke
  python -m scripts.steer_eu_confusion_axes --model qwen3vl --layer 4 \\
      --max_items 36 --n_samples 10 --alphas=-2,-1,1,2
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from config import LOCAL_DATA_DIR, LOCAL_RESULTS_DIR, SEED
from scripts.causal_eu_confusion_axes import pair_key, pair_trial_indices

PatchMode = Literal["last_token", "all_tokens"]
DEFAULT_MECH = LOCAL_RESULTS_DIR / "mech"


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
    """Add alpha * direction to hooked activations."""

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
        raise FileNotFoundError(f"Missing axis {path}. Run causal_eu_confusion_axes.py first.")
    return _unit(np.load(path))


def soft_from_preds(preds: Sequence[Optional[str]], options: Sequence[str]) -> np.ndarray:
    counts = Counter(p for p in preds if p is not None)
    dist = np.array([counts.get(o, 0) for o in options], dtype=float)
    if dist.sum() <= 0:
        return np.ones(len(options), dtype=float) / len(options)
    return dist / dist.sum()


def human_dist_vector(entry: Dict[str, Any], options: Sequence[str]) -> np.ndarray:
    dist = entry["human_response_distribution"]
    return np.array([float(dist.get(o, 0.0)) for o in options], dtype=float)


def trial_pair_membership(row: Dict[str, Any], label_a: str, label_b: str) -> bool:
    return {row["human_target_label"], row["top_foil_label"]} == {label_a, label_b}


def load_trial_table(
    manifest_path: Path,
    human_path: Path,
    data_root: Path,
    *,
    max_items: Optional[int],
    smoke: bool,
    pair_specs: Sequence[Tuple[str, str]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    from scripts.evaluate import load_trials_from_manifest

    manifest_trials, _ = load_trials_from_manifest(manifest_path, data_root)
    human_lookup = json.loads(human_path.read_text(encoding="utf-8"))["trials"]
    meta_rows = {
        row["trial_id"]: row
        for row in json.loads(
            (LOCAL_DATA_DIR / "human_confusion_meta.json").read_text(encoding="utf-8")
        )["per_item"]
    }

    trials: List[Dict[str, Any]] = []
    for i, t in enumerate(manifest_trials):
        tid = str(t["trial_id"])
        he = human_lookup.get(tid)
        if he is None:
            continue
        options = list(he["human_options"])
        meta = meta_rows.get(tid, {})
        trials.append(
            {
                "trial_id": tid,
                "trial_index": i,
                "stimulus_path": t["stimulus_path"],
                "correct_label": t.get("correct_label") or t.get("label"),
                "options": options,
                "human_entry": he,
                "human_target_label": meta.get("human_target_label", options[0]),
                "top_foil_label": meta.get("top_foil_label", ""),
            }
        )

    if smoke:
        max_items = max_items or 3
    if max_items is not None and len(trials) > int(max_items):
        # Prefer pair-confused trials for mech readout, then fill.
        selected: List[Dict[str, Any]] = []
        seen = set()
        for la, lb in pair_specs:
            for tr in trials:
                meta = meta_rows.get(tr["trial_id"], tr)
                if trial_pair_membership(meta, la, lb) and tr["trial_id"] not in seen:
                    selected.append(tr)
                    seen.add(tr["trial_id"])
        for tr in trials:
            if len(selected) >= int(max_items):
                break
            if tr["trial_id"] not in seen:
                selected.append(tr)
                seen.add(tr["trial_id"])
        trials = selected[: int(max_items)]

    return trials, human_lookup, list(meta_rows.values())


def run_steer(
    *,
    model_key: str,
    layer: int,
    mech_dir: Path,
    manifest_path: Path,
    data_root: Path,
    human_path: Path,
    outdir: Path,
    alphas: List[float],
    patch_modes: List[str],
    axis_names: List[str],
    pair_specs: Sequence[Tuple[str, str]],
    n_samples: int,
    sample_temperature: float,
    max_items: Optional[int],
    max_frames: int,
    seed: int,
    smoke: bool,
) -> Dict[str, Any]:
    import torch
    from transformers import AutoProcessor

    from scripts.activation_forward import find_layer_module, find_layer_module_name, prepare_trial_media
    from scripts.emotion_parse import parse_emotion
    from scripts.evaluate import load_hf_model_for_key, resolve_model_path, resolve_trial_media
    from scripts.model_inference import generate_model_response, seed_generation
    from scripts.prompts import build_4afc_prompt

    trials, _human_lookup, meta_rows = load_trial_table(
        manifest_path,
        human_path,
        data_root,
        max_items=max_items,
        smoke=smoke,
        pair_specs=pair_specs,
    )

    device_s = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device_s == "cuda" else torch.float32
    device = torch.device(device_s)

    if device_s != "cuda":
        proto = {
            "status": "planned_only_no_cuda",
            "model": model_key,
            "layer": layer,
            "n_trials": len(trials),
        }
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / f"steer_protocol_{model_key}_layer{layer}.json").write_text(
            json.dumps(proto, indent=2) + "\n", encoding="utf-8"
        )
        return proto

    axes: Dict[str, np.ndarray] = {}
    for name in axis_names:
        axes[name] = load_axis(mech_dir, name, model_key, layer)

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

    meta_by_tid = {r["trial_id"]: r for r in meta_rows if "trial_id" in r}
    pair_tags: Dict[str, str] = {}
    for la, lb in pair_specs:
        pk = pair_key(la, lb)
        for tr in trials:
            row = meta_by_tid.get(tr["trial_id"], tr)
            if trial_pair_membership(row, la, lb):
                pair_tags[tr["trial_id"]] = pk

    rows: List[Dict[str, Any]] = []

    def generate_once(
        trial: Dict[str, Any],
        *,
        steerer: Optional[AxisSteerer],
        tag: str,
        sample_i: int,
    ) -> Optional[str]:
        options = list(trial["options"])
        trial_copy = dict(trial)
        trial_copy["candidate_labels"] = options
        video_path, audio_path, _ = resolve_trial_media(
            trial_copy,
            dataset_key="eu_emotions",
            dataset_root=data_root,
            condition="video_only",
            seed=seed,
        )
        path = Path(trial["stimulus_path"])
        if not path.is_file():
            path = (data_root / path).resolve()
        if video_path is not None and Path(video_path).is_file():
            path = Path(video_path)
        images, images_for_proc, _ = prepare_trial_media(
            path, model_key=model_key, fps=1.0, max_frames=max_frames
        )
        prompt = build_4afc_prompt(options, condition="video_only")
        seed_generation(seed, model_key, f"eu_steer_{trial['trial_id']}_{tag}", sample_i)
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
                images=images,
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
        tid = trial["trial_id"]
        options = list(trial["options"])
        p_human = human_dist_vector(trial["human_entry"], options)
        membership = pair_tags.get(tid, "other")
        print(f"steer {tid}", flush=True)

        for cond_name, axis_label, axis_vec, alpha, pm in conditions:
            steerer = None
            if axis_vec is not None:
                steerer = AxisSteerer(axis_vec, alpha=alpha, patch_mode=pm)  # type: ignore[arg-type]
            preds: List[Optional[str]] = []
            for s_i in range(int(n_samples)):
                preds.append(generate_once(trial, steerer=steerer, tag=cond_name, sample_i=s_i))
            soft = soft_from_preds(preds, options)
            rows.append(
                {
                    "trial_id": tid,
                    "condition": cond_name,
                    "axis": axis_label,
                    "alpha": alpha,
                    "patch_mode": pm,
                    "js_human": _js(soft, p_human),
                    "soft_dist": soft.tolist(),
                    "n_parsed": int(sum(p is not None for p in preds)),
                    "pair_membership": membership,
                    "hook_module": module_name,
                }
            )

    import pandas as pd

    df = pd.DataFrame(rows)
    csv_path = outdir / f"steer_trials_{model_key}_layer{layer}.csv"
    df.to_csv(csv_path, index=False)

    base = df[df["condition"] == "baseline"][["trial_id", "js_human"]].rename(
        columns={"js_human": "js_human_base"}
    )
    merged = df[df["condition"] != "baseline"].merge(base, on="trial_id", how="left")
    merged["delta_js_human"] = merged["js_human"] - merged["js_human_base"]

    summary_rows = []
    for (axis, alpha, pm), g in merged.groupby(["axis", "alpha", "patch_mode"]):
        row: Dict[str, Any] = {
            "axis": axis,
            "alpha": float(alpha),
            "patch_mode": pm,
            "mean_delta_js_human": float(g["delta_js_human"].mean()),
            "n_trials": int(g["trial_id"].nunique()),
        }
        for la, lb in pair_specs:
            pk = pair_key(la, lb)
            own = g[g["pair_membership"] == pk]
            other = g[(g["pair_membership"] != pk) & (g["pair_membership"] != "other")]
            if len(own):
                row[f"own_{pk}_mean_delta_js"] = float(own["delta_js_human"].mean())
            if len(other):
                row[f"reuse_other_pairs_mean_delta_js"] = float(other["delta_js_human"].mean())
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = outdir / f"steer_summary_{model_key}_layer{layer}.csv"
    summary_df.to_csv(summary_csv, index=False)

    result = {
        "status": "ok",
        "model": model_key,
        "layer": layer,
        "hook_module": module_name,
        "n_trials": len(trials),
        "n_samples": n_samples,
        "patch_modes": patch_modes,
        "alphas": alphas,
        "axes": axis_names,
        "pair_specs": [{"label_a": a, "label_b": b} for a, b in pair_specs],
        "csv": str(csv_path),
        "summary_csv": str(summary_csv),
        "readout_note": (
            "Negative mean_delta_js_human = model soft label moved closer to human distribution."
            " Compare own pair vs other pair deltas for reuse."
        ),
    }
    (outdir / f"steer_summary_{model_key}_layer{layer}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--mech_dir", type=Path, default=DEFAULT_MECH)
    ap.add_argument("--manifest", type=Path, default=LOCAL_DATA_DIR / "eu_emotions_full_manifest.json")
    ap.add_argument("--data_root", type=Path, default=LOCAL_DATA_DIR / "eu_emotions")
    ap.add_argument("--human", type=Path, default=LOCAL_DATA_DIR / "eu_emotions_human_entropy.json")
    ap.add_argument("--pairs_json", type=Path, default=LOCAL_DATA_DIR / "human_confused_pairs.json")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_MECH)
    ap.add_argument("--alphas", default="-1,1")
    ap.add_argument("--patch_modes", default="last_token")
    ap.add_argument("--n_samples", type=int, default=5)
    ap.add_argument("--sample_temperature", type=float, default=1.0)
    ap.add_argument("--max_items", type=int, default=None)
    ap.add_argument("--max_frames", type=int, default=4)
    ap.add_argument("--top_pairs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    from scripts.causal_eu_confusion_axes import default_pair_specs

    pair_specs = default_pair_specs(args.pairs_json, top_k=args.top_pairs)
    axis_names = ["confusability", "entropy", "random"]
    for la, lb in pair_specs:
        pk = pair_key(la, lb)
        axis_path = args.mech_dir / f"axis_{pk}_{args.model}_layer{args.layer}.npy"
        if axis_path.exists():
            axis_names.append(pk)

    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    patch_modes = [x.strip() for x in args.patch_modes.split(",") if x.strip()]
    n_samples = 3 if args.smoke and args.n_samples == 5 else args.n_samples

    result = run_steer(
        model_key=args.model,
        layer=args.layer,
        mech_dir=args.mech_dir,
        manifest_path=args.manifest,
        data_root=args.data_root,
        human_path=args.human,
        outdir=args.outdir,
        alphas=alphas,
        patch_modes=patch_modes,
        axis_names=axis_names,
        pair_specs=pair_specs,
        n_samples=n_samples,
        sample_temperature=args.sample_temperature,
        max_items=args.max_items,
        max_frames=args.max_frames,
        seed=args.seed,
        smoke=bool(args.smoke),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
