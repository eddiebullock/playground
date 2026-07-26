"""
Linear probing on extracted activations (Study 2/3).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from config import LOCAL_RESULTS_DIR, SEED


_LAYER_RE = re.compile(r"layer(\d+)_eu_emotions_seed\d+\.npy$")


def layer_index_from_path(path: Path) -> int:
    m = _LAYER_RE.search(path.name)
    if not m:
        raise ValueError(f"Cannot parse layer index from {path.name}")
    return int(m.group(1))


def list_activation_layers(act_dir: Path) -> List[Tuple[int, Path]]:
    files = sorted(act_dir.glob("layer*_eu_emotions_seed*.npy"), key=lambda p: layer_index_from_path(p))
    if not files:
        raise FileNotFoundError(f"No activation .npy in {act_dir}")
    return [(layer_index_from_path(p), p) for p in files]


def load_trial_ids(act_dir: Path, layer_path: Path) -> List[str]:
    layer_idx = layer_index_from_path(layer_path)
    sidecar = act_dir / f"layer{layer_idx}_trial_ids.json"
    if sidecar.is_file():
        return json.loads(sidecar.read_text(encoding="utf-8"))
    sidecars = list(act_dir.glob("layer*_trial_ids.json"))
    if sidecars:
        return json.loads(sidecars[0].read_text(encoding="utf-8"))
    return []


def load_activation_matrix(act_dir: Path, layer_path: Optional[Path] = None) -> Tuple[np.ndarray, List[str]]:
    layers = list_activation_layers(act_dir)
    path = layer_path if layer_path is not None else layers[0][1]
    trial_ids = load_trial_ids(act_dir, path)
    return np.load(path), trial_ids


def load_labels_from_eval(eval_json: Path, trial_ids: List[str]) -> np.ndarray:
    obj = json.loads(eval_json.read_text(encoding="utf-8"))
    label_by_id = {t["trial_id"]: t["label"] for t in obj.get("trials", [])}
    return np.array([label_by_id.get(tid, "unknown") for tid in trial_ids])


def _effective_n_splits(n_samples: int, y: np.ndarray, n_splits: int = 5) -> Optional[int]:
    """StratifiedKFold needs n_splits <= smallest class count and <= n_samples // 2."""
    if n_samples < 2:
        return None
    _, counts = np.unique(y, return_counts=True)
    min_class = int(counts.min())
    if min_class < 2:
        return None
    return max(2, min(n_splits, n_samples // 2, min_class))


def probe_layer(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = SEED,
    n_splits: int = 5,
    *,
    allow_random_fallback: bool = False,
) -> float:
    eff = _effective_n_splits(len(X), y, n_splits=n_splits)
    if eff is None:
        if not allow_random_fallback or len(X) < 8:
            return float("nan")
        classes, counts = np.unique(y, return_counts=True)
        keep = {c for c, n in zip(classes, counts) if int(n) >= 2}
        mask = np.array([yi in keep for yi in y])
        if int(mask.sum()) < 8:
            return float("nan")
        X = X[mask]
        y = y[mask]
        from sklearn.model_selection import train_test_split

        accs: List[float] = []
        for split_seed in range(seed, seed + 3):
            try:
                idx = np.arange(len(X))
                train_idx, test_idx = train_test_split(
                    idx,
                    test_size=0.25,
                    random_state=split_seed,
                    stratify=y,
                )
            except ValueError:
                continue
            if len(test_idx) < 2:
                continue
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X[train_idx])
            Xte = scaler.transform(X[test_idx])
            clf = LogisticRegression(
                multi_class="multinomial",
                max_iter=2000,
                random_state=seed,
            )
            clf.fit(Xtr, y[train_idx])
            accs.append(float(clf.score(Xte, y[test_idx])))
        return float(np.mean(accs)) if accs else float("nan")
    skf = StratifiedKFold(n_splits=eff, shuffle=True, random_state=seed)
    accs = []
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[train_idx])
        Xte = scaler.transform(X[test_idx])
        clf = LogisticRegression(
            multi_class="multinomial",
            max_iter=2000,
            random_state=seed,
        )
        clf.fit(Xtr, y[train_idx])
        accs.append(float(clf.score(Xte, y[test_idx])))
    return float(np.mean(accs))


def entropy_tertiles(eval_json: Path) -> Tuple[set, set]:
    obj = json.loads(eval_json.read_text(encoding="utf-8"))
    ents = []
    for t in obj.get("trials", []):
        s1 = t.get("stage1") or {}
        e = s1.get("semantic_entropy")
        tid = t.get("trial_id")
        if e is not None and e == e and tid:
            ents.append((tid, float(e)))
    if len(ents) < 3:
        return set(), set()
    vals = sorted(ents, key=lambda x: x[1])
    n = len(vals)
    low_cut = n // 3
    high_cut = 2 * n // 3
    low = {tid for tid, _ in vals[:low_cut]}
    high = {tid for tid, _ in vals[high_cut:]}
    return low, high


def run_probing_sweep(
    activations_dir: Path,
    eval_json: Path,
    output: Path,
    seed: int = SEED,
) -> Dict:
    layers = list_activation_layers(activations_dir)
    trial_ids = load_trial_ids(activations_dir, layers[0][1])
    y = load_labels_from_eval(eval_json, trial_ids)
    low_ids, high_ids = entropy_tertiles(eval_json)

    layer_results: List[Dict] = []
    for layer_idx, path in layers:
        X = np.load(path)
        acc = probe_layer(X, y, seed=seed)
        entry: Dict = {
            "layer_index": layer_idx,
            "cv_accuracy": acc,
            "activation_file": path.name,
        }
        if low_ids:
            mask = np.array([tid in low_ids for tid in trial_ids])
            if mask.sum() >= 5:
                low_acc = probe_layer(X[mask], y[mask], seed=seed, allow_random_fallback=True)
                if low_acc == low_acc:
                    entry["low_ambiguity_accuracy"] = low_acc
        if high_ids:
            mask = np.array([tid in high_ids for tid in trial_ids])
            if mask.sum() >= 5:
                high_acc = probe_layer(X[mask], y[mask], seed=seed, allow_random_fallback=True)
                if high_acc == high_acc:
                    entry["high_ambiguity_accuracy"] = high_acc
        layer_results.append(entry)

    peak = max(layer_results, key=lambda r: r["cv_accuracy"])
    peak_entry = next(r for r in layer_results if r["layer_index"] == peak["layer_index"])
    tertile_summary: Dict = {}
    if low_ids or high_ids:
        tertile_summary = {
            "n_low_ambiguity": len(low_ids),
            "n_high_ambiguity": len(high_ids),
            "peak_layer_low_ambiguity_accuracy": peak_entry.get("low_ambiguity_accuracy"),
            "peak_layer_high_ambiguity_accuracy": peak_entry.get("high_ambiguity_accuracy"),
        }
    results = {
        "activations_dir": str(activations_dir),
        "eval_json": str(eval_json),
        "n_trials": len(y),
        "seed": seed,
        "layers": layer_results,
        "peak_layer": peak["layer_index"],
        "peak_accuracy": peak["cv_accuracy"],
        "overall_accuracy": peak["cv_accuracy"],
        "entropy_tertiles": tertile_summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    peak_path = output.parent / "peak_layer.json"
    peak_path.write_text(
        json.dumps(
            {
                "peak_layer": peak["layer_index"],
                "peak_accuracy": peak["cv_accuracy"],
                "best_accuracy": peak["cv_accuracy"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return results


def run_probing(
    activations_dir: Path,
    eval_json: Path,
    output: Path,
    seed: int = SEED,
    sweep: bool = True,
) -> Dict:
    if sweep:
        return run_probing_sweep(activations_dir, eval_json, output, seed=seed)
    X, trial_ids = load_activation_matrix(activations_dir)
    y = load_labels_from_eval(eval_json, trial_ids)
    acc = probe_layer(X, y, seed=seed)
    low_ids, high_ids = entropy_tertiles(eval_json)
    results = {"overall_accuracy": acc, "n_trials": len(y), "seed": seed}
    if low_ids:
        mask = np.array([tid in low_ids for tid in trial_ids])
        if mask.sum() >= 5:
            low_acc = probe_layer(X[mask], y[mask], seed=seed, allow_random_fallback=True)
            if low_acc == low_acc:
                results["low_ambiguity_accuracy"] = low_acc
    if high_ids:
        mask = np.array([tid in high_ids for tid in trial_ids])
        if mask.sum() >= 5:
            high_acc = probe_layer(X[mask], y[mask], seed=seed, allow_random_fallback=True)
            if high_acc == high_acc:
                results["high_ambiguity_accuracy"] = high_acc
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Train probes on activations (multi-layer sweep).")
    ap.add_argument("--activations_dir", type=Path, required=True)
    ap.add_argument("--eval_json", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=LOCAL_RESULTS_DIR / "probes" / "probes_summary.json")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--single_layer_only", action="store_true", help="Legacy: first layer only.")
    args = ap.parse_args()
    run_probing(
        args.activations_dir,
        args.eval_json,
        args.output,
        seed=args.seed,
        sweep=not args.single_layer_only,
    )


if __name__ == "__main__":
    main()
