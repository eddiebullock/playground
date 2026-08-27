"""Load study4 item-level and trial-level tables for the robustness layer."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

STUDY4_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = STUDY4_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Reuse alignment discovery / loaders (do not duplicate schema)
from alignment_analyses import (  # noqa: E402
    DEFAULT_HUMAN,
    discover_model_evals,
    load_model_item_table,
)
from human_item_difficulty import to_long  # noqa: E402

DEFAULT_CARD = STUDY4_ROOT / "data" / "processed" / "card_rmet_item_level.csv"
DEFAULT_A1 = STUDY4_ROOT / "results" / "alignment" / "a1_summary.json"
DEFAULT_A1_CSV = STUDY4_ROOT / "results" / "alignment" / "a1_behavioural_alignment.csv"


def load_human_item_sensitivity(path: Path = DEFAULT_HUMAN) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"item", "trait_sensitivity_coef"}
    if not need.issubset(df.columns):
        raise ValueError(f"Human CSV missing {need - set(df.columns)}")
    return df.sort_values("item").reset_index(drop=True)


def load_a1_perm_summary(path: Path = DEFAULT_A1) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_all_model_item_tables(
    results_model_dir: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    results_model_dir = results_model_dir or (STUDY4_ROOT / "results" / "model")
    paths = discover_model_evals(results_model_dir)
    return {m: load_model_item_table(p) for m, p in paths.items()}


def load_model_eval_json(model: str) -> Tuple[Path, Dict[str, Any]]:
    paths = discover_model_evals(STUDY4_ROOT / "results" / "model")
    if model not in paths:
        raise FileNotFoundError(f"No full eval for {model}")
    p = paths[model]
    return p, json.loads(p.read_text(encoding="utf-8"))


def model_sample_matrix(model: str) -> Tuple[np.ndarray, List[int], Dict[str, Any]]:
    """
    Return (n_items × k) binary correctness matrix from sampled completions,
    item ids, and meta. k is whatever was stored (currently 10 in full evals).
    """
    path, data = load_model_eval_json(model)
    items, rows = [], []
    for t in sorted(data["trials"], key=lambda x: int(x["item"])):
        correct = str(t["correct_label"]).lower()
        preds = t.get("samples", {}).get("predictions") or []
        row = [1 if (p is not None and str(p).lower() == correct) else 0 for p in preds]
        if not row:
            continue
        items.append(int(t["item"]))
        rows.append(row)
    if not rows:
        return np.zeros((0, 0)), [], {"source": str(path), "k": 0}
    # Pad to common k if ragged
    k = max(len(r) for r in rows)
    mat = np.full((len(rows), k), np.nan)
    for i, r in enumerate(rows):
        mat[i, : len(r)] = r
    meta = {
        "source": str(path),
        "k": int(k),
        "n_items": len(items),
        "n_samples_field": data.get("n_samples"),
        "sample_temperature": data.get("sample_temperature"),
    }
    return mat.astype(float), items, meta


def load_human_trials(card_csv: Path = DEFAULT_CARD) -> pd.DataFrame:
    """One row per subject × item: correct, eq_total, VolunteerID, item."""
    wide = pd.read_csv(card_csv)
    long = to_long(wide)
    long = long.dropna(subset=["eq_total", "correct"]).copy()
    long["eq_total"] = pd.to_numeric(long["eq_total"], errors="coerce")
    long = long.dropna(subset=["eq_total"])
    return long


def paired_item_vectors(
    human: pd.DataFrame,
    model_df: pd.DataFrame,
    metric: str = "sample_accuracy",
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return aligned (eq_sensitivity, model_metric) vectors and n."""
    m = human.merge(model_df, on="item", how="inner").sort_values("item")
    if metric not in m.columns:
        metric = "det_correct"
    x = pd.to_numeric(m["trait_sensitivity_coef"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(m[metric], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask], int(mask.sum())
