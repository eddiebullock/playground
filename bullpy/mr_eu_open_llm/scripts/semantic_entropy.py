"""
Semantic entropy over EU-Emotions label embeddings (Stage 1 ambiguity index).

Primary index (protocol v2, updated): spread of free text over base emotions,
after mapping through a fine label pool (27 minus neutral) with rich label prompts,
then collapsing intensity variants (e.g. happy + happy low intensity -> happy).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import (
    ENTROPY_COLLAPSE_INTENSITY,
    ENTROPY_EXCLUDE_LABELS,
    ENTROPY_LOG_BASE,
    ENTROPY_TEMPERATURE,
    ENTROPY_USE_RICH_LABEL_EMBEDDINGS,
    EMBEDDING_MODEL,
    LOCAL_CACHE_DIR,
    PROJECT_ROOT,
)
from scripts.eu_emotion_definitions import embedding_text_for_label

LOW_INTENSITY_SUFFIX = " low intensity"


def _model_cache_id(model_name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name)
    return safe


def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


_ST_MODEL_CACHE: Dict[str, Any] = {}


def _load_sentence_transformer(model_name: str):
    if model_name in _ST_MODEL_CACHE:
        return _ST_MODEL_CACHE[model_name]
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for semantic entropy. "
            "Install via: pip install sentence-transformers"
        ) from e
    _ST_MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _ST_MODEL_CACHE[model_name]


def base_emotion(label: str) -> str:
    s = str(label).strip()
    low = s.casefold()
    if low.endswith(LOW_INTENSITY_SUFFIX):
        return s[: -len(LOW_INTENSITY_SUFFIX)].strip()
    return s


def prepare_entropy_label_pool(
    emotion_pool: Sequence[str],
    exclude: Sequence[str] = ENTROPY_EXCLUDE_LABELS,
) -> List[str]:
    excluded = {str(x).casefold() for x in exclude}
    return sorted(
        [str(l).strip() for l in emotion_pool if str(l).strip() and str(l).casefold() not in excluded],
        key=str.casefold,
    )


def rich_label_embedding_text(label: str) -> str:
    return embedding_text_for_label(label, rich=ENTROPY_USE_RICH_LABEL_EMBEDDINGS)


def _definitions_cache_tag() -> str:
    from scripts.eu_emotion_definitions import definitions_path

    p = definitions_path()
    if not p.is_file():
        return "nodef"
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def _embedding_texts_for_labels(labels: Sequence[str], *, rich_prompts: bool) -> List[str]:
    if rich_prompts:
        return [embedding_text_for_label(l, rich=True) for l in labels]
    return [str(l) for l in labels]


def load_or_compute_label_embeddings(
    labels: Sequence[str],
    model_name: str = EMBEDDING_MODEL,
    cache_dir: Optional[Path] = None,
    *,
    rich_prompts: bool = ENTROPY_USE_RICH_LABEL_EMBEDDINGS,
) -> np.ndarray:
    """Return L2-normalized embeddings E of shape (n_labels, d), cached on disk."""
    labels_list = [str(x) for x in labels]
    texts = _embedding_texts_for_labels(labels_list, rich_prompts=rich_prompts)
    rich_tag = f"richdef_{_definitions_cache_tag()}" if rich_prompts else "plain"
    label_key = hashlib.sha256("|".join(texts).encode("utf-8")).hexdigest()[:16]
    cache_dir = cache_dir or (LOCAL_CACHE_DIR if LOCAL_CACHE_DIR else PROJECT_ROOT / "data" / "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = (
        cache_dir
        / f"label_embeddings_{_model_cache_id(model_name)}_{rich_tag}_{label_key}.npy"
    )

    if cache_path.exists():
        arr = np.load(cache_path)
        if arr.shape[0] == len(labels_list):
            return arr.astype(np.float32)

    model = _load_sentence_transformer(model_name)
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    emb = _l2_normalize(np.asarray(emb, dtype=np.float32))
    np.save(cache_path, emb)
    return emb


def embed_text(text: str, model_name: str = EMBEDDING_MODEL, model=None) -> np.ndarray:
    if model is None:
        model = _load_sentence_transformer(model_name)
    emb = model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]
    return _l2_normalize(np.asarray(emb, dtype=np.float32))


def cosine_probs(
    text_emb: np.ndarray,
    label_embs: np.ndarray,
    temperature: float = ENTROPY_TEMPERATURE,
) -> np.ndarray:
    """Softmax over cosine similarities with temperature tau."""
    text_emb = _l2_normalize(np.asarray(text_emb, dtype=np.float32).reshape(1, -1))[0]
    sims = label_embs @ text_emb
    tau = max(float(temperature), 1e-8)
    logits = sims / tau
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    probs = exp / np.sum(exp)
    return probs.astype(np.float64)


def semantic_entropy(probs: np.ndarray, log_base: str = ENTROPY_LOG_BASE) -> float:
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, 1e-15, 1.0)
    p = p / p.sum()
    if log_base in ("e", "ln", "natural"):
        log_fn = np.log
    elif log_base == "2":
        log_fn = np.log2
    else:
        raise ValueError(f"Unsupported ENTROPY_LOG_BASE: {log_base}")
    return float(-np.sum(p * log_fn(p)))


def collapse_probs_to_base(
    probs: np.ndarray,
    labels: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    """Sum fine-label probabilities into base-emotion bins (intensity collapsed)."""
    buckets: Dict[str, float] = {}
    order: List[str] = []
    for lbl, p in zip(labels, probs):
        base = base_emotion(lbl)
        key = base.casefold()
        if key not in buckets:
            buckets[key] = 0.0
            order.append(base)
        buckets[key] += float(p)
    base_labels = sorted(order, key=str.casefold)
    out = np.array([buckets[b.casefold()] for b in base_labels], dtype=np.float64)
    out = out / max(out.sum(), 1e-15)
    return out, base_labels


def _label_index(labels: Sequence[str], target: str) -> Optional[int]:
    t = str(target).casefold()
    for i, lbl in enumerate(labels):
        if str(lbl).casefold() == t:
            return i
    return None


def correct_label_metrics(
    probs: np.ndarray,
    labels: Sequence[str],
    true_label: Optional[str],
) -> Dict[str, Optional[float]]:
    """
    How aligned free text is with the correct EU label (when that label is in the pool).

    - p_correct: softmax mass on the true fine label
    - margin_correct: p_correct minus the largest wrong-label probability
    """
    if true_label is None or not str(true_label).strip():
        return {"p_correct": None, "margin_correct": None, "correct_in_entropy_pool": False}
    idx = _label_index(labels, true_label)
    if idx is None:
        return {"p_correct": None, "margin_correct": None, "correct_in_entropy_pool": False}
    p = np.asarray(probs, dtype=np.float64)
    p_correct = float(p[idx])
    wrong = np.delete(p, idx)
    margin = float(p_correct - wrong.max()) if wrong.size else p_correct
    return {
        "p_correct": p_correct,
        "margin_correct": margin,
        "correct_in_entropy_pool": True,
    }


def top_k_labels(
    probs: np.ndarray,
    labels: Sequence[str],
    k: int = 5,
) -> List[Tuple[str, float]]:
    order = np.argsort(-probs)[:k]
    return [(str(labels[i]), float(probs[i])) for i in order]


def compute_entropy_bundle(
    free_text: str,
    labels: Sequence[str],
    *,
    true_label: Optional[str] = None,
    label_embeddings: Optional[np.ndarray] = None,
    model_name: str = EMBEDDING_MODEL,
    temperature: float = ENTROPY_TEMPERATURE,
    log_base: str = ENTROPY_LOG_BASE,
    cache_dir: Optional[Path] = None,
    rich_prompts: bool = ENTROPY_USE_RICH_LABEL_EMBEDDINGS,
    collapse_intensity: bool = ENTROPY_COLLAPSE_INTENSITY,
    st_model=None,
) -> Dict[str, Any]:
    """Full Stage 1 entropy outputs for one trial."""
    labels_list = list(labels)
    empty: Dict[str, Any] = {
        "semantic_entropy": float("nan"),
        "semantic_entropy_fine": float("nan"),
        "semantic_entropy_base": float("nan"),
        "label_probs": None,
        "base_label_probs": None,
        "base_labels": None,
        "top_labels": [],
        "p_correct": None,
        "margin_correct": None,
        "correct_in_entropy_pool": False,
        "n_entropy_labels": len(labels_list),
    }
    if not free_text or not str(free_text).strip() or not labels_list:
        return empty

    if label_embeddings is None:
        label_embeddings = load_or_compute_label_embeddings(
            labels_list, model_name, cache_dir, rich_prompts=rich_prompts
        )

    if st_model is None:
        st_model = _load_sentence_transformer(model_name)
    text_emb = embed_text(free_text, model_name=model_name, model=st_model)
    probs = cosine_probs(text_emb, label_embeddings, temperature=temperature)
    h_fine = semantic_entropy(probs, log_base=log_base)

    if collapse_intensity:
        base_probs, base_labels = collapse_probs_to_base(probs, labels_list)
        h_base = semantic_entropy(base_probs, log_base=log_base)
        h_primary = h_base
    else:
        base_probs, base_labels = collapse_probs_to_base(probs, labels_list)
        h_base = semantic_entropy(base_probs, log_base=log_base)
        h_primary = h_fine

    correct = correct_label_metrics(probs, labels_list, true_label)
    return {
        "semantic_entropy": h_primary,
        "semantic_entropy_fine": h_fine,
        "semantic_entropy_base": h_base,
        "label_probs": probs.tolist(),
        "base_label_probs": base_probs.tolist(),
        "base_labels": base_labels,
        "top_labels": [[lbl, p] for lbl, p in top_k_labels(probs, labels_list, k=5)],
        "p_correct": correct["p_correct"],
        "margin_correct": correct["margin_correct"],
        "correct_in_entropy_pool": correct["correct_in_entropy_pool"],
        "n_entropy_labels": len(labels_list),
    }


def compute_semantic_entropy_for_response(
    free_text: str,
    labels: Sequence[str],
    label_embeddings: Optional[np.ndarray] = None,
    model_name: str = EMBEDDING_MODEL,
    temperature: float = ENTROPY_TEMPERATURE,
    log_base: str = ENTROPY_LOG_BASE,
    cache_dir: Optional[Path] = None,
    true_label: Optional[str] = None,
) -> Tuple[float, np.ndarray, List[Tuple[str, float]]]:
    bundle = compute_entropy_bundle(
        free_text,
        labels,
        true_label=true_label,
        label_embeddings=label_embeddings,
        model_name=model_name,
        temperature=temperature,
        log_base=log_base,
        cache_dir=cache_dir,
    )
    probs = np.asarray(bundle["label_probs"], dtype=np.float64)
    tops = [(lbl, p) for lbl, p in bundle["top_labels"]]
    return float(bundle["semantic_entropy"]), probs, tops


def strip_boilerplate_response(text: str) -> str:
    """Light cleanup before embedding (full string by default)."""
    if not text:
        return ""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    drop_prefixes = ("you are ", "describe what mental", "options:", "respond with:")
    kept = []
    for ln in lines:
        low = ln.lower()
        if any(low.startswith(p) for p in drop_prefixes):
            continue
        kept.append(ln)
    return "\n".join(kept) if kept else text.strip()
