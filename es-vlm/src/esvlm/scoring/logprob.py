"""Option log-probability scoring (summed and length-normalised).

Generation is a fallback for API models only. Open-weight evals score
options sequentially (Qwen3-VL OOMs when forced-choice is batched with
video).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from esvlm.types import MCQItem, ScoreRow


def score_from_logits(
    logits: np.ndarray,
    option_token_ids: Sequence[Sequence[int]],
) -> list[float]:
    """Manual, model-free scorer used by tests and as the inner kernel.

    ``logits`` has shape [prompt_len + max_option_len, vocab]. For each
    option we take log-softmax at the positions that generate that option's
    tokens, sum, and divide by token count.
    """

    if logits.ndim != 2:
        raise ValueError(f"expected [T, V] logits, got {logits.shape}")
    # numerically stable log-softmax
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    scores: list[float] = []
    for ids in option_token_ids:
        if not ids:
            scores.append(float("-inf"))
            continue
        total = 0.0
        for t, tok in enumerate(ids):
            if t >= log_probs.shape[0]:
                raise ValueError("option longer than logits sequence")
            if tok < 0 or tok >= log_probs.shape[1]:
                raise ValueError(f"token id {tok} outside vocab {log_probs.shape[1]}")
            total += float(log_probs[t, tok])
        scores.append(total / len(ids))
    return scores


def pick_pred(logprobs: Sequence[float]) -> int:
    return int(np.argmax(np.asarray(logprobs, dtype=np.float64)))


def option_logprobs(
    model: Any,
    processor: Any,
    item: MCQItem,
    *,
    device: Any,
    images: list[Any] | None = None,
) -> tuple[list[float], int]:
    """Score each option by teacher-forced log-prob. Sequential, not batched."""

    import torch
    import torch.nn.functional as F

    tokenizer = getattr(processor, "tokenizer", processor)
    scores: list[float] = []
    n_tokens = 0
    for option in item.options:
        text_prompt = _join_prompt(item.prompt_text, option)
        inputs = _encode(processor, text_prompt, images)
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        opt_ids = tokenizer(option, add_special_tokens=False)["input_ids"]
        if hasattr(opt_ids, "tolist"):
            opt_ids = opt_ids.tolist()
        if opt_ids and isinstance(opt_ids[0], list):
            opt_ids = opt_ids[0]
        n_tokens += len(opt_ids)
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits[0]
        # last len(opt_ids) positions predict the option tokens
        if len(opt_ids) == 0:
            scores.append(float("-inf"))
            continue
        take = logits[-(len(opt_ids)) :, :]
        logp = F.log_softmax(take.float(), dim=-1)
        idx = torch.tensor(opt_ids, device=logp.device)
        gathered = logp.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        scores.append(float(gathered.mean().cpu()))
    return scores, n_tokens


def score_item(
    model: Any,
    processor: Any,
    item: MCQItem,
    *,
    device: Any,
    images: list[Any] | None,
    model_key: str,
    revision: str,
    template_id: str,
    condition: str,
    vector: str,
    layer: int | None,
    alpha: float,
    mode: str,
    frames: int,
    seed: int,
) -> ScoreRow:
    logprobs, tokens = option_logprobs(
        model, processor, item, device=device, images=images
    )
    pred = pick_pred(logprobs)
    return ScoreRow(
        model=model_key,
        revision=revision,
        task=item.task,
        item_id=item.item_id,
        template_id=template_id,
        condition=condition,
        vector=vector,
        layer=layer,
        alpha=alpha,
        mode=mode,
        frames=frames,
        option_logprobs=tuple(logprobs),
        pred=pred,
        correct=pred == item.correct_idx,
        seed=seed,
        tokens=tokens,
    )


def _join_prompt(prompt: str, option: str) -> str:
    if prompt.endswith("\n"):
        return f"{prompt}{option}"
    return f"{prompt}\n{option}"


def _encode(processor: Any, text: str, images: list[Any] | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"return_tensors": "pt"}
    if images:
        try:
            return processor(text=text, images=images, **kwargs)
        except TypeError:
            return processor(text, images=images, **kwargs)
    try:
        return processor(text=text, **kwargs)
    except TypeError:
        return processor(text, **kwargs)
