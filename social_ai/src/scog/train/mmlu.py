"""Score the frozen 200-question MMLU list. Prints accuracy only."""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def score(model_path: str, ids_path: Path, out_path: Path) -> dict:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    payload = json.loads(ids_path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    if len(rows) != 200:
        raise RuntimeError(f"MMLU id list has {len(rows)} rows, expected 200")
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_subject[row["subject"]].append(row)

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("MMLU scoring found no GPU")
    model.to(device)
    model.eval()
    correct = 0
    scored = 0
    with torch.no_grad():
        for subject, subject_rows in by_subject.items():
            dataset = load_dataset("cais/mmlu", subject, split="test")
            for row in subject_rows:
                item = dataset[int(row["index"])]
                choices = list(item["choices"])
                if len(choices) != 4:
                    raise RuntimeError(f"MMLU item {subject} {row['index']} does not have 4 choices")
                prompt = item["question"].strip() + "\n" + "\n".join(
                    f"{letter}. {choice}" for letter, choice in zip("ABCD", choices)
                ) + "\nAnswer:"
                scores = []
                prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
                for letter in "ABCD":
                    cont = tokenizer(f" {letter}", add_special_tokens=False, truncation=False)["input_ids"]
                    if not cont:
                        raise RuntimeError("tokenizer returned an empty continuation")
                    ids = prompt_ids + cont
                    tensor = torch.tensor([ids], dtype=torch.long, device=device)
                    logits = model(tensor).logits[0]
                    total = 0.0
                    for offset, token in enumerate(cont):
                        position = len(prompt_ids) - 1 + offset
                        total += torch.log_softmax(logits[position], dim=-1)[token].item()
                    scores.append(total)
                predicted = int(max(range(4), key=scores.__getitem__))
                answer = int(item["answer"])
                correct += int(predicted == answer)
                scored += 1
    if scored != 200:
        raise RuntimeError(f"scored {scored} MMLU questions, expected 200")
    summary = {"n": scored, "n_correct": correct, "accuracy": correct / scored, "sha256": payload["sha256"]}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"mmlu_n\t{scored}\tmmlu_correct\t{correct}\tmmlu_accuracy\t{summary['accuracy']:.4f}")
    return summary


def main() -> None:
    model = os.environ.get("A2_MODEL", "")
    root = os.environ.get("A2_ROOT", "")
    if not model or not root:
        raise RuntimeError("A2_MODEL and A2_ROOT must be set")
    tag = os.environ.get("A2_MMLU_TAG", "before")
    score(model, Path(root) / "mmlu_ids.json", Path(root) / f"mmlu_{tag}.json")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"mmlu failed: {exc}", file=sys.stderr)
        raise
