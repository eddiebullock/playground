"""Resume a local vLLM generation chunk. Prints counts, never prompt text.

The weights must already be on disk. This module does not download them and
does not call a hosted API.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from scog.a1.parse import parse_completion


def _key(record: dict) -> tuple:
    return (
        record["model"],
        record["persona_id"],
        int(record["condition"]),
        int(record["seed"]),
    )


def load_done(path: Path) -> set[tuple]:
    done: set[tuple] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path.name}:{lineno} is not valid JSON") from exc
            done.add(_key(record))
    return done


def pending(manifest: list[dict], done: set[tuple], model: str) -> list[dict]:
    out = []
    for record in manifest:
        stamped = dict(record)
        stamped["model"] = model
        if _key(stamped) not in done:
            out.append(stamped)
    return out


def summarise(path: Path) -> dict:
    counts = {"complete": 0, "refusal": 0, "unparseable": 0, "n": 0}
    if not path.exists():
        return counts
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            status = record.get("status", "unparseable")
            if status not in counts:
                status = "unparseable"
            counts[status] += 1
            counts["n"] += 1
    return counts


def main() -> None:
    model_path = Path(os.environ["SCOG_MODEL_PATH"])
    manifest_path = Path(os.environ["SCOG_MANIFEST"])
    out = Path(os.environ["SCOG_JSONL"])
    limit = int(os.environ.get("SCOG_MAX_NEW", "40"))
    if not model_path.is_dir():
        raise RuntimeError(f"local model directory not found: {model_path.name}")
    if not manifest_path.is_file():
        raise RuntimeError("prompt manifest is missing")
    manifest = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not manifest:
        raise RuntimeError("prompt manifest is empty")
    model_name = model_path.name
    todo = pending(manifest, load_done(out), model_name)[:limit]
    if not todo:
        print(json.dumps({"generated": 0, **summarise(out)}, sort_keys=True))
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompts = []
    for record in todo:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": record["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(rendered)
    engine = LLM(
        model=str(model_path),
        dtype="bfloat16",
        max_model_len=4096,
        trust_remote_code=True,
    )
    params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=800, seed=int(todo[0]["seed"]))
    outputs = engine.generate(prompts, params)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        for record, output in zip(todo, outputs):
            text = output.outputs[0].text
            parsed = parse_completion(text)
            stored = {
                "model": model_name,
                "persona_id": record["persona_id"],
                "group": record["group"],
                "condition": int(record["condition"]),
                "seed": int(record["seed"]),
                "status": parsed["status"],
                "n_parsed": parsed["n_parsed"],
                "answers": parsed["answers"],
                "text": text,
            }
            handle.write(json.dumps(stored, ensure_ascii=True) + "\n")
            handle.flush()
    print(json.dumps({"generated": len(todo), **summarise(out)}, sort_keys=True))


if __name__ == "__main__":
    main()
