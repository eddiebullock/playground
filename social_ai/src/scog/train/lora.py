"""One resumable LoRA chunk. Exit 0 when the epoch budget is done, 3 on checkpoint."""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

EXIT_DONE = 0
EXIT_CHECKPOINT = 3


def _flag() -> dict:
    return {"stop": False}


_STOP = _flag()


def _on_usr1(signum, frame) -> None:
    del signum, frame
    _STOP["stop"] = True


def install_usr1() -> None:
    import signal

    signal.signal(signal.SIGUSR1, _on_usr1)


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise RuntimeError(f"missing training file {path.name}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"training file {path.name} has zero rows")
    return rows


def encode_example(tokenizer, prompt: str, answer: str, max_length: int) -> dict | None:
    prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False, truncation=False)["input_ids"]
    eos = tokenizer.eos_token_id
    if eos is not None:
        answer_ids = answer_ids + [eos]
    input_ids = prompt_ids + answer_ids
    if len(input_ids) > max_length:
        return None
    labels = [-100] * len(prompt_ids) + answer_ids
    if len(labels) != len(input_ids) or sum(1 for token in labels if token != -100) == 0:
        raise RuntimeError("label mask does not cover the answer")
    return {"input_ids": input_ids, "labels": labels}


def capture_rng(order_gen) -> dict:
    import torch

    payload = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "order_gen": order_gen.get_state(),
    }
    if torch.cuda.is_available():
        payload["cuda"] = [state.detach().to("cpu") for state in torch.cuda.get_rng_state_all()]
    return payload


def restore_rng(payload: dict, order_gen) -> None:
    import torch

    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch_cpu"])
    order_gen.set_state(payload["order_gen"])
    if "cuda" in payload and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.to("cuda") for state in payload["cuda"]])


def _batch(tokenizer, features: list[dict], device):
    import torch

    width = max(len(feature["input_ids"]) for feature in features)
    pad = tokenizer.pad_token_id
    input_ids = []
    labels = []
    attention = []
    for feature in features:
        length = len(feature["input_ids"])
        pad_n = width - length
        input_ids.append(feature["input_ids"] + [pad] * pad_n)
        labels.append(feature["labels"] + [-100] * pad_n)
        attention.append([1] * length + [0] * pad_n)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention, dtype=torch.long, device=device),
    }


def _dev_nll(model, tokenizer, rows: list[dict], max_length: int, device) -> dict:
    import torch

    model.eval()
    total = 0.0
    tokens = 0
    used = 0
    skipped = 0
    with torch.no_grad():
        for row in rows:
            encoded = encode_example(tokenizer, row["prompt"], row["answer"], max_length)
            if encoded is None:
                skipped += 1
                continue
            batch = _batch(tokenizer, [encoded], device)
            loss = model(**batch).loss
            n_tokens = int((batch["labels"] != -100).sum().item())
            total += float(loss.item()) * n_tokens
            tokens += n_tokens
            used += 1
    model.train()
    if used < 10 or tokens == 0:
        raise RuntimeError("dev negative log-likelihood used fewer than 10 examples")
    return {"nll": total / tokens, "n_examples": used, "n_tokens": tokens, "n_skipped_long": skipped}


def train_chunk(config: dict) -> int:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    install_usr1()
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    data_name = "exploratory_train.jsonl" if config["split"] == "exploratory" else "confirmatory_train.jsonl"
    rows = load_jsonl(Path(config["data_dir"]) / data_name)
    dev_rows = []
    if config["split"] == "exploratory":
        dev_rows = load_jsonl(Path(config["data_dir"]) / "exploratory_dev.jsonl")[: int(config["dev_nll_n"])]
        if len(dev_rows) < 10:
            raise RuntimeError("dev file has fewer than 10 rows")

    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base = AutoModelForCausalLM.from_pretrained(
        config["model_path"],
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    base.config.use_cache = False
    adapter_dir = run_dir / "adapter"
    if (adapter_dir / "adapter_config.json").is_file():
        model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=True)
    else:
        model = get_peft_model(
            base,
            LoraConfig(
                r=int(config["rank"]),
                lora_alpha=int(config["rank"]) * 2,
                lora_dropout=float(config["lora_dropout"]),
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=list(config["target_modules"]),
            ),
        )
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("LoRA attached no trainable parameters")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("training chunk found no GPU")
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=float(config["lr"]))
    order_gen = torch.Generator(device="cpu")
    order_gen.manual_seed(int(config["seed"]))
    trainer_path = run_dir / "trainer.pt"
    awaiting_nll = False
    step = 0
    if trainer_path.is_file():
        saved = torch.load(trainer_path, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"], order_gen)
        epochs_completed = int(saved["epochs_completed"])
        cursor = int(saved["cursor"])
        perm = list(saved["perm"])
        step = int(saved["step"])
        awaiting_nll = bool(saved.get("awaiting_nll", False))
    else:
        epochs_completed = 0
        cursor = 0
        perm = torch.randperm(len(rows), generator=order_gen).tolist()

    target_epochs = int(config["epochs"])
    micro = int(config["micro_batch"])
    accum = int(config["grad_accum"])
    max_length = int(config["max_length"])
    deadline = time.monotonic() + float(config["max_minutes"]) * 60
    loss_path = run_dir / "loss.jsonl"

    def persist() -> None:
        model.save_pretrained(adapter_dir)
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "epochs_completed": epochs_completed,
                "cursor": cursor,
                "perm": perm,
                "step": step,
                "awaiting_nll": awaiting_nll,
                "rng": capture_rng(order_gen),
            },
            trainer_path,
        )

    def write_nll(epoch: int) -> None:
        path = run_dir / f"dev_nll_epoch{epoch}.json"
        if path.is_file():
            return
        payload = _dev_nll(model, tokenizer, dev_rows, max_length, device)
        payload["epoch"] = epoch
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        print(f"dev_nll\tepoch\t{epoch}\tnll\t{payload['nll']:.6f}\tn\t{payload['n_examples']}")

    while True:
        if _STOP["stop"] or time.monotonic() >= deadline:
            persist()
            print(f"checkpoint\tstep\t{step}\tepochs_completed\t{epochs_completed}")
            return EXIT_CHECKPOINT
        if cursor >= len(perm) or awaiting_nll:
            if not awaiting_nll:
                epochs_completed += 1
                awaiting_nll = True
                persist()
            if config["split"] == "exploratory":
                write_nll(epochs_completed)
            awaiting_nll = False
            if epochs_completed >= target_epochs:
                persist()
                (run_dir / "COMPLETE").write_text("ok\n", encoding="utf-8")
                print(f"complete\tepochs\t{epochs_completed}\tsteps\t{step}")
                return EXIT_DONE
            perm = torch.randperm(len(rows), generator=order_gen).tolist()
            cursor = 0
            persist()
            continue
        optimizer.zero_grad(set_to_none=True)
        losses = []
        consumed = 0
        while consumed < accum and cursor < len(perm):
            batch_rows = []
            while len(batch_rows) < micro and cursor < len(perm):
                encoded = encode_example(
                    tokenizer,
                    rows[perm[cursor]]["prompt"],
                    rows[perm[cursor]]["answer"],
                    max_length,
                )
                cursor += 1
                if encoded is None:
                    continue
                batch_rows.append(encoded)
            if not batch_rows:
                break
            batch = _batch(tokenizer, batch_rows, device)
            loss = model(**batch).loss
            (loss / accum).backward()
            losses.append(float(loss.item()))
            consumed += 1
        if not losses:
            continue
        optimizer.step()
        step += 1
        if step % 20 == 0:
            with loss_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": step, "loss": sum(losses) / len(losses), "epochs_completed": epochs_completed}) + "\n")
            print(f"step\t{step}\tloss\t{sum(losses) / len(losses):.4f}\tepochs_completed\t{epochs_completed}")


def main() -> int:
    required = ["A2_RUN_DIR", "A2_DATA_DIR", "A2_MODEL", "A2_SPLIT", "A2_RANK", "A2_LR", "A2_EPOCHS", "A2_MAX_MINUTES"]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"missing environment: {missing}")
    grid = json.loads((_repo_config()).read_text(encoding="utf-8"))
    config = {
        "run_dir": os.environ["A2_RUN_DIR"],
        "data_dir": os.environ["A2_DATA_DIR"],
        "model_path": os.environ["A2_MODEL"],
        "split": os.environ["A2_SPLIT"],
        "rank": int(os.environ["A2_RANK"]),
        "lr": float(os.environ["A2_LR"]),
        "epochs": int(os.environ["A2_EPOCHS"]),
        "max_minutes": float(os.environ["A2_MAX_MINUTES"]),
        "seed": int(grid["seed"]),
        "max_length": int(grid["max_length"]),
        "micro_batch": int(grid["micro_batch"]),
        "grad_accum": int(grid["grad_accum"]),
        "lora_dropout": float(grid["lora_dropout"]),
        "dev_nll_n": int(grid["dev_nll_n"]),
        "target_modules": grid["target_modules"],
    }
    if config["split"] not in {"exploratory", "confirmatory"}:
        raise RuntimeError("A2_SPLIT must be exploratory or confirmatory")
    return train_chunk(config)


def _repo_config() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "a2_grid.json"


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"train failed: {exc}", file=sys.stderr)
        raise
