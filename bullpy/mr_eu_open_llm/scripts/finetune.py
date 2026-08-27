"""
LoRA fine-tuning on Mindreading (Study 1). Gemma4 multimodal; Qwen2-VL / LLaVA video-only.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import (
    BEST_MODEL_KEY,
    FINETUNE_MODALITY,
    FINETUNE_MODALITY_BY_MODEL,
    FRAME_POLICY,
    LORA_DEFAULT,
    LORA_TARGET_MODULES,
    MODEL_AUDIO_CAPABILITIES,
    MODELS,
    PROTOCOL_VERSION,
    SEED,
    TRAINING_DEFAULTS,
    lora_alpha_for_rank,
)
from scripts.evaluate import load_hf_model_for_key, resolve_model_path
from scripts.activation_forward import resolve_model_device
from scripts.frame_sampling import load_stimulus_as_images
from scripts.model_inference import _build_gemma4_content, generate_model_response
from scripts.multi_frame import prepare_images_for_model
from scripts.prompts import build_finetune_prompt

# setting up the lora config
def setup_lora_config(
    model_key: str,
    r: int,
    dropout: float,
    alpha: Optional[int] = None,
) -> Dict[str, Any]:
    alpha = alpha if alpha is not None else lora_alpha_for_rank(r)
    target = LORA_TARGET_MODULES.get(model_key, LORA_DEFAULT["target_modules"])
    return {
        "r": r,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "target_modules": target,
    }

# which modality condition should we finetune on?
def finetune_modality_for_model(model_key: str) -> str:
    return FINETUNE_MODALITY_BY_MODEL.get(model_key, FINETUNE_MODALITY)

# loading the json file for finetuning data (train and val) into a list of dictionaries that represent each record
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


# finding the video path for the record
def record_video_path(rec: Dict[str, Any], data_root: Path) -> Path:
    if rec.get("video_path"):
        return Path(rec["video_path"])
    if rec.get("stimulus_path"):
        p = Path(rec["stimulus_path"])
        return p if p.is_absolute() else (data_root / p).resolve()
    rel = rec.get("media_paths", [None])[0]
    if rel is None:
        raise ValueError(f"Record missing media path: {rec}")
    return (data_root / rel).resolve()

# finding the audio path for the record
def record_audio_path(rec: Dict[str, Any]) -> Optional[Path]:
    ap = rec.get("audio_path")
    return Path(ap).resolve() if ap else None

# normalizing the label to lowercase and removing extra spaces
def normalize_label(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().casefold())

# checking if the predicted label matches the true label
def labels_match(pred: str, true_label: str) -> bool:
    p = normalize_label(pred)
    t = normalize_label(true_label)
    if not p or not t:
        return False
    if p == t:
        return True
    return t in p or p in t

# dataset class for finetuning
class MindreadingFinetuneDataset(Dataset):
    def __init__(
        self,
        records: List[Dict[str, Any]],
        *,
        data_root: Path,
        model_key: str,
        condition: str,
    ) -> None:
        self.records = records
        self.data_root = data_root
        self.model_key = model_key
        self.condition = condition
        self.prompt = build_finetune_prompt(condition=condition)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        video_path = record_video_path(rec, self.data_root)
        audio_path = record_audio_path(rec)
        return {
            "id": rec.get("id") or rec.get("trial_id"),
            "label": str(rec.get("label", "")),
            "prompt": self.prompt,
            "video_path": video_path,
            "audio_path": audio_path,
        }


def collate_single(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return batch[0]

# load HF processor for the model
def load_processor(model_key: str, model_path: Path):
    from transformers import AutoProcessor

    if model_key != "gemma4":
        return AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    try:
        return AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        from transformers import Gemma4Processor  # type: ignore

        return Gemma4Processor.from_pretrained(model_path)

# resolve the target modules for the LoRA adapter
def resolve_lora_target_modules(model: Any, model_key: str) -> List[str]:
    """Gemma 4 uses Gemma4ClippableLinear wrappers; target inner nn.Linear by full path."""
    import torch.nn as nn

    if model_key != "gemma4":
        return list(LORA_TARGET_MODULES.get(model_key, LORA_DEFAULT["target_modules"]))

    linear_paths = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and any(tok in name for tok in (".self_attn.", ".mlp.", "attn"))
    ]
    if linear_paths:
        return linear_paths

    fallback = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    if not fallback:
        raise RuntimeError(f"No nn.Linear modules found for LoRA on {model_key}")
    return fallback

# wrap model in peramiter efficient fine-tuning (PEFT) adapter
def apply_lora(model: Any, model_key: str, r: int, alpha: int, dropout: float) -> Any:
    from peft import LoraConfig, get_peft_model, TaskType

    target = resolve_lora_target_modules(model, model_key)
    print(f"LoRA targets: {len(target)} modules (e.g. {target[0] if target else 'none'})")

    def _build_cfg(modules: Any) -> LoraConfig:
        return LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

    try:
        return get_peft_model(model, _build_cfg(target))
    except ValueError as exc:
        if model_key != "gemma4" or "not supported" not in str(exc):
            raise
        print("LoRA full-path targets failed; retrying with target_modules='all-linear'")
        return get_peft_model(model, _build_cfg("all-linear"))

# build the training messages for the gemma4 model (user and assistant messages)    
def _gemma_training_messages(
    prompt: str,
    label: str,
    *,
    images_for_processor: Any,
    audio_path: Optional[Path],
    condition: str,
) -> List[Dict[str, Any]]:
    user_content = _build_gemma4_content(
        prompt,
        condition=condition,
        images_for_processor=images_for_processor,
        audio_path=audio_path,
    )
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": label},
    ]

# mask the labels for the single-sample fine-tuning step
def _mask_labels_for_sft(input_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    labels = input_ids.clone()
    if prompt_len > 0:
        labels[:, :prompt_len] = -100
    return labels


def _gemma_processor_inputs(
    processor: Any,
    text: str,
    *,
    images_for_processor: Any,
    audio_path: Optional[Path],
    condition: str,
) -> Any:
    proc_kwargs: Dict[str, Any] = {"text": text, "return_tensors": "pt"}
    if images_for_processor is not None and condition != "audio_only":
        proc_kwargs["images"] = images_for_processor
    if audio_path is not None and condition in {"audio_only", "multimodal"}:
        proc_kwargs["audio"] = str(audio_path)
    try:
        return processor(**proc_kwargs)
    except TypeError:
        proc_kwargs.pop("audio", None)
        return processor(**proc_kwargs)


def _move_inputs_to_device(
    inputs: Dict[str, Any],
    *,
    device: str,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    return {
        k: v.to(device, dtype=dtype if hasattr(v, "is_floating_point") and v.is_floating_point() else None)
        if hasattr(v, "to")
        else v
        for k, v in inputs.items()
    }


def _load_training_images(
    video_path: Path,
    *,
    model_key: str,
    condition: str,
    fps: float,
    max_frames: int,
) -> Tuple[List[Any], Any]:
    images: List[Any] = []
    images_for_processor: Any = None
    if condition != "audio_only" and video_path.exists():
        images, _ = load_stimulus_as_images(video_path, fps=fps, max_frames=max_frames)
        images_for_processor, _ = prepare_images_for_model(
            model_key,
            images,
            enforce_multi_frame=FRAME_POLICY.get("enforce_multi_frame", True),
        )
    return images, images_for_processor


def _vlm_processor_inputs(
    processor: Any,
    model_key: str,
    text: str,
    *,
    images_for_processor: Any,
    images: List[Any],
    model: Any,
) -> Dict[str, Any]:
    try:
        inputs = processor(images=images_for_processor, text=text, return_tensors="pt")
    except Exception:
        try:
            inputs = processor(images=images_for_processor, text=[text], return_tensors="pt")
        except Exception:
            inputs = processor(text=text, images=images_for_processor, return_tensors="pt")
    if inputs is None:
        raise RuntimeError("Processor returned None for fine-tune inputs.")

    if model_key == "llavanext":
        inputs = dict(inputs)
        inputs.pop("image_sizes", None)

    return dict(inputs)


def _qwen_chat_texts(
    processor: Any,
    prompt: str,
    label: str,
    *,
    n_img: int,
) -> Tuple[str, str]:
    content_q: List[Dict[str, Any]] = [{"type": "image"} for _ in range(n_img)]
    content_q.append({"type": "text", "text": prompt})
    user_messages = [{"role": "user", "content": content_q}]
    full_messages = user_messages + [{"role": "assistant", "content": label}]
    try:
        prompt_text = processor.apply_chat_template(
            user_messages, tokenize=False, add_generation_prompt=True
        )
        full_text = processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        img_block = "\n".join(["<image>"] * n_img)
        prompt_text = f"{img_block}\n{prompt}"
        full_text = f"{prompt_text}\n{label}"
    return prompt_text, full_text


def _llava_chat_texts(
    processor: Any,
    prompt: str,
    label: str,
) -> Tuple[str, str]:
    user_messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image"}]}
    ]
    full_messages = user_messages + [{"role": "assistant", "content": label}]
    try:
        prompt_text = processor.apply_chat_template(
            user_messages, tokenize=False, add_generation_prompt=True
        )
        full_text = processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        prompt_text = f"{prompt}\n<image>"
        full_text = f"{prompt_text}\n{label}"
    return prompt_text, full_text


def vlm_sft_step(
    model: Any,
    processor: Any,
    sample: Dict[str, Any],
    *,
    model_key: str,
    condition: str,
    device: str,
    dtype: torch.dtype,
    fps: float,
    max_frames: int,
) -> torch.Tensor:
    """Single-sample SFT step for vision-only VLMs (Qwen2-VL, LLaVA-NeXT)."""
    video_path = sample["video_path"]
    prompt = sample["prompt"]
    label = sample["label"]

    images, images_for_processor = _load_training_images(
        video_path,
        model_key=model_key,
        condition=condition,
        fps=fps,
        max_frames=max_frames,
    )
    n_img = len(images) if isinstance(images, list) else 1
    if model_key == "qwen2vl":
        prompt_text, full_text = _qwen_chat_texts(
            processor, prompt, label, n_img=max(n_img, 1)
        )
    elif model_key == "llavanext":
        prompt_text, full_text = _llava_chat_texts(processor, prompt, label)
    else:
        raise ValueError(f"vlm_sft_step does not support model_key={model_key}")

    prompt_inputs = _vlm_processor_inputs(
        processor,
        model_key,
        prompt_text,
        images_for_processor=images_for_processor,
        images=images,
        model=model,
    )
    inputs = _vlm_processor_inputs(
        processor,
        model_key,
        full_text,
        images_for_processor=images_for_processor,
        images=images,
        model=model,
    )
    prompt_len = int(prompt_inputs["input_ids"].shape[1])

    model_device = resolve_model_device(model, fallback=device)
    model_dtype = getattr(model, "dtype", None) or dtype
    inputs = _move_inputs_to_device(inputs, device=str(model_device), dtype=model_dtype)
    labels = _mask_labels_for_sft(inputs["input_ids"], prompt_len)
    outputs = model(**inputs, labels=labels)
    return outputs.loss


def gemma_sft_step(
    model: Any,
    processor: Any,
    sample: Dict[str, Any],
    *,
    condition: str,
    device: str,
    dtype: torch.dtype,
    fps: float,
    max_frames: int,
) -> torch.Tensor:
    video_path = sample["video_path"]
    audio_path = sample["audio_path"]
    prompt = sample["prompt"]
    label = sample["label"]

    images: List[Any] = []
    images_for_processor: Any = None
    if condition != "audio_only" and video_path.exists():
        images, images_for_processor = _load_training_images(
            video_path,
            model_key="gemma4",
            condition=condition,
            fps=fps,
            max_frames=max_frames,
        )

    if condition in {"audio_only", "multimodal"} and audio_path is not None and not audio_path.exists():
        audio_path = None

    user_messages = [
        {
            "role": "user",
            "content": _build_gemma4_content(
                prompt,
                condition=condition,
                images_for_processor=images_for_processor,
                audio_path=audio_path,
            ),
        }
    ]
    full_messages = _gemma_training_messages(
        prompt,
        label,
        images_for_processor=images_for_processor,
        audio_path=audio_path,
        condition=condition,
    )

    prompt_text = processor.apply_chat_template(
        user_messages, tokenize=False, add_generation_prompt=True
    )
    full_text = processor.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )

    inputs = _gemma_processor_inputs(
        processor,
        full_text,
        images_for_processor=images_for_processor,
        audio_path=audio_path,
        condition=condition,
    )
    prompt_inputs = _gemma_processor_inputs(
        processor,
        prompt_text,
        images_for_processor=images_for_processor,
        audio_path=audio_path,
        condition=condition,
    )
    prompt_len = int(prompt_inputs["input_ids"].shape[1])

    model_device = resolve_model_device(model, fallback=device)
    model_dtype = getattr(model, "dtype", None) or dtype
    inputs = {
        k: v.to(model_device, dtype=model_dtype if hasattr(v, "is_floating_point") and v.is_floating_point() else None)
        if hasattr(v, "to")
        else v
        for k, v in inputs.items()
    }
    labels = _mask_labels_for_sft(inputs["input_ids"], prompt_len)
    outputs = model(**inputs, labels=labels)
    return outputs.loss


@torch.inference_mode()
def evaluate_val_accuracy(
    model: Any,
    processor: Any,
    tokenizer: Any,
    val_records: List[Dict[str, Any]],
    *,
    model_key: str,
    data_root: Path,
    condition: str,
    device: str,
    dtype: torch.dtype,
    fps: float,
    max_frames: int,
    max_eval: Optional[int] = None,
) -> Tuple[float, int, int]:
    correct = 0
    scored = 0
    subset = val_records[: max_eval or len(val_records)]
    prompt = build_finetune_prompt(condition=condition)

    for rec in subset:
        sample = {
            "video_path": record_video_path(rec, data_root),
            "audio_path": record_audio_path(rec),
            "prompt": prompt,
            "label": str(rec.get("label", "")),
        }
        images: List[Any] = []
        images_for_processor: Any = None
        if condition != "audio_only" and sample["video_path"].exists():
            try:
                images, _ = load_stimulus_as_images(sample["video_path"], fps=fps, max_frames=max_frames)
                images_for_processor, _ = prepare_images_for_model(
                    model_key,
                    images,
                    enforce_multi_frame=FRAME_POLICY.get("enforce_multi_frame", True),
                )
            except RuntimeError as exc:
                if "Failed to open video" in str(exc) or "Failed to extract frames" in str(exc):
                    print(f"Skipping unreadable val video: {sample['video_path']}")
                    continue
                raise
        audio_path = sample["audio_path"]
        if audio_path is not None and not audio_path.exists():
            audio_path = None

        try:
            pred = generate_model_response(
                model_key=model_key,
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                model_path=resolve_model_path(model_key),
                prompt=prompt,
                images=images,
                images_for_processor=images_for_processor,
                device=device,
                dtype=dtype,
                temperature=0.0,
                max_new_tokens=32,
                pipe_cache={},
                audio_path=audio_path,
                condition=condition,
            )
        except Exception:
            pred = ""
        scored += 1
        if labels_match(pred, sample["label"]):
            correct += 1

    acc = correct / scored if scored else 0.0
    return acc, correct, scored


def run_finetuning(
    model_key: str,
    train_file: Path,
    val_file: Path,
    output_dir: Path,
    learning_rate: float,
    r: int,
    alpha: int,
    dropout: float,
    *,
    data_root: Path,
    condition: Optional[str] = None,
    seed: int = SEED,
    epochs: Optional[int] = None,
    max_train_samples: Optional[int] = None,
    max_val_eval: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    condition = condition or finetune_modality_for_model(model_key)
    if condition in {"audio_only", "multimodal"} and not MODEL_AUDIO_CAPABILITIES.get(model_key, False):
        raise ValueError(f"Model {model_key} does not support condition={condition}")

    torch.manual_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    lora_cfg = setup_lora_config(model_key, r, dropout, alpha)

    train_records = load_jsonl(train_file)
    val_records = load_jsonl(val_file)
    if max_train_samples is not None:
        train_records = train_records[: int(max_train_samples)]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model_path = resolve_model_path(model_key)
    processor = load_processor(model_key, model_path)
    tokenizer = getattr(processor, "tokenizer", None)

    model = load_hf_model_for_key(model_key, model_path, device=device, dtype=dtype)
    model = apply_lora(model, model_key, r, alpha, dropout)
    if device != "cuda":
        model = model.to(device)
    model.train()

    fps = float(FRAME_POLICY["fps"])
    max_frames = int(FRAME_POLICY["max_frames"])
    epochs = epochs if epochs is not None else int(TRAINING_DEFAULTS["epochs"])
    batch_size = int(TRAINING_DEFAULTS["batch_size"])
    grad_accum = int(TRAINING_DEFAULTS["grad_accum_steps"])

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
    )

    global_step = 0
    train_losses: List[float] = []
    epoch_metrics: List[Dict[str, Any]] = []

    for epoch in range(epochs):
        model.train()
        running = 0.0
        n_steps = 0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_records, desc=f"epoch {epoch + 1}/{epochs}", leave=False)
        for i, rec in enumerate(pbar):
            sample = {
                "video_path": record_video_path(rec, data_root),
                "audio_path": record_audio_path(rec),
                "prompt": build_finetune_prompt(condition=condition),
                "label": str(rec.get("label", "")),
            }
            if model_key == "gemma4":
                try:
                    loss = gemma_sft_step(
                        model,
                        processor,
                        sample,
                        condition=condition,
                        device=device,
                        dtype=dtype,
                        fps=fps,
                        max_frames=max_frames,
                    )
                except RuntimeError as exc:
                    if "Failed to open video" in str(exc) or "Failed to extract frames" in str(exc):
                        print(f"Skipping unreadable video: {sample['video_path']}")
                        continue
                    raise
            elif model_key in {"qwen2vl", "llavanext"}:
                try:
                    loss = vlm_sft_step(
                        model,
                        processor,
                        sample,
                        model_key=model_key,
                        condition=condition,
                        device=device,
                        dtype=dtype,
                        fps=fps,
                        max_frames=max_frames,
                    )
                except RuntimeError as exc:
                    if "Failed to open video" in str(exc) or "Failed to extract frames" in str(exc):
                        print(f"Skipping unreadable video: {sample['video_path']}")
                        continue
                    raise
            else:
                raise NotImplementedError(f"Fine-tuning not implemented for model_key={model_key}")

            loss = loss / grad_accum
            loss.backward()
            running += float(loss.item()) * grad_accum
            n_steps += 1

            if n_steps % grad_accum == 0 or i + 1 == len(train_records):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                train_losses.append(float(loss.item()) * grad_accum)
                if max_steps is not None and global_step >= max_steps:
                    break

        stopped_early = max_steps is not None and global_step >= max_steps

        model.eval()
        val_acc, val_correct, val_scored = evaluate_val_accuracy(
            model,
            processor,
            tokenizer,
            val_records,
            model_key=model_key,
            data_root=data_root,
            condition=condition,
            device=device,
            dtype=dtype,
            fps=fps,
            max_frames=max_frames,
            max_eval=max_val_eval,
        )
        epoch_metrics.append(
            {
                "epoch": epoch + 1,
                "train_loss_mean": running / max(n_steps, 1),
                "val_accuracy": val_acc,
                "val_correct": val_correct,
                "val_scored": val_scored,
            }
        )
        adapter_dir = output_dir / f"checkpoint-epoch{epoch + 1}"
        model.save_pretrained(adapter_dir)
        processor.save_pretrained(adapter_dir)

        if stopped_early:
            break

    final_adapter = output_dir / "adapter_final"
    model.save_pretrained(final_adapter)
    processor.save_pretrained(final_adapter)

    metrics = {
        "protocol_version": PROTOCOL_VERSION,
        "model_key": model_key,
        "condition": condition,
        "learning_rate": learning_rate,
        "lora": lora_cfg,
        "train_file": str(train_file),
        "val_file": str(val_file),
        "data_root": str(data_root),
        "epochs": epochs,
        "global_steps": global_step,
        "epoch_metrics": epoch_metrics,
        "best_val_accuracy": max((m["val_accuracy"] for m in epoch_metrics), default=None),
        "adapter_final": str(final_adapter),
        "status": "completed",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "finetune_config.json").write_text(
        json.dumps({**metrics, "seed": seed}, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "finetune_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for multimodal mental state recognition.")
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()))
    parser.add_argument("--train_file", type=Path, required=True)
    parser.add_argument("--val_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, default=Path("data/mindreading"))
    parser.add_argument("--condition", type=str, default=None, choices=("video_only", "audio_only", "multimodal"))
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=LORA_DEFAULT["r"])
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=LORA_DEFAULT["dropout"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_eval", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    alpha = args.lora_alpha if args.lora_alpha is not None else lora_alpha_for_rank(args.lora_r)
    cond = args.condition or finetune_modality_for_model(args.model)
    metrics = run_finetuning(
        model_key=args.model,
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        r=args.lora_r,
        alpha=alpha,
        dropout=args.lora_dropout,
        data_root=args.data_root,
        condition=cond,
        seed=args.seed,
        epochs=args.epochs,
        max_train_samples=args.max_train_samples,
        max_val_eval=args.max_val_eval,
        max_steps=args.max_steps,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
