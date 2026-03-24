import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

from config import (
    SEED,
    MODELS,
    DATASETS,
    EVAL,
    LOCAL_RESULTS_DIR,
)

from scripts.statistics import binomial_vs_chance, wilson_ci


MEDIA_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".jpg", ".jpeg", ".png", ".webp"}


def load_model(model_key: str) -> Any:
    """
    Load a multimodal model given a key in MODELS.

    This stub should be replaced with actual transformers / model-loading logic.
    """
    model_cfg = MODELS[model_key]
    _ = model_cfg
    return None


def resolve_model_path(model_key: str) -> Path:
    cfg = MODELS[model_key]
    hpc = Path(cfg["hpc_path"])
    if hpc.exists():
        return hpc
    return Path(cfg["local_path"])


def resolve_dataset_root(dataset_key: str, override_root: Optional[Path] = None) -> Path:
    if override_root is not None:
        return override_root

    cfg = DATASETS[dataset_key]
    hpc = Path(cfg["hpc"])
    if hpc.exists():
        return hpc
    return Path(cfg["local"])


def list_eu_emotions_trials(dataset_root: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    EU-Emotions (faces-only) loader.

    Assumes structure like:
      data/eu_emotions/<EMOTION_NAME>/*.(mov|mp4|jpg|png|...)
      data/eu_emotions/audio/... (ignored here)
    """
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    label_dirs = [p for p in dataset_root.iterdir() if p.is_dir() and p.name.lower() != "audio"]
    labels = sorted([p.name for p in label_dirs])

    trials: List[Dict[str, Any]] = []
    for label in labels:
        label_dir = dataset_root / label
        # Only one level deep for faces-only; still be robust to nested.
        for p in label_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in MEDIA_EXTS:
                rel = p.relative_to(dataset_root).as_posix()
                trials.append(
                    {
                        "trial_id": rel,
                        "stimulus_path": str(p),
                        "label": label,
                    }
                )

    return trials, labels


def load_eu_emotions_manifest(
    manifest_path: Path,
    dataset_root: Path,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Load the 118-trial EU-Emotions stimulus set from a manifest JSON.

    Expected format (as in mr_ts_play):
      {
        "num_trials": 118,
        "trials": [
          {"stimulus_path": "...", "correct_label": "...", "trial_id": "..."},
          ...
        ]
      }
    """
    obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    trials_in = obj.get("trials", [])
    labels = sorted({t["correct_label"] for t in trials_in if "correct_label" in t})

    trials: List[Dict[str, Any]] = []
    for t in trials_in:
        rel_path = t["stimulus_path"]
        abs_path = (dataset_root / rel_path).resolve()
        trials.append(
            {
                "trial_id": t.get("trial_id", rel_path),
                "stimulus_path": str(abs_path),
                "label": t["correct_label"],
                "stimulus_relpath": rel_path,
            }
        )

    return trials, labels


def make_4afc_options(
    correct_label: str,
    all_labels: Sequence[str],
    rng: np.random.Generator,
) -> List[str]:
    others = [l for l in all_labels if l != correct_label]
    if len(others) < 3:
        raise ValueError("Need at least 4 labels total for 4AFC.")
    sampled = rng.choice(np.array(others, dtype=object), size=3, replace=False).tolist()
    options = [correct_label, *sampled]
    rng.shuffle(options)
    return [str(x) for x in options]


def frame_schedule(n_frames: int) -> List[float]:
    if n_frames == 4:
        return list(EVAL["frame_sampling_4"])
    if n_frames == 8:
        return list(EVAL["frame_sampling_8"])
    # Fallback: uniform positions (inclusive endpoints)
    if n_frames <= 1:
        return [0.5]
    return [float(x) for x in np.linspace(0.0, 1.0, n_frames)]


def load_frames_from_video(video_path: Path, n_frames: int) -> List[Image.Image]:
    # Interlaced HD clips can spam FFmpeg swscaler warnings to stderr; job still succeeds.
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        pass

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Video has no frames: {video_path}")

    positions = frame_schedule(n_frames)
    frames: List[Image.Image] = []
    for pos in positions:
        idx = int(round(pos * (frame_count - 1)))
        idx = max(0, min(frame_count - 1, idx))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

    cap.release()
    if not frames:
        raise RuntimeError(f"Failed to extract frames from: {video_path}")
    return frames


def load_stimulus_as_images(stimulus_path: Path, n_frames: int) -> List[Image.Image]:
    ext = stimulus_path.suffix.lower()
    if ext in {".jpg", ".jpeg", ".png", ".webp"}:
        img = Image.open(stimulus_path).convert("RGB")
        return [img]
    if ext in {".mp4", ".mov", ".avi", ".mkv"}:
        return load_frames_from_video(stimulus_path, n_frames=n_frames)
    raise ValueError(f"Unsupported stimulus type: {stimulus_path}")


def build_4afc_prompt(options: Sequence[str]) -> str:
    opts = "\n".join([f"{i+1}) {opt}" for i, opt in enumerate(options)])
    return (
        "You are performing a 4-alternative forced-choice mental state recognition task.\n"
        "Choose exactly one label from the options.\n\n"
        f"OPTIONS:\n{opts}\n\n"
        "Respond with:\n"
        "EMOTION: <one of the option labels exactly>\n"
        "REASONING: <brief justification>\n"
    )


def _is_instruction_emotion_placeholder(raw: str) -> bool:
    s = raw.strip().lower()
    if "<" in raw and ">" in raw:
        return True
    if "option labels" in s or "brief justification" in s:
        return True
    return False


def _match_raw_to_option(raw: str, options: Sequence[str]) -> Optional[str]:
    """
    Map raw EMOTION field text to one of the four options (4AFC).
    Prefer longer labels on substring match so e.g. 'Afraid Low Intensity' beats 'Afraid'.
    """
    raw0 = raw.strip().strip(" .\"'`")
    # "3) Excited" or "2) Proud"
    m_paren = re.match(r"^(\d+)\s*[\)\.:]\s*(.+)$", raw0)
    if m_paren:
        idx = int(m_paren.group(1)) - 1
        rest = m_paren.group(2).strip()
        if 0 <= idx < len(options):
            if rest.lower() == options[idx].lower() or rest.lower() in options[idx].lower():
                return options[idx]
            em = _match_raw_to_option(rest, options)
            if em is not None:
                return em
        em = _match_raw_to_option(rest, options)
        if em is not None:
            return em

    mnum = re.match(r"^(\d+)\s*[\)\.]?\s*$", raw0)
    if mnum:
        idx = int(mnum.group(1)) - 1
        if 0 <= idx < len(options):
            return options[idx]

    for opt in options:
        if raw0.lower() == opt.lower():
            return opt

    for opt in sorted(options, key=lambda x: -len(str(x))):
        if opt.lower() in raw0.lower():
            return opt

    return None


def parse_emotion(output_text: str, options: Sequence[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (emotion, reasoning). Emotion is forced to one of options if possible.

    If output_text still includes the user prompt (should be avoided: decode only new tokens),
    we skip instruction placeholders and use the last EMOTION / REASONING lines.
    """
    emotion = None
    reasoning = None
    text = output_text.strip()

    # Last EMOTION line wins (avoids matching the template line in the user message).
    for m in reversed(list(re.finditer(r"(?im)^\s*EMOTION\s*[:\-]\s*(.+?)\s*$", text))):
        raw = m.group(1).strip()
        if _is_instruction_emotion_placeholder(raw):
            continue
        emotion = _match_raw_to_option(raw, options)
        if emotion is not None:
            break

    for r in reversed(list(re.finditer(r"(?im)^\s*REASONING\s*:\s*(.+?)\s*$", text))):
        rs = r.group(1).strip()
        if rs.lower() in {"<brief justification>", "brief justification"}:
            continue
        if "<" in rs and "brief" in rs.lower():
            continue
        reasoning = rs
        break

    return emotion, reasoning


def run_evaluation(
    model_key: str,
    dataset_key: str,
    n_frames: int,
    output_path: Path,
    seed: int = SEED,
    data_root: Optional[Path] = None,
    manifest: Optional[Path] = None,
    max_trials: Optional[int] = None,
    temperature: float = EVAL["temperature"],
    max_new_tokens: int = 128,
) -> Dict[str, Any]:
    """
    Run the full 4AFC evaluation pipeline for a model/dataset combination.

    Returns a dictionary of aggregate metrics and metadata which is also saved to output_path.
    """
    rng = np.random.default_rng(seed)

    dataset_root = resolve_dataset_root(dataset_key, override_root=data_root)
    if dataset_key == "eu_emotions":
        if manifest is not None:
            trials_raw, labels = load_eu_emotions_manifest(manifest, dataset_root)
        else:
            trials_raw, labels = list_eu_emotions_trials(dataset_root)
    else:
        # Stub for other datasets; will be implemented later.
        trials_raw, labels = ([], [])

    # Optional: cap number of trials for smoke tests.
    trials_raw = list(trials_raw)
    if max_trials is not None:
        trials_raw = trials_raw[: int(max_trials)]

    # Load model (Phase B: implemented for Qwen2-VL only).
    model = None
    processor = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    if model_key == "qwen2vl":
        model_path = resolve_model_path(model_key)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        # Compatibility: transformers versions differ in exposed model classes.
        model = None
        # Try the specific Qwen2-VL class first (most reliable when available).
        try:
            from transformers import Qwen2VLForConditionalGeneration  # type: ignore

            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
            )
        except Exception:
            pass

        # Fall back to a Vision2Seq auto class if present.
        if model is None:
            try:
                from transformers import AutoModelForVision2Seq  # type: ignore

                model = AutoModelForVision2Seq.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    device_map="auto" if device == "cuda" else None,
                    trust_remote_code=True,
                )
            except Exception:
                pass

        # Last resort: CausalLM auto class (may still work for some versions).
        if model is None:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
            )
        if device != "cuda":
            model = model.to(device)
        model.eval()
    else:
        raise NotImplementedError(f"Model inference not implemented yet for: {model_key}")

    trials: List[Dict[str, Any]] = []
    n_correct = 0
    n_scored = 0
    for t in trials_raw:
        options = make_4afc_options(t["label"], labels, rng) if labels else []
        stimulus_path = Path(t["stimulus_path"])
        prompt = build_4afc_prompt(options)

        try:
            images = load_stimulus_as_images(stimulus_path, n_frames=n_frames)

            # Qwen2-VL: build a prompt + embed as many <image> tokens as we have frames.
            content: List[Dict[str, Any]] = [{"type": "image"} for _ in images]
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]

            try:
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                # Fallback: some processors don't support the chat template interface.
                # We include explicit <image> placeholders then the text prompt.
                image_placeholder = "<image>"
                joined = "\n".join([image_placeholder] * len(images))
                text = f"{joined}\n{prompt}"

            inputs = processor(text=[text], images=images, return_tensors="pt")
            inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": int(max_new_tokens),
            }
            if float(temperature) > 0:
                gen_kwargs.update(
                    {
                        "do_sample": True,
                        "temperature": float(temperature),
                        "top_p": float(EVAL["top_p"]),
                    }
                )
            else:
                gen_kwargs["do_sample"] = False

            with torch.inference_mode():
                out_ids = model.generate(**inputs, **gen_kwargs)
            # Decode only generated tokens so we do not parse EMOTION/REASONING from the prompt.
            input_ids = inputs["input_ids"]
            in_len = int(input_ids.shape[1])
            gen_ids = out_ids[:, in_len:]
            if gen_ids.shape[1] > 0:
                out_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
            else:
                out_text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]

            pred, reasoning = parse_emotion(out_text, options)
            # Enforce 4AFC scoring: if the model didn't output one of the provided options,
            # we treat it as a wrong response rather than "unscored".
            if pred is None:
                correct = False
            else:
                correct = pred == t["label"]

            n_scored += 1
            n_correct += int(correct)

            trials.append(
                {
                    **t,
                    "options": options,
                    "prediction": pred,
                    "correct": bool(correct),
                    "reasoning": reasoning,
                    "raw_model_output": out_text,
                }
            )
        except Exception as e:
            trials.append(
                {
                    **t,
                    "options": options,
                    "prediction": None,
                    "correct": None,
                    "error": str(e),
                }
            )

    n_trials = len(trials)
    if n_scored > 0:
        accuracy = n_correct / n_scored
        ci_low, ci_high = wilson_ci(n_correct, n_scored)
        p_binom = binomial_vs_chance(n_correct, n_scored)
    else:
        accuracy = None
        ci_low, ci_high = (None, None)
        p_binom = None

    metrics: Dict[str, Any] = {
        "accuracy": accuracy,
        "accuracy_wilson_ci_95": [ci_low, ci_high],
        "p_binom_gt_chance": p_binom,
        "n_trials": n_trials,
        "n_scored": n_scored,
        "n_correct": n_correct,
        "seed": seed,
        "model": model_key,
        "dataset": dataset_key,
        "n_frames": n_frames,
        "temperature": float(temperature),
        "device": str(device),
        "dataset_root": str(dataset_root),
        "manifest": str(manifest) if manifest is not None else None,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "trials": trials,
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multimodal LLMs on mental state recognition datasets.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODELS.keys()),
        help="Model key defined in config.MODELS.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASETS.keys()),
        help="Dataset key defined in config.DATASETS.",
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=EVAL["n_frames_default"],
        help="Number of frames per video to include in the prompt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save the JSON results. If not provided, a default path in results/baseline is used.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for evaluation.",
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=None,
        help="Optional override for dataset root path (useful on HPC).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest JSON listing exact trials (e.g., 118-trial EU-Emotions stimulus set).",
    )
    parser.add_argument(
        "--max_trials",
        type=int,
        default=None,
        help="Optional cap on number of trials (for quick tests).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=EVAL["temperature"],
        help="Sampling temperature for generation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Max new tokens to generate per trial.",
    )

    args = parser.parse_args()

    if args.output is None:
        default_dir = LOCAL_RESULTS_DIR / "baseline" / args.dataset / args.model
        default_dir.mkdir(parents=True, exist_ok=True)
        args.output = default_dir / f"baseline_{args.dataset}_{args.model}_frames{args.n_frames}_seed{args.seed}.json"
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    metrics = run_evaluation(
        model_key=args.model,
        dataset_key=args.dataset,
        n_frames=args.n_frames,
        output_path=args.output,
        seed=args.seed,
        data_root=args.data_root,
        manifest=args.manifest,
        max_trials=args.max_trials,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    # Always write a results artifact (even if metrics are placeholders).
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True, default=str)
        f.write("\n")


if __name__ == "__main__":
    main()

