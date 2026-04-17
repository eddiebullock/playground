import argparse
import json
import re
import os
import socket
import subprocess
import traceback
import types
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


def _safe_cmd(cmd: List[str]) -> Optional[str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
        return out or None
    except Exception:
        return None


def collect_run_metadata() -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_submit_dir": os.environ.get("SLURM_SUBMIT_DIR"),
        "python": _safe_cmd(["python", "-V"]) or _safe_cmd(["python3", "-V"]),
        "git_commit": _safe_cmd(["git", "rev-parse", "HEAD"]),
        "transformers_version": None,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    try:
        import transformers  # type: ignore

        meta["transformers_version"] = getattr(transformers, "__version__", None)
    except Exception:
        pass

    freeze = _safe_cmd(["python", "-m", "pip", "freeze"]) or _safe_cmd(["python3", "-m", "pip", "freeze"])
    if isinstance(freeze, str):
        meta["pip_freeze_head"] = "\n".join(freeze.splitlines()[:50])
    else:
        meta["pip_freeze_head"] = None
    return meta


def load_model(model_key: str) -> Any:
    """
    Load a multimodal model given a key in MODELS.

    This stub should be replaced with actual transformers / model-loading logic.
    """
    model_cfg = MODELS[model_key]
    _ = model_cfg
    return None


def _internvl2_fallback_generation_config(mod: Any) -> Any:
    """
    InternLM2 configs often break GenerationConfig.from_model_config (None._from_model_config).
    Always fall back to a plain GenerationConfig with token ids copied from the LM config.
    """
    from transformers import GenerationConfig  # type: ignore

    cfg = getattr(mod, "config", None)
    if cfg is not None:
        try:
            return GenerationConfig.from_model_config(cfg)  # type: ignore[attr-defined]
        except Exception:
            pass
    gc = GenerationConfig()
    if cfg is not None:
        for name in ("eos_token_id", "pad_token_id", "bos_token_id"):
            val = getattr(cfg, name, None)
            if val is not None:
                setattr(gc, name, val)
    return gc


def _internvl2_ensure_module_generation_config(mod: Any) -> None:
    if mod is None or getattr(mod, "generation_config", None) is not None:
        return
    mod.generation_config = _internvl2_fallback_generation_config(mod)


def _internvl2_past_seq_len(past_key_values: Any) -> int:
    """
    InternLM2 remote `prepare_inputs_for_generation` uses past_key_values[0][0].shape[2].
    Transformers 4.5+ may pass DynamicCache where layer key tensors are still None (lazy init);
    use the Cache API when available.
    """
    if past_key_values is None:
        return 0
    gsq = getattr(past_key_values, "get_seq_length", None)
    if callable(gsq):
        try:
            return int(gsq(0))
        except TypeError:
            try:
                return int(gsq())
            except Exception:
                pass
        except Exception:
            pass
    try:
        layer0 = past_key_values[0]
        first = layer0[0] if layer0 is not None else None
        if first is not None and hasattr(first, "shape") and len(first.shape) > 2:
            return int(first.shape[2])
    except Exception:
        pass
    return 0


def _internvl2_patch_prepare_inputs_for_generation(lm: Any) -> None:
    """
    Replace broken legacy cache indexing with cache.get_seq_length(); fix position_ids slicing when
    input_ids is empty (inputs_embeds prefill path).
    """
    if getattr(lm, "_mr_eu_internvl_prep_patch", False):
        return

    def prepare_inputs_for_generation(
        self: Any,
        input_ids: Any,
        past_key_values: Any = None,
        attention_mask: Any = None,
        inputs_embeds: Any = None,
        **kwargs: Any,
    ) -> Any:
        if past_key_values is not None:
            past_length = _internvl2_past_seq_len(past_key_values)
            if input_ids is not None and input_ids.shape[1] > 0:
                if input_ids.shape[1] > past_length:
                    remove_prefix_length = past_length
                else:
                    remove_prefix_length = input_ids.shape[1] - 1
                input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                if input_ids is not None and input_ids.shape[1] > 0:
                    seq_w = int(input_ids.shape[1])
                else:
                    seq_w = int(attention_mask.shape[1])
                position_ids = position_ids[:, -seq_w:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    lm.prepare_inputs_for_generation = types.MethodType(prepare_inputs_for_generation, lm)
    setattr(lm, "_mr_eu_internvl_prep_patch", True)


def _internvl2_wrap_language_model_generate(lm: Any) -> None:
    """
    InternVL batch_chat does self.generate(..., **dict), so InternVLChatModel passes
    max_new_tokens etc. as kwargs while generation_config stays None. language_model.generate
    must not receive generation_config=None under GenerationMixin on recent Transformers.
    """
    if getattr(lm, "_mr_eu_internvl_gc_wrap", False):
        return
    orig = lm.generate

    def generate(self: Any, *args: Any, generation_config: Any = None, **kwargs: Any) -> Any:
        if generation_config is None:
            _internvl2_ensure_module_generation_config(self)
            generation_config = getattr(self, "generation_config", None)
            if generation_config is None:
                self.generation_config = _internvl2_fallback_generation_config(self)
                generation_config = self.generation_config
        return orig(*args, generation_config=generation_config, **kwargs)

    lm.generate = types.MethodType(generate, lm)
    setattr(lm, "_mr_eu_internvl_gc_wrap", True)


def patch_internvl2_language_model_generation(model: Any) -> None:
    """
    InternVL's chat/batch_chat call self.language_model.generate(...). On Transformers >=4.50,
    some remote-code LMs (e.g. InternLM2ForCausalLM) no longer inherit GenerationMixin, so
    `.generate` is missing. Mix GenerationMixin onto the *instance* so InternVL's forward path works.
    """
    _internvl2_ensure_module_generation_config(model)
    lm = getattr(model, "language_model", None)
    if lm is None:
        return
    _internvl2_ensure_module_generation_config(lm)

    if not callable(getattr(lm, "generate", None)):
        try:
            from transformers.generation.utils import GenerationMixin  # type: ignore

            base_cls = lm.__class__
            if not issubclass(base_cls, GenerationMixin):
                merged_name = base_cls.__name__ + "_WithGenerationMixin"
                merged_cls = type(merged_name, (base_cls, GenerationMixin), {})
                lm.__class__ = merged_cls  # type: ignore[assignment]
                _internvl2_ensure_module_generation_config(lm)
        except Exception as e:
            raise RuntimeError(
                "InternVL2 language model has no .generate() under this Transformers version "
                "(common with Transformers>=4.50). Could not attach GenerationMixin. "
                "Try a dedicated env with transformers<4.50, or an updated InternVL checkpoint."
            ) from e

    _internvl2_patch_prepare_inputs_for_generation(lm)
    _internvl2_wrap_language_model_generate(lm)


def resolve_model_path(model_key: str) -> Path:
    cfg = MODELS[model_key]
    hpc = Path(cfg["hpc_path"])
    if hpc.exists():
        return hpc
    return Path(cfg["local_path"])


def load_hf_model_for_key(model_key: str, model_path: Path, device: str, dtype: torch.dtype) -> Any:
    """
    Best-effort loader across HF model class variants.
    """
    # Qwen2-VL dedicated class first (where available).
    if model_key == "qwen2vl":
        try:
            from transformers import Qwen2VLForConditionalGeneration  # type: ignore

            return Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
            )
        except Exception:
            pass

    # InternVL2: prefer its custom AutoModel (InternVLChatModel).
    if model_key == "internvl2":
        try:
            from transformers import AutoModel  # type: ignore

            return AutoModel.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
            )
        except Exception:
            pass

    # LLaVA-NeXT is not always mapped correctly by older Transformers versions.
    # Try common explicit classes first.
    if model_key == "llavanext":
        for cls_name in (
            "LlavaForConditionalGeneration",
            "LlavaNextForConditionalGeneration",
            "LlavaQwenForConditionalGeneration",
        ):
            try:
                mod = __import__("transformers", fromlist=[cls_name])
                cls = getattr(mod, cls_name)
                return cls.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    device_map="auto" if device == "cuda" else None,
                    trust_remote_code=True,
                )
            except Exception:
                pass

    # Most recent multimodal auto class.
    try:
        from transformers import AutoModelForImageTextToText  # type: ignore

        return AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
    except Exception:
        pass

    # Older multimodal class.
    try:
        from transformers import AutoModelForVision2Seq  # type: ignore

        return AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
    except Exception:
        pass

    # Last resort fallback.
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )


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


def parse_emotion(output_text: Any, options: Sequence[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (emotion, reasoning). Emotion is forced to one of options if possible.

    If output_text still includes the user prompt (should be avoided: decode only new tokens),
    we skip instruction placeholders and use the last EMOTION / REASONING lines.
    """
    emotion = None
    reasoning = None
    if not isinstance(output_text, str):
        try:
            text = json.dumps(output_text, ensure_ascii=False)
        except Exception:
            text = str(output_text)
    else:
        text = output_text
    text = text.strip()

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

    # Fallback: some models ignore the EMOTION/REASONING format and just answer with
    # an option label (or include it inline). Since we decode only generated tokens,
    # scanning the completion is safe (it won't match the OPTIONS block from the prompt).
    if emotion is None and text:
        lower = text.lower()
        for opt in sorted(options, key=lambda x: -len(str(x))):
            if str(opt).lower() in lower:
                emotion = str(opt)
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

    model_path = resolve_model_path(model_key)
    # Some models (e.g. Gemma4) may require a newer Transformers version for AutoProcessor to resolve the
    # correct processing class. Only load a processor when we actually need it, and provide a targeted
    # error for Gemma4 if the installed Transformers is too old.
    processor = None
    if model_key not in {"gemma4"}:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    else:
        try:
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        except Exception as e:
            try:
                # If available, use the explicit processor class (newer Transformers).
                from transformers import Gemma4Processor  # type: ignore

                processor = Gemma4Processor.from_pretrained(model_path)
            except Exception:
                raise RuntimeError(
                    "Gemma4 processor could not be loaded from the local model folder. "
                    "This usually means your Transformers version is too old for Gemma4. "
                    "On CSD3, try: `python -m pip install --upgrade transformers` in the mr_eu_open_llm env, "
                    "then re-run the job."
                ) from e

    # InternVL2 is best driven via its custom chat() API.
    tokenizer = None
    if model_key == "internvl2":
        from transformers import AutoTokenizer  # type: ignore

        tok_err: Optional[Exception] = None
        for use_fast in (True, False):
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=use_fast)
                break
            except Exception as e:
                tok_err = e
                tokenizer = None
        if tokenizer is None:
            raise RuntimeError(f"InternVL2 tokenizer failed to load: {tok_err}")

    model = load_hf_model_for_key(model_key=model_key, model_path=model_path, device=device, dtype=dtype)
    if device != "cuda":
        model = model.to(device)
    model.eval()

    # Model-family specific compatibility tweaks.
    if model_key == "llavanext":
        # Some LLaVA checkpoints use a SigLIP vision tower that lacks `.patch_size`,
        # but downstream utilities may assume it exists.
        try:
            vt = getattr(model, "vision_tower", None)
            if vt is None and hasattr(model, "get_vision_tower"):
                vt = model.get_vision_tower()
            if vt is not None and not hasattr(vt, "patch_size"):
                cfg = getattr(vt, "config", None)
                ps = getattr(cfg, "patch_size", None) if cfg is not None else None
                if ps is None:
                    vision_cfg = getattr(cfg, "vision_config", None) if cfg is not None else None
                    ps = getattr(vision_cfg, "patch_size", None) if vision_cfg is not None else None
                if ps is not None:
                    setattr(vt, "patch_size", ps)
        except Exception:
            pass

    if model_key == "internvl2":
        # If we didn't get the chat wrapper, retry a couple of other auto-classes.
        if not hasattr(model, "chat"):
            try:
                from transformers import AutoModel  # type: ignore

                model = AutoModel.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                )
                if device == "cuda":
                    model = model.to(device)
                model.eval()
            except Exception:
                pass
        if not hasattr(model, "chat"):
            raise RuntimeError(
                "InternVL2 loaded without `chat()` (got a base LM). "
                "Your local model snapshot likely isn't the InternVL chat wrapper class. "
                "Re-download the model to ensure `auto_map` is present, or switch to an InternVL2 chat checkpoint."
            )
        try:
            tok = getattr(processor, "tokenizer", None)
            if tok is not None and hasattr(model, "img_context_token_id"):
                if getattr(model, "img_context_token_id", None) is None:
                    candidates = ["<IMG_CONTEXT>", "<img_context>", "<image>", "<IMAGE>"]
                    for cand in candidates:
                        try:
                            tid = tok.convert_tokens_to_ids(cand)
                            if isinstance(tid, int) and tid >= 0 and tid != getattr(tok, "unk_token_id", -1):
                                setattr(model, "img_context_token_id", tid)
                                break
                        except Exception:
                            continue
        except Exception:
            pass

        # batch_chat/chat -> self.generate -> language_model.generate; LM must expose .generate().
        patch_internvl2_language_model_generation(model)

    trials: List[Dict[str, Any]] = []
    n_correct = 0
    n_scored = 0
    pipe_cache: Dict[str, Any] = {}
    for t in trials_raw:
        options = make_4afc_options(t["label"], labels, rng) if labels else []
        stimulus_path = Path(t["stimulus_path"])
        prompt = build_4afc_prompt(options)

        try:
            images = load_stimulus_as_images(stimulus_path, n_frames=n_frames)
            # Some model families (notably many LLaVA and InternVL variants) behave as
            # single-image chat models; passing multiple frames can cause failures.
            if model_key in {"llavanext", "internvl2", "gemma4"}:
                images = [images[0]]
                images_for_processor: Any = images[0]  # pass a single PIL.Image
            else:
                images_for_processor = images

            # InternVL2: always use the checkpoint's chat() API (no fallback to generate()).
            if model_key == "internvl2":
                if tokenizer is None or not hasattr(model, "chat"):
                    raise RuntimeError(
                        f"InternVL2 missing chat() or tokenizer (tokenizer={tokenizer is not None}, has_chat={hasattr(model, 'chat')})."
                    )
                img_proc = getattr(processor, "image_processor", None) or getattr(processor, "vision_processor", None)
                if img_proc is None:
                    try:
                        from transformers import AutoImageProcessor  # type: ignore

                        img_proc = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
                    except Exception:
                        img_proc = None
                if img_proc is None:
                    raise RuntimeError("InternVL2 processor missing image processor.")

                pv = img_proc(images_for_processor, return_tensors="pt")
                pixel_values = pv.get("pixel_values")
                if pixel_values is None:
                    raise RuntimeError("InternVL2 image processor did not return pixel_values.")
                pixel_values = pixel_values.to(model.device, dtype=dtype)

                # InternVL remote code expects a *mutable dict* passed as generation_config; batch_chat
                # does `generation_output = self.generate(..., **generation_config)`. Never pass
                # generation kwargs as **kwargs to batch_chat (that triggers "unexpected keyword ...").
                gen_cfg: Dict[str, Any] = {"max_new_tokens": int(max_new_tokens)}
                if float(temperature) > 0:
                    gen_cfg.update({"do_sample": True, "temperature": float(temperature), "top_p": float(EVAL["top_p"])})
                else:
                    gen_cfg["do_sample"] = False
                with torch.inference_mode():
                    # batch_chat requires num_patches_list (see OpenGVLab modeling_internvl_chat.py).
                    num_patches_list = [int(pixel_values.shape[0])] if pixel_values is not None else []
                    if hasattr(model, "batch_chat") and num_patches_list:
                        gc = dict(gen_cfg)
                        out_list = model.batch_chat(
                            tokenizer,
                            pixel_values,
                            [prompt],
                            gc,
                            num_patches_list=num_patches_list,
                        )
                        out_text = out_list[0] if isinstance(out_list, list) and out_list else str(out_list)
                    else:
                        out_text = model.chat(tokenizer, pixel_values, prompt, generation_config=dict(gen_cfg))
                if not isinstance(out_text, str):
                    out_text = str(out_text)
            elif model_key == "gemma4":
                # Gemma 4: prefer the Transformers-native image-text-to-text pipeline for robustness.
                # If the local Transformers is too old to resolve Gemma4Processor, pipeline construction will fail;
                # in that case, instruct the user to upgrade Transformers in the HPC env.
                try:
                    from transformers import pipeline  # type: ignore

                    pipe = pipe_cache.get("gemma4")
                    if pipe is None:
                        pipe = pipeline(
                            "image-text-to-text",
                            model=model_path,
                            device=0 if device == "cuda" else -1,
                            torch_dtype=dtype if device == "cuda" else None,
                        )
                        pipe_cache["gemma4"] = pipe

                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": images_for_processor},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]
                    out = pipe(text=messages, max_new_tokens=int(max_new_tokens))
                    if isinstance(out, list) and out:
                        out0 = out[0]
                        out_text = out0.get("generated_text") if isinstance(out0, dict) else None
                        if out_text is None:
                            out_text = str(out0)
                    else:
                        out_text = str(out)
                except Exception as e:
                    raise RuntimeError(
                        f"Gemma4 pipeline inference failed: {e}. "
                        "If this mentions an unrecognized Gemma4 processor/config, upgrade Transformers "
                        "in the mr_eu_open_llm env (e.g. `python -m pip install --upgrade transformers`)."
                    ) from e
            elif model_key == "llavanext":
                # LLaVA Interleave: the HF pipeline handles image/text alignment more robustly than
                # hand-rolled processor + generate for some backbone/processor combinations.
                try:
                    from transformers import pipeline  # type: ignore

                    pipe = pipe_cache.get("llavanext")
                    if pipe is None:
                        pipe = pipeline(
                            "image-text-to-text",
                            model=model_path,
                            device=0 if device == "cuda" else -1,
                            torch_dtype=dtype if device == "cuda" else None,
                        )
                        pipe_cache["llavanext"] = pipe
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": images_for_processor},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]
                    out = pipe(text=messages, max_new_tokens=int(max_new_tokens))
                    if isinstance(out, list) and out:
                        out0 = out[0]
                        out_text = out0.get("generated_text") if isinstance(out0, dict) else None
                        if out_text is None:
                            out_text = str(out0)
                    else:
                        out_text = str(out)
                except Exception as e:
                    raise RuntimeError(f"LLaVA pipeline inference failed: {e}") from e
            else:
                # Qwen2-VL: structured chat template with {"type": "image"} blocks.
                # LLaVA(-Next): also prefers structured chat template; avoid manual <image> token counting.
                text: str
                if model_key == "llavanext":
                    # Follow the official LLaVA Interleave HF example: text first, then image placeholder.
                    # This avoids internal image/text alignment errors seen with other orderings.
                    conversation = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image"}]}]
                    try:
                        text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
                    except Exception:
                        text = f"{prompt}\n<image>"
                elif model_key == "qwen2vl":
                    content_q: List[Dict[str, Any]] = [{"type": "image"} for _ in images]
                    content_q.append({"type": "text", "text": prompt})
                    messages_q = [{"role": "user", "content": content_q}]
                    try:
                        text = processor.apply_chat_template(messages_q, tokenize=False, add_generation_prompt=True)
                    except Exception:
                        image_placeholder = "<image>"
                        joined = "\n".join([image_placeholder] * len(images))
                        text = f"{joined}\n{prompt}"
                else:
                    image_placeholder = "<image>"
                    joined = "\n".join([image_placeholder] * len(images))
                    text = f"{joined}\n{prompt}"

                # Some processors expect `text` (str) not `text=[...]`.
                try:
                    inputs = processor(images=images_for_processor, text=text, return_tensors="pt")
                except Exception:
                    try:
                        inputs = processor(images=images_for_processor, text=[text], return_tensors="pt")
                    except Exception:
                        inputs = processor(text=text, images=images_for_processor, return_tensors="pt")
                if inputs is None:
                    raise RuntimeError("Processor returned None.")

                # LLaVA(-Next) generation often expects image sizes; ensure present and non-None.
                if model_key == "llavanext":
                    try:
                        image_sizes = inputs.get("image_sizes") if hasattr(inputs, "get") else inputs["image_sizes"]  # type: ignore[index]
                    except Exception:
                        image_sizes = None
                    if image_sizes is None:
                        h = int(images[0].size[1])
                        w = int(images[0].size[0])
                        try:
                            inputs["image_sizes"] = torch.tensor([[h, w]], device=model.device)  # type: ignore[index]
                        except Exception:
                            pass

                inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

                gen_kwargs: Dict[str, Any] = {"max_new_tokens": int(max_new_tokens)}
                if float(temperature) > 0:
                    gen_kwargs.update({"do_sample": True, "temperature": float(temperature), "top_p": float(EVAL["top_p"])})
                else:
                    gen_kwargs["do_sample"] = False

                with torch.inference_mode():
                    out_ids = model.generate(**inputs, **gen_kwargs)

                in_len = 0
                try:
                    input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else inputs["input_ids"]
                    if input_ids is not None:
                        in_len = int(input_ids.shape[1])
                except Exception:
                    in_len = 0
                try:
                    gen_ids = out_ids[:, in_len:] if in_len > 0 else out_ids
                    out_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
                except Exception:
                    out_text = str(out_ids)

            # Ensure parsing always receives a string-like value.
            if not isinstance(out_text, str):
                try:
                    out_text = json.dumps(out_text, ensure_ascii=False)
                except Exception:
                    out_text = str(out_text)
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
            msg = str(e).strip()
            if not msg:
                msg = f"{type(e).__name__} (empty message)"
            else:
                msg = f"{type(e).__name__}: {msg}"
            trials.append(
                {
                    **t,
                    "options": options,
                    "prediction": None,
                    "correct": None,
                    "error": msg,
                    "traceback": traceback.format_exc(limit=6),
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
        "max_new_tokens": int(max_new_tokens),
        "evaluator_version": "2026-04-01-internvl-prepare-inputs-cache",
        "run_metadata": collect_run_metadata(),
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

