import argparse
import json
import re
import os
import socket
import subprocess
import traceback
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
    PROTOCOL_VERSION,
    FRAME_POLICY,
    EMBEDDING_MODEL,
    ENTROPY_COLLAPSE_INTENSITY,
    ENTROPY_EXCLUDE_LABELS,
    ENTROPY_TEMPERATURE,
    ENTROPY_LOG_BASE,
    ENTROPY_USE_RICH_LABEL_EMBEDDINGS,
    CHAIN_STAGES,
    STAGE1_MAX_NEW_TOKENS,
    STAGE2_MAX_NEW_TOKENS,
    HUMAN_BENCHMARKS,
    MODALITY_CONDITIONS,
    MODEL_AUDIO_CAPABILITIES,
    EU_EMOTION_LABELS_FILE,
    CHANCE_LEVEL,
    CONFIRMATORY_N_MODELS,
)

from scripts.eu_audio_resolver import (
    build_audio_mapping_audit as build_eu_audio_audit,
    resolve_eu_audio_only,
    resolve_eu_multimodal_audio,
    save_audio_mapping_audit as save_eu_audio_audit,
)
from scripts.mindreading_audio_resolver import (
    LeakageAudioPathError,
    build_audio_mapping_audit as build_mr_audio_audit,
    extract_audio_from_video,
    resolve_item_folder_audio,
    resolve_mindreading_v_video,
    save_audio_mapping_audit as save_mr_audio_audit,
)
from scripts.trial_foils import (
    build_emotion_pool_from_trials,
    resolve_eu_emotion_pool,
    resolve_candidate_labels,
)
from scripts.tolerant_parse import parse_emotion_tolerant
from scripts.emotion_parse import parse_emotion
from scripts.prompts import build_free_response_prompt, build_4afc_prompt, build_finetune_prompt
from scripts.frame_sampling import load_stimulus_as_images, frame_policy_tag
from scripts.multi_frame import prepare_images_for_model
from scripts.model_inference import generate_model_response
from scripts.semantic_entropy import (
    compute_entropy_bundle,
    load_or_compute_label_embeddings,
    prepare_entropy_label_pool,
    strip_boilerplate_response,
)
from scripts.statistics import (
    binomial_vs_chance,
    wilson_ci,
    two_proportion_ztest_vs_human,
    bonferroni_correction,
)


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


def resolve_frame_mode_policy(
    frame_mode: Optional[str],
    fps: float,
    max_frames: int,
) -> Tuple[float, int, bool, str]:
    """Resolve fps, max_frames, enforce_multi_frame, and mode key from FRAME_POLICY."""
    mode_key = (frame_mode or FRAME_POLICY.get("default_mode", "composite_grid")).strip()
    modes = FRAME_POLICY.get("modes") or {}
    if mode_key in modes:
        spec = modes[mode_key]
        return (
            float(spec.get("fps", fps)),
            int(spec.get("max_frames", max_frames)),
            bool(spec.get("enforce_multi_frame", True)),
            mode_key,
        )
    return fps, max_frames, bool(FRAME_POLICY.get("enforce_multi_frame", True)), mode_key


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


def _patch_remote_code_tied_weights(model: Any) -> None:
    """Instance-level fallback after load (device_map paths)."""
    if model is not None and not hasattr(model, "all_tied_weights_keys"):
        tied = getattr(model, "_tied_weights_keys", None)
        if tied is not None:
            model.all_tied_weights_keys = tied


def _from_pretrained_with_device_fallback(loader: Any, model_path: Path, *, device: str, dtype: torch.dtype) -> Any:
    """Load with device_map=auto on CUDA; retry without auto map on older remote-code models."""
    base_kw = {"torch_dtype": dtype, "trust_remote_code": True}
    if device == "cuda":
        try:
            model = loader.from_pretrained(model_path, device_map="auto", **base_kw)
            _patch_remote_code_tied_weights(model)
            return model
        except (AttributeError, TypeError, ValueError, RuntimeError):
            pass
    model = loader.from_pretrained(model_path, device_map=None, **base_kw)
    _patch_remote_code_tied_weights(model)
    if device == "cuda":
        model = model.to(device)
    return model


def load_hf_model_for_key(model_key: str, model_path: Path, device: str, dtype: torch.dtype) -> Any:
    """
    Best-effort loader across HF model class variants.
    """
    # Qwen2-VL dedicated class first (where available).
    if model_key == "qwen2vl":
        try:
            from transformers import Qwen2VLForConditionalGeneration  # type: ignore

            return _from_pretrained_with_device_fallback(
                Qwen2VLForConditionalGeneration, model_path, device=device, dtype=dtype
            )
        except Exception:
            pass

    if model_key == "gemma4":
        try:
            from transformers import AutoModelForMultimodalLM  # type: ignore

            return _from_pretrained_with_device_fallback(
                AutoModelForMultimodalLM, model_path, device=device, dtype=dtype
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
                return _from_pretrained_with_device_fallback(
                    cls, model_path, device=device, dtype=dtype
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

    # Last resort fallback (vision/multimodal keys must not use CausalLM).
    if model_key in {"qwen2vl", "llavanext", "gemma4"}:
        raise RuntimeError(
            f"Failed to load {model_key} from {model_path}. "
            "Check model files and transformers version on the compute node."
        )
    from transformers import AutoModelForCausalLM

    return _from_pretrained_with_device_fallback(
        AutoModelForCausalLM, model_path, device=device, dtype=dtype
    )


def _dataset_benchmark_key(dataset_key: str) -> str:
    return "eu_emotion" if dataset_key == "eu_emotions" else dataset_key


def _lookup_human_benchmark(dataset_key: str, condition: str) -> Optional[Dict[str, Any]]:
    ds = _dataset_benchmark_key(dataset_key)
    bench = (HUMAN_BENCHMARKS.get(ds) or {}).get(condition)
    if not bench:
        return None
    if bench.get("accuracy") is None or bench.get("n") is None:
        return None
    return bench


def _guard_no_leakage_audio(audio_path: Optional[Path]) -> None:
    if audio_path is not None and "/Emotions/Audio/" in str(audio_path).replace("\\", "/"):
        raise LeakageAudioPathError(f"Refusing leakage audio path: {audio_path}")


def load_trials_from_manifest(manifest_path: Path, dataset_root: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    trials_in = obj.get("trials", [])
    labels = sorted(
        {
            str(t.get("correct_label") or t.get("emotion") or t.get("label"))
            for t in trials_in
            if t.get("correct_label") or t.get("emotion") or t.get("label")
        }
    )
    trials: List[Dict[str, Any]] = []
    for t in trials_in:
        rel_path = t.get("stimulus_path") or t.get("video_path")
        if not rel_path:
            continue
        abs_path = Path(rel_path)
        if not abs_path.is_absolute():
            abs_path = (dataset_root / rel_path).resolve()
        label = t.get("correct_label") or t.get("emotion") or t.get("label")
        trials.append(
            {
                **t,
                "trial_id": t.get("trial_id", rel_path),
                "stimulus_path": str(abs_path),
                "label": label,
                "stimulus_relpath": rel_path if not Path(rel_path).is_absolute() else None,
            }
        )
    return trials, labels


def manifest_n_options(manifest_path: Path) -> Optional[int]:
    """Read optional top-level n_options from a manifest (e.g. 6 for study3 full-EU)."""
    obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw = obj.get("n_options")
    if raw is None:
        return None
    return int(raw)


def resolve_trial_media(
    trial: Dict[str, Any],
    *,
    dataset_key: str,
    dataset_root: Path,
    condition: str,
    seed: int,
) -> Tuple[Optional[Path], Optional[Path], str]:
    """Return (video_path, audio_path, audio_rule) for a trial under the given condition."""
    stimulus = Path(str(trial["stimulus_path"]))
    trial_id = str(trial.get("trial_id", stimulus.name))
    label = str(trial.get("label") or trial.get("correct_label") or trial.get("emotion") or "")

    video_path: Optional[Path] = None
    audio_path: Optional[Path] = None
    audio_rule = "none"

    if dataset_key == "mindreading":
        video_path = resolve_mindreading_v_video(stimulus)
    else:
        video_path = stimulus

    if condition == "video_only":
        return video_path, None, "video_only"

    if dataset_key == "eu_emotions":
        if condition == "audio_only":
            ap, audio_rule = resolve_eu_audio_only(
                emotion_label=label, base_data_dir=dataset_root, trial_id=trial_id, seed=seed
            )
            audio_path = ap
            video_path = None
        elif condition == "multimodal":
            ap, audio_rule = resolve_eu_multimodal_audio(
                video_path,
                emotion_label=label,
                base_data_dir=dataset_root,
                trial_id=trial_id,
                seed=seed,
            )
            audio_path = ap
    elif dataset_key == "mindreading" and condition in {"audio_only", "multimodal"}:
        ap, audio_rule = resolve_item_folder_audio(video_path)
        if ap is not None and ap.suffix.lower() == ".mov":
            ap = extract_audio_from_video(ap) or ap
        audio_path = ap
        if condition == "audio_only":
            video_path = None

    _guard_no_leakage_audio(audio_path)
    return video_path, audio_path, audio_rule


def write_results_csv(metrics: Dict[str, Any], csv_path: Path) -> None:
    import csv

    rows: List[Dict[str, Any]] = []
    for t in metrics.get("trials", []):
        s2 = t.get("stage2") or {}
        s1 = t.get("stage1") or {}
        rows.append(
            {
                "trial_id": t.get("trial_id"),
                "model": metrics.get("model"),
                "dataset": metrics.get("dataset"),
                "condition": metrics.get("condition"),
                "correct_label": t.get("label"),
                "predicted_label": s2.get("prediction"),
                "is_correct": s2.get("correct"),
                "semantic_entropy": s1.get("semantic_entropy"),
                "p_correct": s1.get("p_correct"),
                "margin_correct": s1.get("margin_correct"),
                "video_path": t.get("video_path"),
                "audio_path": t.get("audio_path"),
            }
        )
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


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
    return load_trials_from_manifest(manifest_path, dataset_root)


def run_evaluation(
    model_key: str,
    dataset_key: str,
    output_path: Path,
    seed: int = SEED,
    data_root: Optional[Path] = None,
    manifest: Optional[Path] = None,
    max_trials: Optional[int] = None,
    temperature: float = EVAL["temperature"],
    max_frames: int = FRAME_POLICY["max_frames"],
    fps: float = FRAME_POLICY["fps"],
    stage: str = "both",
    condition: str = "video_only",
    skip_entropy: bool = False,
    n_frames: Optional[int] = None,
    lora_adapter: Optional[Path] = None,
    stage2_prompt_mode: str = "4afc",
    frame_mode: Optional[str] = None,
    n_options: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Protocol v2 two-stage evaluation per trial.

    Stage 1 (semantic entropy) runs only for condition=video_only (cross-model comparability).
    Stage 2 uses condition-aware prompts and modality gating.
    n_options: Stage-2 forced-choice size (default 4; study3 full-EU uses 6). If None,
    uses manifest n_options when present, else 4.
    """
    if n_frames is not None:
        max_frames = int(n_frames)
    fps, max_frames, enforce_multi_frame, frame_mode_key = resolve_frame_mode_policy(
        frame_mode, fps, max_frames
    )
    stage = stage.lower()
    condition = (condition or "video_only").strip().lower()
    if condition not in MODALITY_CONDITIONS:
        raise ValueError(f"Invalid condition={condition}; use one of {MODALITY_CONDITIONS}")
    if stage not in {"both", "stage1", "stage2"}:
        raise ValueError(f"Invalid stage={stage}; use both|stage1|stage2")
    stage2_prompt_mode = (stage2_prompt_mode or "4afc").strip().lower()
    if stage2_prompt_mode not in {"4afc", "finetune_label"}:
        raise ValueError(f"Invalid stage2_prompt_mode={stage2_prompt_mode}; use 4afc|finetune_label")

    if condition in {"audio_only", "multimodal"} and not MODEL_AUDIO_CAPABILITIES.get(model_key, False):
        raise ValueError(
            f"Model {model_key} does not support native audio input in model_inference.py; "
            f"cannot run condition={condition}. Set MODEL_AUDIO_CAPABILITIES or use video_only."
        )

    dataset_root = resolve_dataset_root(dataset_key, override_root=data_root)
    if manifest is not None:
        trials_raw, labels = load_trials_from_manifest(manifest, dataset_root)
        if n_options is None:
            n_options = manifest_n_options(manifest)
    elif dataset_key == "eu_emotions":
        trials_raw, labels = list_eu_emotions_trials(dataset_root)
    else:
        trials_raw, labels = ([], [])

    if n_options is None:
        n_options = 4
    n_options = int(n_options)
    if n_options < 2:
        raise ValueError(f"n_options must be >= 2, got {n_options}")
    chance_level = 1.0 / float(n_options)

    trials_all = list(trials_raw)
    if max_trials is not None:
        trials_raw = trials_all[: int(max_trials)]
    else:
        trials_raw = trials_all

    if dataset_key == "eu_emotions":
        data_labels = None
        if data_root is not None:
            data_labels = Path(data_root).parent / "eu_emotion_states_list.txt"
        emotion_pool = resolve_eu_emotion_pool(
            label_paths=[EU_EMOTION_LABELS_FILE, data_labels],
            trials_fallback=trials_all,
        )
    else:
        emotion_pool = build_emotion_pool_from_trials(trials_raw)

    if condition in {"audio_only", "multimodal"}:
        audit_path = output_path.with_name(
            f"{model_key}_{dataset_key}_{condition}_audio_mapping_audit.json"
        )
        try:
            if dataset_key == "mindreading":
                audit = build_mr_audio_audit(trials_raw, base_data_dir=dataset_root)
                save_mr_audio_audit(audit, audit_path)
            elif dataset_key == "eu_emotions":
                audit = build_eu_audio_audit(
                    trials_raw, base_data_dir=dataset_root, condition=condition, seed=seed
                )
                save_eu_audio_audit(audit, audit_path)
        except Exception:
            traceback.print_exc()

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

    tokenizer = None
    model = load_hf_model_for_key(model_key=model_key, model_path=model_path, device=device, dtype=dtype)
    if lora_adapter is not None:
        from peft import PeftModel

        adapter_path = Path(lora_adapter)
        if not adapter_path.exists():
            raise FileNotFoundError(f"LoRA adapter not found: {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    if device != "cuda":
        model = model.to(device)
    model.eval()

    # Model-family specific compatibility tweaks.
    from scripts.model_compat import apply_llavanext_compat, is_peft_model

    apply_llavanext_compat(model, model_key)

    use_peft = lora_adapter is not None or is_peft_model(model)

    if dataset_key == "eu_emotions" and emotion_pool:
        entropy_labels = prepare_entropy_label_pool(emotion_pool, exclude=ENTROPY_EXCLUDE_LABELS)
    else:
        entropy_labels = list(labels)
    label_embeddings = None
    if entropy_labels and stage in {"both", "stage1"} and not skip_entropy:
        label_embeddings = load_or_compute_label_embeddings(
            entropy_labels,
            EMBEDDING_MODEL,
            rich_prompts=ENTROPY_USE_RICH_LABEL_EMBEDDINGS,
        )

    trials: List[Dict[str, Any]] = []
    n_correct = 0
    n_scored = 0
    entropies: List[float] = []
    pipe_cache: Dict[str, Any] = {}
    fp_tag = frame_policy_tag(fps, max_frames, frame_mode_key)

    for trial_idx, t in enumerate(trials_raw):
        trial_copy = dict(t)
        options = (
            resolve_candidate_labels(
                trial_copy, emotion_pool, seed=seed, trial_index=trial_idx, n_options=n_options
            )
            if stage in {"both", "stage2"}
            else []
        )
        video_path, audio_path, audio_rule = resolve_trial_media(
            trial_copy,
            dataset_key=dataset_key,
            dataset_root=dataset_root,
            condition=condition,
            seed=seed,
        )
        stage1_prompt = build_free_response_prompt(condition=condition)
        if options and stage2_prompt_mode == "finetune_label":
            stage2_prompt = build_finetune_prompt(condition=condition)
        else:
            stage2_prompt = build_4afc_prompt(options, condition=condition) if options else ""

        try:
            images: List[Any] = []
            frame_indices: List[int] = []
            mf_meta: Dict[str, Any] = {}
            images_for_processor: Any = None

            if video_path is not None and condition != "audio_only":
                images, frame_indices = load_stimulus_as_images(
                    video_path, fps=fps, max_frames=max_frames
                )
                images_for_processor, mf_meta = prepare_images_for_model(
                    model_key,
                    images,
                    enforce_multi_frame=enforce_multi_frame,
                )
            elif condition == "audio_only":
                if audio_path is None:
                    raise FileNotFoundError(
                        f"audio_only requires resolved audio for trial {t.get('trial_id')}"
                    )

            stage1_block: Dict[str, Any] = {}
            stage2_block: Dict[str, Any] = {}

            if stage in {"both", "stage1"}:
                out_stage1 = generate_model_response(
                    model_key=model_key,
                    model=model,
                    processor=processor,
                    tokenizer=tokenizer,
                    model_path=model_path,
                    prompt=stage1_prompt,
                    images=images,
                    images_for_processor=images_for_processor,
                    device=device,
                    dtype=dtype,
                    temperature=temperature,
                    max_new_tokens=STAGE1_MAX_NEW_TOKENS,
                    pipe_cache=pipe_cache,
                    audio_path=audio_path,
                    condition=condition,
                    prefer_loaded_model=use_peft,
                )
                free_text = strip_boilerplate_response(out_stage1)
                entropy_bundle: Dict[str, Any] = {}
                if not skip_entropy and label_embeddings is not None:
                    entropy_bundle = compute_entropy_bundle(
                        free_text,
                        entropy_labels,
                        true_label=t.get("label"),
                        label_embeddings=label_embeddings,
                        model_name=EMBEDDING_MODEL,
                        temperature=ENTROPY_TEMPERATURE,
                        log_base=ENTROPY_LOG_BASE,
                        rich_prompts=ENTROPY_USE_RICH_LABEL_EMBEDDINGS,
                        collapse_intensity=ENTROPY_COLLAPSE_INTENSITY,
                    )
                    h_sem = entropy_bundle.get("semantic_entropy")
                    if h_sem is not None and h_sem == h_sem:
                        entropies.append(float(h_sem))
                stage1_block = {
                    "prompt": stage1_prompt,
                    "free_response_text": free_text,
                    "raw_model_output": out_stage1,
                    "semantic_entropy": entropy_bundle.get("semantic_entropy") if not skip_entropy else None,
                    "semantic_entropy_fine": entropy_bundle.get("semantic_entropy_fine") if not skip_entropy else None,
                    "semantic_entropy_base": entropy_bundle.get("semantic_entropy_base") if not skip_entropy else None,
                    "label_probs": entropy_bundle.get("label_probs") if not skip_entropy else None,
                    "base_label_probs": entropy_bundle.get("base_label_probs") if not skip_entropy else None,
                    "base_labels": entropy_bundle.get("base_labels") if not skip_entropy else None,
                    "top_labels": entropy_bundle.get("top_labels") if not skip_entropy else None,
                    "p_correct": entropy_bundle.get("p_correct") if not skip_entropy else None,
                    "margin_correct": entropy_bundle.get("margin_correct") if not skip_entropy else None,
                    "correct_in_entropy_pool": entropy_bundle.get("correct_in_entropy_pool") if not skip_entropy else None,
                    "n_entropy_labels": entropy_bundle.get("n_entropy_labels") if not skip_entropy else None,
                    "embedding_model": EMBEDDING_MODEL,
                    "entropy_temperature": ENTROPY_TEMPERATURE,
                    "entropy_rich_label_embeddings": ENTROPY_USE_RICH_LABEL_EMBEDDINGS,
                    "entropy_exclude_labels": list(ENTROPY_EXCLUDE_LABELS),
                    "entropy_collapse_intensity": ENTROPY_COLLAPSE_INTENSITY,
                }

            if stage in {"both", "stage2"}:
                if CHAIN_STAGES and stage1_block.get("free_response_text"):
                    stage2_prompt = (
                        f"Prior description:\n{stage1_block['free_response_text']}\n\n{stage2_prompt}"
                    )
                out_stage2 = generate_model_response(
                    model_key=model_key,
                    model=model,
                    processor=processor,
                    tokenizer=tokenizer,
                    model_path=model_path,
                    prompt=stage2_prompt,
                    images=images,
                    images_for_processor=images_for_processor,
                    device=device,
                    dtype=dtype,
                    temperature=temperature,
                    max_new_tokens=STAGE2_MAX_NEW_TOKENS,
                    pipe_cache=pipe_cache,
                    audio_path=audio_path,
                    condition=condition,
                    prefer_loaded_model=use_peft,
                )
                pred, reasoning, parse_method = parse_emotion_tolerant(
                    out_stage2, options, full_label_pool=emotion_pool
                )
                if pred is None and stage2_prompt_mode == "4afc":
                    pred, reasoning = parse_emotion(out_stage2, options)
                    parse_method = "strict_4afc"
                if pred is None:
                    correct = False
                else:
                    correct = pred == t["label"]
                n_scored += 1
                n_correct += int(correct)
                stage2_block = {
                    "options": options,
                    "prompt": stage2_prompt,
                    "prompt_mode": stage2_prompt_mode,
                    "prediction": pred,
                    "correct": bool(correct),
                    "reasoning": reasoning,
                    "parse_method": parse_method if stage2_prompt_mode == "finetune_label" else None,
                    "raw_model_output": out_stage2,
                }

            trials.append(
                {
                    "trial_id": t.get("trial_id"),
                    "stimulus_path": t.get("stimulus_path"),
                    "stimulus_relpath": t.get("stimulus_relpath"),
                    "label": t.get("label"),
                    "video_path": str(video_path) if video_path else None,
                    "audio_path": str(audio_path) if audio_path else None,
                    "audio_resolution_rule": audio_rule,
                    "frame_indices": frame_indices,
                    "n_frames_used": mf_meta.get("n_frames_used", len(images)),
                    "multi_frame_strategy": mf_meta.get("multi_frame_strategy"),
                    "stage1": stage1_block if stage1_block else None,
                    "stage2": stage2_block if stage2_block else None,
                    "error": None,
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
                    "trial_id": t.get("trial_id"),
                    "stimulus_path": t.get("stimulus_path"),
                    "label": t.get("label"),
                    "stage1": None,
                    "stage2": {
                        "options": options,
                        "prediction": None,
                        "correct": None,
                    }
                    if stage in {"both", "stage2"}
                    else None,
                    "error": msg,
                    "traceback": traceback.format_exc(limit=6),
                }
            )

    n_trials = len(trials)
    if n_scored > 0:
        accuracy = n_correct / n_scored
        ci_low, ci_high = wilson_ci(n_correct, n_scored)
        p_binom = binomial_vs_chance(n_correct, n_scored, p0=chance_level)
    else:
        accuracy = None
        ci_low, ci_high = (None, None)
        p_binom = None

    human_bench = _lookup_human_benchmark(dataset_key, condition)
    p_vs_human = None
    p_vs_human_bonf = None
    if n_scored > 0 and human_bench is not None:
        h_acc = float(human_bench["accuracy"])
        h_n = int(human_bench["n"])
        p_vs_human = two_proportion_ztest_vs_human(
            n_correct, n_scored, int(round(h_acc * h_n)), h_n
        )
        p_vs_human_bonf = bonferroni_correction([p_vs_human], CONFIRMATORY_N_MODELS)[0]

    ent_arr = [x for x in entropies if x == x]
    if ent_arr:
        mean_sem = float(np.mean(ent_arr))
        median_sem = float(np.median(ent_arr))
        std_sem = float(np.std(ent_arr))
    else:
        mean_sem = median_sem = std_sem = None

    metrics: Dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "frame_policy": fp_tag,
        "frame_mode": frame_mode_key,
        "enforce_multi_frame": enforce_multi_frame,
        "fps": fps,
        "max_frames": max_frames,
        "primary_outcomes": ["accuracy", "semantic_entropy"],
        "accuracy": accuracy,
        "accuracy_wilson_ci_95": [ci_low, ci_high],
        "p_binom_gt_chance": p_binom,
        "p_vs_human_benchmark": p_vs_human,
        "p_vs_human_benchmark_bonferroni": p_vs_human_bonf,
        "human_benchmark": human_bench,
        "condition": condition,
        "stage1_policy": "free_response_semantic_entropy_all_conditions",
        "stage2_prompt_mode": stage2_prompt_mode,
        "n_options": n_options,
        "chance_level": chance_level,
        "mean_semantic_entropy": mean_sem,
        "median_semantic_entropy": median_sem,
        "std_semantic_entropy": std_sem,
        "n_trials": n_trials,
        "n_scored": n_scored,
        "n_correct": n_correct,
        "seed": seed,
        "model": model_key,
        "dataset": dataset_key,
        "temperature": float(temperature),
        "embedding_model": EMBEDDING_MODEL,
        "entropy_temperature": ENTROPY_TEMPERATURE,
        "entropy_log_base": ENTROPY_LOG_BASE,
        "entropy_exclude_labels": list(ENTROPY_EXCLUDE_LABELS),
        "entropy_rich_label_embeddings": ENTROPY_USE_RICH_LABEL_EMBEDDINGS,
        "entropy_collapse_intensity": ENTROPY_COLLAPSE_INTENSITY,
        "entropy_definition": (
            "primary semantic_entropy = H over base emotions after softmax on "
            f"{len(entropy_labels)} fine labels (neutral excluded), rich label prompts, "
            "intensity collapsed"
            if ENTROPY_COLLAPSE_INTENSITY
            else "semantic_entropy over fine labels (neutral excluded)"
        ),
        "chain_stages": CHAIN_STAGES,
        "stage": stage,
        "device": str(device),
        "dataset_root": str(dataset_root),
        "manifest": str(manifest) if manifest is not None else None,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluator_version": "protocol-v2-two-stage",
        "run_metadata": collect_run_metadata(),
        "lora_adapter": str(lora_adapter) if lora_adapter is not None else None,
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
        "--max_frames",
        type=int,
        default=FRAME_POLICY["max_frames"],
        help="Max frames per video (protocol v2: 1 fps, cap at this value).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=FRAME_POLICY["fps"],
        help="Frames per second of video duration to sample.",
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=None,
        help="Deprecated alias for --max_frames.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="both",
        choices=["both", "stage1", "stage2"],
        help="Run Stage 1 (free response), Stage 2 (N-AFC), or both.",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="video_only",
        choices=list(MODALITY_CONDITIONS),
        help="Modality ablation: video_only, audio_only, or multimodal.",
    )
    parser.add_argument(
        "--skip_entropy",
        action="store_true",
        help="Debug only: skip semantic entropy computation.",
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
        "--n_options",
        type=int,
        default=None,
        help="Stage-2 forced-choice size (default: manifest n_options if set, else 4). Use 6 for study3.",
    )
    parser.add_argument(
        "--stage2_prompt_mode",
        type=str,
        default="4afc",
        choices=["4afc", "finetune_label"],
        help="Stage 2 prompt: standard N-AFC (4afc mode) or finetune-style LABEL: (in-distribution readout).",
    )
    parser.add_argument(
        "--frame_mode",
        type=str,
        default=None,
        choices=list(FRAME_POLICY.get("modes", {}).keys()) or None,
        help="Frame presentation mode: composite_grid (default) or native_video.",
    )
    parser.add_argument(
        "--lora_adapter",
        type=Path,
        default=None,
        help="Path to PEFT LoRA adapter directory (fine-tuned checkpoint).",
    )
    args = parser.parse_args()
    max_frames = args.max_frames if args.n_frames is None else args.n_frames
    fps, max_frames, _, frame_mode_key = resolve_frame_mode_policy(
        args.frame_mode, args.fps, max_frames
    )
    fp = frame_policy_tag(fps, max_frames, frame_mode_key)

    if args.output is None:
        default_dir = LOCAL_RESULTS_DIR / "baseline" / args.dataset / args.model
        default_dir.mkdir(parents=True, exist_ok=True)
        if args.stage2_prompt_mode == "finetune_label":
            default_dir = LOCAL_RESULTS_DIR / "finetune" / "eu_post_ft"
            args.output = (
                default_dir
                / f"eval_v2_{args.dataset}_{args.model}_{args.condition}_finetune_prompt_seed{args.seed}.json"
            )
        else:
            args.output = (
                default_dir
                / f"eval_v2_{args.dataset}_{args.model}_{args.condition}_{fp}_two_stage_seed{args.seed}.json"
            )
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    metrics = run_evaluation(
        model_key=args.model,
        dataset_key=args.dataset,
        output_path=args.output,
        seed=args.seed,
        data_root=args.data_root,
        manifest=args.manifest,
        max_trials=args.max_trials,
        temperature=args.temperature,
        max_frames=max_frames,
        fps=args.fps,
        stage=args.stage,
        condition=args.condition,
        skip_entropy=args.skip_entropy,
        lora_adapter=args.lora_adapter,
        stage2_prompt_mode=args.stage2_prompt_mode,
        frame_mode=args.frame_mode,
        n_options=args.n_options,
    )

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True, default=str)
        f.write("\n")

    csv_path = args.output.with_name(
        f"{args.model}_{args.dataset}_{args.condition}_results.csv"
    )
    write_results_csv(metrics, csv_path)


if __name__ == "__main__":
    main()

