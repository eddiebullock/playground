"""
Commercial API arm for study4 RMET (A1 / A3 only — no activations).

Same stimuli, prompt, scoring schema as evaluate_rmet.py so alignment_analyses.py
can ingest the JSON/CSV unchanged.

Models (study keys → provider API ids):
  gpt5          → gpt-5                 (OpenAI)
  claude_opus   → claude-opus-4-5       (Anthropic)
  gemini_flash  → gemini-3-flash-preview (Google)

API keys loaded from (first hit wins, no override of already-set env):
  1) process environment
  2) study4_rmet/.env
  3) mr_ts_play/experiments/cam_human_like/training/.env
  4) mr_ts_play/.env (if present)

Usage (from mr_eu_open_llm repo root):
  python study4_rmet/scripts/evaluate_rmet_api.py --model gpt5 --max_items 3
  python study4_rmet/scripts/evaluate_rmet_api.py --model claude_opus
  python study4_rmet/scripts/evaluate_rmet_api.py --model gemini_flash

Or run the panel:
  ./study4_rmet/scripts/run_rmet_api_panel.sh
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

STUDY4_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from parse_rmet_emotion import parse_rmet_emotion  # noqa: E402
from rmet_prompts import build_rmet_4afc_prompt  # noqa: E402

# Short study keys → provider + API model id
MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "gpt5": {
        "provider": "openai",
        "api_model": "gpt-5",
        "env_key": "OPENAI_API_KEY",
    },
    "claude_opus": {
        "provider": "anthropic",
        "api_model": "claude-opus-4-5",
        "env_key": "ANTHROPIC_API_KEY",
    },
    "gemini_flash": {
        "provider": "google",
        "api_model": "gemini-3-flash-preview",
        "env_key": "GOOGLE_API_KEY",
    },
}

DOTENV_CANDIDATES = [
    STUDY4_ROOT / ".env",
    Path("/Users/eb2007/playground/bullpy/mr_ts_play/experiments/cam_human_like/training/.env"),
    Path("/Users/eb2007/playground/bullpy/mr_ts_play/.env"),
]


def _load_dotenv_files() -> List[str]:
    """
    Load keys from candidate .env files.
    study4_rmet/.env always wins (override=True) so a fresh Gemini key there
    replaces a stale key from mr_ts_play.
    Other candidates only fill missing vars.
    """
    loaded: List[str] = []
    study4_env = STUDY4_ROOT / ".env"
    others = [p for p in DOTENV_CANDIDATES if p.resolve() != study4_env.resolve()]

    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None  # type: ignore

    if load_dotenv is not None:
        for p in others:
            if p.is_file():
                load_dotenv(dotenv_path=p, override=False)
                loaded.append(str(p))
        if study4_env.is_file():
            load_dotenv(dotenv_path=study4_env, override=True)
            loaded.append(str(study4_env) + " (override)")
        return loaded

    # Minimal fallback parser (KEY=VALUE)
    def _parse(p: Path, *, override: bool) -> None:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if not k:
                continue
            if override or k not in os.environ:
                os.environ[k] = v

    for p in others:
        if p.is_file():
            _parse(p, override=False)
            loaded.append(str(p))
    if study4_env.is_file():
        _parse(study4_env, override=True)
        loaded.append(str(study4_env) + " (override)")
    return loaded


def _entropy(counts: Counter, options: Sequence[str]) -> float:
    total = sum(counts[o] for o in options)
    if total <= 0:
        return float("nan")
    h = 0.0
    for o in options:
        p = counts[o] / total
        if p > 0:
            h -= p * math.log(p)
    return float(h)


def _image_b64(path: Path) -> tuple[str, str]:
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return b64, mime


def _resolve_image(trial: Dict[str, Any], stim_root: Path, manifest_path: Path) -> Path:
    img_rel = trial["image"]
    name = Path(img_rel).name
    for cand in (
        stim_root / name,
        STUDY4_ROOT / img_rel,
        manifest_path.parent / name,
    ):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Missing stimulus for item {trial['item']}: {img_rel}")


def _call_openai(*, api_model: str, prompt: str, image_b64: str, mime: str, temperature: Optional[float], max_tokens: int) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    content = [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{image_b64}", "detail": "high"},
        },
    ]
    kwargs: Dict[str, Any] = {
        "model": api_model,
        "messages": [{"role": "user", "content": content}],
    }
    # GPT-5 family: OpenAI rejects non-default temperature; use max_completion_tokens.
    # Reasoning models can consume a large share of the budget before visible text —
    # give them headroom (mr_ts_play uses 1000; we use at least 2048 for gpt-5).
    if api_model.startswith("gpt-5"):
        kwargs["max_completion_tokens"] = max(int(max_tokens), 2048)
    else:
        kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = float(temperature)

    resp = client.chat.completions.create(**kwargs)
    if not resp.choices:
        return ""
    msg = resp.choices[0].message
    text = getattr(msg, "content", None) or ""
    if not text:
        # Some SDK versions expose refusal / nested content parts
        refusal = getattr(msg, "refusal", None)
        if refusal:
            text = str(refusal)
        else:
            parts = getattr(msg, "content", None)
            if isinstance(parts, list):
                text = "\n".join(
                    str(p.get("text", p)) if isinstance(p, dict) else getattr(p, "text", str(p))
                    for p in parts
                )
    if not text.strip():
        # Treat empty visible content as a retryable failure (common when
        # reasoning tokens exhaust the completion budget).
        finish = getattr(resp.choices[0], "finish_reason", None)
        raise RuntimeError(f"OpenAI returned empty content (finish_reason={finish})")
    return text


def _call_anthropic(*, api_model: str, prompt: str, image_b64: str, mime: str, temperature: float, max_tokens: int) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    content = [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": mime, "data": image_b64},
        },
        {"type": "text", "text": prompt},
    ]
    resp = client.messages.create(
        model=api_model,
        max_tokens=max_tokens,
        temperature=float(temperature),
        messages=[{"role": "user", "content": content}],
    )
    blocks = getattr(resp, "content", []) or []
    texts = [b.text for b in blocks if getattr(b, "type", None) == "text" and getattr(b, "text", None)]
    return "\n".join(texts) if texts else str(resp)


def _call_gemini(*, api_model: str, prompt: str, image_b64: str, mime: str, temperature: float, max_tokens: int) -> str:
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    try:
        from google.generativeai.types import HarmBlockThreshold, HarmCategory

        safety_settings = [
            {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
            {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
            {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_NONE},
            {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
        ]
    except Exception:
        safety_settings = None

    model_id = api_model if api_model.startswith("models/") else f"models/{api_model}"
    model = genai.GenerativeModel(model_name=model_id, safety_settings=safety_settings)
    parts = [
        prompt,
        {"inline_data": {"mime_type": mime, "data": base64.b64decode(image_b64)}},
    ]
    resp = model.generate_content(
        parts,
        generation_config={"temperature": float(temperature), "max_output_tokens": int(max_tokens)},
    )
    return getattr(resp, "text", "") or ""


def call_vision_model(
    *,
    provider: str,
    api_model: str,
    prompt: str,
    image_path: Path,
    temperature: Optional[float],
    max_tokens: int,
    retries: int = 6,
) -> str:
    image_b64, mime = _image_b64(image_path)
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            if provider == "openai":
                return _call_openai(
                    api_model=api_model,
                    prompt=prompt,
                    image_b64=image_b64,
                    mime=mime,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            if provider == "anthropic":
                return _call_anthropic(
                    api_model=api_model,
                    prompt=prompt,
                    image_b64=image_b64,
                    mime=mime,
                    temperature=float(temperature if temperature is not None else 0.0),
                    max_tokens=max_tokens,
                )
            if provider == "google":
                return _call_gemini(
                    api_model=api_model,
                    prompt=prompt,
                    image_b64=image_b64,
                    mime=mime,
                    temperature=float(temperature if temperature is not None else 0.0),
                    max_tokens=max_tokens,
                )
            raise ValueError(f"Unknown provider {provider}")
        except Exception as e:
            last_err = e
            err = str(e).lower()
            # Hard stop: bad key / daily quota — retries won't help
            if (
                "api_key_invalid" in err
                or "api key not valid" in err
                or "incorrect api key" in err
                or "invalid_api_key" in err
                or "per_day" in err
                or "generaterequestsperday" in err
            ):
                raise RuntimeError(
                    f"Non-retryable API error for {provider}/{api_model}: {e}\n"
                    "Fix: put a valid key in study4_rmet/.env "
                    "(GOOGLE_API_KEY or GEMINI_API_KEY for Gemini; "
                    "OPENAI_API_KEY / ANTHROPIC_API_KEY for the others)."
                ) from e
            if attempt == retries:
                break
            delay = min(60.0, 2.0 ** (attempt - 1))
            print(f"  retry {attempt}/{retries} after {delay:.1f}s ({type(e).__name__}: {e})", flush=True)
            time.sleep(delay)
    raise RuntimeError(f"API call failed after {retries} attempts: {last_err}")


def _write_outputs(payload: Dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    table_path = output.with_suffix(".csv")
    with table_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "item",
                "correct_label",
                "det_prediction",
                "det_correct",
                "det_parse_fail",
                "sample_accuracy",
                "sample_entropy",
                "sample_parse_failures",
            ],
        )
        w.writeheader()
        for t in payload["trials"]:
            w.writerow(
                {
                    "item": t["item"],
                    "correct_label": t["correct_label"],
                    "det_prediction": t["deterministic"]["prediction"],
                    "det_correct": int(t["deterministic"]["correct"]),
                    "det_parse_fail": int(t["deterministic"]["parse_fail"]),
                    "sample_accuracy": t["samples"]["accuracy"],
                    "sample_entropy": t["samples"]["entropy"],
                    "sample_parse_failures": t["samples"]["parse_failures"],
                }
            )
    payload["table_csv"] = str(table_path)


def _summarize(trials: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_scored = sum(1 for t in trials if not t["deterministic"]["parse_fail"])
    n_correct = sum(1 for t in trials if t["deterministic"]["correct"])
    return {
        "n_items": len(trials),
        "n_scored_deterministic": n_scored,
        "accuracy_deterministic": (n_correct / n_scored) if n_scored else None,
    }


def run_eval(
    *,
    model_key: str,
    manifest_path: Path,
    stim_root: Path,
    output: Path,
    seed: int = 42,
    temperature_det: float = 0.0,
    n_samples: int = 10,
    sample_temperature: float = 0.7,
    max_items: Optional[int] = None,
    max_tokens: int = 256,
    request_pause_s: float = 0.4,
    resume: bool = True,
) -> Dict[str, Any]:
    if model_key not in MODEL_REGISTRY:
        raise SystemExit(f"Unknown model {model_key}; choose from {sorted(MODEL_REGISTRY)}")

    meta = MODEL_REGISTRY[model_key]
    env_key = meta["env_key"]
    # Gemini: accept either GOOGLE_API_KEY or GEMINI_API_KEY
    if meta["provider"] == "google":
        if not os.environ.get("GOOGLE_API_KEY") and os.environ.get("GEMINI_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
    if not os.environ.get(env_key):
        raise SystemExit(
            f"Missing {env_key} in environment / .env (searched {DOTENV_CANDIDATES}). "
            "For Gemini you may also set GEMINI_API_KEY."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    trials_spec = list(manifest["trials"])
    if max_items is not None:
        trials_spec = trials_spec[: int(max_items)]

    done_by_item: Dict[int, Dict[str, Any]] = {}
    if resume and output.exists():
        try:
            prev = json.loads(output.read_text(encoding="utf-8"))
            for t in prev.get("trials", []):
                done_by_item[int(t["item"])] = t
            print(f"Resuming {output}: {len(done_by_item)} items already done", flush=True)
        except Exception as e:
            print(f"Could not resume from {output}: {e}", flush=True)

    provider = meta["provider"]
    api_model = meta["api_model"]
    # GPT-5: provider rejects temperature != default; omit for both det and samples.
    gpt5_style = api_model.startswith("gpt-5")

    results_trials: List[Dict[str, Any]] = []
    for trial in trials_spec:
        item = int(trial["item"])
        if item in done_by_item:
            results_trials.append(done_by_item[item])
            continue

        options = list(trial["options"])
        correct_label = trial["correct_label"]
        img_path = _resolve_image(trial, stim_root, manifest_path)
        prompt = build_rmet_4afc_prompt(options)

        print(f"[{model_key}] item {item:02d}/{len(trials_spec)} …", flush=True)
        det_temp: Optional[float] = None if gpt5_style else float(temperature_det)
        det_text = call_vision_model(
            provider=provider,
            api_model=api_model,
            prompt=prompt,
            image_path=img_path,
            temperature=det_temp,
            max_tokens=max_tokens,
        )
        time.sleep(request_pause_s)
        det_pred, det_reason = parse_rmet_emotion(det_text, options)
        det_correct = bool(det_pred is not None and det_pred.lower() == correct_label.lower())

        sample_preds: List[Optional[str]] = []
        sample_texts: List[str] = []
        samp_temp: Optional[float] = None if gpt5_style else float(sample_temperature)
        for _ in range(int(n_samples)):
            t = call_vision_model(
                provider=provider,
                api_model=api_model,
                prompt=prompt,
                image_path=img_path,
                temperature=samp_temp,
                max_tokens=max_tokens,
            )
            time.sleep(request_pause_s)
            sample_texts.append(t)
            pred, _ = parse_rmet_emotion(t, options)
            sample_preds.append(pred)

        counts = Counter(p for p in sample_preds if p is not None)
        parse_fail = sum(1 for p in sample_preds if p is None)
        ent = _entropy(counts, options) if sample_preds else float("nan")
        n_ok = sum(1 for p in sample_preds if p is not None)
        sample_acc = (
            sum(1 for p in sample_preds if p is not None and p.lower() == correct_label.lower()) / max(1, n_ok)
        )

        results_trials.append(
            {
                "trial_id": trial["trial_id"],
                "item": item,
                "image": str(img_path),
                "options": options,
                "correct_label": correct_label,
                "multi_frame_strategy": "single_image_api",
                "prompt": prompt,
                "deterministic": {
                    "temperature": det_temp,
                    "raw_output": det_text,
                    "prediction": det_pred,
                    "reasoning": det_reason,
                    "correct": det_correct,
                    "parse_fail": det_pred is None,
                    "note": "gpt-5 uses API default temperature" if gpt5_style else None,
                },
                "samples": {
                    "n_samples": int(n_samples),
                    "temperature": samp_temp,
                    "predictions": sample_preds,
                    "raw_outputs": sample_texts,
                    "distribution": {k: int(v) for k, v in counts.items()},
                    "parse_failures": int(parse_fail),
                    "accuracy": float(sample_acc) if sample_preds else None,
                    "entropy": ent,
                },
            }
        )

        # Checkpoint after each item
        stats = _summarize(results_trials)
        payload = {
            "study": "study4_rmet",
            "arm": "commercial_api",
            "model": model_key,
            "api_model": api_model,
            "provider": provider,
            "condition": "image_only",
            "seed": seed,
            "chance_level": 0.25,
            "n_samples": int(n_samples),
            "sample_temperature": samp_temp,
            "manifest": str(manifest_path),
            "a2_activations": False,
            **stats,
            "trials": results_trials,
        }
        _write_outputs(payload, output)

    stats = _summarize(results_trials)
    payload = {
        "study": "study4_rmet",
        "arm": "commercial_api",
        "model": model_key,
        "api_model": api_model,
        "provider": provider,
        "condition": "image_only",
        "seed": seed,
        "chance_level": 0.25,
        "n_samples": int(n_samples),
        "sample_temperature": None if gpt5_style else float(sample_temperature),
        "manifest": str(manifest_path),
        "a2_activations": False,
        **stats,
        "trials": results_trials,
    }
    _write_outputs(payload, output)
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    loaded = _load_dotenv_files()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY.keys()))
    ap.add_argument(
        "--manifest",
        type=Path,
        default=STUDY4_ROOT / "data" / "rmet" / "stimuli" / "manifest.json",
    )
    ap.add_argument(
        "--stim_root",
        type=Path,
        default=STUDY4_ROOT / "data" / "rmet" / "stimuli",
    )
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_samples", type=int, default=10)
    ap.add_argument("--sample_temperature", type=float, default=0.7)
    ap.add_argument("--temperature_det", type=float, default=0.0)
    ap.add_argument("--max_items", type=int, default=None, help="Smoke: limit items")
    ap.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="Max output tokens (gpt-5 uses max(this, 2048) as max_completion_tokens)",
    )
    ap.add_argument("--request_pause_s", type=float, default=0.4)
    ap.add_argument("--no_resume", action="store_true")
    args = ap.parse_args(argv)

    print(f"dotenv loaded from: {loaded or '(none — using process env only)'}", flush=True)

    out = args.output
    if out is None:
        tag = f"max{args.max_items}" if args.max_items else "full"
        out = (
            STUDY4_ROOT
            / "results"
            / "model"
            / args.model
            / f"rmet_eval_{args.model}_{tag}_seed{args.seed}.json"
        )

    payload = run_eval(
        model_key=args.model,
        manifest_path=args.manifest,
        stim_root=args.stim_root,
        output=out,
        seed=args.seed,
        temperature_det=args.temperature_det,
        n_samples=args.n_samples,
        sample_temperature=args.sample_temperature,
        max_items=args.max_items,
        max_tokens=args.max_tokens,
        request_pause_s=args.request_pause_s,
        resume=not args.no_resume,
    )
    print(
        f"model={payload['model']} api={payload['api_model']} "
        f"acc_det={payload['accuracy_deterministic']} n={payload['n_items']} -> {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
