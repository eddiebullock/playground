"""Load MCQ items from manifests or synthetic fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from esvlm.tasks.prompts import render_prompt
from esvlm.tasks.synthetic import items_from_fixtures
from esvlm.types import Domain, MCQItem, MediaType


def load_synthetic_items(root: Path, *, template_id: str = "color") -> list[MCQItem]:
    raw = items_from_fixtures(root)
    out: list[MCQItem] = []
    for item in raw:
        tid = "shape" if item.meta.get("kind") == "shape" else template_id
        prompt = render_prompt("synthetic", tid if tid in {"color", "shape", "numbered"} else "color", item.options)
        out.append(
            MCQItem(
                item_id=item.item_id,
                task=item.task,
                domain=item.domain,
                media=item.media,
                media_type=item.media_type,
                prompt_text=prompt,
                options=item.options,
                correct_idx=item.correct_idx,
                meta={**item.meta, "template_id": tid},
            )
        )
    return out


def resolve_media(media_root: Path, rel: str) -> str:
    """Resolve a manifest path in place. Never copy the file."""

    if not rel:
        return ""
    direct = media_root / rel
    if direct.is_file():
        return str(direct.resolve())
    named = media_root / Path(rel).name
    if named.is_file():
        return str(named.resolve())
    found = next(media_root.rglob(Path(rel).name), None)
    if found is not None and found.is_file():
        return str(found.resolve())
    return str(direct.resolve())


def parse_manifest_trial(
    trial: dict[str, Any],
    *,
    task: str,
    domain: Domain,
    media_root: Path,
    default_media_type: MediaType,
    template_id: str,
    prompt_text: str | None = None,
) -> MCQItem:
    """Map a heterogeneous on-disk trial dict onto ``MCQItem``.

    Licensed item text is not stored here. Options come from the existing
    manifest on disk and are not committed.
    """

    item_id = str(trial.get("item_id") or trial.get("item") or trial.get("trial_id"))
    options = tuple(str(o) for o in trial["options"])
    if "correct_idx" in trial:
        correct_idx = int(trial["correct_idx"])
    else:
        label = trial.get("correct_label") or trial.get("correct") or trial.get("label")
        correct_idx = options.index(str(label))
    media_rel = (
        trial.get("image")
        or trial.get("media")
        or trial.get("video")
        or trial.get("stimulus_path")
        or trial.get("stimulus_relpath")
        or ""
    )
    media = resolve_media(media_root, str(media_rel)) if media_rel else ""
    media_type: MediaType = trial.get("media_type") or default_media_type
    if not media:
        media_type = "none"
    return MCQItem(
        item_id=item_id,
        task=task,
        domain=domain,
        media=media,
        media_type=media_type,
        prompt_text=prompt_text or "",
        options=options,
        correct_idx=correct_idx,
        meta={"template_id": template_id, "raw_keys": sorted(trial)},
    )


def load_json_manifest(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def load_rmet_items(
    manifest_path: Path,
    answer_key_path: Path,
    stim_root: Path,
    *,
    template_id: str = "classic",
) -> list[MCQItem]:
    """Read RMET ``manifest.json`` + answer key in place. No copies."""

    from esvlm.tasks.prompts import render_prompt

    manifest = load_json_manifest(manifest_path)
    key = load_json_manifest(answer_key_path)
    answers = key.get("answers") or key
    trials = manifest.get("trials") or manifest.get("items") or []
    items: list[MCQItem] = []
    for trial in trials:
        item_no = trial.get("item") or trial.get("item_id")
        if "options" not in trial:
            raise ValueError(f"RMET trial {item_no} missing options")
        if "correct_label" not in trial and "correct_idx" not in trial:
            label = answers.get(str(item_no)) or answers.get(int(item_no))
            trial = {**trial, "correct_label": label}
        prompt = render_prompt("rmet", template_id, trial["options"])
        items.append(
            parse_manifest_trial(
                trial,
                task="rmet",
                domain="E",
                media_root=stim_root,
                default_media_type="image",
                template_id=template_id,
                prompt_text=prompt,
            )
        )
    return items


def load_eu_emotions_items(
    manifest_path: Path,
    stim_root: Path,
    *,
    template_id: str = "direct",
    entropy_path: Path | None = None,
) -> list[MCQItem]:
    """Full EU-Emotions manifest. Never the 118-trial subset."""

    from esvlm.tasks.prompts import render_prompt

    if "118" in manifest_path.name:
        raise ValueError(
            f"refusing study-subset manifest {manifest_path.name}; "
            "use eu_emotions_full_manifest.json"
        )
    manifest = load_json_manifest(manifest_path)
    entropy: dict[str, Any] = {}
    if entropy_path is not None and entropy_path.exists():
        payload = load_json_manifest(entropy_path)
        entropy = payload.get("trials") or payload
    trials = manifest.get("trials") or manifest.get("items") or []
    items: list[MCQItem] = []
    for trial in trials:
        extra = entropy.get(str(trial.get("trial_id") or trial.get("item_id") or "")) or {}
        options = [str(o) for o in (trial.get("options") or extra.get("human_options") or [])]
        label = str(trial.get("correct_label") or trial.get("label") or extra.get("manifest_label") or "")
        if label and label not in options:
            options = [label, *options]
        if len(options) < 2:
            raise ValueError(f"EU-Emotions trial {trial.get('trial_id')} has no option set")
        trial = {**trial, "options": options, "correct_label": label or trial.get("correct_label")}
        prompt = render_prompt("eu_emotions", template_id, trial["options"])
        item = parse_manifest_trial(
            trial,
            task="eu_emotions",
            domain="E",
            media_root=stim_root,
            default_media_type="video",
            template_id=template_id,
            prompt_text=prompt,
        )
        tid = item.item_id
        extra = entropy.get(tid) or entropy.get(str(tid)) or {}
        items.append(
            MCQItem(
                item_id=item.item_id,
                task=item.task,
                domain=item.domain,
                media=item.media,
                media_type=item.media_type,
                prompt_text=item.prompt_text,
                options=item.options,
                correct_idx=item.correct_idx,
                meta={**item.meta, "human_entropy": extra},
            )
        )
    return items


def load_mindreading_items(
    manifest_path: Path,
    stim_root: Path,
    *,
    template_id: str = "direct",
) -> list[MCQItem]:
    """Mind Reading item dirs / jsonl. Requires a forced-choice option set."""

    import json

    from esvlm.tasks.prompts import render_prompt

    text = manifest_path.read_text(encoding="utf-8")
    if manifest_path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        data = json.loads(text)
        rows = data.get("trials") or data.get("items") or data
        if not isinstance(rows, list):
            raise ValueError("mindreading manifest has no trial list")
    items: list[MCQItem] = []
    for trial in rows:
        options = trial.get("options") or trial.get("choices")
        if not options or len(options) < 2:
            raise ValueError(
                "mindreading dump has no MCQ options; not scored until a "
                "forced-choice form exists"
            )
        trial = {**trial, "options": list(options)}
        prompt = render_prompt("mindreading", template_id, trial["options"])
        items.append(
            parse_manifest_trial(
                trial,
                task="mindreading",
                domain="E",
                media_root=stim_root,
                default_media_type="video",
                template_id=template_id,
                prompt_text=prompt,
            )
        )
    return items
