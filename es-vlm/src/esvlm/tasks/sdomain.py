"""Public S-domain tasks: PhysBench, CLEVRER, PIQA.

Media and source JSON stay under $ESVLM_PUBLIC_DIR, never in a licensed
stimuli tree. Loaders only keep items with textual options so sequential
option log-probs work.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from esvlm.tasks.prompts import render_prompt
from esvlm.types import MCQItem, MediaType

PHYS_KEEP_TASK = frozenset({"relationships", "relationship", "dynamics", "phenomena"})
PHYS_KEEP_SUB = frozenset(
    {
        "collision",
        "support",
        "containment",
        "motion",
        "fluid",
        "deform",
        "stability",
        "rolling",
        "sliding",
        "falling",
        "spatial",
        "prediction",
    }
)
PHYS_KEEP_ABILITY = frozenset({"prediction", "causality", "dynamics", "relationship"})
CLEVRER_MCQ_TYPES = frozenset({"explanatory", "predictive", "counterfactual"})
LETTER_RE = re.compile(r"(?m)^\s*([A-D])\.\s*(.*)$")
IMAGE_ONLY = frozenset({"", "<image>", "<video>", "<image>.", "<video>."})


def clevrer_item_id(media: str, qid: object) -> str:
    """Stable id: CLEVRER question_id repeats inside every video."""

    stem = Path(str(media)).stem or "video"
    return f"clevrer_{stem}_{qid}"


def uniquify_clevrer_item_id(item_id: str, media: str) -> str:
    """Rewrite legacy `clevrer_<qid>` ids that collide across videos."""

    stem = Path(str(media)).stem
    if not stem:
        return item_id
    prefix = f"clevrer_{stem}_"
    if item_id.startswith(prefix):
        return item_id
    qid = item_id.removeprefix("clevrer_")
    return f"{prefix}{qid}"


def _resolve_media_rel(media_root: Path, name: str) -> str:
    direct = media_root / name
    if direct.is_file():
        return name
    for folder in ("video", "image", "videos", "images"):
        candidate = media_root / folder / name
        if candidate.is_file():
            return str(candidate.relative_to(media_root))
    found = next(media_root.rglob(Path(name).name), None)
    if found is not None and found.is_file():
        try:
            return str(found.relative_to(media_root))
        except ValueError:
            return str(found)
    return name


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def parse_lettered_options(question: str) -> tuple[str, list[str]]:
    """Split a PhysBench stem into prompt text and A-D option strings."""

    matches = list(LETTER_RE.finditer(question))
    if len(matches) < 2:
        return question.strip(), []
    stem = question[: matches[0].start()].strip()
    options = [m.group(2).strip() for m in matches]
    return stem, options


def is_text_option(text: str) -> bool:
    return text.strip().lower() not in IMAGE_ONLY and "<image>" not in text.lower()


def merge_physbench_labels(rows: list[dict[str, Any]], labeled: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach answers / task tags from a labeled dump onto media rows."""

    by_idx: dict[int, dict[str, Any]] = {}
    for extra in labeled:
        if extra.get("idx") is None:
            continue
        by_idx[int(extra["idx"])] = extra
    out: list[dict[str, Any]] = []
    for row in rows:
        merged = dict(row)
        extra = by_idx.get(int(row["idx"])) if row.get("idx") is not None else None
        if extra:
            for key in ("answer", "task_type", "sub_type", "ability_type", "question", "file_name"):
                if not merged.get(key) and extra.get(key) not in (None, ""):
                    merged[key] = extra[key]
        out.append(merged)
    return out


def keep_physbench_item(row: dict[str, Any]) -> bool:
    task = _norm(row.get("task_type") or row.get("task"))
    sub = _norm(row.get("sub_type") or row.get("subtype"))
    ability = _norm(row.get("ability_type") or row.get("ability"))
    if task:
        return task in PHYS_KEEP_TASK
    if sub in PHYS_KEEP_SUB or ability in PHYS_KEEP_ABILITY:
        return True
    blob = " ".join((task, sub, ability))
    return any(mark in blob for mark in ("dynamic", "relation", "collis", "caus"))


def physbench_to_trial(row: dict[str, Any], *, media_root: Path) -> dict[str, Any] | None:
    if not keep_physbench_item(row):
        return None
    stem, options = parse_lettered_options(str(row.get("question") or ""))
    if len(options) < 2 or not all(is_text_option(opt) for opt in options):
        return None
    answer = str(row.get("answer") or "").strip().upper()
    if answer not in "ABCD"[: len(options)]:
        return None
    names = row.get("file_name") or []
    if isinstance(names, str):
        names = [names]
    media_name = next((n for n in names if str(n).strip()), "")
    if not media_name:
        return None
    suffix = Path(str(media_name)).suffix.lower()
    media_type: MediaType = "video" if suffix in {".mp4", ".mov", ".webm", ".avi"} else "image"
    rel = _resolve_media_rel(media_root, str(media_name))
    item_key = row.get("idx", row.get("id", Path(rel).stem))
    return {
        "item_id": f"physbench_{item_key}",
        "question": stem,
        "options": options,
        "correct_idx": "ABCD".index(answer),
        "media": rel,
        "media_type": media_type,
        "task_type": row.get("task_type"),
        "sub_type": row.get("sub_type"),
        "ability_type": row.get("ability_type"),
    }


def clevrer_to_trials(video_row: dict[str, Any]) -> list[dict[str, Any]]:
    video_index = video_row.get("video_index") or video_row.get("video_id")
    if video_row.get("video_filename") or video_row.get("video"):
        video_filename = video_row.get("video_filename") or video_row.get("video")
    elif video_index is not None and str(video_index).isdigit():
        video_filename = f"video_{int(video_index):05d}.mp4"
    else:
        video_filename = f"video_{video_index}.mp4"
    out: list[dict[str, Any]] = []
    for q in video_row.get("questions") or []:
        qtype = _norm(q.get("question_type") or q.get("type"))
        if qtype not in CLEVRER_MCQ_TYPES:
            continue
        choices = q.get("choices") or q.get("options") or []
        texts: list[str] = []
        correct_idx = None
        for i, choice in enumerate(choices):
            if isinstance(choice, str):
                texts.append(choice)
                continue
            texts.append(str(choice.get("choice") or choice.get("text") or ""))
            mark = choice.get("answer")
            if mark in {True, "correct", "yes", "true", 1, "1"}:
                correct_idx = i
        if correct_idx is None:
            ans = q.get("answer")
            if isinstance(ans, int):
                correct_idx = ans
            elif isinstance(ans, str) and ans.isdigit():
                correct_idx = int(ans)
        if correct_idx is None or len(texts) < 2:
            continue
        if not all(is_text_option(t) for t in texts):
            continue
        qid = q.get("question_id") or q.get("qid") or len(out)
        out.append(
            {
                "item_id": clevrer_item_id(video_filename, qid),
                "question": str(q.get("question") or "").strip(),
                "options": texts,
                "correct_idx": int(correct_idx),
                "media": str(video_filename),
                "media_type": "video",
                "question_type": qtype,
                "video_index": video_index,
            }
        )
    return out


def piqa_to_trial(row: dict[str, Any], *, split: str, idx: int) -> dict[str, Any] | None:
    goal = str(row.get("goal") or row.get("question") or "").strip()
    sol1 = str(row.get("sol1") or "").strip()
    sol2 = str(row.get("sol2") or "").strip()
    if not goal or not sol1 or not sol2:
        return None
    label = row.get("label")
    if label is None:
        label = row.get("correct_idx", 0)
    return {
        "item_id": f"piqa_{split}_{idx}",
        "question": goal,
        "options": [sol1, sol2],
        "correct_idx": int(label),
        "media": "",
        "media_type": "none",
        "split": split,
    }


def write_manifest(path: Path, task: str, trials: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"task": task, "domain": "S", "n": len(trials), "trials": trials}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_sdomain_items(
    manifest_path: Path,
    media_root: Path,
    *,
    task: str,
    template_id: str = "mechanism",
) -> list[MCQItem]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    trials = data.get("trials") or []
    items: list[MCQItem] = []
    template_task = "s_task"
    for trial in trials:
        options = tuple(str(o) for o in trial["options"])
        prompt = render_prompt(template_task, template_id, options)
        if trial.get("question"):
            prompt = f"{trial['question']}\n\n{prompt}"
        media_rel = str(trial.get("media") or "")
        media_type: MediaType = trial.get("media_type") or "none"
        media = str((media_root / media_rel).resolve()) if media_rel else ""
        if not media:
            media_type = "none"
        item_id = str(trial["item_id"])
        if task == "clevrer":
            item_id = uniquify_clevrer_item_id(item_id, media_rel)
        items.append(
            MCQItem(
                item_id=item_id,
                task=task,
                domain="S",
                media=media,
                media_type=media_type,
                prompt_text=prompt,
                options=options,
                correct_idx=int(trial["correct_idx"]),
                meta={
                    "template_id": template_id,
                    "question_type": trial.get("question_type") or trial.get("sub_type"),
                },
            )
        )
    return items
