"""MMToM-QA text-only control for Paper 2 visual ToM.

Public JSONL (MIT). Media is empty: this is the text control, not the
257GB video zip. Manifests go under $ESVLM_PUBLIC_DIR via fetch_mmtom_text.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from esvlm.tasks.prompts import format_options, render_prompt
from esvlm.tasks.sdomain import write_manifest
from esvlm.types import MCQItem

MMTOM_QUESTIONS_URL = (
    "https://raw.githubusercontent.com/chuanyangjin/MMToM-QA/main/Benchmark/questions.json"
)
_PLEASE_RE = re.compile(r"\s*Please respond with either a or b\.?\s*$", re.I)
_LABEL_PAIR = frozenset({"a", "b"})


def parse_ab_options(question: str) -> tuple[str, str]:
    """Extract the two statements after (a) and (b). Strip 'Please respond...'."""

    text = _PLEASE_RE.sub("", str(question).strip()).strip()
    lower = text.lower()
    idx_a = lower.rfind("(a)")
    idx_b = lower.rfind("(b)")
    if idx_a < 0 or idx_b < 0 or idx_b <= idx_a:
        raise ValueError("could not parse (a)/(b) options from question")
    left = text[idx_a + 3 : idx_b].strip()
    right = text[idx_b + 3 :].strip()
    if not left or not right:
        raise ValueError("empty (a)/(b) option")
    return left, right


def _question_stem(question: str) -> str:
    text = str(question)
    lower = text.lower()
    idx_a = lower.rfind("(a)")
    head = text[:idx_a] if idx_a >= 0 else text
    marker = "question:"
    pos = head.lower().rfind(marker)
    if pos >= 0:
        stem = head[pos + len(marker) :].strip()
        if stem:
            return stem
    stem = head.strip()
    return stem if stem else text.strip()


def _options_for_row(row: dict[str, Any], question: str) -> tuple[str, str]:
    raw = row.get("answer_list")
    if isinstance(raw, list) and len(raw) == 2 and all(isinstance(x, str) for x in raw):
        left, right = raw[0].strip(), raw[1].strip()
        # Official file stores labels ["a", "b"], not the belief statements.
        if left and right and {left.lower(), right.lower()} != _LABEL_PAIR:
            return left, right
    return parse_ab_options(question)


def mmtom_to_trial(row: dict[str, Any], *, idx: int) -> dict[str, Any]:
    question = str(row.get("question") or "")
    options = list(_options_for_row(row, question))
    answer = str(row.get("answer") or "").strip().lower()
    episode = row.get("episode", "x")
    return {
        "item_id": f"mmtom_{idx}_{episode}",
        "question": _question_stem(question) or question,
        "options": options,
        "correct_idx": 0 if answer.startswith("a") else 1,
        "media": "",
        "media_type": "none",
        "question_type": row.get("question_type"),
        "episode": row.get("episode"),
        "end_time": row.get("end_time"),
    }


def load_mmtom_jsonl(path: Path) -> list[dict]:
    """Load JSONL (official dump) or a JSON list of question dicts."""

    text = Path(path).read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{path}: expected a JSON list")
        return data
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("trials", "questions", "items"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    raise ValueError(f"{path}: unsupported JSON")


def _which(name: str) -> str | None:
    from shutil import which

    return which(name)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    wget = _which("wget")
    if wget:
        subprocess.check_call([wget, "-c", "-O", str(dest), url])
        return
    with urlopen(url) as src, dest.open("wb") as out:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def fetch_mmtom_text(root: Path, log) -> Path:
    """Download MMToM-QA questions.json and write a text-only manifest."""

    raw = Path(root) / "mmtom" / "raw" / "questions.json"
    log.info("mmtom questions %s", MMTOM_QUESTIONS_URL)
    _download(MMTOM_QUESTIONS_URL, raw)
    if not raw.is_file() or raw.stat().st_size == 0:
        raise FileNotFoundError(f"mmtom questions missing at {raw}")
    rows = load_mmtom_jsonl(raw)
    trials = [mmtom_to_trial(row, idx=i) for i, row in enumerate(rows)]
    dest = Path(root) / "mmtom" / "manifest.json"
    write_manifest(dest, "mmtom", trials)
    log.info("mmtom kept %s text-control items", len(trials))
    return dest


def _render_mmtom_prompt(question: str, template_id: str, options: tuple[str, ...]) -> str:
    from esvlm.tasks.prompts import TEMPLATES

    TEMPLATES.setdefault(
        "mmtom",
        {
            "direct": (
                "Which statement is more likely to be true?\n\n{options}\n\n"
                "Reply with the label only."
            ),
        },
    )
    try:
        rendered = render_prompt("mmtom", template_id, options)
    except KeyError:
        rendered = format_options(options)
    if question:
        return f"{question}\n\n{rendered}"
    return rendered


def load_mmtom_items(manifest_path: Path, *, template_id: str = "direct") -> list[MCQItem]:
    rows = load_mmtom_jsonl(Path(manifest_path))
    trials: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        if "options" in row and "correct_idx" in row:
            trials.append(row)
        else:
            trials.append(mmtom_to_trial(row, idx=int(row.get("idx", i))))
    items: list[MCQItem] = []
    for trial in trials:
        options = tuple(str(o) for o in trial["options"])
        prompt = _render_mmtom_prompt(str(trial.get("question") or "").strip(), template_id, options)
        items.append(
            MCQItem(
                item_id=str(trial["item_id"]),
                task="mmtom",
                domain="E",
                media="",
                media_type="none",
                prompt_text=prompt,
                options=options,
                correct_idx=int(trial["correct_idx"]),
                meta={
                    "template_id": template_id,
                    "question_type": trial.get("question_type"),
                    "episode": trial.get("episode"),
                    "end_time": trial.get("end_time"),
                },
            )
        )
    return items
