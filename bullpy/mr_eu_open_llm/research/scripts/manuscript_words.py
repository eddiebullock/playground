#!/usr/bin/env python3
"""Count the manuscript's words against the journal's limits, and write a clean reading copy.

The paper's parts in research/paper/v2/ carry drafting material a reader never sees: HTML comments
(drafting notes, the scaffold of each Results section, branches kept for later) and registry
markers [REG: ...]. This script removes them and counts what is left.

Counted against the main-text limit: introduction.md, results.md and discussion.md. The title and
abstract come from abstract.md (headings "# Title" and "# Abstract"). Figure legends and Methods
are counted and reported, but the limit does not cover them.

What counts as a word: a whitespace-separated token with at least one letter or digit, after
removing comments, registry markers, [PENDING: ...] slots, numbered citations such as [1,2] or
[8-13], headings and table rows. This is close to a word processor's count of the prose.

Usage:
  python research/scripts/manuscript_words.py              # table of counts
  python research/scripts/manuscript_words.py --json       # counts as JSON
  python research/scripts/manuscript_words.py --out FILE   # also write the clean reading copy

Limits (Nature Human Behaviour Articles, as recorded in research/paper/v2/outline.md, "Journal
limits"): abstract up to 150 words; main text up to 5,000 words, excluding abstract, Methods,
references and figure legends.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "research/paper/v2"
LIMITS = {"abstract": 150, "main_text": 5000}
MAIN_PARTS = ("introduction", "results", "discussion")

COMMENT = re.compile(r"<!--.*?-->", re.S)
REG = re.compile(r"[ \t]*\[REG:[^\]]*\]")
PENDING = re.compile(r"\[PENDING:[^\]]*\]")
CITE = re.compile(r"\[\d+(?:\s*[-–,]\s*\d+)*\]")
HAS_ALNUM = re.compile(r"[A-Za-z0-9]")


def clean(text: str) -> str:
    """The text a reader sees: no HTML comments and no registry markers."""
    text = COMMENT.sub("", text)
    text = REG.sub("", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def count_words(text: str) -> int:
    """Words of prose in `text`, by the rule in the module docstring."""
    lines = [ln for ln in clean(text).splitlines() if not ln.lstrip().startswith(("#", "|"))]
    body = PENDING.sub(" ", "\n".join(lines))
    body = CITE.sub(" ", body)
    return sum(1 for token in body.split() if HAS_ALNUM.search(token))


def split_title_abstract(text: str) -> tuple[str, str]:
    """Return (title, abstract) from abstract.md; the title is empty if there is no "# Title"."""
    body = COMMENT.sub("", text)
    title = ""
    m = re.search(r"^# Title\s*\n(.*?)(?=^# |\Z)", body, re.S | re.M)
    if m:
        title = m.group(1).strip()
    m = re.search(r"^# Abstract\s*\n(.*?)(?=^# |\Z)", body, re.S | re.M)
    abstract = m.group(1).strip() if m else body.strip()
    return title, abstract


def read_part(paper: Path, name: str) -> str:
    path = paper / f"{name}.md"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def counts(paper: Path = PAPER) -> dict:
    title, abstract = split_title_abstract(read_part(paper, "abstract"))
    parts = {name: count_words(read_part(paper, name)) for name in MAIN_PARTS}
    out = {
        "title": title,
        "title_words": count_words(title),
        "abstract": count_words(abstract),
        "introduction": parts["introduction"],
        "results": parts["results"],
        "discussion": parts["discussion"],
        "main_text": sum(parts.values()),
        "figure_legends": count_words(read_part(paper, "figure_legends")),
        "methods": count_words(read_part(paper, "methods")),
        "limits": dict(LIMITS),
    }
    out["abstract_over"] = max(0, out["abstract"] - LIMITS["abstract"])
    out["main_text_over"] = max(0, out["main_text"] - LIMITS["main_text"])
    return out


def reading_copy(paper: Path = PAPER) -> str:
    """Title, abstract, main text, figure legends, Table 1 and Methods, cleaned, in journal order."""
    title, abstract = split_title_abstract(read_part(paper, "abstract"))
    chunks = []
    if title:
        chunks.append(f"# {clean(title).strip()}\n")
    chunks.append("## Abstract\n\n" + clean(abstract))
    for name in ("introduction", "results", "discussion", "figure_legends"):
        text = read_part(paper, name)
        if text:
            chunks.append(clean(text))
    table1 = read_part(paper, "table1")
    if table1:
        chunks.append(clean(table1.split("\nInputs:", 1)[0]))
    methods = read_part(paper, "methods")
    if methods:
        chunks.append(clean(methods))
    return "\n".join(chunks)


def _table(c: dict) -> str:
    rows = [
        ("Title", c["title_words"], ""),
        ("Abstract", c["abstract"], f"limit {c['limits']['abstract']}"),
        ("Introduction", c["introduction"], ""),
        ("Results", c["results"], ""),
        ("Discussion", c["discussion"], ""),
        ("Main text", c["main_text"], f"limit {c['limits']['main_text']}"),
        ("Figure legends", c["figure_legends"], "not counted in the limit"),
        ("Methods", c["methods"], "not counted in the limit"),
    ]
    lines = [f"{name:<15}{n:>7}  {note}".rstrip() for name, n, note in rows]
    if c["abstract_over"]:
        lines.append(f"Abstract is {c['abstract_over']} words over its limit.")
    if c["main_text_over"]:
        lines.append(f"Main text is {c['main_text_over']} words over its limit.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--paper", type=Path, default=PAPER, help="folder holding the paper's parts")
    ap.add_argument("--json", action="store_true", help="print the counts as JSON")
    ap.add_argument("--out", type=Path, help="also write the clean reading copy here")
    args = ap.parse_args(argv)
    c = counts(args.paper)
    print(json.dumps(c, indent=2) if args.json else _table(c))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(reading_copy(args.paper), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
