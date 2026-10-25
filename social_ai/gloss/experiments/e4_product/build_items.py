"""E4: turn the Hu et al. stories into Gloss Reflect inputs.

Each story becomes {person, context, text}: the last quoted remark is what
the person said, everything before it is what came before. The original
options and their labels are kept for grading, and the human accuracy per item
for ambiguity alignment. Writes items.json.

  python build_items.py
"""

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HU = ROOT / "data/raw/lm-pragmatics"
TASKS = ["Irony", "IndirectSpeech", "Deceits", "Maxims", "Metaphor"]
QUOTE = re.compile(r'["“]([^"”]+)["”]')
NAME = re.compile(r"\b(?:did|does|is|was|has|might|do)\s+([A-Z][a-z]+)\b")


def human(task):
    by = defaultdict(list)
    for r in csv.DictReader(open(HU / f"human_data/fj/Human_{task}.csv")):
        by[int(r["itemNum"])].append(int(r["Correct"]))
    return {k: sum(v) / len(v) for k, v in by.items()}


def options(row):
    """Original-order option texts from the seed-0 prompt."""
    body = row["prompt"].split("Options:", 1)[1]
    opts = [m.group(1).strip() for m in re.finditer(r"^\s*\d\)\s*(.+)$", body, re.M)]
    order = json.loads(row["randomized_option_order"])
    orig = [None] * len(order)
    for pos, o in enumerate(order):
        orig[o - 1] = opts[pos]
    return orig


def split(content):
    content = " ".join(content.split()).replace("''", '"')
    quotes = list(QUOTE.finditer(content))
    if not quotes:
        return None
    last = quotes[-1]
    question = content[last.end():].strip(" .")
    m = NAME.search(question)
    person = m.group(1) if m else None
    lead = content[: last.start()].rstrip()
    # Pronoun questions ("What might he be trying to convey?") leave person
    # unset, as a user might; the lead-in says who is speaking.
    # Keep the lead-in ("Richard points at a building and asks his wife:"),
    # so the scene reads naturally; the remark itself goes in text.
    lead = lead.rstrip(" ,:") + ":"
    return {"person": person, "context": lead, "text": last.group(1).strip(), "question": question}


def main():
    items, skipped = [], Counter()
    sincere = json.load(open(ROOT / "data/sincere_irony.json"))
    for task in TASKS:
        hum = human(task)
        for row in csv.DictReader(open(HU / f"prompts/{task}_prompts_seed0_examples0.csv")):
            s = split(row["content"])
            if not s or not s["question"].endswith("?"):
                skipped[task] += 1
                continue
            labels = json.loads(row["original_labels_complex"].replace("'", '"'))
            base = {"task": task, "item": int(row["item_id"]), "options": options(row), "labels": labels,
                    "human_acc": hum.get(int(row["item_id"]))}
            items.append({"id": f"{task}:{row['item_id']}", **base, **s, "gold": next(l for l in labels if l.startswith("Correct"))})
            if task == "Irony" and row["item_id"] in sincere:
                t = split(sincere[row["item_id"]])
                if t:
                    items.append({"id": f"Sincere:{row['item_id']}", **base, **t, "task": "Sincere",
                                  "gold": "IncorrectLiteral", "human_acc": None})
    json.dump(items, open(Path(__file__).parent / "items.json", "w"), indent=1)
    print(len(items), "items", dict(Counter(i["task"] for i in items)), "skipped", dict(skipped), file=sys.stderr)


if __name__ == "__main__":
    main()
