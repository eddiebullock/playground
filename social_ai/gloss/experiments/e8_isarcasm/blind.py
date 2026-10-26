"""Blind labelling for runs on any item set with a free-text intended meaning
(E8 tweets) or the E4 items (intended option text).

  python3 blind.py make <sheet_dir> <items.json> results/a.jsonl [results/b.jsonl ...]
  python3 blind.py split <sheet_dir> <labels.txt> <hand_labels_dir>

Every written reading is labelled, including readings a wide run wrote but
did not show (hidden), so recall can be computed for both. The sheet shows no
run, rank, probability or hidden flag. Label rule: 1 if the reading gives the
speaker the intended meaning (for sincere items: that they mean what they
say), even if framed as hidden or polite; 0 otherwise.
"""

import json
import random
import sys
from pathlib import Path


def intended(it):
    if "intended" in it:
        return it["intended"]
    return it["options"][it["labels"].index(it["gold"])]


def make(sheet_dir, items_path, files):
    items = {i["id"]: i for i in json.load(open(items_path))}
    cases = []
    for f in files:
        for line in open(f):
            r = json.loads(line)
            cases.append((Path(f).stem, r["id"], [x["meaning"] for x in r["readings"]]))
    rng = random.Random(20261026)
    rng.shuffle(cases)
    key, lines = {}, []
    for n, (run, iid, readings) in enumerate(cases):
        code = f"c{n:04d}"
        perm = list(range(len(readings)))
        rng.shuffle(perm)
        key[code] = {"run": run, "id": iid, "perm": perm}
        it = items[iid]
        lines.append(f'## {code} | {it.get("context", "")[-260:]} "{it["text"]}"\nINTENDED: {intended(it)}')
        lines += [f"  {k}. {readings[p]}" for k, p in enumerate(perm)]
    Path(sheet_dir).mkdir(parents=True, exist_ok=True)
    (Path(sheet_dir) / "sheet.txt").write_text("\n".join(lines) + "\n")
    json.dump(key, open(Path(sheet_dir) / "key.json", "w"), indent=0)
    print(f"{len(cases)} cases, {sum(len(c[2]) for c in cases)} readings")


def split(sheet_dir, labels, out_dir):
    key = json.load(open(Path(sheet_dir) / "key.json"))
    out = {}
    seen = set()
    for line in open(labels):
        if not line.strip():
            continue
        code, bits = line.split()
        seen.add(code)
        k = key[code]
        assert len(bits) == len(k["perm"]), code
        orig = [""] * len(bits)
        for pos, p in enumerate(k["perm"]):
            orig[p] = bits[pos]
        out.setdefault(k["run"], []).append(f'{k["id"]} {"".join(orig)}')
    print("unlabelled cases:", len(set(key) - seen))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for run, rows in out.items():
        path = Path(out_dir) / f"{run}_blind.txt"
        merged = dict(l.split() for l in open(path) if l.strip()) if path.exists() else {}
        merged.update(r.split() for r in rows)
        path.write_text("".join(f"{k} {v}\n" for k, v in merged.items()))
        print(run, len(rows), "->", len(merged))


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:]
    make(rest[0], rest[1], rest[2:]) if cmd == "make" else split(*rest)
