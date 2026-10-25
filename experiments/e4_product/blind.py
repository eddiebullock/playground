"""Blind hand-labelling for E4: mix readings from several runs into one sheet.

  python blind.py make sheet_dir results/a.jsonl results/b.jsonl ...
  python blind.py split sheet_dir labels.txt

`make` writes sheet_dir/sheet.txt (cases in random order, readings within a
case in random order, no engine name, no rank or probability) and
sheet_dir/key.json. Label each case with one line "<case> <0/1 per reading,
in sheet order>", 1 meaning the reading says what the story's intended
option says. `split` maps the labels back and writes
hand_labels/<run>_blind.txt in the format analyze.py --hand reads.
"""

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent
ITEMS = {i["id"]: i for i in json.load(open(HERE / "items.json"))}


def make(sheet_dir, files):
    cases = []
    for f in files:
        for line in open(f):
            r = json.loads(line)
            cases.append((Path(f).stem, r["id"], [x["meaning"] for x in r["readings"]]))
    rng = random.Random(20261025)
    rng.shuffle(cases)
    key, lines = {}, []
    for n, (run, iid, readings) in enumerate(cases):
        code = f"c{n:04d}"
        perm = list(range(len(readings)))
        rng.shuffle(perm)
        key[code] = {"run": run, "id": iid, "perm": perm}
        it = ITEMS[iid]
        gold = it["options"][it["labels"].index(it["gold"])]
        lines.append(f'## {code} | {it["context"][-260:]} "{it["text"]}"\nINTENDED: {gold}')
        lines += [f"  {k}. {readings[p]}" for k, p in enumerate(perm)]
    Path(sheet_dir).mkdir(parents=True, exist_ok=True)
    (Path(sheet_dir) / "sheet.txt").write_text("\n".join(lines) + "\n")
    json.dump(key, open(Path(sheet_dir) / "key.json", "w"), indent=0)
    print(f"{len(cases)} cases, {sum(len(c[2]) for c in cases)} readings")


def split(sheet_dir, labels):
    key = json.load(open(Path(sheet_dir) / "key.json"))
    out = {}
    for line in open(labels):
        if not line.strip():
            continue
        code, bits = line.split()
        k = key[code]
        assert len(bits) == len(k["perm"]), code
        orig = [""] * len(bits)
        for pos, p in enumerate(k["perm"]):
            orig[p] = bits[pos]
        out.setdefault(k["run"], []).append(f'{k["id"]} {"".join(orig)}')
    missing = set(key) - {l.split()[0] for l in open(labels) if l.strip()}
    print("unlabelled cases:", len(missing))
    (HERE / "hand_labels").mkdir(exist_ok=True)
    for run, rows in out.items():
        (HERE / "hand_labels" / f"{run}_blind.txt").write_text("\n".join(rows) + "\n")
        print(run, len(rows))


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:]
    make(rest[0], rest[1:]) if cmd == "make" else split(*rest)
