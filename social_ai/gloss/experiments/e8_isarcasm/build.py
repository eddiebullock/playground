"""E8: freeze the external held-out set from iSarcasmEval (Abu Farha et al.,
SemEval 2022 Task 6). Real tweets labelled by their own authors; for
sarcastic ones the author also wrote what they meant ("rephrase"). This set
is never used for tuning. Seed and filters are fixed before any run.

  python3 -I build.py ../../data/raw/isarcasm/train.En.csv > items.json
"""

import csv
import json
import random
import sys

rows = [r for r in csv.DictReader(open(sys.argv[1]))
        if 30 <= len(r["tweet"]) <= 280 and "http" not in r["tweet"]]
sar = [r for r in rows if r["sarcastic"] == "1" and len(r["rephrase"].strip()) >= 10]
sin = [r for r in rows if r["sarcastic"] == "0"]
rng = random.Random(20261025)
pick = [("Sarcastic", r) for r in rng.sample(sar, 80)] + [("Sincere", r) for r in rng.sample(sin, 80)]
rng.shuffle(pick)
items = [{"id": f"isarc:{r['']}", "task": kind, "text": r["tweet"].strip(),
          "intended": r["rephrase"].strip() if kind == "Sarcastic" else "The author means what the tweet says.",
          "context": "Someone I follow posted this.", "channel": "social media", "person": None, "gold": "Intended"}
         for kind, r in pick]
json.dump(items, sys.stdout, indent=1, ensure_ascii=False)
