"""E3: abstention thresholds for the product.

For each cutoff on the top reading's probability (order-averaged, as in the
app), report coverage (share of items where Gloss would commit to a favourite)
and selective accuracy (right when it commits), with bootstrap CIs.

  python thresholds.py results/hu_qwen3-8b.jsonl [...]
"""

import json
import random
import sys
from collections import defaultdict

CUTS = [0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9]


def items(path, mode_filter=None):
    by = defaultdict(list)
    for r in map(json.loads, open(path)):
        if not r["p"]:
            continue
        orig = [0.0] * r["n_opts"]
        for pos, o in enumerate(r["order"]):
            orig[o - 1] = r["p"][pos]
        by[r["id"]].append((orig, r["order"][r["true"] - 1] - 1))
    out = []
    for rows in by.values():
        n = len(rows[0][0])
        avg = [sum(x[0][k] for x in rows) / len(rows) for k in range(n)]
        out.append((max(avg), int(avg.index(max(avg)) == rows[0][1])))
    return out


def table(data, reps=2000, seed=0):
    rnd = random.Random(seed)
    res = []
    for c in CUTS:
        kept = [ok for conf, ok in data if conf >= c]
        cov = len(kept) / len(data)
        acc = sum(kept) / len(kept) if kept else float("nan")
        boots = []
        for _ in range(reps):
            s = [rnd.choice(data) for _ in data]
            k = [ok for conf, ok in s if conf >= c]
            if k:
                boots.append(sum(k) / len(k))
        boots.sort()
        lo, hi = (boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]) if boots else (float("nan"),) * 2
        res.append({"cut": c, "coverage": cov, "sel_acc": acc, "ci": [lo, hi], "n_kept": len(kept)})
    return res


if __name__ == "__main__":
    for p in sys.argv[1:]:
        d = items(p)
        print(f"\n{p.split('/')[-1]}  items={len(d)}  overall acc={sum(ok for _, ok in d) / len(d):.3f}")
        print("cut   coverage  sel_acc  95% CI")
        for r in table(d):
            print(f"{r['cut']:.2f}  {r['coverage']:.2f}      {r['sel_acc']:.3f}    [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]  n={r['n_kept']}")
