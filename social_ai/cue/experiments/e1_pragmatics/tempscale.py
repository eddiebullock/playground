"""Temperature scaling of the order-averaged option distribution, 5-fold CV
over items. One scalar T per model; the app would apply p^(1/T), renormalised.

  python tempscale.py results/hu_qwen3-8b.jsonl
"""

import json
import math
import random
import sys
from collections import defaultdict

from analyze import auroc, ece

GRID = [0.5 + 0.1 * i for i in range(46)]  # T from 0.5 to 5.0


def items(path):
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
        out.append(([sum(x[0][k] for x in rows) / len(rows) for k in range(n)], rows[0][1]))
    return out


def scale(p, t):
    q = [max(x, 1e-9) ** (1 / t) for x in p]
    z = sum(q)
    return [x / z for x in q]


def nll(data, t):
    return -sum(math.log(scale(p, t)[y]) for p, y in data) / len(data)


def fit(data):
    return min(GRID, key=lambda t: nll(data, t))


def main(path, folds=5, seed=0):
    data = items(path)
    idx = list(range(len(data)))
    random.Random(seed).shuffle(idx)
    conf_raw, conf_cal, corr, ts = [], [], [], []
    for f in range(folds):
        test = idx[f::folds]
        train = [data[i] for i in idx if i not in set(test)]
        t = fit(train)
        ts.append(t)
        for i in test:
            p, y = data[i]
            q = scale(p, t)
            conf_raw.append(max(p))
            conf_cal.append(max(q))
            corr.append(int(p.index(max(p)) == y))
    print(f"{path.split('/')[-1]}: T per fold {[round(t, 1) for t in ts]}, T on all items {fit(data):.1f}")
    print(f"  ECE raw {ece(conf_raw, corr):.3f} -> calibrated {ece(conf_cal, corr):.3f}  "
          f"(AUROC unchanged by monotone scaling: {auroc(conf_cal, corr):.3f})")
    print("  cut   coverage  sel_acc   (calibrated confidence)")
    for c in [0.5, 0.6, 0.7, 0.8, 0.9]:
        kept = [ok for cf, ok in zip(conf_cal, corr) if cf >= c]
        if kept:
            print(f"  {c:.1f}   {len(kept) / len(corr):.2f}      {sum(kept) / len(kept):.3f}  n={len(kept)}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
