"""Item-level accuracy and calibration after averaging the option distribution
over the 5 rotated orders, which is what the product does."""
import json, sys
from collections import defaultdict
from analyze import auroc, ece, selective, boot_ci

def averaged(path):
    by = defaultdict(list)
    for r in map(json.loads, open(path)):
        if not r["p"]:
            continue
        orig = [0.0] * r["n_opts"]
        for pos, o in enumerate(r["order"]):
            orig[o - 1] = r["p"][pos]
        true_orig = r["order"][r["true"] - 1] - 1
        by[r["id"]].append((orig, true_orig))
    conf, corr = [], []
    for rows in by.values():
        n = len(rows[0][0])
        avg = [sum(x[0][k] for x in rows) / len(rows) for k in range(n)]
        conf.append(max(avg))
        corr.append(int(avg.index(max(avg)) == rows[0][1]))
    return dict(n=len(corr), acc=sum(corr) / len(corr), ci=boot_ci(corr), auroc=auroc(conf, corr),
                ece=ece(conf, corr), sel80=selective(conf, corr, .8), sel50=selective(conf, corr, .5))

for p in sys.argv[1:]:
    print(p.split("/")[-1], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in averaged(p).items()})
