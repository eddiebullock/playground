"""Paper 1: sensitivity versus bias on the minimal pairs.

Every Hu et al. item has a non-literal correct answer, so accuracy cannot
tell pragmatic sensitivity from a preference for non-literal readings. The
minimal pairs (same remark, context rewritten so it is sincere) can. For each
item version we take q = p(non-literal) / (p(non-literal) + p(literal)),
the model's preference between the item's CorrectNonLiteral and
IncorrectLiteral options, averaged over option orders. Then:

- hit rate H: share of non-literal versions with q > 0.5
- false-alarm rate F: share of sincere versions with q > 0.5
- d' = z(H) - z(F) (sensitivity), c = -(z(H) + z(F)) / 2 (bias; positive =
  literal-leaning), with the log-linear correction for rates of 0 or 1
- AUC: P(q_nonliteral > q_sincere) over pairs, a threshold-free sensitivity
- mean q on each version (continuous bias)

Bootstrap 95% CIs over pairs. Only items that have both versions count.

  python sdt.py                 # every model and mode found in results/
"""

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

OUT = Path(__file__).resolve().parent / "results"
Z = NormalDist().inv_cdf
SIN2TASK = {"Sincere": "Irony", "SincereIndirectSpeech": "IndirectSpeech", "SincereDeceits": "Deceits"}


def load(path):
    """Per item: order-averaged probability by label."""
    acc = defaultdict(lambda: defaultdict(float))
    n = defaultdict(int)
    for line in open(path):
        r = json.loads(line)
        if r["p"] is None:
            continue
        for lab, p in zip(r["labels"], r["p"]):
            acc[r["id"]][lab] += p
        n[r["id"]] += 1
    return {k: {lab: v / n[k] for lab, v in d.items()} for k, d in acc.items()}


def q(d):
    nl, lit = d.get("CorrectNonLiteral", 0), d.get("IncorrectLiteral", 0)
    return nl / (nl + lit) if nl + lit > 0 else 0.5


def stats(pairs):
    """pairs: list of (q_nonliteral_version, q_sincere_version, acc_nl, acc_sin)."""
    k = len(pairs)
    h = (sum(a > 0.5 for a, _, _, _ in pairs) + 0.5) / (k + 1)
    f = (sum(b > 0.5 for _, b, _, _ in pairs) + 0.5) / (k + 1)
    auc = sum((a > b) + 0.5 * (a == b) for a, b, _, _ in pairs) / k
    return {
        "H": h, "F": f, "dprime": Z(h) - Z(f), "c": -(Z(h) + Z(f)) / 2, "auc": auc,
        "q_nonliteral": sum(a for a, _, _, _ in pairs) / k, "q_sincere": sum(b for _, b, _, _ in pairs) / k,
        "acc_nonliteral": sum(x for _, _, x, _ in pairs) / k, "acc_sincere": sum(y for _, _, _, y in pairs) / k,
        "balanced_acc": (sum(x for _, _, x, _ in pairs) + sum(y for _, _, _, y in pairs)) / (2 * k),
    }


def boot(pairs, n=2000, seed=0):
    r = random.Random(seed)
    draws = [stats([r.choice(pairs) for _ in pairs]) for _ in range(n)]
    ci = {}
    for key in draws[0]:
        xs = sorted(d[key] for d in draws)
        ci[key] = [round(xs[int(0.025 * n)], 3), round(xs[int(0.975 * n)], 3)]
    return ci


def top(d):
    return max(d, key=d.get)


def run():
    files = {p.stem: p for p in OUT.glob("*.jsonl")}
    res = {}
    for stem, path in sorted(files.items()):
        m = re.match(r"hu_(.+?)(_stance)?$", stem)
        if not m:
            continue
        model, mode = m.group(1), m.group(2) or ""
        hu = load(path)
        sincere = {}
        for pre in ("sincere", "sincere_all"):
            sp = files.get(f"{pre}_{model}{mode}")
            if sp:
                sincere.update(load(sp))
        by_task = defaultdict(list)
        for sid, d in sincere.items():
            name, num = sid.split(":")
            hid = f"{SIN2TASK[name]}:{num}"
            if hid not in hu:
                continue
            h = hu[hid]
            by_task[SIN2TASK[name]].append(
                (q(h), q(d), float(top(h) == "CorrectNonLiteral"), float(top(d) == "IncorrectLiteral")))
        if not by_task:
            continue
        key = f"{model}{' stance' if mode else ' direct'}"
        res[key] = {}
        everything = [p for v in by_task.values() for p in v]
        for task, pairs in [("All", everything)] + sorted(by_task.items()):
            s = {k: round(v, 3) for k, v in stats(pairs).items()}
            s["n_pairs"] = len(pairs)
            s["ci"] = boot(pairs)
            res[key][task] = s
    return res


if __name__ == "__main__":
    res = run()
    json.dump(res, open(OUT / "sdt_summary.json", "w"), indent=1)
    print(f"{'model/mode':24} {'task':15} {'n':>3} {'acc_NL':>6} {'acc_S':>6} {'d_prime':>14} {'c':>14} {'AUC':>5}")
    for key, tasks in res.items():
        for task, s in tasks.items():
            print(f"{key:24} {task:15} {s['n_pairs']:3d} {s['acc_nonliteral']:6.2f} {s['acc_sincere']:6.2f} "
                  f"{s['dprime']:5.2f} [{s['ci']['dprime'][0]:.2f},{s['ci']['dprime'][1]:.2f}] "
                  f"{s['c']:5.2f} [{s['ci']['c'][0]:.2f},{s['ci']['c'][1]:.2f}] {s['auc']:5.2f}")
