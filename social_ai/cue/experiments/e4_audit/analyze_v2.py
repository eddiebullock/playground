"""Readings audit: does what the reader says about themselves move the readings?

  python3 analyze_v2.py v2_qwen3-8b_stance.json [more.json ...]

For each disclosure against no disclosure, paired by scenario: mean total
variation (TV) distance, top-1 flips, shift in the negativity index
(mild = 1, hostile = 2) and in the probability of the sincere reading. The
placebo ("I am left-handed") measures what any extra sentence about the
reader does, so each condition is also reported net of the placebo. Bootstrap
95% CIs over scenarios.
"""

import json
import random
import sys
from pathlib import Path


def tv(p, q):
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


def neg(p):
    return p[1] + 2 * p[2]


def ci(xs, n=2000, seed=0):
    r = random.Random(seed)
    ms = sorted(sum(r.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return [round(ms[int(0.025 * n)], 3), round(ms[int(0.975 * n)], 3)]


def mean(xs):
    return sum(xs) / len(xs)


def analyse(path):
    d = json.load(open(path))
    sc = d["scenarios"]
    conds = [k for k in sc[0] if k not in ("i", "text", "none")]
    out = {"file": Path(path).name, "model": d["model"], "variant": d.get("variant", "stance"), "n": len(sc)}
    for c in conds:
        tvs = [tv(s[c], s["none"]) for s in sc]
        flips = sum(s[c].index(max(s[c])) != s["none"].index(max(s["none"])) for s in sc)
        dneg = [neg(s[c]) - neg(s["none"]) for s in sc]
        dsin = [s[c][0] - s["none"][0] for s in sc]
        row = {"tv": round(mean(tvs), 3), "tv_ci": ci(tvs), "flips": flips,
               "neg_shift": round(mean(dneg), 3), "neg_ci": ci(dneg),
               "sincere_shift": round(mean(dsin), 3), "sincere_ci": ci(dsin)}
        if "placebo" in sc[0] and c != "placebo":
            net_tv = [tv(s[c], s["none"]) - tv(s["placebo"], s["none"]) for s in sc]
            net_neg = [neg(s[c]) - neg(s["placebo"]) for s in sc]
            row.update({"tv_minus_placebo": round(mean(net_tv), 3), "tv_minus_placebo_ci": ci(net_tv),
                        "neg_vs_placebo": round(mean(net_neg), 3), "neg_vs_placebo_ci": ci(net_neg)})
        out[c] = row
    return out


if __name__ == "__main__":
    for f in sys.argv[1:]:
        r = analyse(f)
        print(f"\n{r['model']} ({r['variant']}), {r['n']} scenarios")
        print(f"{'disclosure':10} {'TV':>6} {'TV CI':>16} {'flips':>5} {'neg shift':>9} {'CI':>17} {'TV-placebo':>10} {'CI':>17}")
        for c, v in r.items():
            if not isinstance(v, dict):
                continue
            extra = f"{v['tv_minus_placebo']:10.3f} {str(v['tv_minus_placebo_ci']):>17}" if "tv_minus_placebo" in v else ""
            print(f"{c:10} {v['tv']:6.3f} {str(v['tv_ci']):>16} {v['flips']:5d} {v['neg_shift']:9.3f} {str(v['neg_ci']):>17} {extra}")
