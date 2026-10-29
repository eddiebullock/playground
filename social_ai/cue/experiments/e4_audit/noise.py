"""Replies audit across seeds: disclosure effect vs run-to-run noise.

  python noise.py replies_qwen3-8b.json replies_qwen3-8b_seed1.json ...

Per seed, the mean share of each reply kind per message and condition. The
disclosure effect is (condition - none), averaged over seeds, paired by
message, with a bootstrap CI over messages. The noise floor is the same
quantity between two seeds of the *same* condition (none vs none, etc.):
how far a rerun moves the numbers when nothing about the user changed.
"""

import itertools
import json
import random
import sys

KINDS = ["engage", "check", "hold back", "apologise"]
CONDS = ["none", "autistic", "anxious"]
runs = [json.load(open(p))["scenarios"] for p in sys.argv[1:]]
M = len(runs[0])


def share(s, c, j):
    ks = s[c]["kinds"]
    return sum(k[j] for k in ks) / len(ks)


def ci(diffs):
    rng = random.Random(0)
    bs = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(5000))
    return bs[125], bs[4875]


print(f"{len(runs)} seeds, {M} messages")
for c in CONDS:
    tot = [sum(share(r[m], c, j) for r in runs for m in range(M)) / (len(runs) * M) for j in range(4)]
    print(f"  {c:9s}", " ".join(f"{k} {v:.3f}" for k, v in zip(KINDS, tot)))
for j in (2, 3):
    print(f"\n{KINDS[j]}:")
    for c in CONDS[1:]:
        d = [sum(share(r[m], c, j) - share(r[m], "none", j) for r in runs) / len(runs) for m in range(M)]
        lo, hi = ci(d)
        print(f"  effect {c:9s} {sum(d) / M:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    noise = []
    for c in CONDS:
        for a, b in itertools.combinations(range(len(runs)), 2):
            noise.append(abs(sum(share(runs[a][m], c, j) - share(runs[b][m], c, j) for m in range(M)) / M))
    print(f"  noise floor: mean |seed-to-seed shift| of the condition mean {sum(noise) / len(noise):.3f}, max {max(noise):.3f} (n={len(noise)})")
