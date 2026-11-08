"""
A5, after the run 4 audit: can a skew-targeted test hold B's error rate when every column of
a 4-cell family carries portrayal-gain skew? Synthetic observers only, no model output.

Compares, on the same data: A2's sign-flip maxT, A5's bootstrap-t maxT, a bootstrap-t on
Hall's (1992) cubic transformation of t (the monotone successor to Johnson's 1978
skewness correction), and the max-p pairs A5 uses (sign-flip with either bootstrap).
Null arms: 4 columns sharing actor gains (and 1 column), at gain .3 with signal 2 and at
gain .5, priors from run 4's calibration. Power arms: Phi .3 and .7 at signal 1 and 2, the
observer column plus 3 null companions, as in recovery. E* uses 5 shuffles, since only B
is read.

Usage (run from a scratch copy, kept here with only this docstring and the path line changed):
    python -m scripts.a5_skew_diagnostic 1000 400 research/reproduction/a5_run4_skew_diagnostic.json
"""
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from scripts import a5_recovery as rec
from scripts import a5_analyse as a5
from scripts import stimulus_interventions as si

R = rec.R
items = rec.real_items(rec.ROOT / "data/eu_emotions_full_manifest.json", rec.ROOT / "data/eu_emotions_human_entropy.json")
W = rec.World(items)
NULL = rec.base_spec(prior=0.4841)
ARMS = {  # name: (column spec, n columns, shared gains, companions are null spec)
    "null_shared4_gain0.3_sig2": (rec.base_spec(signal=2.0, gain=0.3, prior=0.8082), 4, True, False),
    "null_shared4_gain0.5": (rec.base_spec(signal=2.0, gain=0.5, prior=0.94), 4, True, False),
    "null_single_gain0.5": (rec.base_spec(signal=2.0, gain=0.5, prior=0.94), 1, True, False),
    "null_single_gain0.3_sig2": (rec.base_spec(signal=2.0, gain=0.3, prior=0.8082), 1, True, False),
    "pow_phi0.3_sig2": (rec.base_spec(signal=2.0, prior=1.0927), 4, False, True),
    "pow_phi0.7_sig2": (rec.base_spec(signal=2.0, prior=0.5969), 4, False, True),
    "pow_phi0.3_sig1": (rec.base_spec(signal=1.0, prior=0.7794), 4, False, True),
    "pow_phi0.7_sig1": (rec.base_spec(signal=1.0, prior=0.3891), 4, False, True),
}

def scores(spec, rng, gains):
    s = spec["prior"] * W.prior[None, :] + spec["item_sd"] * W.family_noise(rng, W.n)
    s = s + (spec["actor_sd"] * W.family_noise(rng, len(W.actors)))[W.actor]
    if spec["signal"]:
        s[np.arange(W.n), W.acted] += np.exp(gains)[W.actor] * spec["signal"]
    return s

def hall_maxt(x, draws=10000, seed=42):
    n, k = x.shape
    def g(t, gamma):
        a = gamma / (3 * np.sqrt(n)); b = gamma / (6 * np.sqrt(n))
        return t + a * t ** 2 + a ** 2 * t ** 3 / 3 + b
    m, s = x.mean(0), x.std(0, ddof=1)
    gam = ((x - m) ** 3).mean(0) / s ** 3
    obs = np.abs(g(m / (s / np.sqrt(n)), gam))
    idx = np.random.default_rng(seed).integers(0, n, size=(draws, n))
    smp = x[idx]
    mm, ss = smp.mean(1), smp.std(1, ddof=1)
    gg = ((smp - mm[:, None, :]) ** 3).mean(1) / ss ** 3
    nul = np.abs(g((mm - m) / (ss / np.sqrt(n)), gg))
    nul = np.nan_to_num(nul)
    order = np.argsort(-obs, kind="stable"); running = np.full(draws, -np.inf); adj = np.empty(k)
    for t in order[::-1]:
        running = np.maximum(running, nul[:, t]); adj[t] = ((running >= obs[t] - 1e-12).sum() + 1) / (draws + 1)
    prev = 0.0
    for t in order:
        prev = max(prev, adj[t]); adj[t] = prev
    return adj

def one(args):
    name, k = args
    spec, cols, shared, comp = ARMS[name]
    rng = np.random.default_rng([7300, list(ARMS).index(name), k])
    gains = rng.normal(0, spec["gain"], len(W.actors))
    rows = []
    for c in range(cols):
        if comp and c > 0:
            sp, g = NULL, rng.normal(0, NULL["gain"], len(W.actors))
        else:
            sp, g = spec, (gains if shared else rng.normal(0, spec["gain"], len(W.actors)))
        rows.append(a5.estimate(W.judgement(scores(sp, rng, g)), shuffles=5).B_row)
    x = np.column_stack(rows)
    f = np.array(si.signflip_maxt(x, alternative=[0] * cols)["p_maxt"])
    b = np.array(a5.bootstrap_t_maxt(x, [0] * cols)["p_maxt"])
    h = hall_maxt(x)
    return name, {"flip": f, "boot": b, "hall": h, "flip_boot": np.maximum(f, b), "flip_hall": np.maximum(f, h)}

if __name__ == "__main__":
    n_null, n_pow = int(sys.argv[1]), int(sys.argv[2])
    tasks = [(nm, k) for nm in ARMS for k in range(n_null if nm.startswith("null") else n_pow)]
    t0 = time.time()
    with Pool(3) as p:
        res = list(p.imap_unordered(one, tasks, chunksize=5))
    out = {}
    for nm in ARMS:
        rs = [r for n, r in res if n == nm]
        out[nm] = {"n": len(rs)}
        for test in rs[0]:
            ps = np.array([r[test] for r in rs])
            out[nm][test] = {"familywise": float((ps < 0.05).any(1).mean()), "per_column": float((ps < 0.05).mean()),
                             "column0": float((ps[:, 0] < 0.05).mean())}
    print(json.dumps(out, indent=1)); print("secs", round(time.time() - t0))
    json.dump(out, open(sys.argv[3], "w"), indent=1)
