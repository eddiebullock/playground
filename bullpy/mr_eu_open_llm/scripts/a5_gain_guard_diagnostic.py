"""
A5, after recovery run 5: can a guard on estimated portrayal-gain heterogeneity hold B's
familywise rate at gain .5 without firing often at the validated gain .3? Synthetic
observers only, no model output.

Uses a5_analyse.gain_heterogeneity and family_gain_log_sd. A guard at log-SD .4 (halfway
between the validated .3 and the failing .5) withholds a balance claim when the family's
pooled estimate exceeds it. Arms: single columns at gain 0, .3, .5 and signal .5, 1, 2
(tracking); 4 shared-gain null columns at gain .3 and .5 (familywise rate with and without
the guard); grid points at Phi .3 and .7, gain .3, signal .5, 1, 2 (how often it fires).
The guard was not adopted (draft, Changes after recovery run 5).

Usage (run from a scratch copy, kept here with only this docstring, the path line and the
cut, which was a block setting while it ran, changed):
    python -m scripts.a5_gain_guard_diagnostic 400 research/reproduction/a5_run5_gain_guard_diagnostic.json
"""
import sys, json, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from scripts import a5_recovery as rec
from scripts import a5_analyse as a5
items = rec.real_items(rec.ROOT / "data/eu_emotions_full_manifest.json", rec.ROOT / "data/eu_emotions_human_entropy.json")
W = rec.World(items)
CUT = 0.4  # the dropped guard's cut
ARMS = {}
for g in (0.0, 0.3, 0.5):
    for s in (0.5, 1.0, 2.0):
        ARMS[f"track_gain{g}_sig{s}"] = (rec.prior_for_phi(W, 0.5, rec.base_spec(signal=s, gain=g)), 1)
for g in (0.3, 0.5):
    ARMS[f"shared4_gain{g}"] = (rec.base_spec(signal=2.0, gain=g, prior={0.3: 0.8082, 0.5: 0.94}[g]), 4)
for phi in (0.3, 0.7):
    for s in (0.5, 1.0, 2.0):
        ARMS[f"grid_phi{phi}_sig{s}"] = (rec.prior_for_phi(W, phi, rec.base_spec(signal=s)), 1)

def one(args):
    name, k = args
    spec, cols = ARMS[name]
    rng = np.random.default_rng([4411, list(ARMS).index(name), k])
    shared = rng.normal(0, spec["gain"], len(W.actors))
    js = [W.judgement(W.scores(spec, rng, shared)[0]) for _ in range(cols)]
    hs = [a5.gain_heterogeneity(j) for j in js]
    fam = a5.family_gain_log_sd(js)
    out = {"own": hs[0]["log_sd"], "estimable": hs[0]["estimable"], "family": fam, "fires": fam is not None and fam > CUT}
    if cols == 4:
        t = a5.family_tests([a5.estimate(j, shuffles=5) for j in js])
        sig = [x["p_B"] < 0.05 for x in t]
        out["any_sig"] = any(sig); out["any_claim"] = any(sig) and not out["fires"]
    return name, out

if __name__ == "__main__":
    n = int(sys.argv[1])
    with Pool(3) as p:
        res = p.map(one, [(nm, k) for nm in ARMS for k in range(n)], chunksize=10)
    out = {}
    for nm in ARMS:
        rs = [r for x, r in res if x == nm]
        est = [r["own"] for r in rs if r["own"] is not None]
        d = {"n": len(rs), "estimable": float(np.mean([r["estimable"] for r in rs])),
             "mean": float(np.mean(est)) if est else None, "sd": float(np.std(est)) if est else None,
             "fires": float(np.mean([r["fires"] for r in rs]))}
        if "any_sig" in rs[0]:
            d["fwer_raw"] = float(np.mean([r["any_sig"] for r in rs])); d["fwer_guarded"] = float(np.mean([r["any_claim"] for r in rs]))
            fam = [r["family"] for r in rs if r["family"] is not None]; d["family_mean"] = float(np.mean(fam)); d["family_sd"] = float(np.std(fam))
        out[nm] = d; print(nm, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}, flush=True)
    json.dump(out, open(sys.argv[2], "w"), indent=1)
