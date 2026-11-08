"""
A5 recovery run 3 diagnostic: is B's excess false-positive rate in small families caused by
the leave-one-out default (actor rows depending on other actors), or by portrayal-gain
heterogeneity? Synthetic observers only, no model output.

Arm `oracle` replaces the leave-one-out default with a fixed population default per spec,
which depends on no actor's rows. Arm `loo` keeps the real default. Both recalibrate each
observer to population B = 0 on 3000 draws, and use the same replicate seeds, so the arms
are paired. B only enters through the default, so templates are left as they are.

Usage (run from a scratch copy, kept here with only this docstring and the path line changed):
    python -m scripts.a5_dependence_diagnostic oracle research/reproduction/a5_run3_dependence_oracle.json
    python -m scripts.a5_dependence_diagnostic loo research/reproduction/a5_run3_dependence_loo.json
It was run against a5_analyse before `min_balance_cells` was added, which does not change
the raw B test it reports.
"""
import sys, json, multiprocessing
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np
from pathlib import Path
from scripts import a5_recovery as rc
from scripts import a5_analyse as a5
R = rc.R
REAL_DEFAULT = a5.default_for_clips
STATE = {"spec": None, "world": None, "cache": {}}

def oracle_vector(spec):
    key = tuple(sorted((k, round(v, 10) if isinstance(v, float) else v) for k, v in spec.items()))
    if key not in STATE["cache"]:
        w = STATE["world"]; acc = 0
        for k in range(400):
            s, _, _ = w.scores(spec, np.random.default_rng([R["seed"], 9, k]))
            q = a5.softmax_of_log(s)
            means = np.array([q[w.actor == a].mean(0) for a in range(len(w.actors))])
            acc = acc + means.mean(0)
        STATE["cache"][key] = acc / 400
    return STATE["cache"][key]

def oracle_default(q, actor, n_actors, leave_out=True):
    return np.broadcast_to(oracle_vector(STATE["spec"]), q.shape)

def install(world, oracle):
    STATE["world"] = world
    if oracle:
        real_scores = rc.World.scores
        def scores(self, spec, rng):
            STATE["spec"] = dict(spec); return real_scores(self, spec, rng)
        rc.World.scores = scores
        a5.default_for_clips = oracle_default

def init(items, companion, oracle):
    rc._init(items, companion)
    install(rc._WORLD, oracle)

def task(t):
    obs, i, k = t
    r = rc.replicate(rc._WORLD, obs, rc._COMPANION, i, k)
    return obs["name"], r["b_sig"], r["b_over_t"]

if __name__ == "__main__":
    oracle = sys.argv[1] == "oracle"
    items = rc.real_items(Path("data/eu_emotions_full_manifest.json"), Path("data/eu_emotions_human_entropy.json"))
    world = rc.World(items)
    install(world, oracle)
    R["balance_zero_calibration_draws"] = 3000
    null = rc.prior_for_balance_zero(world, rc.base_spec())
    obs = []
    specs = {"gain0.5_phi0.5": rc.base_spec(signal=2.0, gain=0.5), "mixed_phi0.5_signal2.0": rc.base_spec(signal=2.0),
             "gain0.0_phi0.5": rc.base_spec(signal=2.0, gain=0.0)}
    for name, base in specs.items():
        spec = rc.prior_for_balance_zero(world, base)
        print(name, "prior", spec["prior"], "B/T fresh", rc.balance_over_total(world, spec), flush=True)
        for c in (0, 1):
            obs.append({"name": f"{name}|companions{c}", "spec": spec, "companions": c})
    tasks = [(o, 100 + i, k) for i, o in enumerate(obs) for k in range(400)]
    with multiprocessing.Pool(3, init, (items, null, oracle)) as pool:
        res = pool.map(task, tasks, chunksize=20)
    out = {o["name"]: {"b_sig": float(np.mean([b for n, b, _ in res if n == o["name"]])),
                       "b_over_t": float(np.mean([x for n, _, x in res if n == o["name"]]))} for o in obs}
    print(json.dumps(out, indent=1))
    json.dump(out, open(sys.argv[2], "w"), indent=1)
