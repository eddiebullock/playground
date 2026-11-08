"""
Does A2's frozen actor sign-flip step-down maxT stay at alpha under per-actor portrayal gain?

Read-only for the repo. Synthetic cells in A2's format, over the real 243 items and 17 actors,
go through the frozen scripts.analyse_stimulus_interventions.test_family with the frozen
default flip settings (all 2^17 flips, as the prereg states). Nothing in the repo is changed.

Null mechanisms (all with A2 recovery's prior/actor/item/cell noise, EVIDENCE 2 on the acted
family in every condition):
  sharp      : acted evidence scaled by per-actor gain g_a; no condition changes anything.
               Gain cancels within item. Control.
  hetero     : as sharp, plus an actor-by-condition-by-family shift tau * g_a * N(0,1),
               tau = .3. Symmetric, heteroscedastic across actors.
  skew_k{k}_s{s}: A5 analogue. Each confirmatory a-side condition moves the outcome family of
               each contrast by s * dir_k * k * (g_a - E[g]): the edit removes evidence in
               proportion to the actor's gain, offset by a constant artefact, so the population
               mean actor effect is exactly zero but the actor distribution is skewed (log-normal).
               s=+1 and s=-1 put the long tail on the predicted and on the opposite side.
Gain g_a = exp(N(0, sd)), sd in {0, .3, .5}, as in scripts/a5_recovery.py.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import os
os.chdir(REPO)

from scripts import analyse_stimulus_interventions as an  # noqa: E402
from scripts.run_stimulus_interventions import INPUTS, LEXICON, OPTION_TO_FAMILY, load_items  # noqa: E402
from scripts.recovery_stimulus_interventions import ACTOR_SD, ITEM_SD, CELL_SD, EVIDENCE  # noqa: E402

FAMILIES = sorted(LEXICON)
F = len(FAMILIES)
FI = {f: i for i, f in enumerate(FAMILIES)}
ROUTES = ("single_face", "video_face")
OUT = Path(os.environ.get("A2SKEW_OUT", "/tmp/a2skew"))
OUT.mkdir(parents=True, exist_ok=True)


def fam(label):
    return FI[OPTION_TO_FAMILY[label]]


def skew_shifts(condition, label, low):
    """(family index, unit sign) pairs: unit sign is dir_k for every contrast this touches,
    built so each confirmatory contrast's actor effect is c * dir_k * (g_a - m), c > 0."""
    out = []
    if condition == "mask_mouth_neutral":
        if label in ("Happy", "Disgusted"):
            out.append((fam(label), -1))          # P1, P2, P6 (dir -1)
        if label in ("Happy", "Sad", "Angry"):
            out.append((FI["Neutral"], +1))       # P7 (dir +1)
        if label == "Disgusted":
            out.append((fam("Angry"), +1))        # P8 (dir +1)
    if condition == "mask_eyes_neutral" and label in ("Afraid", "Sad"):
        out.append((fam(label), -1))              # P3, P4; P6 via Afraid on b side -> -2
    if condition == "mask_brows_neutral" and label == "Angry":
        out.append((fam(label), -1))              # P5
    if condition == "sf_low" and label == "Happy":
        out.append((fam(label), +1))              # P9 (dir +1)
    if condition == "sf_low" and label == "Sad":
        out.append((fam(label), -1))              # P10 (dir -1)
    if condition in an.SHUFFLES and (label in ("Happy", "Surprised") or low):
        out.append((fam(label), -1))              # P12, P13
    if condition == "trunc_first_third":
        partner = {"Afraid": "Surprised", "Surprised": "Afraid", "Angry": "Disgusted", "Disgusted": "Angry"}.get(label)
        if partner:
            out.append((fam(partner), +1))        # P14a/b, P15a/b (dir +1)
    return out


def rows_for(items, mech, gain_sd, seed):
    rng = np.random.default_rng(seed)
    actors = sorted({i["actor"] for i in items})
    gain = dict(zip(actors, np.exp(rng.normal(0, gain_sd, len(actors)))))
    m = float(np.exp(gain_sd ** 2 / 2))
    prior = rng.normal(0, 1, F)
    actor_effect = {a: rng.normal(0, ACTOR_SD, F) for a in actors}
    hetero = {}
    if mech == "hetero":
        for a in actors:
            for route in ROUTES:
                for c in INPUTS[route]["conditions"]:
                    hetero[(a, route, c)] = 0.3 * gain[a] * rng.normal(0, 1, F) if c != "original" else np.zeros(F)
    k = s = 0.0
    if mech.startswith("skew"):
        _, kk, ss = mech.split("_")
        k, s = float(kk[1:]), float(ss[1:])
    rows = []
    for item in items:
        a = item["actor"]
        f = fam(item["label"])
        low = an.is_low_intensity(item["trial_id"])
        base = prior + actor_effect[a] + rng.normal(0, ITEM_SD, F)
        base[f] += EVIDENCE * gain[a]
        for route in ROUTES:
            for c in INPUTS[route]["conditions"]:
                scores = base + rng.normal(0, CELL_SD, F)
                if mech == "hetero":
                    scores = scores + hetero[(a, route, c)]
                if k:
                    for fi, unit in skew_shifts(c, item["label"], low):
                        scores[fi] += s * unit * k * (gain[a] - m)
                rows.append({**item, "input_name": route, "condition": c, "status": "ok",
                             "blind": {"family_logprob": dict(zip(FAMILIES, scores.tolist())), "lexicon_mass": 0.5}})
    return rows, actors


ITEMS = None


def init():
    global ITEMS
    items = load_items(Path("data/eu_emotions_full_manifest.json"), Path("data/eu_emotions_human_entropy.json"), Path("."))
    keep = ("trial_id", "actor", "label", "human_modal_label", "weak_label")
    ITEMS = [{k: i[k] for k in keep} for i in items]


def one(args):
    mech, gain_sd, rep = args
    seed = [hash(mech) % 2**31, int(gain_sd * 100), rep]
    seed = [abs(int.from_bytes(mech.encode(), "little")) % 2**31, int(round(gain_sd * 100)), rep]
    rows, actors = rows_for(ITEMS, mech, gain_sd, seed)
    cells = an.Cells(rows)
    out = {}
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for route in ROUTES:
            family = [c for c in an.CONFIRMATORY if c.route == route]
            res = an.test_family(cells, family, actors)  # frozen defaults: exact 2^17 flips
            out[route] = {cid: (r["p_raw"], r["p_maxt"], r["n_actors"]) for cid, r in res.items()}
    return mech, gain_sd, rep, out


def wilson(x, n, z=1.959964):
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(p, 4), round(c - h, 4), round(c + h, 4)]


def summarise(results):
    table = {}
    for (mech, g), reps in results.items():
        n = len(reps)
        entry = {"n": n}
        for route in ROUTES:
            ids = list(reps[0][route])
            raw = {cid: sum(r[route][cid][0] < an.ALPHA for r in reps) for cid in ids}
            adj = {cid: sum(r[route][cid][1] < an.ALPHA for r in reps) for cid in ids}
            fwer = sum(any(r[route][cid][1] < an.ALPHA for cid in ids) for r in reps)
            entry[route] = {
                "n_contrasts": len(ids),
                "n_actors": {cid: reps[0][route][cid][2] for cid in ids},
                "fwer_stepdown": wilson(fwer, n),
                "per_test_stepdown": {cid: wilson(adj[cid], n) for cid in ids},
                "per_test_unadjusted": {cid: wilson(raw[cid], n) for cid in ids},
                "max_per_test_unadjusted": max(wilson(raw[cid], n)[0] for cid in ids),
            }
        entry["fwer_either_route"] = wilson(sum(any(r[rt][cid][1] < an.ALPHA for rt in ROUTES for cid in r[rt]) for r in reps), n)
        table[f"{mech}|gain{g}"] = entry
    return table


if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    mechs = sys.argv[2].split(",") if len(sys.argv) > 2 else ["sharp", "hetero", "skew_k0.5_s1", "skew_k0.5_s-1", "skew_k2_s1", "skew_k2_s-1"]
    gains = (0.0, 0.3, 0.5)
    tag = sys.argv[3] if len(sys.argv) > 3 else "full"
    log = OUT / f"a2_gain_null_{tag}.jsonl"
    done = set()
    results = {}
    if log.exists():
        for line in log.read_text().splitlines():
            d = json.loads(line)
            done.add((d["m"], d["g"], d["r"]))
            results.setdefault((d["m"], d["g"]), []).append(d["out"])
    jobs = [(m, g, r) for m in mechs for g in gains for r in range(reps) if (m, g, r) not in done]
    t0 = time.time()
    with open(log, "a") as fh, mp.Pool(int(os.environ.get("WORKERS", 7)), initializer=init) as pool:
        for i, (m, g, r, out) in enumerate(pool.imap_unordered(one, jobs, chunksize=4)):
            results.setdefault((m, g), []).append(out)
            fh.write(json.dumps({"m": m, "g": g, "r": r, "out": out}) + "\n"); fh.flush()
            if (i + 1) % 500 == 0:
                print(f"{i+1}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    results = {k: v for k, v in results.items() if k[0] in mechs}
    summary = summarise(results)
    (OUT / f"a2_gain_null_{tag}.json").write_text(json.dumps({"replicates": reps, "alpha": an.ALPHA, "flips": "exact 2^17 (frozen default)", "results": summary}, indent=1))
    for key, e in summary.items():
        print(key, "FWER sf", e["single_face"]["fwer_stepdown"], "vf", e["video_face"]["fwer_stepdown"],
              "max raw sf", e["single_face"]["max_per_test_unadjusted"], "vf", e["video_face"]["max_per_test_unadjusted"])
    print(f"done {time.time()-t0:.0f}s")
