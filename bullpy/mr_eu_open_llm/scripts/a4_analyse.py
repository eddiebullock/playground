"""
Amendment A4: analysis of model responses to rendered Snoek AU configurations.

Effects are fitted as in research/scripts/human_au_effects.py: one-vs-rest ridge logistic on
33 AU amplitudes plus centred face gender, then G-computation of do(AU=1) minus do(AU=0). The
model counts as one subject, with fractional responses. Ridge strength is the human one for
that dataset. See research/PREREG_A4_DRAFT.md and research/theory/causal_feature_alignment.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "scripts"))
sys.path.insert(0, str(ROOT))
import human_au_effects as hae  # noqa: E402
from analyse_rotation import holm  # noqa: E402

from scripts import a4_design as d  # noqa: E402

NAMES, MEMBERS = hae.effect_index()
FAM = hae.family_positions(NAMES)
FAMILY_MEMBERS = [MEMBERS[i] for i in FAM]
FAMILIES = list(hae.FAMILIES)
OUTCOMES = ("Aligned", "Not aligned", "Inconclusive")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.ravel(a), np.ravel(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 0 else float("nan")


def softmax(values: Sequence[float]) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    x = np.exp(x - x.max())
    return x / x.sum()


def stimulus_responses(rows: Sequence[Mapping[str, Any]], task: str) -> Dict[str, np.ndarray]:
    """Mean option probability over a complete rotation set, in the canonical option order.

    A stimulus needs a usable response on every rotation, so position cannot return through it.
    """
    wanted = d.rotations(task)
    canon = d.OPTIONS[task]
    seen: Dict[str, Dict[int, np.ndarray]] = {}
    for r in rows:
        if r.get("task") != task or r.get("readout", "mcq") != "mcq" or r.get("status") != "ok":
            continue
        p = softmax(r["joint_logprob"])
        order = [list(r["options"]).index(o) for o in canon]
        seen.setdefault(r["stim_id"], {})[r["rotation"]] = p[order]
    return {s: np.mean([v[k] for k in wanted], axis=0)
            for s, v in seen.items() if all(k in v for k in wanted)}


def design(manifest_rows: Sequence[Mapping[str, Any]]) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Stimulus ids, the 33-column AU matrix and centred gender."""
    ids, au, gender = [], [], []
    for r in manifest_rows:
        config = json.loads(r["au_json"])
        ids.append(r["stim_id"])
        au.append([float(config.get(c, 0.0)) for c in hae.AU_COLS])
        gender.append((r["gender"] == "M") - 0.5)
    return ids, np.array(au, dtype=float).reshape(-1, len(hae.AU_COLS)), np.array(gender, dtype=float)


def fit_tau(P: np.ndarray, X_au: np.ndarray, gender: np.ndarray, lam: float) -> np.ndarray:
    """Effect matrix, options by 25 families, from responses P (stimuli by options)."""
    X_cov = np.column_stack([np.ones(len(P)), gender])
    X = np.column_stack([X_cov, X_au])
    pen = np.r_[0.0, 0.0, np.full(X_au.shape[1], lam)]
    fitted = np.vstack([hae.fit_ridge_logit(X, P[:, k], pen=pen) for k in range(P.shape[1])])
    return hae.do_effects(X_au, X_cov, fitted, FAMILY_MEMBERS)


def cue_null_p(tau: np.ndarray, human: np.ndarray, n: int, rng: np.random.Generator) -> Tuple[float, float]:
    """One-sided p from shuffling cue labels. Returns (p, null 95th percentile).

    Shuffling columns keeps each cue's profile and any effect common to all cues, so a system
    that responds to every cue alike cannot pass, however similar its mean profile is.
    """
    observed = cosine(tau, human)
    norm = np.linalg.norm(tau) * np.linalg.norm(human)
    order = np.argsort(rng.random((n, tau.shape[1])), axis=1)
    null = np.einsum("nkj,kj->n", tau[:, order].transpose(1, 0, 2), human) / norm
    return float((1 + np.sum(null >= observed - 1e-12)) / (n + 1)), float(np.quantile(null, 0.95))


def signal_test(P: np.ndarray, X_au: np.ndarray, gender: np.ndarray, n: int, rng: np.random.Generator) -> float:
    """p that AU amplitudes move responses at all, from shuffling stimuli against responses."""
    Y = P - P.mean(axis=0)
    Z = np.column_stack([np.ones(len(P)), gender, X_au])
    H = np.linalg.pinv(Z.T @ Z + 1e-6 * np.eye(Z.shape[1])) @ Z.T
    stat = lambda M: float(np.sum((H[2:] @ M) ** 2))  # noqa: E731
    observed = stat(Y)
    hits = sum(stat(Y[rng.permutation(len(Y))]) >= observed for _ in range(n))
    return float((1 + hits) / (n + 1))


def bootstrap_ci(P, X_au, gender, lam, human, draws, rng) -> Tuple[float, float]:
    """Percentile CI of CFA, resampling stimuli."""
    values = []
    for _ in range(draws):
        i = rng.integers(0, len(P), len(P))
        values.append(cosine(fit_tau(P[i], X_au[i], gender[i], lam), human))
    low, high = np.nanquantile(values, [0.025, 0.975])
    return float(low), float(high)


def split_half(P, X_au, gender, lam, splits, rng) -> float:
    """Reliability of the model's matrix over stimuli, Spearman-Brown corrected."""
    rs = []
    for _ in range(splits):
        order = rng.permutation(len(P))
        a, b = order[: len(P) // 2], order[len(P) // 2 :]
        rs.append(cosine(fit_tau(P[a], X_au[a], gender[a], lam), fit_tau(P[b], X_au[b], gender[b], lam)))
    return float(hae.spearman_brown(np.mean(rs)))


def decide(gates: Mapping[str, Any], p_signal: float, p_holm: float, cfa: float,
           ci: Sequence[float]) -> str:
    """Missing or failed gates give Inconclusive, as does no measurable AU signal."""
    if not gates or not all(v is True for v in gates.values()):
        return "Inconclusive"
    if not p_signal < d.ALPHA:
        return "Inconclusive"
    if p_holm < d.ALPHA and cfa >= d.SESOI_CFA and ci[0] > 0:
        return "Aligned"
    if ci[1] < d.SESOI_CFA:
        return "Not aligned"
    return "Inconclusive"


def estimate(P, X_au, gender, lam, human, *, rng, cue_n=d.CUE_PERMUTATIONS,
             signal_n=d.SIGNAL_PERMUTATIONS, draws=d.BOOTSTRAP_DRAWS) -> Dict[str, Any]:
    tau = fit_tau(P, X_au, gender, lam)
    p, q95 = cue_null_p(tau, human, cue_n, rng)
    return {"cfa": cosine(tau, human), "p_cue": p, "null_q95": q95,
            "p_signal": signal_test(P, X_au, gender, signal_n, rng),
            "ci": bootstrap_ci(P, X_au, gender, lam, human, draws, rng), "tau": tau}


def keep_families(tau: np.ndarray, drop: Sequence[str]) -> np.ndarray:
    return tau[:, [j for j, f in enumerate(FAMILIES) if f not in drop]]


def sensitivities(tau: np.ndarray, human: np.ndarray) -> Dict[str, float]:
    return {"without_AU22_AU27": cosine(keep_families(tau, d.APPROXIMATED_FAMILIES),
                                        keep_families(human, d.APPROXIMATED_FAMILIES)),
            "without_other": cosine(tau[:-1], human[:-1])}


def direct_effects(responses: Mapping[str, np.ndarray], rows: Sequence[Mapping[str, Any]]) -> Tuple[np.ndarray, List[float]]:
    """One-AU contrast at amplitude 1 against the same identity's neutral, on the family basis.

    Bilateral families take the mean of L and R summed, as the human composite sets both.
    Also returns the mean absolute effect by amplitude, the dose check.
    """
    neutral = {(r["identity"], r["seed"]): responses.get(r["stim_id"]) for r in rows if r["set"] == "neutral"}
    table: Dict[Tuple[str, float], List[np.ndarray]] = {}
    for r in rows:
        if r["set"] != "one_au" or r["stim_id"] not in responses:
            continue
        base = neutral.get((r["identity"], r["seed"]))
        if base is None:
            continue
        amplitude = float(next(iter(json.loads(r["au_json"]).values())))
        table.setdefault((r["cfg"], amplitude), []).append(responses[r["stim_id"]] - base)
    size = len(next(iter(neutral.values()))) if neutral and next(iter(neutral.values())) is not None else 0
    out = np.zeros((size, len(FAMILIES)))
    for j, members in enumerate(FAMILY_MEMBERS):
        cols = [hae.AU_COLS[m] for m in members]
        parts = [np.mean(table[(c, 1.0)], axis=0) for c in cols if (c, 1.0) in table]
        if parts:
            out[:, j] = np.sum(parts, axis=0)
    dose = [float(np.mean([np.abs(np.mean(v, axis=0)).sum() for k, v in table.items() if k[1] == a]))
            for a in d.ONE_AU_AMPLITUDES if any(k[1] == a for k in table)]
    return out, dose


def js(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = 0.5 * (p + q)
    kl = lambda a: np.sum(a * (np.log(np.clip(a, 1e-12, None)) - np.log(np.clip(m, 1e-12, None))), axis=-1)  # noqa: E731
    return 0.5 * (kl(p) + kl(q))


def replicate_check(first: np.ndarray, second: np.ndarray, rng: np.random.Generator, draws: int = 9999) -> Dict[str, float]:
    """Does a configuration carry across identities? V3, and B4's analogue.

    first and second are the same configurations on two identities. The mismatch pairs each
    configuration with the next one's replicate, so it uses no human target. Sign-flip over
    configurations, one-sided.
    """
    diff = js(first, np.roll(second, -1, axis=0)) - js(first, second)
    observed = float(diff.mean())
    signs = rng.choice((-1.0, 1.0), size=(draws, len(diff)))
    null = (signs * diff).mean(axis=1)
    return {"delta_js": observed, "p": float((1 + np.sum(null >= observed)) / (draws + 1))}


def detector_gate(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Share of AU and identity pairs whose detected intensity rises across the three doses.

    rows hold au, identity, amplitude and intensity. AUs the detector lacks are not in rows
    and are listed as unconfirmed, not as failures.
    """
    groups: Dict[Tuple[str, str], Dict[float, float]] = {}
    for r in rows:
        groups.setdefault((r["au"], r["identity"]), {})[float(r["amplitude"])] = float(r["intensity"])
    rising = {}
    for key, values in groups.items():
        series = [values.get(a) for a in d.ONE_AU_AMPLITUDES]
        rising[key] = None not in series and series[0] <= series[1] <= series[2] and series[2] > series[0]
    share = float(np.mean(list(rising.values()))) if rising else 0.0
    covered = {k[0] for k in rising}
    return {"share_rising": share, "pass": share >= d.V_RULES["detector_monotonic_share"],
            "unconfirmed_columns": sorted(set(hae.AU_COLS) - covered),
            "failing": sorted(f"{a}:{i}" for (a, i), ok in rising.items() if not ok)}


def validity_gate(P_main: np.ndarray, replicate: Mapping[str, float], image_rms: Optional[float],
                  detector: Optional[Mapping[str, Any]]) -> Dict[str, bool]:
    """V1 to V4. Anything missing is not passed. No check uses a human target."""
    mean = P_main.mean(axis=0)
    return {"v1_detector": bool(detector and detector.get("pass") is True),
            "v2_image_dependence": image_rms is not None and image_rms >= d.V_RULES["image_rms_nats"],
            "v3_config_carries": bool(replicate.get("p", 1.0) < d.V_RULES["replicate_p"]
                                      and replicate.get("delta_js", 0.0) > 0),
            "v4_domain": bool(mean[-1] <= d.V_RULES["max_abstain_mass"]
                              and mean.max() <= d.V_RULES["max_option_mass"])}


def behaviour_metrics(model: np.ndarray, human: np.ndarray) -> Dict[str, float]:
    """Output-level agreement on the same stimuli. Descriptive, beside CFA.

    Snoek has no correct answer, so error consistency proper is undefined. Chance-corrected
    modal agreement is its closest analogue here.
    """
    a, b = model.argmax(axis=1), human.argmax(axis=1)
    observed = float(np.mean(a == b))
    pa, pb = [np.bincount(x, minlength=model.shape[1]) / len(x) for x in (a, b)]
    chance = float(pa @ pb)
    return {"output_r": hae.pearson(model - model.mean(axis=0), human - human.mean(axis=0)),
            "modal_agreement": observed,
            "modal_kappa": (observed - chance) / (1 - chance) if chance < 1 else float("nan")}


def human_matrix(store: Mapping[str, Any], dataset: str) -> Tuple[np.ndarray, float]:
    block = store["datasets"][dataset]
    return np.array(block["do_effect_probability_au_full_vs_none"])[:, FAM], float(block["ridge_lambda_selected"])


def human_reference(store: Mapping[str, Any]) -> Dict[str, float]:
    return {task: cosine(human_matrix(store, f"{task}_WC")[0], human_matrix(store, f"{task}_EA")[0])
            for task in ("basic", "conversational")}


def pooled_do(table: Any, subjects: np.ndarray, stimuli: np.ndarray, lam: float) -> np.ndarray:
    """table.fit's pooled fit, returning do effects on the family basis."""
    N = table.N[subjects].sum(axis=0)[stimuli]
    keep = N > 0
    X = table.X[stimuli][keep]
    pen = np.r_[np.zeros(table.n_cov), np.full(X.shape[1] - table.n_cov, lam)]
    fitted = []
    for l in range(table.C.shape[0]):
        y = table.C[l][subjects].sum(axis=0)[stimuli][keep] / N[keep]
        fitted.append(hae.fit_ridge_logit(X, y, w=N[keep], pen=pen))
    return hae.do_effects(X[:, table.n_cov:], X[:, : table.n_cov], np.vstack(fitted), FAMILY_MEMBERS)


def human_do_ceiling(frame: Any, labels: Sequence[str], label_col: str, splits: int, seed: int) -> Dict[str, float]:
    """The subjects_and_stimuli ceiling, on do effects: halves share neither."""
    table = hae.StimulusTable(frame, labels, label_col)
    rng = np.random.default_rng(seed)
    groups_sub = [np.where(table.sub_set == s)[0] for s in np.unique(table.sub_set)]
    groups_stim = [np.where(table.stim_strata == s)[0] for s in np.unique(table.stim_strata)]
    rs = []
    for _ in range(splits):
        mask = np.zeros(len(table.subs), dtype=bool)
        for g in groups_sub:
            mask[rng.permutation(g)[: len(g) // 2]] = True
        a, b = [], []
        for g in groups_stim:
            perm = rng.permutation(g)
            a.extend(perm[: len(perm) // 2])
            b.extend(perm[len(perm) // 2 :])
        rs.append(cosine(pooled_do(table, mask, np.array(a), 1.0), pooled_do(table, ~mask, np.array(b), 1.0)))
    r = float(np.mean(rs))
    brown = hae.spearman_brown(r)
    return {"split_half_r": round(r, 4), "spearman_brown": round(brown, 4),
            "ceiling": round(float(np.sqrt(max(brown, 0))), 4), "splits": splits}


def analyse_dataset(rows, manifest_rows, dataset, store, ceiling, seed=d.SEED, **kw) -> Dict[str, Any]:
    """One model on one dataset. Holm and gates are applied by the caller."""
    task = d.TASK_OF[dataset]
    responses = stimulus_responses(rows, task)
    wanted = [r for r in manifest_rows if r["set"] == "main" and r["dataset"] == dataset]
    kept = [r for r in wanted if r["stim_id"] in responses]
    lost = 1 - len(kept) / len(wanted)
    ids, X_au, gender = design(kept)
    P = np.vstack([responses[s] for s in ids])
    human, lam = human_matrix(store, dataset)
    rng = np.random.default_rng(seed)
    out = estimate(P, X_au, gender, lam, human, rng=rng, **kw)
    tau = out.pop("tau")
    out.update(share_lost=round(lost, 4), lost_ok=lost <= d.MAX_LOST_SHARE,
               model_split_half=split_half(P, X_au, gender, lam, 20, rng),
               sensitivities=sensitivities(tau, human), tau=np.round(tau, 4).tolist())
    r_s = max(out["model_split_half"], 1e-6)
    out["cfa_over_ceiling"] = out["cfa"] / (float(ceiling[dataset]["ceiling"]) * np.sqrt(r_s))
    pairs = {}
    for r in wanted:
        pairs.setdefault(r["cfg"], {})["main"] = responses.get(r["stim_id"])
    for r in manifest_rows:
        if r["set"] == "replicate" and r["dataset"] == dataset and r["cfg"] in pairs:
            pairs[r["cfg"]]["replicate"] = responses.get(r["stim_id"])
    usable = [v for v in pairs.values() if v.get("main") is not None and v.get("replicate") is not None]
    out["replicate"] = (replicate_check(np.vstack([v["main"] for v in usable]),
                                        np.vstack([v["replicate"] for v in usable]), rng)
                        if len(usable) > 1 else {})
    out["P_main_mean"] = P.mean(axis=0).tolist()
    return out


def apply_rule(results: Dict[str, Dict[str, Any]], gates: Mapping[str, Mapping[str, Any]]) -> None:
    """Holm across models within one dataset, then the decision per model."""
    eligible = [m for m in d.MODELS if m in results]
    adjusted = dict(zip(eligible, holm([results[m]["p_cue"] for m in eligible]))) if eligible else {}
    for model, entry in results.items():
        entry["p_holm"] = adjusted.get(model, float("nan"))
        entry["gates"] = {**dict(gates.get(model, {})), "clips_kept": entry.get("lost_ok") is True}
        entry["decision"] = ("Descriptive" if model not in d.MODELS else
                             decide(entry["gates"], entry["p_signal"], entry["p_holm"], entry["cfa"], entry["ci"]))


def image_rms(rows: Sequence[Mapping[str, Any]], task: str) -> Optional[float]:
    """Mean RMS change in option log probability, render against pixels removed. V2."""
    free = {r["stim_id"]: r for r in rows if r.get("readout") == "image_free" and r.get("task") == task
            and r.get("status") == "ok"}
    changes = []
    for r in rows:
        other = free.get(r["stim_id"])
        if other is None or r.get("readout", "mcq") != "mcq" or r.get("task") != task or r.get("rotation") != 0:
            continue
        if r.get("status") != "ok" or list(r["options"]) != list(other["options"]):
            continue
        change = np.log(softmax(r["joint_logprob"])) - np.log(softmax(other["joint_logprob"]))
        changes.append(float(np.sqrt(np.mean(change**2))))
    return float(np.mean(changes)) if changes else None


def blind_summary(rows, manifest_rows, store) -> Dict[str, Any]:
    """Secondary route, on six basic labels. "other" has no blind family, so it is left out."""
    families = [d.BLIND_FAMILY[k] for k in d.HUMAN_LABELS["basic"][:-1]]
    blind = {r["stim_id"]: r for r in rows if r.get("readout") == "blind" and r.get("status") == "ok"}
    wanted = [r for r in manifest_rows if r["set"] == "blind" and r["stim_id"] in blind]
    if not wanted:
        return {}
    ids, X_au, gender = design(wanted)
    P = np.vstack([softmax([blind[s]["family_logprob"][f] for f in families]) for s in ids])
    mass = float(np.median([blind[s]["lexicon_mass"] for s in ids]))
    human, lam = human_matrix(store, d.BLIND_DATASET)
    tau = fit_tau(P, X_au, gender, lam)
    pairs = {}
    for r in manifest_rows:
        if r["dataset"] == d.BLIND_DATASET and r["set"] in ("main", "replicate") and r["stim_id"] in blind:
            pairs.setdefault(r["cfg"], {})[r["set"]] = softmax([blind[r["stim_id"]]["family_logprob"][f] for f in families])
    usable = [v for v in pairs.values() if len(v) == 2]
    rng = np.random.default_rng(d.SEED)
    check = (replicate_check(np.vstack([v["main"] for v in usable]), np.vstack([v["replicate"] for v in usable]), rng)
             if len(usable) > 1 else {})
    repeat = [r.get("repeat_change") for r in blind.values() if r.get("repeat_change") is not None]
    gate = {"b1_repeat": bool(repeat) and max(repeat) <= d.B_RULES["repeat"],
            "b2_coverage": mass >= d.B_RULES["coverage"],
            "b4_config_carries": check.get("p", 1.0) < d.B_RULES["sign_flip_p"] and check.get("delta_js", 0.0) > 0}
    return {"median_lexicon_mass": mass, "replicate": check, "gate": gate,
            "cfa_six_labels": cosine(tau, human[:-1])}


def scale_summary(rows, manifest_rows, store) -> Dict[str, float]:
    """Amplitude sensitivity, with the 1x fit on the same configurations beside it."""
    out = {}
    for dataset in d.SCALE_DATASETS:
        responses = stimulus_responses(rows, d.TASK_OF[dataset])
        human, lam = human_matrix(store, dataset)
        main = {r["cfg"]: r for r in manifest_rows if r["set"] == "main" and r["dataset"] == dataset}
        for scale in d.SCALES:
            chosen = [r for r in manifest_rows if r["set"] == f"scale_{scale}" and r["dataset"] == dataset
                      and r["stim_id"] in responses]
            base = [main[r["cfg"]] for r in chosen if main[r["cfg"]]["stim_id"] in responses]
            if chosen:
                ids, X_au, gender = design(chosen)
                out[f"{dataset}_{scale}x"] = cosine(fit_tau(np.vstack([responses[s] for s in ids]), X_au, gender, lam), human)
            if base and f"{dataset}_1x_same_configs" not in out:
                ids, X_au, gender = design(base)
                out[f"{dataset}_1x_same_configs"] = cosine(fit_tau(np.vstack([responses[s] for s in ids]), X_au, gender, lam), human)
    return out


def human_stimulus_table(frame: Any, labels: Sequence[str], label_col: str) -> Dict[str, np.ndarray]:
    """Mean human response per configuration, for the output-level metrics."""
    counts = frame.groupby(["cfg", label_col]).size().unstack(fill_value=0).reindex(columns=labels, fill_value=0)
    values = counts.to_numpy(float)
    return dict(zip(counts.index, values / values.sum(axis=1, keepdims=True)))


def read_shards(paths: Sequence[Path], expected_hash: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            rows.append(json.loads(line))
    hashes = {r.get("design_sha256") for r in rows}
    if hashes != {expected_hash}:
        raise SystemExit(f"Cells carry design hashes {sorted(map(str, hashes))[:3]}, not {expected_hash}")
    return rows


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, default=Path("results/a4/manifest.csv"))
    parser.add_argument("--cells_dir", type=Path, default=Path("results/a4"))
    parser.add_argument("--human", type=Path, default=Path("research/reproduction/human_au_effects.json"))
    parser.add_argument("--ceiling", type=Path, default=Path("research/reproduction/a4_human_ceiling.json"))
    parser.add_argument("--gates", type=Path, required=True,
                        help="JSON, per model: rq0 and g2_g4 as true or false. Missing means not passed.")
    parser.add_argument("--detector", type=Path, help="CSV of au, identity, amplitude, intensity.")
    parser.add_argument("--zip", type=Path, default=Path("data/external/snoek_au/au_emotion_data.zip"))
    parser.add_argument("--output", type=Path, default=Path("results/a4/summary.json"))
    parser.add_argument("--design_sha256", required=True)
    args = parser.parse_args()
    if args.design_sha256 != d.design_hash():
        raise SystemExit("--design_sha256 is not this design block's hash.")
    store = json.loads(args.human.read_text(encoding="utf-8"))
    ceiling = json.loads(args.ceiling.read_text(encoding="utf-8"))["datasets"]
    manifest = pd.read_csv(args.manifest).to_dict("records")
    recorded = json.loads(args.gates.read_text(encoding="utf-8"))
    detector = detector_gate(pd.read_csv(args.detector).to_dict("records")) if args.detector else None
    frames, _ = hae.load_zip(args.zip)
    out: Dict[str, Any] = {"design_sha256": args.design_sha256, "human_reference": human_reference(store),
                           "detector": detector, "datasets": {}, "models": {}}
    models = [m for m in (*d.MODELS, *d.DESCRIPTIVE_MODELS) if list(args.cells_dir.glob(f"cells_{m}_*.jsonl"))]
    cells = {m: read_shards(sorted(args.cells_dir.glob(f"cells_{m}_*.jsonl")), args.design_sha256) for m in models}
    for dataset in d.DATASETS:
        task = d.TASK_OF[dataset]
        frame, label_col = frames[dataset]
        human_table = human_stimulus_table(frame, d.HUMAN_LABELS[task], label_col)
        results = {m: analyse_dataset(cells[m], manifest, dataset, store, ceiling) for m in models}
        gates = {}
        for model, entry in results.items():
            responses = stimulus_responses(cells[model], task)
            chosen = [r for r in manifest if r["set"] == "main" and r["dataset"] == dataset and r["stim_id"] in responses]
            P = np.vstack([responses[r["stim_id"]] for r in chosen])
            entry["behaviour"] = behaviour_metrics(P, np.vstack([human_table[r["cfg"]] for r in chosen]))
            direct, dose = direct_effects(responses, manifest)
            entry["direct_cfa"], entry["direct_dose"] = cosine(direct, human_matrix(store, dataset)[0]), dose
            given = recorded.get(model, {})
            gates[model] = {"rq0": given.get("rq0") is True, "g2_g4": given.get("g2_g4") is True,
                            **validity_gate(P, entry["replicate"], image_rms(cells[model], task), detector)}
        apply_rule(results, gates)
        out["datasets"][dataset] = results
    for model in models:
        out["models"][model] = {"blind": blind_summary(cells[model], manifest, store),
                                "scale": scale_summary(cells[model], manifest, store)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8")
    print(args.output, hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
