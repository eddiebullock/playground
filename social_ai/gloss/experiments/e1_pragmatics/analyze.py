"""Analyse E1 results: accuracy vs humans, calibration, deferral, human alignment.

  python analyze.py            # every results/*.jsonl
Writes results/summary.json and prints a table.
"""

import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from run import HU, OUT, HU_TASKS

TASK_CODE = {"Irony": "IV", "IndirectSpeech": "ISV", "Deceits": "DV", "Maxims": "MV",
             "Metaphor": "MPV", "Humour": "HV", "CoherenceInference": "CIV"}


def rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va == 0 or vb == 0:
        return float("nan")
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)


def spearman(a, b):
    return pearson(rank(a), rank(b))


def auroc(scores, labels):
    """P(score of a correct item > score of an incorrect item)."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    r = rank(scores)
    rp = sum(r[i] + 1 for i, l in enumerate(labels) if l)
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def ece(conf, correct, bins=10):
    tot, n = 0.0, len(conf)
    for b in range(bins):
        idx = [i for i, c in enumerate(conf) if b / bins < c <= (b + 1) / bins or (b == 0 and c == 0)]
        if idx:
            acc = sum(correct[i] for i in idx) / len(idx)
            cf = sum(conf[i] for i in idx) / len(idx)
            tot += len(idx) / n * abs(acc - cf)
    return tot


def selective(conf, correct, coverage):
    idx = sorted(range(len(conf)), key=lambda i: -conf[i])[: max(1, int(len(conf) * coverage))]
    return sum(correct[i] for i in idx) / len(idx)


def boot_ci(vals, n=2000, seed=0):
    rnd = random.Random(seed)
    ms = sorted(sum(rnd.choice(vals) for _ in vals) / len(vals) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n)]


def entropy(p):
    return -sum(x * math.log(x) for x in p if x > 0)


def human_items(task):
    """item -> (p_correct, distribution over original options or None)."""
    rows = list(csv.DictReader(open(HU / f"human_data/fj/Human_{task}.csv")))
    by = defaultdict(list)
    for r in rows:
        by[int(r["itemNum"])].append(r)
    out = {}
    for item, rs in by.items():
        pc = sum(int(r["Correct"]) for r in rs) / len(rs)
        c = Counter(r["OptionChosen"] for r in rs)
        dist = None
        if all(k.startswith("Answer") for k in c):
            n = max(int(k[6:]) for k in c)
            dist = [c.get(f"Answer{k}", 0) / len(rs) for k in range(1, n + 1)]
        out[item] = (pc, dist)
    return out


def calib_block(recs):
    conf = [max(r["p"]) for r in recs]
    corr = [int(r["p"].index(max(r["p"])) + 1 == r["true"]) for r in recs]
    return {
        "n": len(recs),
        "acc": sum(corr) / len(corr),
        "acc_ci": boot_ci(corr),
        "auroc_conf": auroc(conf, corr),
        "ece": ece(conf, corr),
        "sel_acc_80": selective(conf, corr, 0.8),
        "sel_acc_50": selective(conf, corr, 0.5),
        "mean_conf": sum(conf) / len(conf),
    }


def analyse_hu(recs):
    res = {"overall": calib_block(recs), "tasks": {}}
    errors = Counter()
    align_m, align_h, ent_m, ent_h = [], [], [], []
    for task in HU_TASKS:
        tr = [r for r in recs if r["task"] == task]
        if not tr:
            continue
        block = calib_block(tr)
        hum = human_items(task)
        block["human_acc"] = sum(pc for pc, _ in hum.values()) / len(hum)
        # Seed-averaged model distribution over original options per item.
        per_item = defaultdict(list)
        for r in tr:
            n = r["n_opts"]
            orig = [0.0] * n
            for pos, o in enumerate(r["order"]):
                orig[o - 1] = r["p"][pos]
            per_item[int(r["id"].split(":")[1])].append(orig)
            pred_pos = r["p"].index(max(r["p"]))
            if pred_pos + 1 != r["true"]:
                errors[r["labels"][pred_pos]] += 1
        m_pc, h_pc = [], []
        for item, dists in per_item.items():
            if item not in hum:
                continue
            avg = [sum(d[k] for d in dists) / len(dists) for k in range(len(dists[0]))]
            correct_orig = next(r for r in tr if int(r["id"].split(":")[1]) == item)
            ci = correct_orig["order"][correct_orig["true"] - 1] - 1
            m_pc.append(avg[ci])
            h_pc.append(hum[item][0])
            hd = hum[item][1]
            if hd and len(hd) == len(avg):
                ent_m.append(entropy(avg))
                ent_h.append(entropy(hd))
        block["item_rho_pcorrect"] = spearman(m_pc, h_pc) if len(m_pc) > 3 else float("nan")
        align_m += m_pc
        align_h += h_pc
        res["tasks"][task] = block
    res["overall"]["human_acc_mean_of_tasks"] = sum(b["human_acc"] for b in res["tasks"].values()) / len(res["tasks"])
    res["overall"]["item_rho_pcorrect"] = spearman(align_m, align_h)
    res["overall"]["item_rho_entropy"] = spearman(ent_m, ent_h)
    res["overall"]["n_items_entropy"] = len(ent_m)
    res["error_types"] = dict(errors.most_common())
    return res


def analyse_mustard(recs):
    res = {"overall": calib_block(recs)}
    tp = sum(1 for r in recs if r["sarcasm"] and r["p"].index(max(r["p"])) + 1 == r["true"])
    fn = sum(1 for r in recs if r["sarcasm"]) - tp
    fp = sum(1 for r in recs if not r["sarcasm"] and r["p"].index(max(r["p"])) + 1 != r["true"])
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    res["overall"]["sarcasm_f1"] = 2 * prec * rec / (prec + rec) if prec + rec else 0
    res["overall"]["precision"] = prec
    res["overall"]["recall"] = rec
    # Order bias: does accuracy depend on which slot "sarcastic" sits in?
    res["by_seed"] = {s: calib_block([r for r in recs if r["seed"] == s])["acc"] for s in sorted({r["seed"] for r in recs})}
    res["by_show"] = {sh: calib_block([r for r in recs if r["show"] == sh])["acc"] for sh in sorted({r["show"] for r in recs})}
    return res


def main():
    summary = {}
    for f in sorted(OUT.glob("*.jsonl")):
        recs = [r for r in map(json.loads, open(f)) if r["p"]]
        n_all = sum(1 for _ in open(f))
        key = f.stem
        if key.startswith("hu_"):
            summary[key] = analyse_hu(recs)
        elif key.startswith("sincere_"):
            summary[key] = {"overall": calib_block(recs), "chosen": dict(Counter(
                r["labels"][r["p"].index(max(r["p"]))] for r in recs))}
        else:
            summary[key] = analyse_mustard(recs)
        summary[key]["unparsed"] = n_all - len(recs)
        o = summary[key]["overall"]
        print(f"\n== {key}  n={o['n']} unparsed={summary[key]['unparsed']}")
        print(f"acc {o['acc']:.3f} [{o['acc_ci'][0]:.3f},{o['acc_ci'][1]:.3f}]  AUROC(conf->correct) {o['auroc_conf']:.3f}  "
              f"ECE {o['ece']:.3f}  sel@80% {o['sel_acc_80']:.3f}  sel@50% {o['sel_acc_50']:.3f}")
        if "chosen" in summary[key]:
            print("  chosen option types:", summary[key]["chosen"])
        elif "tasks" in summary[key]:
            print(f"human acc (mean of tasks) {o['human_acc_mean_of_tasks']:.3f}  item rho P(correct) {o['item_rho_pcorrect']:.3f}  "
                  f"item rho entropy {o['item_rho_entropy']:.3f} (n={o['n_items_entropy']})")
            for t, b in summary[key]["tasks"].items():
                print(f"  {t:20s} model {b['acc']:.3f}  human {b['human_acc']:.3f}  AUROC {b['auroc_conf']:.2f}  rho {b['item_rho_pcorrect']:.2f}")
            print("  errors by chosen option type:", summary[key]["error_types"])
        else:
            print(f"F1 {o['sarcasm_f1']:.3f} P {o['precision']:.3f} R {o['recall']:.3f}  by seed {summary[key]['by_seed']}  by show {summary[key]['by_show']}")
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
