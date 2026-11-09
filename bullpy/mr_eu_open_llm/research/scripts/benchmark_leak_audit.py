#!/usr/bin/env python
"""Audit public multiple-choice affect benchmarks for the leaks found in our own 6AFC task.

Our EU-Emotion 6AFC prompt had the target at option 1 on every item and one fixed option set
per target label (research/reproduction/prompt_confound_audit.json, DECISIONS 2026-11-04).
This script asks whether public VLM / MLLM benchmarks with facial-emotion or affect items share
either leak. For each benchmark subset it computes, from annotation text only:

1. the distribution of the correct answer's position, with a chi-square goodness-of-fit test
   against "each item's answer is uniform over its own options" (analytic p, if scipy is
   installed, and a Monte Carlo p that is exact in expectation for mixed option counts);
2. whether the (unordered) option set identifies the target label: distinct option sets per
   label, labels per option set, and the accuracy of a leave-one-out option-set-only predictor
   (majority correct label among the other items that show the same option set);
3. text-only baselines against chance: always-same-position (in-sample best position, and a
   leave-one-out choice), the option-set predictor, and two supplementary predictors
   (leave-one-out answer prior: P(label is correct | label offered); longest option).

Only annotation text is downloaded, into a scratch directory outside the repo; images, videos
and image columns are never fetched or are dropped while streaming. Each source records URL,
revision and sha256. Writes research/reproduction/benchmark_leak_audit.json. Exploratory,
post hoc; no GPU, no model calls.

Usage: python research/scripts/benchmark_leak_audit.py [--scratch DIR] [--out PATH]
"""

from __future__ import annotations

import argparse
import ast
import collections
import csv
import datetime as dt
import hashlib
import io
import json
import math
import random
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCRATCH = Path(
    "/private/tmp/claude-502/-Users-eb2007/cf56ec16-1588-406d-9db5-6872ba6bedc3/scratchpad/bench_audit"
)
DEFAULT_OUT = ROOT / "research/reproduction/benchmark_leak_audit.json"
SEED = 20261109
N_SIM = 20000
UA = {"User-Agent": "benchmark-leak-audit/1.0 (research; annotation text only)"}
HF = "https://huggingface.co/datasets"
GH = "https://raw.githubusercontent.com"

# ---------------------------------------------------------------------------
# Metric functions (pure; tested in tests/test_benchmark_leak_audit.py)
# ---------------------------------------------------------------------------


def norm(text) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def chance_rate(n_options):
    return sum(1.0 / k for k in n_options) / len(n_options)


def _chi2_stat(counts, expected):
    return sum((c - e) ** 2 / e for c, e in zip(counts, expected) if e > 0)


def position_test(answer_idx, n_options, n_sim=N_SIM, seed=SEED):
    """Chi-square GOF of answer positions against per-item uniform positions.

    Expected count at position j is sum_i [j < k_i] / k_i, so items with different numbers of
    options are handled. The Monte Carlo p draws each item's position uniformly from its own
    k_i options and compares the same statistic.
    """
    if len(answer_idx) != len(n_options) or not answer_idx:
        raise ValueError("need equal-length, non-empty inputs")
    kmax = max(n_options)
    counts = [0] * kmax
    for a, k in zip(answer_idx, n_options):
        if not 0 <= a < k:
            raise ValueError(f"answer index {a} outside {k} options")
        counts[a] += 1
    expected = [sum(1.0 / k for k in n_options if j < k) for j in range(kmax)]
    stat = _chi2_stat(counts, expected)
    df = sum(e > 0 for e in expected) - 1
    p_chi2 = None
    try:
        from scipy.stats import chi2

        p_chi2 = float(chi2.sf(stat, df)) if df > 0 else None
    except ImportError:  # pragma: no cover
        pass
    p_mc = None
    if df > 0 and n_sim:
        rng = random.Random(seed)
        hits = 0
        for _ in range(n_sim):
            sim = [0] * kmax
            for k in n_options:
                sim[rng.randrange(k)] += 1
            hits += _chi2_stat(sim, expected) >= stat - 1e-12
        p_mc = (hits + 1) / (n_sim + 1)
    n = len(answer_idx)
    by_k = collections.defaultdict(lambda: None)
    for k in sorted(set(n_options)):
        c = [0] * k
        for a, kk in zip(answer_idx, n_options):
            if kk == k:
                c[a] += 1
        by_k[str(k)] = c
    return {
        "n": n,
        "position_counts_by_n_options": dict(by_k),
        "n_options_counts": {str(k): v for k, v in sorted(collections.Counter(n_options).items())},
        "position_counts": counts,
        "expected_counts": [round(e, 3) for e in expected],
        "max_position_share": max(counts) / n,
        "chi2": stat,
        "df": df,
        "p_chi2": p_chi2,
        "p_mc": p_mc,
        "n_sim": n_sim if df > 0 else 0,
    }


def _tie_score(true_value, candidates_counter, allowed=None):
    """Expected accuracy of picking uniformly among the most common candidates."""
    if allowed is not None:
        candidates_counter = collections.Counter({c: v for c, v in candidates_counter.items() if c in allowed})
    if not candidates_counter:
        return None
    top = max(candidates_counter.values())
    tied = [c for c, v in candidates_counter.items() if v == top]
    return (1.0 / len(tied)) if true_value in tied else 0.0


def option_set_leak(labels, option_sets):
    """Does the unordered option set identify the target label?

    labels: normalised correct-answer text per item. option_sets: hashable unordered key per item
    (e.g. tuple(sorted(normalised options))). The leave-one-out predictor guesses the most
    common correct label among the *other* items with the same option set, restricted to labels
    on offer; with no such peer it guesses uniformly among the item's options (1/k).
    """
    if len(labels) != len(option_sets) or not labels:
        raise ValueError("need equal-length, non-empty inputs")
    by_set = collections.defaultdict(collections.Counter)
    sets_by_label = collections.defaultdict(set)
    for lab, s in zip(labels, option_sets):
        by_set[s][lab] += 1
        sets_by_label[lab].add(s)
    scores, fallback, memo = [], 0, {}
    for lab, s in zip(labels, option_sets):
        if (lab, s) not in memo:  # the LOO score depends only on (label, set)
            peers = by_set[s].copy()
            peers[lab] -= 1
            peers = +peers
            memo[(lab, s)] = _tie_score(lab, peers, allowed=set(s))
        sc = memo[(lab, s)]
        if sc is None:
            sc = 1.0 / len(s)
            fallback += 1
        scores.append(sc)
    n = len(labels)
    labels_per_set = [len(c) for c in by_set.values()]
    sets_per_label = [len(v) for v in sets_by_label.values()]
    single_label_sets = {s for s, c in by_set.items() if len(c) == 1 and sum(c.values()) >= 2}
    return {
        "n": n,
        "n_labels": len(sets_by_label),
        "n_option_sets": len(by_set),
        "option_sets_per_label_mean": sum(sets_per_label) / len(sets_per_label),
        "option_sets_per_label_min": min(sets_per_label),
        "option_sets_per_label_max": max(sets_per_label),
        "labels_per_option_set_mean": sum(labels_per_set) / len(labels_per_set),
        "labels_per_option_set_max": max(labels_per_set),
        "n_labels_with_one_option_set": sum(v == 1 for v in sets_per_label),
        "share_items_in_repeated_single_label_sets": sum(s in single_label_sets for s in option_sets) / n,
        "share_items_no_loo_peer": fallback / n,
        "loo_option_set_accuracy": sum(scores) / n,
        "_scores": scores,
    }


def fixed_position_baseline(answer_idx, n_options):
    """Always answer the same position: in-sample best, and a leave-one-out choice."""
    n = len(answer_idx)
    counts = collections.Counter(answer_idx)
    best_pos = max(sorted(counts), key=lambda j: counts[j])
    in_sample = counts[best_pos] / n
    loo, memo = [], {}
    for a, k in zip(answer_idx, n_options):
        if (a, k) not in memo:
            others = counts.copy()
            others[a] -= 1
            others = +others
            sc = _tie_score(a, others, allowed=set(range(k)))
            memo[(a, k)] = 1.0 / k if sc is None else sc
        loo.append(memo[(a, k)])
    return {"best_position": best_pos, "in_sample_accuracy": in_sample, "loo_accuracy": sum(loo) / n, "_scores": loo}


def position_given_k_baseline(answer_idx, n_options):
    """Leave-one-out modal answer position among other items with the same number of options."""
    groups = collections.defaultdict(collections.Counter)
    for a, k in zip(answer_idx, n_options):
        groups[k][a] += 1
    scores, memo = [], {}
    for a, k in zip(answer_idx, n_options):
        if (a, k) not in memo:
            others = groups[k].copy()
            others[a] -= 1
            others = +others
            sc = _tie_score(a, others)
            memo[(a, k)] = 1.0 / k if sc is None else sc
        scores.append(memo[(a, k)])
    return {"loo_accuracy": sum(scores) / len(scores), "_scores": scores}


def predictor_accuracies(answer_idx, opts_norm):
    """Mean accuracy of each text-only predictor for given answers and normalised options."""
    n_options = [len(o) for o in opts_norm]
    labels = [o[a] for o, a in zip(opts_norm, answer_idx)]
    sets = [tuple(sorted(set(o))) for o in opts_norm]
    fixed = fixed_position_baseline(answer_idx, n_options)
    return {
        "always_same_position_in_sample": fixed["in_sample_accuracy"],
        "always_same_position_loo": fixed["loo_accuracy"],
        "same_position_given_n_options_loo": position_given_k_baseline(answer_idx, n_options)["loo_accuracy"],
        "option_set_only_loo": option_set_leak(labels, sets)["loo_option_set_accuracy"],
        "answer_prior_loo_supplementary": answer_prior_baseline(labels, opts_norm)["loo_accuracy"],
    }


def null_simulation(answer_idx, opts_norm, n_null=1000, seed=SEED):
    """Calibrate each predictor against answers drawn uniformly from each item's own options.

    Leave-one-out majority rules are biased below 1/k when classes are balanced, and in-sample
    best position is biased above it, so each observed accuracy is compared with its own null
    distribution. p = (1 + #null >= observed) / (1 + n_null).
    """
    obs = predictor_accuracies(answer_idx, opts_norm)
    rng = random.Random(seed)
    sims = {k: [] for k in obs}
    for _ in range(n_null):
        sim_idx = [rng.randrange(len(o)) for o in opts_norm]
        for k, v in predictor_accuracies(sim_idx, opts_norm).items():
            sims[k].append(v)
    out = {}
    for k, v in obs.items():
        null = sorted(sims[k])
        out[k] = {
            "accuracy": v,
            "null_mean": sum(null) / n_null,
            "null_q975": null[min(n_null - 1, int(0.975 * n_null))],
            "p_null": (1 + sum(x >= v - 1e-12 for x in null)) / (1 + n_null),
        }
    return out


def answer_prior_baseline(labels, options_norm):
    """Leave-one-out: pick the offered option with the highest P(correct | offered) elsewhere.

    Captures leaks where a label is offered mostly when it is the answer, even if option sets
    differ. Labels never seen elsewhere get the pooled rate. Ties are scored in expectation.
    """
    offered, correct = collections.Counter(), collections.Counter()
    for lab, opts in zip(labels, options_norm):
        for o in set(opts):
            offered[o] += 1
        correct[lab] += 1
    tot_off, tot_cor = sum(offered.values()), sum(correct.values())
    scores, memo = [], {}
    for lab, opts in zip(labels, options_norm):
        key = (lab, frozenset(opts))
        if key in memo:
            scores.append(memo[key])
            continue
        rates = {}
        o_off, o_cor = tot_off - len(set(opts)), tot_cor - 1
        pooled = o_cor / o_off if o_off else 0.0
        for o in set(opts):
            off = offered[o] - 1
            cor = correct[o] - (1 if o == lab else 0)
            rates[o] = cor / off if off > 0 else pooled
        top = max(rates.values())
        tied = [o for o, r in rates.items() if abs(r - top) < 1e-12]
        memo[key] = 1.0 / len(tied) if lab in tied else 0.0
        scores.append(memo[key])
    return {"loo_accuracy": sum(scores) / len(scores), "_scores": scores}


def longest_option_baseline(options, answer_idx):
    scores = []
    for opts, a in zip(options, answer_idx):
        lens = [len(str(o)) for o in opts]
        top = max(lens)
        tied = [j for j, v in enumerate(lens) if v == top]
        scores.append(1.0 / len(tied) if a in tied else 0.0)
    return {"accuracy": sum(scores) / len(scores), "_scores": scores}


def versus_chance(scores, n_options):
    """Normal approximation to a Poisson-binomial test of accuracy against per-item 1/k.

    Approximate for leave-one-out predictors (items are not independent); reported as a
    screening statistic, not a decision test.
    """
    p = [1.0 / k for k in n_options]
    n = len(scores)
    obs, exp = sum(scores), sum(p)
    var = sum(q * (1 - q) for q in p)
    z = (obs - exp) / math.sqrt(var) if var > 0 else float("nan")
    p_one = 0.5 * math.erfc(z / math.sqrt(2)) if var > 0 else float("nan")
    return {"accuracy": obs / n, "chance": exp / n, "excess": (obs - exp) / n, "z": z, "p_one_sided_approx": p_one}


def holm(pvalues):
    """Holm step-down adjusted p values, in input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adj[i] = running
    return adj


HOLM_KEYS = ("p_position_mc", "p_same_position_given_k", "p_option_set", "p_answer_prior")


def add_holm(summary, keys=HOLM_KEYS):
    """Holm adjustment across every subset x test listed in keys (one family)."""
    cells = [(i, k) for i, row in enumerate(summary) for k in keys if row.get(k) is not None]
    adj = holm([summary[i][k] for i, k in cells])
    for (i, k), a in zip(cells, adj):
        summary[i][k + "_holm"] = a
    return len(cells)


def audit_items(items, n_sim=N_SIM, n_null=1000):
    """items: list of dicts with 'options' (presented order) and 'answer_idx'."""
    options = [list(it["options"]) for it in items]
    answer_idx = [int(it["answer_idx"]) for it in items]
    n_options = [len(o) for o in options]
    opts_norm = [[norm(x) for x in o] for o in options]
    labels = [o[a] for o, a in zip(opts_norm, answer_idx)]
    sets = [tuple(sorted(set(o))) for o in opts_norm]
    pos = position_test(answer_idx, n_options, n_sim=n_sim)
    leak = option_set_leak(labels, sets)
    fixed = fixed_position_baseline(answer_idx, n_options)
    longest = longest_option_baseline(options, answer_idx)
    calibrated = null_simulation(answer_idx, opts_norm, n_null=n_null)
    label_counts = collections.Counter(labels)
    return {
        "n_items": len(items),
        "chance": chance_rate(n_options),
        "majority_label_share": max(label_counts.values()) / len(labels),
        "n_distinct_option_lists_ordered": len({tuple(o) for o in opts_norm}),
        "fixed_option_list": len({tuple(o) for o in opts_norm}) == 1,
        "position": pos,
        "option_set": {k: v for k, v in leak.items() if not k.startswith("_")},
        "baselines": dict(
            calibrated,
            always_same_position_in_sample=dict(calibrated["always_same_position_in_sample"],
                                                best_position=fixed["best_position"]),
            longest_option_supplementary=versus_chance(longest["_scores"], n_options),
        ),
        "n_null": n_null,
        "label_counts": dict(label_counts.most_common(40)),
    }


# ---------------------------------------------------------------------------
# Download helpers (annotation text only)
# ---------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(url, dest: Path) -> dict:
    """Download a small text file once into the scratch dir; return provenance."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=300) as r:
            data = r.read()
        dest.write_bytes(data)
    data = dest.read_bytes()
    return {"url": url, "local": str(dest), "bytes": len(data), "sha256": sha256_bytes(data)}


class HTTPRange(io.RawIOBase):
    """Seekable read-only HTTP file using Range requests (for zip central directories)."""

    def __init__(self, url):
        self.url, self.pos, self.fetched = url, 0, 0
        req = urllib.request.Request(url, method="HEAD", headers=UA)
        with urllib.request.urlopen(req, timeout=120) as r:
            self.size = int(r.headers["Content-Length"])
            self.final = r.geturl()

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size or len(b) == 0:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        req = urllib.request.Request(self.final, headers={**UA, "Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, timeout=300) as r:
            data = r.read()
        b[: len(data)] = data
        self.pos += len(data)
        self.fetched += len(data)
        return len(data)


class HashingReader(io.RawIOBase):
    def __init__(self, raw):
        self.raw, self.h, self.n = raw, hashlib.sha256(), 0

    def readable(self):
        return True

    def readinto(self, b):
        data = self.raw.read(len(b))
        if not data:
            return 0
        self.h.update(data)
        self.n += len(data)
        b[: len(data)] = data
        return len(data)


def cached(scratch: Path, name: str, builder):
    """Cache extracted annotation rows (text only) plus provenance as JSON in scratch."""
    path = scratch / "extracted" / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    obj = builder()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))
    return obj


def parquet_text_columns(url, scratch: Path, name: str, drop=("image",)):
    """Download a parquet file to scratch, hash it, keep non-image columns, delete the file."""
    import pyarrow.parquet as pq

    tmp = scratch / "tmp" / f"{name}.parquet"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    h, n = hashlib.sha256(), 0
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            n += len(chunk)
            f.write(chunk)
    pf = pq.ParquetFile(tmp)
    cols = [c for c in pf.schema_arrow.names if c not in drop]
    df = pf.read(columns=cols).to_pandas()
    tmp.unlink()
    rows = json.loads(df.to_json(orient="records", force_ascii=False))
    return rows, {"url": url, "bytes": n, "sha256": h.hexdigest(), "dropped_columns": list(drop),
                  "note": "parquet downloaded to scratch, hashed, image column dropped, file deleted"}


# ---------------------------------------------------------------------------
# Benchmark loaders: each returns (subsets: {name: [items]}, sources: [provenance], notes)
# ---------------------------------------------------------------------------

LETTERS = "ABCDEFGHIJ"


def load_mmbench(scratch):
    rev = "ecd23e001f5c649c959eda84b846a01cc6fafbde"
    url = f"{HF}/lmms-lab/MMBench_EN/resolve/{rev}/data/dev-00000-of-00001-75b6649fb044d38b.parquet"

    def build():
        rows, prov = parquet_text_columns(url, scratch, "mmbench_en_dev")
        keep = [r for r in rows if r.get("category") == "image_emotion"]
        return {"rows": keep, "prov": prov, "n_rows_file": len(rows)}

    obj = cached(scratch, "mmbench_en_dev_image_emotion", build)
    # The file stores each question four times (index + k * 1e6), options rotated for
    # CircularEval. Keep the base copy (index < 1e6) so leave-one-out peers are other questions.
    n_copies = collections.Counter(int(r["index"]) % 1_000_000 for r in obj["rows"])
    items = []
    for r in obj["rows"]:
        if int(r["index"]) >= 1_000_000:
            continue
        opts = [(L, r[L]) for L in "ABCD" if r.get(L) not in (None, "", "nan")]
        letters = [L for L, _ in opts]
        if r["answer"] not in letters:
            continue
        items.append({"id": r["index"], "options": [str(v) for _, v in opts], "answer_idx": letters.index(r["answer"])})
    prov = dict(obj["prov"], repo="lmms-lab/MMBench_EN", revision=rev, filter="category == image_emotion",
                n_rows_file=obj["n_rows_file"], n_rows_image_emotion=len(obj["rows"]),
                n_unique_questions=len(n_copies), copies_per_question=dict(collections.Counter(n_copies.values())),
                dedup="kept base copy (index < 1e6) of each CircularEval group")
    return {"image_emotion (dev, unique questions, base order)": items}, [prov]


def parse_inline_options(question):
    """MMStar-style 'Options: A: x, B: y, C: z' -> (stem, [x, y, z])."""
    if "Options:" not in question:
        return None
    stem, tail = question.split("Options:", 1)
    parts = re.split(r"(?:^|,\s*|\n\s*)([A-F])\s*[:.]\s+", tail.strip())
    # parts = ['', 'A', 'x', 'B', 'y', ...]
    letters, texts = parts[1::2], [p.strip().rstrip(",") for p in parts[2::2]]
    if not letters or letters != list(LETTERS[: len(letters)]):
        return None
    return stem.strip(), texts


def load_mmstar(scratch):
    rev = "bc98d668301da7b14f648724866e57302778ab27"
    url = f"{HF}/Lin-Chen/MMStar/resolve/{rev}/mmstar.parquet"

    def build():
        rows, prov = parquet_text_columns(url, scratch, "mmstar")
        keep = [r for r in rows if r.get("l2_category") == "image emotion"]
        return {"rows": keep, "prov": prov, "n_rows_file": len(rows)}

    obj = cached(scratch, "mmstar_image_emotion", build)
    items, bad = [], 0
    for r in obj["rows"]:
        parsed = parse_inline_options(r["question"])
        if not parsed or r["answer"] not in LETTERS[: len(parsed[1])]:
            bad += 1
            continue
        items.append({"id": r["index"], "options": parsed[1], "answer_idx": LETTERS.index(r["answer"])})
    prov = dict(obj["prov"], repo="Lin-Chen/MMStar", revision=rev, filter="l2_category == image emotion",
                n_rows_file=obj["n_rows_file"], n_unparsed=bad)
    return {"image emotion": items}, [prov]


MMT_PATTERN = re.compile(r"emotion|expression|affect|sentiment|facial", re.I)


def load_mmt_bench(scratch):
    rev = "2255d3a7b250ae6d87383dbb9ab0d6416432ece7"
    url = f"{HF}/OpenGVLab/MMT-Bench/resolve/{rev}/MMT-Bench_VAL.tsv"

    def build():
        csv.field_size_limit(sys.maxsize)
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=1200) as r:
            hr = HashingReader(r)
            text = io.TextIOWrapper(io.BufferedReader(hr, buffer_size=1 << 20), encoding="utf-8", newline="")
            reader = csv.reader(text, delimiter="\t")
            header = next(reader)
            img = header.index("image")
            cats = collections.Counter()
            keep = []
            for row in reader:
                rec = {h: v for j, (h, v) in enumerate(zip(header, row)) if j != img}
                cats[(rec.get("category"), rec.get("l2-category"))] += 1
                if MMT_PATTERN.search(rec.get("category", "") or "") or MMT_PATTERN.search(rec.get("l2-category", "") or ""):
                    keep.append(rec)
            text.detach()
        prov = {"url": url, "bytes": hr.n, "sha256": hr.h.hexdigest(),
                "note": "TSV streamed and hashed in memory; image column dropped; nothing written but text rows"}
        return {"rows": keep, "prov": prov, "categories": [[a, b, c] for (a, b), c in sorted(cats.items())]}

    obj = cached(scratch, "mmt_bench_val_affect", build)
    subsets = collections.defaultdict(list)
    for r in obj["rows"]:
        if r.get("l2-category") != "emotion":  # regex also caught e.g. handwritten math "expression"
            continue
        opts = [(L, r[L]) for L in "ABCDEFGHI" if r.get(L) not in (None, "", "nan")]
        letters = [L for L, _ in opts]
        if r.get("answer") not in letters:
            continue
        subsets[r["category"]].append({"id": r["index"], "options": [v for _, v in opts], "answer_idx": letters.index(r["answer"])})
    matched = sorted({(r["category"], r["l2-category"]) for r in obj["rows"]})
    prov = dict(obj["prov"], repo="OpenGVLab/MMT-Bench", revision=rev,
                filter="l2-category == emotion (rows first screened by "
                       f"/{MMT_PATTERN.pattern}/i on category or l2-category)",
                screened_categories=matched, n_rows_file=sum(c for _, _, c in obj["categories"]))
    return dict(subsets), [prov]


SEED_PATTERN = re.compile(r"\bemotion|\bemotional\b|facial expression|\bmood\b|\bfeeling\b", re.I)


def load_seed_bench(scratch):
    rev = "81cb1571b8a3a30af2e62073b8d2054418350f0a"
    url = f"{HF}/AILab-CVC/SEED-Bench/resolve/{rev}/SEED-Bench.json"
    prov = fetch(url, scratch / "raw/seed_bench/SEED-Bench.json")
    qs = json.loads(Path(prov["local"]).read_text())["questions"]
    items = []
    for q in qs:
        if q.get("data_type") != "image" or not SEED_PATTERN.search(q["question"]):
            continue
        opts = [q[f"choice_{c}"] for c in "abcd"]
        items.append({"id": q["question_id"], "options": opts, "answer_idx": "ABCD".index(q["answer"])})
    prov.update(repo="AILab-CVC/SEED-Bench", revision=rev,
                filter=f"image questions whose text matches /{SEED_PATTERN.pattern}/i (post hoc keyword filter; SEED-Bench v1 has no emotion dimension)")
    return {"emotion-keyword image questions": items}, [prov]


def load_facexbench(scratch):
    rev = "52021254a43b9a5b7f2a83f0966a51e9fb585915"
    files = ["affectnet_single_mcq", "rafdb_single_mcq_train", "affectnet_multiple_mcq", "rafdb_multiple_mcq_train"]
    subsets, provs = {}, []
    for f in files:
        url = f"{GH}/Kartik-3004/facexbench/{rev}/benchmark/{f}.json"
        prov = fetch(url, scratch / f"raw/facexbench/{f}.json")
        d = json.loads(Path(prov["local"]).read_text())
        items = []
        for qid, q in d["questions"].items():
            idx = LETTERS.index(q["correct_answer_option"])
            items.append({"id": qid, "options": q["options"], "answer_idx": idx})
        provs.append(dict(prov, repo="github.com/Kartik-3004/facexbench", revision=rev, category=d.get("category")))
        subsets[f] = items
    return subsets, provs


FHB_EVAL_ORDER = [  # copied from code/evaluation/eval.py at the pinned commit
    ("face_age_basic_5", "original"), ("face_age_basic_5", "cropped"), ("face_age_basic_10", "original"),
    ("face_age_basic_10", "cropped"), ("face_age_basic_15", "original"), ("face_age_basic_15", "cropped"),
    ("face_attack_digital_expression_swap", "cropped"), ("face_attack_digital_identity_swap", "cropped"),
    ("face_attack_physical_paper", "cropped"), ("face_attack_physical_replay", "cropped"),
    ("face_attribute_compound", "original"), ("face_attribute_compound", "cropped"),
    ("face_expression_basic", "original"), ("face_expression_basic", "cropped"),
    ("face_expression_compound", "cropped"), ("face_recognition_basic", "cropped"),
    ("face_recognition_cross_age", "cropped"), ("face_recognition_cross_pose", "cropped"),
    ("face_recognition_mask", "cropped"), ("face_recognition_similar_looking", "cropped"),
    ("human_action_basic", "boxed"), ("human_attribute_compound", "boxed"),
    ("human_attribute_compound_cropped", "cropped"), ("human_reid_basic", "cropped"),
    ("human_social_relation_occupation", "boxed"), ("human_social_relation_relationship", "boxed"),
    ("human_spatial_relation_count_10", "original"), ("human_spatial_relation_count_100", "original"),
    ("human_spatial_relation_count_100plus", "original"), ("human_spatial_relation_location", "boxed"),
]


def fhb_presented_positions(data_by_subset, order=FHB_EVAL_ORDER, seed=100):
    """Replay eval.py: random.seed(100); per sample round_k = randint(0, k-1) in eval order (en first).

    The presented option i is options[(i + round_k) % k], so the answer letter index is
    (answer - round_k) % k. Returns {(subset, version): [(presented_options, answer_idx), ...]}.
    """
    rng = random.Random(seed)
    out = {}
    for subset, version in order:
        rows = []
        for s in data_by_subset[subset]:
            opts, ans = s["QA"]["options"], s["QA"]["answer"]
            k = len(opts)
            rk = rng.randint(0, k - 1)
            rows.append(([opts[(i + rk) % k] for i in range(k)], (ans - rk) % k))
        out[(subset, version)] = rows
    return out


def load_face_human_bench(scratch):
    rev = "622f1f36ed1c4aeeb48e5af6b6a14f91b2fed238"
    hf_rev = "4d5218c2cdb859e940ccd550fb108983c02b4ae6"
    subsets, provs = {}, []
    names = sorted({s for s, _ in FHB_EVAL_ORDER})
    for split in ("dev", "test"):
        data = {}
        for name in names:
            url = f"{GH}/lxq1000/Face-Human-Bench/{rev}/data/Face-Human-Bench/{split}/json/en/{name}.json"
            prov = fetch(url, scratch / f"raw/face_human_bench/{split}/{name}.json")
            data[name] = json.loads(Path(prov["local"]).read_text())["data"]
            if "expression" in name and "attack" not in name:
                provs.append(dict(prov, repo="github.com/lxq1000/Face-Human-Bench", revision=rev))
        other = [n for n in names if not ("expression" in n and "attack" not in n)]
        provs.append({"repo": "github.com/lxq1000/Face-Human-Bench", "revision": rev, "split": split,
                      "note": f"{len(other)} non-expression en files also downloaded only to replay the eval RNG sequence",
                      "files_sha256": {n: sha256_bytes((scratch / f'raw/face_human_bench/{split}/{n}.json').read_bytes()) for n in other}})
        for name in ("face_expression_basic", "face_expression_compound"):
            subsets[f"{split}/{name} [raw JSON order]"] = [
                {"id": i, "options": s["QA"]["options"], "answer_idx": s["QA"]["answer"]} for i, s in enumerate(data[name])]
        presented = fhb_presented_positions(data)
        for (name, version), rows in presented.items():
            if name in ("face_expression_basic", "face_expression_compound"):
                subsets[f"{split}/{name} [{version}; order as presented by official eval.py]"] = [
                    {"id": i, "options": o, "answer_idx": a} for i, (o, a) in enumerate(rows)]
        # Hugging Face mirror (InQ2025/Face-Human-Bench): raw order only, as a cross-check.
        for name in ("face_expression_basic", "face_expression_compound"):
            url = f"{HF}/InQ2025/Face-Human-Bench/resolve/{hf_rev}/{split}/json/en/{name}.json"
            prov = fetch(url, scratch / f"raw/face_human_bench_hf/{split}/{name}.json")
            d = json.loads(Path(prov["local"]).read_text())["data"]
            subsets[f"{split}/{name} [HF mirror raw JSON order]"] = [
                {"id": i, "options": s["QA"]["options"], "answer_idx": s["QA"]["answer"]} for i, s in enumerate(d)]
            provs.append(dict(prov, repo="InQ2025/Face-Human-Bench", revision=hf_rev))
    return subsets, provs


def load_facebench(scratch):
    rev = "a4f72f7eba638af19476f62139462579e19df23a"
    url = f"{HF}/wxqlab/FaceBench/resolve/{rev}/test.jsonl"
    prov = fetch(url, scratch / "raw/facebench/test.jsonl")
    items, bad = [], 0
    for line in Path(prov["local"]).read_text().splitlines():
        r = json.loads(line)
        if r["question_id"] != "raf_db_q0":
            continue
        if r["gt_answer"] not in r["options"]:
            bad += 1
            continue
        items.append({"id": r["image_id"], "options": r["options"], "answer_idx": r["options"].index(r["gt_answer"])})
    prov.update(repo="wxqlab/FaceBench", revision=rev, filter="question_id == raf_db_q0 (emotion SCQ)", n_unparsed=bad)
    return {"raf_db_q0 emotion (test)": items}, [prov]


def load_eemo(scratch):
    rev = "9795f7cdc66404cb8a4fd56db73f531a99243187"
    url = f"{HF}/Workerred/EEmo-Bench_single/resolve/{rev}/EEmo-Bench(single).json"
    prov = fetch(url, scratch / "raw/eemo/EEmo-Bench_single.json")
    subsets = collections.defaultdict(list)
    for r in json.loads(Path(prov["local"]).read_text()):
        if "options" not in r:
            continue
        opts = ast.literal_eval(r["options"])
        letters = list(opts)
        if r["correct_answer"] not in letters:
            continue
        key = f"{r['perceptual_dimension']} / {r['question_type']}"
        subsets[key].append({"id": r["id"], "options": [opts[L] for L in letters], "answer_idx": letters.index(r["correct_answer"])})
    prov.update(repo="Workerred/EEmo-Bench_single", revision=rev,
                note="image-evoked emotion (viewer affect), not facial expression; descriptive (open) items skipped")
    return dict(subsets), [prov]


def load_humanvbench(scratch):
    rev = "9aea56e899086f54dc3cb81898521f4189045e4b"
    subsets, provs = {}, []
    for task in ("Emotion_Recognition", "Emotion_Temporal_Analysis", "Emotion_Intensity_Compare", "Attitude_Recognition"):
        url = f"{HF}/datajuicer/HumanVBench/resolve/{rev}/for_Eval/{task}.jsonl"
        prov = fetch(url, scratch / f"raw/humanvbench/{task}.jsonl")
        items = []
        for line in Path(prov["local"]).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            pairs = re.findall(r"Option ([A-Z]): \{(.*?)\}\s*(?=\n\s*Option [A-Z]:|\s*$)", r["choices"], flags=re.S)
            letters = [L for L, _ in pairs]
            if r["Answer"] not in letters:
                continue
            items.append({"id": r["Question_id"], "options": [t for _, t in pairs], "answer_idx": letters.index(r["Answer"])})
        subsets[task] = items
        provs.append(dict(prov, repo="datajuicer/HumanVBench", revision=rev))
    return subsets, provs


def load_mmaffben(scratch):
    rev = "26d21c25a3046f8142135d8d692abbb9984096ba"
    subsets, provs = {}, []
    for f in ("FER2013_test", "CFAPS_EC_test"):
        url = f"{HF}/lzw1008/MMAFFBen/resolve/{rev}/images/{f}.json"
        prov = fetch(url, scratch / f"raw/mmaffben/{f}.json")
        items, bad = [], 0
        for i, r in enumerate(json.loads(Path(prov["local"]).read_text())):
            prompt, ans = r["messages"][0]["content"], r["messages"][1]["content"]
            listed = []
            for num, lab in re.findall(r"(\d)\.\s*([a-z]+)", prompt):
                if lab not in listed:
                    listed.append(lab)
            m = re.match(r"\s*\d\.\s*([a-z]+)", ans)
            if not m or m.group(1) not in listed:
                bad += 1
                continue
            items.append({"id": i, "options": listed, "answer_idx": listed.index(m.group(1))})
        subsets[f] = items
        provs.append(dict(prov, repo="lzw1008/MMAFFBen", revision=rev, n_unparsed=bad,
                          note="fixed numbered label list in the prompt; position = order of mention"))
    return subsets, provs


def _emobm_options(prompt):
    m = re.search(r"final answer:\s*(.*?)\.(?:\s*Respond|\s*$)", prompt, flags=re.S)
    if m:
        return [x.strip() for x in m.group(1).split(",") if x.strip()]
    m = re.search(r"Choose one emotion:\s*(.*?)\.", prompt)
    if m:
        return [re.sub(r"^and\s+", "", x.strip()) for x in m.group(1).split(",") if x.strip()]
    return None


def load_emobench_m(scratch):
    rev = "5659068f9573bad8dbe2fe86b4a40fc09030c2c7"
    url = f"{HF}/GMLHUHE/Emobench-M/resolve/{rev}/Emobench_M.zip"
    lfs_sha = "377c434d2afe244621dfa4ff5ada424c5784042221b3230bb9b6edb9d2e90755"
    names = ["FGMSA", "MC-EIU", "MELD", "MOSEI", "MOSI", "MUSTARD", "RAVDSS_song", "RAVDSS_speech", "SIMS",
             "ch-simsv2s", "funny", "mer2023"]
    outdir = scratch / "raw/emobench_m"
    provs = []
    if not all((outdir / f"{n}.json").exists() for n in names):
        outdir.mkdir(parents=True, exist_ok=True)
        f = HTTPRange(url)
        z = zipfile.ZipFile(io.BufferedReader(f, buffer_size=1 << 16))
        for n in names:
            (outdir / f"{n}.json").write_bytes(z.read(f"Emobench_M/benchmark_json/{n}.json"))
    subsets = {}
    for n in names:
        data = (outdir / f"{n}.json").read_bytes()
        provs.append({"url": url, "zip_member": f"Emobench_M/benchmark_json/{n}.json", "repo": "GMLHUHE/Emobench-M",
                      "revision": rev, "member_sha256": sha256_bytes(data), "member_bytes": len(data),
                      "container_lfs_sha256_reported_by_hf_api": lfs_sha,
                      "note": "member extracted via HTTP range requests from the 12 GB zip; videos never fetched"})
        items, bad = [], 0
        for r in json.loads(data):
            prompt, ans = r["conversations"][0]["value"], r["conversations"][1]["value"]
            opts = _emobm_options(prompt)
            if n == "MC-EIU":
                m = re.search(r"Predict emotion:\s*([a-z]+)", ans)
                ans = m.group(1) if m else ""
            lab = norm(ans)
            on = [norm(o) for o in opts] if opts else []
            if lab not in on:
                bad += 1
                continue
            items.append({"id": r["id"], "options": opts, "answer_idx": on.index(lab)})
        subsets[n + (" (emotion part)" if n == "MC-EIU" else "")] = items
        provs[-1]["n_unparsed"] = bad
    return subsets, provs


BENCHMARKS = [
    ("MMBench (EN dev, image_emotion)", load_mmbench,
     "Official VLMEvalKit scoring uses CircularEval (each question asked with options rotated), which neutralises answer position but not option-set content."),
    ("MMStar (image emotion)", load_mmstar, "Questions selected by the authors to require the image; scored as stored."),
    ("MMT-Bench (VAL, emotion/expression tasks)", load_mmt_bench, "VAL split is the one with public answers."),
    ("SEED-Bench v1 (emotion-keyword subset)", load_seed_bench, "No emotion dimension; keyword filter is post hoc and includes some non-facial affect items."),
    ("FaceXBench (expression)", load_facexbench, "Options presented as stored (VLMEvalKit)."),
    ("Face-Human-Bench (expression)", load_face_human_bench,
     "Raw JSON stores the answer first; the official eval.py rotates options with random.seed(100). Both orders are audited."),
    ("FaceBench (RAF-DB emotion)", load_facebench, "Official inference.py presents options as stored."),
    ("EEmo-Bench (single image)", load_eemo, "Evoked-emotion benchmark (not facial)."),
    ("HumanVBench (emotion tasks)", load_humanvbench, "Free-text options per item."),
    ("MMAFFBen (image: FER2013, CFAPS)", load_mmaffben, "Fixed full label list, not shuffled MCQ."),
    ("EmoBench-M (classification sets)", load_emobench_m,
     "Fixed label list per dataset; prompts also contain the spoken transcript, a text channel this audit does not test."),
]

NOT_AUDITED = [
    {"benchmark": "MME", "reason": "No emotion or expression subtask; yes/no format. Category counts checked via datasets-server statistics on the lmms-lab-encoder/MME mirror (rev d6c9023f) during scoping, not by this script."},
    {"benchmark": "EmotionBench (CUHK-ARISE, Huang et al. 2023)", "reason": "Text-only LLM benchmark, not a VLM task; Hugging Face repo is gated (auto). Not accessed."},
    {"benchmark": "EmoBench (SahandSab/EmoBench, Sabour et al. 2024)", "reason": "Text-only LLM benchmark; out of scope."},
    {"benchmark": "FABA-Bench (FABA-Instruct)", "reason": "No public annotation file located on Hugging Face or GitHub search during this audit; format UNVERIFIED."},
    {"benchmark": "FERBench (arXiv 2511.00389)", "reason": "Annotation release not located; UNVERIFIED."},
    {"benchmark": "FEABench (FEALLM, arXiv 2505.13419)", "reason": "Annotation release not located; UNVERIFIED."},
    {"benchmark": "MMAFFBen EMOTIC / text subsets", "reason": "Context emotion (EMOTIC, 18 MB) and text-only sets out of scope for facial affect."},
    {"benchmark": "EmoBench-M smile.json", "reason": "Open-ended generation (laughter reasoning), not multiple choice."},
    {"benchmark": "FaceBench raf_au_q0", "reason": "Multi-select action-unit question; single-answer metrics do not apply."},
]


def summarise(name, audit):
    b = audit["baselines"]
    r = lambda x: round(x, 4)
    return {
        "subset": name,
        "n": audit["n_items"],
        "chance": r(audit["chance"]),
        "max_position_share": r(audit["position"]["max_position_share"]),
        "p_position_mc": audit["position"]["p_mc"],
        "same_position_in_sample_acc": r(b["always_same_position_in_sample"]["accuracy"]),
        "same_position_given_k_loo_acc": r(b["same_position_given_n_options_loo"]["accuracy"]),
        "p_same_position_given_k": b["same_position_given_n_options_loo"]["p_null"],
        "option_set_loo_acc": r(b["option_set_only_loo"]["accuracy"]),
        "option_set_null_mean": r(b["option_set_only_loo"]["null_mean"]),
        "p_option_set": b["option_set_only_loo"]["p_null"],
        "answer_prior_loo_acc": r(b["answer_prior_loo_supplementary"]["accuracy"]),
        "p_answer_prior": b["answer_prior_loo_supplementary"]["p_null"],
        "longest_option_acc": r(b["longest_option_supplementary"]["accuracy"]),
        "n_option_sets": audit["option_set"]["n_option_sets"],
        "n_labels": audit["option_set"]["n_labels"],
        "fixed_option_list": audit["fixed_option_list"],
        "majority_label_share": r(audit["majority_label_share"]),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--only", nargs="*", help="substring filter on benchmark names")
    ap.add_argument("--n-sim", type=int, default=N_SIM)
    ap.add_argument("--n-null", type=int, default=1000)
    args = ap.parse_args(argv)
    scratch = args.scratch.resolve()
    if ROOT in scratch.parents or scratch == ROOT:
        raise SystemExit("scratch must be outside the repo")
    results, summary, failures = [], [], []
    for name, loader, note in BENCHMARKS:
        if args.only and not any(s.lower() in name.lower() for s in args.only):
            continue
        print(f"[{name}] loading", flush=True)
        try:
            subsets, sources = loader(scratch)
        except Exception as exc:  # record and continue
            failures.append({"benchmark": name, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  FAILED: {exc}", flush=True)
            continue
        audits = {}
        for sub, items in subsets.items():
            if len(items) < 2:
                continue
            audits[sub] = audit_items(items, n_sim=args.n_sim, n_null=args.n_null)
            summary.append(dict(summarise(sub, audits[sub]), benchmark=name))
            print(f"  {sub}: n={len(items)} {summary[-1]}", flush=True)
        results.append({"benchmark": name, "note": note, "sources": sources, "subsets": audits})
    n_tests = add_holm(summary)
    out = {
        "holm_family_size": n_tests,
        "holm_family": "all subsets x " + ", ".join(HOLM_KEYS),
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "script": "research/scripts/benchmark_leak_audit.py",
        "seed": SEED,
        "n_sim": args.n_sim,
        "n_null": args.n_null,
        "status": "exploratory, post hoc; annotation text only; no model was run",
        "definitions": {
            "position": "0-based index of the correct answer in the order presented to the model",
            "option_set": "unordered set of normalised option strings (lowercase, whitespace collapsed)",
            "option_set_only_loo": "majority correct label among other items with the same option set (restricted to labels on offer); uniform 1/k if no peer; ties scored in expectation",
            "always_same_position_loo": "most common answer position among the other items, among positions the item has",
            "same_position_given_n_options_loo": "most common answer position among the other items with the same number of options",
            "p_null": "share of null simulations (answers redrawn uniformly from each item's own options, predictor recomputed) with accuracy >= observed",
            "answer_prior_loo_supplementary": "offered option with highest P(correct | offered) on the other items",
            "longest_option_supplementary": "longest option string",
            "chance": "mean over items of 1/k",
            "p_one_sided_approx": "longest-option baseline only: normal approximation to Poisson-binomial",
        },
        "summary": summary,
        "benchmarks": results,
        "load_failures": failures,
        "not_audited": NOT_AUDITED,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
