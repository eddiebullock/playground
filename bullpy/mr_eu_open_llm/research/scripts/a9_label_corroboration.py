"""
A9: label-order corroboration from human data only (descriptive).

Written 2026-11-12 after the independent draft review (research/reviews/draft_review_2026-11-13.md,
finding 12), so that numbers already stated in PREREG_A9_DRAFT.md ("Response codes", checks i and ii) and in
Supplementary Methods S7 come from a committed output file. Reads only the two committed human-target
files, `research/reproduction/a9_human_target.json` (Liu, `datasets.liu_WC`) and
`research/reproduction/human_au_effects.json` (Snoek, `datasets.basic_WC`), and computes:

1. For each Liu label row of do effects (`do_effect_probability_au_full_vs_none`), on the 25 AU families
   picked by the frozen `human_au_effects.family_positions` (the bilateral composite where one exists), the
   cosine with every Snoek basic WC label row, and which Snoek row is closest.
2. The anger and disgust log-odds weights (`beta_log_odds_per_unit_amplitude`) on the families AU04,
   AU07, AU23, AU09 and AU10, read at the same family positions.

No model output is read. Writes research/reproduction/a9_label_corroboration.json with input hashes.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import human_au_effects as hae  # noqa: E402

LIU = ROOT / "research/reproduction/a9_human_target.json"
SNOEK = ROOT / "research/reproduction/human_au_effects.json"
WEIGHT_FAMILIES = ("AU04", "AU07", "AU23", "AU09", "AU10")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--output", type=Path, default=ROOT / "research/reproduction/a9_label_corroboration.json")
    args = ap.parse_args(argv)
    liu = json.loads(LIU.read_text(encoding="utf-8"))["datasets"]["liu_WC"]
    snoek = json.loads(SNOEK.read_text(encoding="utf-8"))["datasets"]["basic_WC"]
    if liu["effect_names"] != snoek["effect_names"]:
        raise SystemExit("The two files name their effects differently.")
    names = liu["effect_names"]
    fam = hae.family_positions(names)
    if len(fam) != 25:
        raise SystemExit(f"{len(fam)} families, not 25.")
    L = np.asarray(liu["do_effect_probability_au_full_vs_none"], dtype=float)[:, fam]
    S = np.asarray(snoek["do_effect_probability_au_full_vs_none"], dtype=float)[:, fam]
    rows = {}
    for i, label in enumerate(liu["labels"]):
        cos = {s_label: round(cosine(L[i], S[j]), 4) for j, s_label in enumerate(snoek["labels"])}
        ranked = sorted(cos.items(), key=lambda kv: -kv[1])
        rows[label] = {"cosine_with_snoek_basic_WC": cos, "closest": ranked[0][0], "same_label_closest": ranked[0][0] == label,
                       "next": ranked[1][0], "next_cosine": ranked[1][1]}
    beta = np.asarray(liu["beta_log_odds_per_unit_amplitude"], dtype=float)
    fam_names = [names[k].replace("_bilateral", "") for k in fam]
    weights = {lab: {f: round(float(beta[liu["labels"].index(lab)][fam[fam_names.index(f)]]), 2) for f in WEIGHT_FAMILIES}
               for lab in ("anger", "disgust")}
    out = {"description": __doc__.strip().splitlines()[0], "families": fam_names, "label_rows": rows,
           "all_same_label_closest": all(r["same_label_closest"] for r in rows.values()),
           "anger_disgust_log_odds_weights": weights,
           "inputs_sha256": {str(LIU.relative_to(ROOT)): sha256(LIU), str(SNOEK.relative_to(ROOT)): sha256(SNOEK)},
           "script_sha256": sha256(__file__)}
    args.output.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for lab, r in rows.items():
        print(lab, "own", r["cosine_with_snoek_basic_WC"].get(lab), "closest", r["closest"], "next", r["next"], r["next_cosine"])
    print("weights", weights)


if __name__ == "__main__":
    main()
