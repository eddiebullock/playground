"""
A3 descriptive: share of cells whose argmax option is "None of the above".

Reads the frozen A3 cell files (results/face_input_validity/cells_{model}.jsonl, written by
scripts/face_input_validity.py) and, per model and condition (real, mismatch), counts the
scored cells whose highest joint logprob falls on "None of the above". This is the
descriptive line under the A3 table in research/RESULTS_REGISTRY.md. It is outside the
frozen A3 summary and no decision uses it.

Rules, matching the A3 script: a row counts only if it is valid JSON ending in a newline
(one bad final line is dropped and counted); only status "ok" cells enter; the denominator
is all ok cells in that condition (243 items by 5 rotations when nothing is lost). Ties at
the maximum are counted separately; the main count uses a strict, unique argmax, and the
first-index argmax count is reported alongside so a tie cannot change the share silently.

Standard library only, so it runs on the cluster login node (Python 3.6). Writes
research/reproduction/a3_nota_shares.json with the sha256 of every input file.
"""

import argparse
import hashlib
import json
from pathlib import Path

MODELS = ("gemma4", "qwen3vl", "molmo2")
CONDITIONS = ("real", "mismatch")
NOTA = "None of the above"


def read_rows(path):
    data = path.read_bytes()
    lines = data.split(b"\n")
    tail = lines.pop()
    rows, dropped = [], int(bool(tail))
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line.decode("utf-8")))
        except (ValueError, UnicodeDecodeError):
            if index == len(lines) - 1 and not tail:
                dropped = 1
                break
            raise ValueError("{}: bad line {} is not the final line".format(path, index + 1))
    return rows, dropped, hashlib.sha256(data).hexdigest()


def shares(rows):
    out = {}
    for condition in CONDITIONS:
        n = unique = first = ties = 0
        for row in rows:
            if row.get("condition") != condition or row.get("status") != "ok":
                continue
            values = row["joint_logprob"]
            top = max(values)
            best = [i for i, v in enumerate(values) if v == top]
            nota = row["options"].index(NOTA)
            n += 1
            ties += int(len(best) > 1)
            unique += int(best == [nota])
            first += int(best[0] == nota)
        out[condition] = {
            "count": unique,
            "n": n,
            "share": unique / n if n else None,
            "count_first_index_argmax": first,
            "n_ties_at_max": ties,
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--cells_dir", type=Path, default=Path("results/face_input_validity"))
    parser.add_argument("--out", type=Path, default=Path("research/reproduction/a3_nota_shares.json"))
    args = parser.parse_args()
    result = {
        "description": "Share of A3 single_face cells whose unique argmax of joint logprob is "
        "'None of the above', per model and condition. Descriptive, outside the frozen A3 summary.",
        "script": "research/scripts/a3_nota_shares.py",
        "inputs": {},
        "models": {},
    }
    for model in MODELS:
        path = args.cells_dir / "cells_{}.jsonl".format(model)
        rows, dropped, digest = read_rows(path)
        result["inputs"][model] = {
            "file": "results/face_input_validity/cells_{}.jsonl".format(model),
            "sha256": digest,
            "n_rows": len(rows),
            "n_partial_lines_dropped": dropped,
            "n_not_ok": sum(1 for r in rows if r.get("status") != "ok"),
        }
        result["models"][model] = shares(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
