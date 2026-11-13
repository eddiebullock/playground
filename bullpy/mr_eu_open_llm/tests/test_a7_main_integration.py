"""The frozen A7 entry point reads the inputs file the A7 pre-run check writes (2026-11-13).

Synthetic Stage 1 and A2 rows from A7's recovery, the real locked Stage 0 tables, and a tree laid out as
`post_phase3.py` leaves it; `a7_prerun_check.check` builds the inputs, and the frozen
`python -m scripts.a7_analyse --a7_frozen --a2_frozen` runs on them in a fresh process, as the runbook runs it.
"""
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts import a7_analyse as a7  # noqa: E402
from scripts import a7_design  # noqa: E402
from scripts import a7_recovery as rc  # noqa: E402
from scripts import run_stimulus_interventions as run  # noqa: E402
from scripts.analyse_stimulus_interventions import LEXICON, lexicon_words  # noqa: E402

WORDS = lexicon_words()
PLACES = {family: WORDS.index((family, LEXICON[family][0])) for family in LEXICON}

spec_ = importlib.util.spec_from_file_location("a7_prerun_for_main", ROOT / "research/scripts/a7_prerun_check.py")
pre = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(pre)


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_frozen_a7_main_runs_on_the_prerun_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "check_geometry", lambda path: None)  # synthetic geometry, not the pinned file
    items = rc.real_items()
    rc._init(items)
    world = rc._WORLD
    spec = dict(next(o for o in rc.observers() if o["name"] == "a_only"))
    rng = np.random.default_rng(0)
    root = tmp_path
    (root / "scripts").mkdir()
    (root / pre.ROOT_MARKER).write_text("")
    repro = root / "research" / "reproduction"
    repro.mkdir(parents=True)
    (root / "research" / "DECISIONS.md").write_text(pre.DECISIONS_HEADING + ", recorded after Stage 0\n")
    a2_hash = a7.D["a2_design_sha256"]
    order = [i["trial_id"] for i in a7.stage1_items(items)]
    steps, outputs = [], {}
    for m in a7.MODELS:
        null = m != "gemma4"
        rows, fills = rc.model_rows(world, rng, 2.0, spec, null)
        for r in rows:
            r.update(a7_design_sha256=a7_design.DESIGN_SHA256, decision_date=pre.DECISION_DATE)
            blind = r.setdefault("blind", {})
            if "word_logprob" not in blind:  # real Stage 1 rows always carry the 106 word logprobs (a7_run.row_for)
                vector = [-20.0] * len(WORDS)
                for family, score in (blind.get("family_logprob") or {}).items():
                    vector[PLACES[family]] = score
                blind["word_logprob"] = vector
        for r in fills:
            r["design_sha256"] = a2_hash
        shutil.copy(ROOT / "research/reproduction" / f"a7_stage0_{m}.json", repro)
        rank = {t: k for k, t in enumerate(order)}
        for k in range(pre.N_SHARDS):
            _jsonl(root / "results/a7" / f"a7_cells_{m}_shard{k}.jsonl",
                   [r for r in rows if rank.get(r["trial_id"], 0) % pre.N_SHARDS == k])
        cells = f"results/stimulus_interventions/cells_{m}_shard0.jsonl"
        report = f"results/stimulus_interventions/run_{m}_shard0.json"
        out = f"research/reproduction/stimulus_interventions_{m}.json"
        _jsonl(root / cells, fills)
        (root / report).write_text(json.dumps({"delivery": [{a7.ROUTE: {"tokens_across_view": 16.0}}]}))
        (root / out).write_text(json.dumps({"design_sha256": a2_hash, **rc.a2_output_for(spec, null)}))
        (repro / f"a2_prerun_{m}.json").write_text(json.dumps({"sha256": {cells: _sha(root / cells), report: _sha(root / report)}}))
        steps.append({"command": ["python3", "-m", "scripts.analyse_stimulus_interventions", "--model", m,
                                  "--cells", cells, "--run_report", report, "--output", out, "--a2_frozen"]})
        outputs[out] = _sha(root / out)
    (repro / "a2_run_note.json").write_text(json.dumps({"status": "complete", "steps": steps, "outputs": outputs}))
    trials = sorted({r["trial_id"] for r in fills})
    geometry = {t: {"face_box": [0, 0, 100, 100], "face_width_px": 40 if k % 10 == 0 else 100} for k, t in enumerate(trials)}
    (root / pre.GEOMETRY).write_text(json.dumps({"items": geometry}))

    record, inputs = pre.check(root)
    assert record["failures"] == [], record["failures"]
    assert 0 < len(inputs["below_token_floor"]["gemma4"]) < len(trials)

    def absolute(value):
        if isinstance(value, str) and (value.startswith("results/") or value.startswith("research/")):
            return str(root / value)
        if isinstance(value, list):
            return [absolute(v) for v in value]
        return value

    for m in a7.MODELS:
        inputs[m] = {k: absolute(v) for k, v in inputs[m].items()}
    inputs_path = repro / "a7_inputs.json"
    inputs_path.write_text(json.dumps(inputs))
    out_path = repro / "a7_analysis.json"
    env = {**os.environ, "PYTHONPATH": f"{ROOT}:{ROOT / 'src'}",
           **{v: "1" for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}}
    result = subprocess.run([sys.executable, "-m", "scripts.a7_analyse", "--inputs", str(inputs_path), "--output", str(out_path),
                             "--a7_frozen", "--a2_frozen"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
    assert result.returncode == 0, result.stderr[-3000:]
    body = json.loads(out_path.read_text())
    gemma = body["models"]["gemma4"]["arms"]
    assert gemma["route"]["verdict"] == "Carried by A"
    assert gemma["token"]["verdict"] == "Not testable"  # the real Stage 0 table did not admit the token arm
    assert "clips_at_8_tokens_or_more" in body["sensitivities"]
