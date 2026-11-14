"""Registry tables printed from Phase 3 outputs (2026-11-12). Synthetic outputs only."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rt = _load("registry_tables_under_test", ROOT / "research/scripts/registry_tables.py")


def test_formatting_follows_the_paper():
    assert rt.r3(0.4021) == ".402" and rt.r3(-0.2184) == "-.218" and rt.r3(1.25) == "1.250" and rt.r3(None) == "-"
    assert rt.p5(0.00009) == ".00009" and rt.p5(0.0123) == ".012" and rt.p5(None) == "-"


def test_a2_table_shows_frozen_and_fill_rule_verdicts_side_by_side(tmp_path):
    mf = _load("make_figures_for_tables", ROOT / "research/scripts/make_figures.py")
    synthetic = list(mf.synthetic_fig6().values())[:2]
    forced = None
    for m, out in zip(rt.MODELS, synthetic):
        path = tmp_path / f"stimulus_interventions_{m}.json"
        path.write_text(json.dumps(out), encoding="utf-8")
        extras = {"a2_output_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for route in rt.ROUTES:
            rows = out[route]["results"]["primary_blind_acted"]
            contrasts = {}
            for cid, row in rows.items():
                after = row["verdict"]
                contrasts[cid] = {"frozen_verdict": row["verdict"], "verdict_after_fill_rule": after,
                                  "equivalence_interval_90": [-0.1, 0.1]}
            extras[route] = {"contrasts": contrasts}
        # Force one row to the fill-rule outcome: frozen Opposes, reported Inconclusive.
        route = rt.ROUTES[0]
        cid = next(iter(out[route]["results"]["primary_blind_acted"]))
        out[route]["results"]["primary_blind_acted"][cid]["verdict"] = "opposes_prediction"
        path.write_text(json.dumps(out), encoding="utf-8")
        extras["a2_output_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        extras[route]["contrasts"][cid].update(frozen_verdict="opposes_prediction", verdict_after_fill_rule="inconclusive")
        forced = (route, cid)
        (tmp_path / f"a2_extras_{m}.json").write_text(json.dumps(extras), encoding="utf-8")
    text = rt.table_a2(repro=tmp_path)
    assert "After fill rule" in text and "### A2, Gemma 4" in text and "### A2, Qwen3-VL" in text
    row = next(l for l in text.splitlines() if l.startswith(f"| {forced[0]} | {forced[1]} |"))
    assert "| Opposes | Inconclusive | - |" in row
    (tmp_path / f"a2_extras_{rt.MODELS[0]}.json").write_text(json.dumps({"a2_output_sha256": "other"}), encoding="utf-8")
    with pytest.raises(SystemExit):
        rt.table_a2(repro=tmp_path)


def test_a4_table_and_refusal_before_the_run_note(tmp_path, monkeypatch):
    mf = _load("make_figures_for_tables_a4", ROOT / "research/scripts/make_figures.py")
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(mf.synthetic_fig5()), encoding="utf-8")
    text = rt.table_a4(path)
    assert "| basic_WC |" in text and "Basic headline" in text
    monkeypatch.setattr(rt, "REPRO", tmp_path)
    with pytest.raises(SystemExit):
        rt.main(["a4"])


def _synthetic_a9():
    """A summary in A9's output schema (scripts/a9_analyse.py), with made-up values."""
    def block(cfa, lo, hi, rs):
        return {"cfa": cfa, "p_cue": 0.5, "null_q95": 0.1, "ci": [lo, hi], "r_s": rs, "over_ceiling": cfa / 0.98,
                "lower_bound_only": rs < 0.85}
    gates = {"rq0": True, "g2_g4": True, "v1_detector": True, "v2_image_dependence": True, "v3_config_carries": True,
             "v4_domain": True, "clips_kept": True}
    models = {
        "gemma4": {"p_signal": 0.001, "raw": block(0.21, 0.18, 0.24, 0.99), "residual": block(0.35, 0.31, 0.39, 0.98),
                   "design": "B", "n_configurations": 1200, "share_lost": 0.0, "lost_ok": True,
                   "face_shared_residual_correlation": 0.9, "p_holm": 0.0002, "gates": dict(gates),
                   "decision": "Aligned", "earlier_rule_reading_descriptive": "outside the recovery-validated conditions",
                   "replication": {"a9_decision": "Aligned", "a4_basic_WC_decision": "Aligned",
                                   "a4_basic_headline_beside": "Aligned",
                                   "reading": "replicated on a second human sample (same lab and generator)", "reason": None}},
        "qwen3vl": {"p_signal": 0.001, "raw": block(0.05, 0.01, 0.09, 0.98), "residual": block(0.04, -0.02, 0.1, 0.6),
                    "design": "B", "n_configurations": 1200, "share_lost": 0.0, "lost_ok": True,
                    "face_shared_residual_correlation": 0.9, "p_holm": 0.4, "gates": dict(gates, v4_domain=False),
                    "decision": "Inconclusive", "earlier_rule_reading_descriptive": "outside the recovery-validated conditions",
                    "replication": {"a9_decision": "Inconclusive", "a4_basic_WC_decision": "Inconclusive",
                                    "a4_basic_headline_beside": "Inconclusive", "reading": "replication not testable",
                                    "reason": "gate v4_domain"}},
    }
    return {"amendment": "A9", "statistic": "residual", "design": "B", "models": models}


def test_a9_table_and_refusal_before_the_run_note(tmp_path, monkeypatch):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(_synthetic_a9()), encoding="utf-8")
    text = rt.table_a9(path)
    assert "| Gemma 4 | .001 | .350 | .310 to .390 | .980 | .00020 | .210 | 1200 (.000) | Aligned |" in text
    assert "| Qwen3-VL | v4_domain | replication not testable | Inconclusive | Inconclusive | Inconclusive | gate v4_domain |" in text
    assert "(lower bound only)" in text  # Qwen's R_S below .85
    monkeypatch.setattr(rt, "REPRO", tmp_path)
    with pytest.raises(SystemExit):
        rt.main(["a9"])

