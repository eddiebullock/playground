"""Supplementary A7 Stage 0 table from the locked files (2026-11-13)."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("stage0_table_under_test", ROOT / "research/scripts/make_stage0_table.py")
st = importlib.util.module_from_spec(spec)
spec.loader.exec_module(st)


def test_stage0_table_copies_the_locked_values():
    text = st.build()
    assert "| C1, determinism, native against a repeat: worst change (nats); pass | .000; pass | .000; pass |" in text
    assert "| C5, block A live: ratio of denominator (floor .10) | .230; live | .802; live |" in text
    assert "| Region test: pass; reason | False; leak_over_limit | False; token_map_failed |" in text
    assert "| Token map: hit rate over trials | 1.000 of 96 | .760 of 96 |" in text
    assert "| Arms admitted: route; token | True; False | True; False |" in text
