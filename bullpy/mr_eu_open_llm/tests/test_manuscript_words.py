import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("manuscript_words_under_test", ROOT / "research/scripts/manuscript_words.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mw = _load()


def test_clean_removes_comments_and_registry_markers_only():
    text = "A <!-- note\nover lines --> b [REG: A4 table, headlines]. See [1,2] and [PENDING: slot].\n"
    out = mw.clean(text)
    assert "note" not in out and "REG" not in out
    assert "[1,2]" in out and "[PENDING: slot]" in out


def test_count_words_skips_headings_tables_citations_and_slots():
    text = ("# Heading words here\n\n| a | b |\n|---|---|\n\n"
            "Two words [3-5]. <!-- hidden words --> Then .386 (95% CI) [REG: x] [PENDING: y] = here.\n")
    # Counted: Two, words, Then, .386, (95%, CI), here.  The bare "=" has no letter or digit.
    assert mw.count_words(text) == 7


def test_split_title_abstract():
    text = "<!-- c -->\n# Title\n\nA short title\n\n<!-- alt -->\n\n# Abstract\n\nOne two three.\n"
    assert mw.split_title_abstract(text) == ("A short title", "One two three.")
    assert mw.split_title_abstract("# Abstract\n\nOnly this.\n") == ("", "Only this.")


def test_counts_on_a_paper_folder(tmp_path):
    (tmp_path / "abstract.md").write_text("# Title\n\nT\n\n# Abstract\n\none two three\n")
    (tmp_path / "introduction.md").write_text("alpha beta\n")
    (tmp_path / "results.md").write_text("# Results\n\ngamma [REG: r]\n")
    (tmp_path / "discussion.md").write_text("delta <!-- x y z -->\n")
    c = mw.counts(tmp_path)
    assert (c["abstract"], c["introduction"], c["results"], c["discussion"], c["main_text"]) == (3, 2, 1, 1, 4)
    assert c["abstract_over"] == 0 and c["main_text_over"] == 0
    copy = mw.reading_copy(tmp_path)
    assert copy.startswith("# T") and "REG" not in copy and "x y z" not in copy


def test_the_paper_abstract_is_within_its_limit():
    c = mw.counts()
    assert c["abstract"] <= mw.LIMITS["abstract"]
