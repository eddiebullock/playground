from scripts.eu_emotion_synonyms import accepted_terms_for_label, match_label_in_text
from scripts.free_response_judge import judge_free_response


def test_synonym_match_joking():
    assert match_label_in_text("The person seems playful and is teasing others.", "Joking")


def test_synonym_match_low_intensity_collapses():
    terms = accepted_terms_for_label("Afraid Low Intensity")
    assert "afraid" in terms
    assert "frightened" in terms


def test_judge_correct():
    r = judge_free_response("She looks jealous of the other person.", "Jealous", use_llm=False)
    assert r["correct"] is True
    assert r["method"] == "rule_synonym"


def test_judge_incorrect():
    r = judge_free_response("The person is clearly bored.", "Excited", use_llm=False)
    assert r["correct"] is False
