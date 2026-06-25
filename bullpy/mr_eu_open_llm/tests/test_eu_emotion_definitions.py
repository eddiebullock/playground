from scripts.eu_emotion_definitions import embedding_text_for_label, load_eu_emotion_definitions


def test_definitions_load():
    defs = load_eu_emotion_definitions()
    assert "afraid" in defs
    assert "definition" in defs["afraid"]


def test_embedding_uses_definition():
    text = embedding_text_for_label("afraid low intensity", rich=True)
    assert "Fear" in text or "fear" in text.casefold()
    assert "low emotional intensity" in text.casefold()
