from scripts.tolerant_parse import extract_free_text_label, parse_emotion_tolerant


def test_finetune_label_format():
    pool = ["Joking", "Proud", "Jealous", "Sad"]
    hit = extract_free_text_label("LABEL: joking\n", pool)
    assert hit == "Joking"


def test_tolerant_falls_back_when_strict_fails():
    opts = ["Joking", "Proud", "Afraid", "Sad"]
    pred, _, method = parse_emotion_tolerant("LABEL: joking\n", opts, full_label_pool=opts)
    assert pred == "Joking"
    assert method in ("tolerant_4afc_option", "strict_4afc", "tolerant_4afc_base")
