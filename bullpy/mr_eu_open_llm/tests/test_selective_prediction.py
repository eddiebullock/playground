from scripts.selective_prediction import selective_prediction_report, expected_calibration_error


def test_selective_prediction_basic():
    trials = [
        {"stage1": {"semantic_entropy": 0.5}, "stage2": {"correct": True}},
        {"stage1": {"semantic_entropy": 0.6}, "stage2": {"correct": False}},
        {"stage1": {"semantic_entropy": 2.0}, "stage2": {"correct": False}},
        {"stage1": {"semantic_entropy": 2.5}, "stage2": {"correct": False}},
    ]
    rep = selective_prediction_report(trials, low_entropy_quantile=0.5)
    assert rep["n_scored"] == 4
    assert rep["overall_accuracy"] == 0.25
    assert rep["low_entropy_subset_n"] == 2
    assert rep["low_entropy_subset_accuracy"] == 0.5


def test_ece_bounded():
    conf = [0.2, 0.2, 0.8, 0.8]
    correct = [False, False, True, True]
    ece = expected_calibration_error(conf, correct, n_bins=10)
    assert ece is not None
    assert 0.0 <= ece <= 1.0
