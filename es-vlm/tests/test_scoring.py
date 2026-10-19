from __future__ import annotations

import numpy as np

from esvlm.scoring.logprob import pick_pred, score_from_logits


def test_manual_two_option_logprob() -> None:
    # vocab: 0,1,2. Two options "A"=[1], "B"=[2].
    # At the single option position, logits favour token 2.
    logits = np.array([[0.0, 1.0, 3.0]], dtype=np.float64)
    scores = score_from_logits(logits, [[1], [2]])
    assert scores[1] > scores[0]
    assert pick_pred(scores) == 1
    # length-norm: two-token option averages
    logits2 = np.array([[0.0, 2.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float64)
    s_short = score_from_logits(logits2[:1], [[1]])[0]
    s_long = score_from_logits(logits2, [[1, 1]])[0]
    assert abs(s_short - s_long) < 1e-6


def test_empty_option_is_neg_inf() -> None:
    logits = np.zeros((1, 3))
    scores = score_from_logits(logits, [[], [1]])
    assert scores[0] == float("-inf")
    assert pick_pred(scores) == 1
