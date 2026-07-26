"""Tests for generation decode helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import torch

from scripts.model_inference import _decode_generated_ids


def test_decode_new_tokens_only_output() -> None:
    processor = MagicMock()
    processor.batch_decode.return_value = ["Joking"]
    out_ids = torch.tensor([[101, 202, 303]])
    inputs = {"input_ids": torch.tensor([[1] * 512])}
    text = _decode_generated_ids(processor, out_ids, inputs)
    assert text == "Joking"
    processor.batch_decode.assert_called_once()
    call_ids = processor.batch_decode.call_args[0][0]
    assert call_ids.shape == out_ids.shape


def test_decode_full_sequence_output() -> None:
    processor = MagicMock()
    processor.batch_decode.return_value = ["answer"]
    prompt = torch.tensor([[1, 2, 3]])
    out_ids = torch.tensor([[1, 2, 3, 9, 10]])
    inputs = {"input_ids": prompt}
    text = _decode_generated_ids(processor, out_ids, inputs)
    assert text == "answer"
    call_ids = processor.batch_decode.call_args[0][0]
    assert call_ids.tolist() == [[9, 10]]
