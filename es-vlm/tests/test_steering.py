from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from esvlm.steering.hooks import SteeringHook  # noqa: E402


class Tiny(torch.nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def test_alpha_zero_is_bit_identical() -> None:
    torch.manual_seed(0)
    m = Tiny()
    x = torch.randn(1, 4, 8)
    with torch.no_grad():
        base = m(x).clone()
    direction = np.ones(8, dtype=np.float32)
    with SteeringHook(m, direction, alpha=0.0, mode="add", position="last_token"), torch.no_grad():
        hooked = m(x)
    assert torch.equal(base, hooked)


def test_ablation_removes_projection() -> None:
    torch.manual_seed(1)
    m = Tiny()
    # identity so we can control hidden states
    with torch.no_grad():
        m.proj.weight.copy_(torch.eye(8))
    x = torch.zeros(1, 3, 8)
    x[..., :] = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    direction = np.array([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    with SteeringHook(m, direction, mode="ablate", position="all"), torch.no_grad():
        out = m(x)
    # projection onto e0 should be ~0
    proj = (out * torch.tensor(direction)).sum(dim=-1)
    assert torch.max(torch.abs(proj)) < 1e-5


def test_add_changes_last_token_only() -> None:
    torch.manual_seed(2)
    m = Tiny()
    with torch.no_grad():
        m.proj.weight.copy_(torch.eye(8))
    x = torch.zeros(1, 3, 8)
    direction = np.zeros(8, dtype=np.float32)
    direction[0] = 1.0
    with torch.no_grad():
        base = m(x).clone()
    with (
        SteeringHook(m, direction, alpha=1.0, mode="add", position="last_token", scale="unit"),
        torch.no_grad(),
    ):
        out = m(x)
    assert torch.equal(base[:, :-1], out[:, :-1])
    assert not torch.equal(base[:, -1], out[:, -1])
