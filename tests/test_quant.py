"""Unit tests for the quantisers and bit allocators used in results/run_real.py (CPU only, no model)."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "results"))
import run_real as r  # noqa: E402


def test_token_quant_levels_and_passthrough():
    x = torch.randn(1, 2, 16, 128)
    for b in (2, 3, 4):
        y = r.quant_tokens(x, torch.tensor([b, b]))
        assert max(len(torch.unique(y[0, h, t])) for h in range(2) for t in range(16)) <= 2 ** b
    assert torch.equal(r.quant_tokens(x, torch.tensor([16, 16])), x)


def test_channel_quant_levels_and_passthrough():
    x = torch.randn(1, 2, 70, 128) * torch.linspace(0.1, 10, 128)  # outlier channels, ragged length
    for b in (2, 3, 4):
        y = r.quant_channels(x, torch.tensor([b, b]), group=32)
        assert y.shape == x.shape
        assert max(len(torch.unique(y[0, 0, :32, c])) for c in range(128)) <= 2 ** b
    assert torch.equal(r.quant_channels(x, torch.tensor([16, 16])), x)


def test_more_bits_less_error():
    x = torch.randn(1, 1, 64, 128)
    errs = [(r.quant_tokens(x, torch.tensor([b])) - x).pow(2).mean() for b in (2, 3, 4, 8)]
    assert all(a > b for a, b in zip(errs, errs[1:]))


def test_split_alloc_hits_budget():
    for budget in (2.5, 3.0, 3.5):
        bits = r.split_alloc(torch.randperm(288), budget, 288)
        assert abs(bits.float().mean().item() - budget) < 0.01
        assert set(bits.tolist()) <= {2, 4}


def test_greedy_alloc_hits_budget_and_prefers_sensitive_units():
    rng = np.random.default_rng(0)
    scale = rng.uniform(0.1, 10, 100)
    err = scale[:, None] * np.array([1.0, 0.25, 0.06, 0.0002])[None, :]
    for budget in (2.5, 3.0, 3.5):
        bits = r.greedy_alloc(err, budget)
        assert abs(bits.float().mean().item() - budget) < 1e-6
        hi, lo = np.argsort(scale)[-10:], np.argsort(scale)[:10]
        assert bits[hi].float().mean() > bits[lo].float().mean()
