"""Tests for HeadVarianceProfile — online Welford variance."""

import torch

from adaptive_kv_cache import HeadVarianceProfile


def test_welford_converges_to_batched_variance():
    """Online Welford should match torch.var after seeing all data."""
    torch.manual_seed(0)
    data = torch.randn(50, 128)

    online = HeadVarianceProfile()
    for d in data:
        online.update(d)

    batched_var = data.var(dim=0, correction=1)
    online_var = online.variance

    assert online_var is not None
    assert torch.allclose(online_var, batched_var, atol=1e-6)


def test_single_element_returns_none():
    """Variance is undefined for count < 2."""
    prof = HeadVarianceProfile()
    prof.update(torch.randn(128))
    assert prof.variance is None
    assert prof.std is None


def test_constant_tensor_zero_variance():
    """A constant key across all tokens yields zero variance."""
    prof = HeadVarianceProfile()
    const = torch.ones(128) * 3.14
    for _ in range(10):
        prof.update(const)

    assert prof.variance is not None
    assert prof.variance.sum() < 1e-10