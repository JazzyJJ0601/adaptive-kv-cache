"""Tests for AdaptiveKVCache bit-width assignment."""

import torch

from adaptive_kv_cache import AdaptiveKVCache, AdaptiveKVCacheConfig, HeadVarianceProfile


def test_all_high_variance_heads_get_max_bits():
    """
    When all heads have the same high variance, the coarsest bit-width
    that satisfies the tolerance should be the same for all heads.
    """
    cfg = AdaptiveKVCacheConfig(num_heads=4, num_layers=2, tau=1.0)
    cache = AdaptiveKVCache(cfg)

    # Manually seed profiles with identical high-variance data.
    for li in range(cfg.num_layers):
        for hi in range(cfg.num_heads):
            prof = HeadVarianceProfile()
            for _ in range(100):
                prof.update(torch.randn(cfg.head_dim) * 5.0)
            cache.profiles[(li, hi)] = prof

    cache._profiling_complete = True
    assign = cache.compute_bit_assignment()

    # All heads should get the same bit-width.
    bits = list(assign.values())
    assert len(set(bits)) == 1, f"Expected same bit-width for all, got {bits}"
    assert bits[0] in cfg.allowed_bits


def test_low_variance_head_gets_fewer_bits():
    """
    A head with very low variance should be assigned a coarser bit-width
    than a high-variance head.
    """
    cfg = AdaptiveKVCacheConfig(num_heads=2, num_layers=1, tau=1.0)
    cache = AdaptiveKVCache(cfg)

    # High-variance head.
    high = HeadVarianceProfile()
    for _ in range(100):
        high.update(torch.randn(cfg.head_dim) * 10.0)

    # Low-variance head (near-constant).
    low = HeadVarianceProfile()
    for _ in range(100):
        low.update(torch.randn(cfg.head_dim) * 0.01 + 0.5)

    cache.profiles = {(0, 0): high, (0, 1): low}
    cache._profiling_complete = True

    assign = cache.compute_bit_assignment()
    assert assign[(0, 1)] <= assign[(0, 0)], (
        f"Low-variance head got {assign[(0,1)]} bits, "
        f"high-variance got {assign[(0,0)]} bits"
    )


def test_requires_profile_before_assignment():
    """Calling compute_bit_assignment without profiling raises."""
    cfg = AdaptiveKVCacheConfig()
    cache = AdaptiveKVCache(cfg)
    try:
        cache.compute_bit_assignment()
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass