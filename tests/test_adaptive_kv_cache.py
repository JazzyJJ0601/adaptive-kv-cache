import torch
import sys
sys.path.insert(0, '..')
from adaptive_kv_cache import AdaptiveKVCache

def test_init():
    """Test class initialization."""
    cache = AdaptiveKVCache(model=None, num_heads=32, hidden_dim=128)
    assert cache.num_heads == 32
    assert cache.min_bits == 2

def test_variance_computation():
    """Test per-head variance computation."""
    cache = AdaptiveKVCache(model=None, num_heads=4, hidden_dim=64)
    # Create synthetic KV states
    k = torch.randn(1, 4, 128, 64)
    v = torch.randn(1, 4, 128, 64)
    
    variances = cache.compute_head_variance(k, v)
    assert variances.shape == (4,)
    assert variances.min() >= 0

def test_bit_assignment():
    """Test bit-width assignment logic."""
    cache = AdaptiveKVCache(model=None, num_heads=8, hidden_dim=128)
    variances = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
    
    bits = cache.assign_bitwidths(variances)
    assert bits.min() >= cache.min_bits
    assert bits.max() <= cache.max_bits

def test_quantization_round_trip():
    """Test that quantization maintains shape."""
    cache = AdaptiveKVCache(model=None, num_heads=2, hidden_dim=32)
    head = torch.randn(1, 1, 16, 32)
    
    q = cache.quantize_head(head, bit_width=4)
    assert q.shape == head.shape
    assert q.dtype == head.dtype

if __name__ == "__main__":
    test_init()
    test_variance_computation()
    test_bit_assignment()
    test_quantization_round_trip()
    print("All tests passed.")
