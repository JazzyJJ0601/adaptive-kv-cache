"""Smoke test for adaptive KV-cache quantization"""

import torch
from adaptive_kv import QuantizationConfig, PerHeadKVQuantizer, GroupedKVQuantizer, StreamingKVCache

def test_per_head_quantizer():
    """Test per-head quantization with attention-score gating."""
    config = QuantizationConfig(bits=8, gating_threshold=0.5)
    quantizer = PerHeadKVQuantizer(config)
    
    # Create dummy data: batch=2, heads=4, seq=8, dim=64
    k = torch.randn(2, 4, 8, 64)
    v = torch.randn(2, 4, 8, 64)
    scores = torch.rand(2, 4, 8)
    
    k_q, v_q = quantizer.forward(k, v, scores)
    
    assert k_q.shape == k.shape, f"Expected {k.shape}, got {k_q.shape}"
    assert v_q.shape == v.shape, f"Expected {v.shape}, got {v_q.shape}"
    print("✓ Per-head quantizer test passed")

def test_grouped_quantizer():
    """Test grouped quantization."""
    config = QuantizationConfig(bits=8, group_size=32)
    quantizer = GroupedKVQuantizer(config)
    
    k = torch.randn(2, 4, 8, 64)
    v = torch.randn(2, 4, 8, 64)
    
    k_q, v_q = quantizer.forward(k, v)
    
    assert k_q.shape == k.shape, f"Expected {k.shape}, got {k_q.shape}"
    assert v_q.shape == v.shape, f"Expected {v.shape}, got {v_q.shape}"
    print("✓ Grouped quantizer test passed")

def test_streaming_cache():
    """Test streaming cache with eviction."""
    config = QuantizationConfig(bits=8, max_context=16)
    cache = StreamingKVCache(config, n_heads=4, head_dim=64)
    
    # Add data within context
    k1 = torch.randn(2, 4, 8, 64)
    v1 = torch.randn(2, 4, 8, 64)
    scores1 = torch.rand(2, 4, 8)
    
    k_q1, v_q1 = cache.update(k1, v1, scores1)
    assert k_q1.shape == k1.shape
    
    # Add more data to overflow (8 + 12 = 20 > 16)
    k2 = torch.randn(2, 4, 12, 64)
    v2 = torch.randn(2, 4, 12, 64)
    
    k_q2, v_q2 = cache.update(k2, v2)
    assert k_q2.shape == k2.shape
    
    # Check cache position (after eviction and new write)
    assert cache.position == 12, f"Expected position 12 after eviction, got {cache.position}"
    print("✓ Streaming cache test passed")

def test_full_pipeline():
    """Test full pipeline with all features."""
    config = QuantizationConfig(bits=8, max_context=64)
    cache = StreamingKVCache(config, n_heads=8, head_dim=128)
    
    # Simulate several sequence additions
    for i in range(5):
        k = torch.randn(1, 8, 16, 128)
        v = torch.randn(1, 8, 16, 128)
        scores = torch.rand(1, 8, 16)
        k_q, v_q = cache.update(k, v, scores, use_per_head=True)
        assert k_q.shape == k.shape
        assert v_q.shape == v.shape
    
    print("✓ Full pipeline test passed")

if __name__ == "__main__":
    print("Running adaptive KV-cache quantization smoke tests...\n")
    test_per_head_quantizer()
    test_grouped_quantizer()
    test_streaming_cache()
    test_full_pipeline()
    print("\n✅ All smoke tests passed!")
