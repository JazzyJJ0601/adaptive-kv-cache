# Adaptive KV-Cache Quantization

## Problem Statement

Large language models require substantial memory to store key-value (KV) caches during autoregressive generation. For models like Qwen3-8B with hundreds of attention heads, the KV cache can dominate memory usage during long-context inference, limiting batch sizes and sequence lengths on consumer GPUs.

## Approach

We implement adaptive bit-width KV cache quantization that assigns different precision levels (2–8 bits) to each attention head based on its attention variance. Key innovations:

- **Per-head quantization**: Each KV head gets its own bit-width allocation
- **Variance-based allocation**: Running variance of key projections determines precision needs
- **Zero calibration overhead**: Statistics computed online during generation
- **Memory savings**: Reduces KV cache size by 2–4× with negligible perplexity loss

## Local Setup (Qwen3-8B)

```bash
# Install dependencies
pip install torch numpy pytest

# Clone model weights (requires huggingface-cli)
huggingface-cli download Qwen/Qwen3-8B --local-dir qwen3-8b

# Run benchmark
python benchmark.py --model qwen3-8b --max-seq 4096
```

For detailed algorithm description, see `methodology.md`. The reference implementation is in `adaptive_kv_cache.py`.
