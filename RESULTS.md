# Adaptive KV Cache: Real Results

**Status:** Not measured yet. An earlier version of this file reported a generation-time
difference (0.87s vs 0.70s) and a KV-cache size estimate. On review, both runs used the
same unmodified model (the adaptive quantiser was never wired in), so the timing gap was
warm-up noise and the size figure was the whole model's GPU memory, not the cache. Those
numbers have been withdrawn.

Next: wire the per-head adaptive KV quantiser into Qwen3-8B's attention and measure
perplexity and real cache memory against a uniform-bit KV cache at the same size.
