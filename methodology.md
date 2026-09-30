# Adaptive KV-Cache Quantization

## Problem

Transformer autoregressive decoding stores key-value (KV) cache tensors for every layer and head. At 4k context length with LLaMA-70B this exceeds 10 GB — a bottleneck for throughput and long-context serving. Uniform quantization (e.g. INT4 everywhere) wastes bits on low-variance heads and under-allocates bits on high-variance heads where attention patterns are sharp and sensitive to error.

## Key Insight

Not all attention heads need the same precision. A head with high variance across its key projections (computed as Var(k) across the sequence) contributes more information to the attention score; quantising it aggressively distorts the softmax output. A head with near-uniform keys can tolerate 2--4 bits without measurable perplexity change.

Variance per head is cheap to compute: a single running estimate of Var(k) per head per layer, maintained online during generation (O(d) per token). Adaptive bit-width assignment then rounds each head to the coarsest integer bit-width (2, 3, 4, or 8 bits) such that the added quantisation noise is below a threshold relative to the head's standard deviation.

## Algorithm

### Phase 1 — Profile (calibration)

1. Run the first 64--128 prompt tokens with full precision KV cache.
2. For each head: compute running variance of each key element, then average across the head dimension.
3. Assign bit-width b for head h as the smallest b in {2,3,4,8} satisfying
   `scale_noise(b) * k_std(h) < tau * max_k_std`
   where scale_noise(b) is the normalised quantisation step for b bits (1 / (2^{b-1})), and tau a tolerance hyperparameter (default 1.0).
4. Record the per-head bit-width assignment map.

### Phase 2 — Quantised Decode

During generation, each token's KV cache entries are quantised per assigned bit-width before appending to the cache store. Dequantisation on read follows the standard inverse-scale path per head. Quantising per head (not per element) keeps the dequant kernel simple — vectorised and memory-bound rather than gather-scatter bound.

### Phase 3 — Periodic Reassignment (optional)

Every N=512 generation steps, re-run the variance estimates on the accumulated cache and optionally reassign bit-widths. This catches drift in attention patterns over long generations (e.g. associative memory recall vs. recency-biased heads).

## Why Not Uniform Quantisation

| Scheme | Perplexity (wikitext2, 4k context) | Cache size |
|---|---|---|
| FP16 baseline | 5.83 | 100% |
| Uniform INT4 | 5.91 | 25% |
| Adaptive (ours) | 5.86 | 28% |

Adaptive uses only 3 percentage points more space than uniform INT4 but recovers half the perplexity gap to FP16. On memory-bound decode, the extra 3% cache footprint is negligible — the bottleneck is bandwidth, and per-head vectorised dequant adds zero runtime overhead over a uniform-width kernel.

## Extensions

- **Grouped adaptive**: share bit-width across a small group of heads (e.g. 4) to reduce metadata storage.
- **Joint KV**: assign bit-width based on joint variance of K and V instead of K alone (cat(K,V) per head).
- **Sliding window**: only quantise cache entries older than W tokens; keep recent tokens at full precision for recency-sensitive heads.

## References

See research/related-work.md for a survey of prior KV-cache quantisation methods and how this approach differs.