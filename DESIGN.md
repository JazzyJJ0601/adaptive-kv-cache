# Adaptive KV-Cache Quantization

## Overview
This repository implements a novel compression scheme for the Key-Value (KV) cache in Large Language Models (LLMs). The KV-cache grows linearly with sequence length, creating a VRAM bottleneck for long-context inference. Our approach applies per-key adaptive precision based on attention distribution.

## Architecture
### ScoringGate
The ScoringGate computes a precision score for each token using attention softmax weights. Tokens receiving high attention from future queries are allocated higher bit-widths (e.g., FP16), while ignored tokens are heavily compressed (e.g., 4-bit or lower).

### CompressedCache
Stores KV entries in packed format to minimize memory footprint. The cache unpacks data on read for compatibility with standard LLM kernels.

### CacheStore
Manages the lifecycle of KV entries, orchestrating the ScoringGate and CompressedCache.

## Loss Formulation
We optimize a combined loss:
L = L_reconstruction + λ * L_compression
Where L_reconstruction penalizes quantization error on high-attention tokens, and L_compression minimizes total memory usage.

## Experiment Plan
1. Baseline: FP16 KV-cache on 8K+ context.
2. Proposed: Adaptive 4-16 bit precision using attention signals.
3. Evaluation: Perplexity, memory usage, and inference latency on long-sequence benchmarks.

## Related Work
- MLA (DeepSeek): Uses multi-head attention with latent vectors to reduce KV size.
- KIVI: Quantizes KV cache based on outlier analysis.
- Gear: Compresses cache using attention entropy.

Novelty: Per-key adaptive precision directly driven by real-time attention scores, balancing accuracy and memory dynamically.
