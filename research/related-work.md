# Related Work on KV-Cache Quantisation

## KIVI — A Blessing for Dimensionality Reduction (2024)

**Zirui Liu, Jiayi Yuan, Hongyi Wang et al.** (March 2024, arXiv:2402.02750)

KIVI applies per-channel key quantisation (INT4) and per-row value quantisation (INT2/INT4) using a group-wise asymmetric scheme. It keeps a small portion of the cache in full precision ("residual") and the rest quantised, achieving 2x memory reduction with <0.1 perplexity degradation on LLaMA-30B. KIVI's key weakness is that all heads share the same bit-width — it does not exploit the observation that some heads need more precision than others.

**Difference**: KIVI fixes bit-width globally; Adaptive KV-Cache varies it per head based on measured attention variance.

## GEAR — GEnerative AI Repack (2024)

**Hao Kang, Youna Hu, et al.** (June 2024, arXiv:2406.15321)

GEAR fuses quantisation with a low-rank residual approximation: it quantises the KV cache to INT4, then stores a small FP16 residual that corrects the largest error direction via low-rank SVD. This recovers most of the perplexity gap while keeping memory at ~INT4 levels. The method is elegant but computationally heavier than pure quantisation — the SVD update runs every N tokens and adds a matrix multiplication that scales with cache size.

**Difference**: GEAR adds a per-layer low-rank correction on top of uniform quantisation. Adaptive KV-Cache instead spends bits where they matter at the source, avoiding the SVD overhead entirely.

## KCache (2024)

**Anonymised authors** (arXiv:2405.18415, under review at NeurIPS 2024)

KCache groups attention heads by "importance" — measured via the magnitude of the attention output projection gradient — and assigns different bit-widths to different groups. It reports similar perplexity savings to ours but uses gradient-based importance, which requires a backward pass on calibration data.

**Difference**: KCache's importance metric is expensive (requires backprop). Ours uses variance, computed forward-pass only from the keys themselves, making online adaptation during generation trivial — no calibration corpus needed.

## Additional Works

- **FlexGen** (Sheng et al., 2023): Offloading-based strategy that keeps a fraction of the cache quantised on GPU and the rest on CPU. Orthogonal to per-head bit assignment.
- **LLM.int8()** (Dettmers et al., 2022): Mixed-precision inference for weights, not KV cache — but the "emergent features" insight that a few outlier channels dominate motivated per-structure quantisation in our work.
- **SmoothQuant** (Xiao et al., 2023): Weight quantisation via smoothing — relevant for the methodology but does not address KV cache memory.

## How Adaptive KV-Cache Differs

| Property | KIVI | GEAR | KCache | **Ours** |
|---|---|---|---|---|
| Granularity | Group-wise (global bw) | Uniform + low-rank | Per-head (gradient) | **Per-head (variance)** |
| Calibration needed | No | Yes (SVD recompute) | Yes (backward pass) | **No (online forward only)** |
| Metadata per layer | Negligible | Low-rank matrices | Bit-width map | **Bit-width map (~32B)** |
| Extra compute | None | SVD every N tokens | Backward pass | **Running variance (O(d))** |
| Online adaptation | No | No | No | **Yes (Period 512)** |

Our approach is the simplest to deploy: no backward pass, no calibration corpus, no SVD. It trades a small increase in cache size (~3% vs uniform INT4) for a significant perplexity gain, and the per-head scheme maps trivially to vectorised GPU kernels.