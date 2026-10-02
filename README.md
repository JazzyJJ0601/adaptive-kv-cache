# Adaptive KV Cache

Per-head bit-widths for a quantised KV cache. Every (layer, KV head, K or V) gets its own number of
bits under a fixed memory budget; a loss-calibrated score decides who gets the extra bits.

Measured on **Qwen3-8B** (WikiText-2 test, 40 × 512 tokens, bf16 = 12.03):

- **With a simple per-token quantiser, the allocation is a large win.** At 3.5 bits the calibrated
  allocation gives **18.79** perplexity against **54.7** for random heads at the same budget, and it beats
  *uniform 4-bit* (19.85) while using 12% less cache.
- **With a strong quantiser (KIVI-style per-channel keys) it does not help yet.** Everything is already
  close to bf16 (uniform 3-bit 12.34), and the calibrated allocation only ties a random split of
  neighbouring bit-widths (12.25 vs 12.18 at 3.5 bits). Written up below, not hidden.

## Method

A unit is one (layer, KV head, K or V): Qwen3-8B has 36 × 8 × 2 = 576 units. Post-RoPE keys and values
(exactly what the cache stores) are fake-quantised inside a custom attention function registered with
`transformers.AttentionInterface`, so the model's own forward pass sees the quantised cache.

1. **Local error.** On 8 × 512 tokens of WikiText-2 *train*, for each unit and each choice of
   {2, 3, 4, 8} bits, measure the squared error of that head's attention output when only that unit is
   quantised.
2. **Loss probe.** For each layer, measure the calibration loss rise when that layer's K (or V) goes to
   2 bits. This converts local error into loss: each head's error is rescaled so its layer's heads add up
   to the measured loss rise.
3. **Greedy allocation.** Start every unit at 2 bits and repeatedly buy the upgrade with the largest loss
   reduction per extra bit until the budget is spent.

Step 2 is the fix for the first version, which allocated on raw local error and lost to random (see
[What failed first](#what-failed-first)).

## Results

Packed cache per token includes a fp16 scale + zero-point per group. bf16 cache = 144 KB/token.

### Per-token keys and values (simple quantiser)

| Allocation | Avg bits | KB/token | Perplexity |
|---|---:|---:|---:|
| Uniform 2-bit | 2.0 | 20.25 | 1046 |
| Random, 2/3 split (3 seeds) | 2.5 | 24.75 | 752 / 607 / 653 (mean 671) |
| Random, 2/4 split (3 seeds) | 2.5 | 24.75 | 620 / 606 / 495 (mean 574) |
| Variance heuristic (repo's original idea) | 2.5 | 24.75 | 439 |
| **Calibrated (ours)** | 2.5 | 24.75 | **72.4** |
| Uniform 3-bit | 3.0 | 29.25 | 333 |
| Random, 2/4 split (3 seeds) | 3.0 | 29.25 | 355 / 205 / 180 (mean 246) |
| Variance heuristic | 3.0 | 29.25 | 94.9 |
| **Calibrated (ours)** | 3.0 | 29.25 | **28.3** |
| Random, 3/4 split (3 seeds) | 3.5 | 33.75 | 76.1 / 42.5 / 45.3 (mean 54.7) |
| Random, 2/4 split (3 seeds) | 3.5 | 33.75 | 45.7 / 49.3 / 33.8 (mean 43.0) |
| Variance heuristic | 3.5 | 33.75 | 41.9 |
| **Calibrated (ours)** | 3.5 | 33.75 | **18.8** |
| Uniform 4-bit | 4.0 | 38.25 | 19.85 |
| Uniform 8-bit | 8.0 | 74.25 | 12.04 |

The calibrated allocation beats every random seed and the variance heuristic at every budget: about 8×
lower perplexity than the random mean at 2.5 and 3 bits, and 2–6× lower than the variance heuristic.

### KIVI-style keys (per-channel over 32-token groups), per-token values

| Allocation | Avg bits | Perplexity |
|---|---:|---:|
| Uniform 2-bit | 2.0 | 15.64 |
| Random, 2/3 split (3 seeds) | 2.5 | 13.28 / 12.98 / 13.21 (mean 13.16) |
| Calibrated (ours) | 2.5 | 13.20 |
| Uniform 3-bit | 3.0 | 12.34 |
| Calibrated (ours) | 3.0 | 12.31 |
| Random, 3/4 split (3 seeds) | 3.5 | 12.20 / 12.17 / 12.18 (mean 12.18) |
| Calibrated (ours) | 3.5 | 12.25 |
| Uniform 4-bit | 4.0 | 12.06 |

A tie at 2.5 and 3 bits, a small loss at 3.5. Why: once keys are quantised per channel, putting a whole
layer's K or V at 2 bits moves calibration loss by at most 0.02 nats, and many layers measure as
slightly *negative*. The loss probe is then at its noise floor (8 × 512 tokens), and the greedy also
spends budget on 8-bit upgrades that buy almost nothing. A v3 run (no 8-bit choice, 4× more probe
data) is in progress; this section will be updated with its result either way.

## What failed first

Version 1 allocated on raw attention-output error. It lost to random heads (250 vs 43 at 3.5 bits,
per-token mode). Late layers have much larger activations, so their heads got 8 bits while layers 0–5
were left at 2 bits, but early-layer error propagates through the whole network. Measuring each layer's
real loss rise (step 2) fixed it: the same budget went from 250 to 18.8.

The original repo described an online variance-based scheme with no measurements; the variance heuristic
row above is that idea measured honestly. Earlier generation-time and memory numbers in this repo were
withdrawn: the quantiser had never been wired into the model.

## Reproduce

```bash
python results/run_real.py token            # -> results/real.json
python results/run_real.py channel          # -> results/real_channel.json
python results/run_real.py token adjacent   # adds the random 2/3 and 3/4 baselines
python -m pytest -q tests
```

Needs a GPU with ~20 GB (Qwen3-8B in bf16); the model path is set in `results/qcommon.py`.

## Honest limits

- Fake quantisation: perplexity is exact for the stated format, but there is no packed mixed-bit
  kernel, so no speed numbers. KB/token is computed, not measured.
- 512-token windows. Long-context behaviour is not tested.
- One model, one dataset.
