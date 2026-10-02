# Adaptive KV Cache: results

All numbers: Qwen3-8B, bf16 weights, WikiText-2 test 40 × 512 tokens (20,440 scored tokens),
calibration on WikiText-2 train 8 × 512. bf16 perplexity 12.0346. Raw rows, per-layer loss probes and
every allocation are in [`results/real.json`](results/real.json) (per-token keys) and
[`results/real_channel.json`](results/real_channel.json) (KIVI-style keys).

## Summary at equal memory

| Avg bits | Per-token: best random mean | Per-token: calibrated | KIVI: best random/uniform | KIVI: calibrated |
|---:|---:|---:|---:|---:|
| 2.5 | 574 | **72.4** | 13.16 | 13.20 |
| 3.0 | 246 (uniform 333) | **28.3** | 12.34 (uniform) | 12.31 |
| 3.5 | 43.0 | **18.8** | 12.18 | 12.25 |

## Per-layer loss probe (calibration loss rise, nats, when one layer's K or V is 2-bit)

- Per-token keys: layer 0 K = **0.72**; every other layer's K is below 0.06; V never above 0.02.
  Layer 0's keys carry outlier channels that per-token quantisation destroys, which is why
  allocation matters so much in this mode.
- KIVI keys: largest K value is 0.023 (layer 32); 13 of 36 layers measure below zero (noise).

## Bits each method ends up with (3.5-bit budget)

| Method | K mean bits | V mean bits |
|---|---:|---:|
| calibrated-local (v1, failed) | 4.03 | 2.97 |
| calibrated, per-token keys | 4.35 | 2.65 |
| calibrated, KIVI keys | 3.56 | 3.44 |

## Version history

- v1 (calibrated on raw attention-output error): 491 / 382 / 250 at 2.5 / 3 / 3.5 bits, per-token mode.
  Lost to random. Diagnosis in the README.
- v2 (loss-calibrated): numbers above.
- v3 (no 8-bit choice, 32 × 512 probe): running.
