# Adaptive KV Cache: results

All numbers: Qwen3-8B, bf16 weights, WikiText-2 test 40 × 512 tokens (20,440 scored tokens),
calibration on WikiText-2 train 8 × 512. bf16 perplexity 12.0346. Raw rows, per-layer loss probes and
every allocation are in [`results/real.json`](results/real.json) (per-token keys) and
[`results/real_channel.json`](results/real_channel.json) (KIVI-style keys, v2) and
[`results/real_channel_v3.json`](results/real_channel_v3.json) (KIVI-style keys, v3) and
[`results/real_v3.json`](results/real_v3.json) (per-token keys, v3 settings).

## Summary at equal memory

| Avg bits | Per-token: best random mean | Per-token: calibrated | KIVI: best random/uniform | KIVI: calibrated v3 |
|---:|---:|---:|---:|---:|
| 2.5 | 574 | **72.4** | 13.16 | **12.48** |
| 3.0 | 246 (uniform 333) | **28.3** | 12.34 (uniform) | **12.17** |
| 3.5 | 43.0 | **18.8** | 12.18 | **12.10** |

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
| calibrated v2, KIVI keys | 3.56 | 3.44 |
| calibrated v3, KIVI keys | 3.39 | 3.62 |

## Version history

- v1 (calibrated on raw attention-output error): 491 / 382 / 250 at 2.5 / 3 / 3.5 bits, per-token mode.
  Lost to random. Diagnosis in the README.
- v2 (loss-calibrated, bits {2,3,4,8}, 8 × 512 probe): per-token numbers above; KIVI 13.20 / 12.31 /
  12.25, a tie with random.
- v3 (bits {2,3,4}, 32 × 512 loss probe), KIVI keys: 12.48 / 12.17 / 12.10, beats every random seed
  and uniform 3-bit.
- v3 settings on per-token keys ([`results/real_v3.json`](results/real_v3.json)): 94.7 / 36.2 / 32.5,
  worse than v2 (72.4 / 28.3 / 18.8) though still far ahead of random. In this mode layer 0's outlier
  keys need the 8-bit option that v3 removes, so the per-token headline stays on v2 (bits {2,3,4,8}).
  The best setting depends on the quantiser: keep 8-bit for per-token keys, drop it for KIVI keys.
