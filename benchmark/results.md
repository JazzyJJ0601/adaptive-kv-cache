# Adaptive KV Cache Perplexity Benchmark

Model: <local-path-to-Qwen3-8B>
Samples: 50
Perplexity (baseline 4-bit): <to-run-evaluate>
Perplexity (adaptive): <to-run-evaluate>
Mean Bit Width: <to-run-evaluate>
Total Memory Savings: <to-run-evaluate> MB

## Notes
- Run with: `python run_perplexity.py --model-path /path/to/Qwen3-8B`
- KV cache variance is captured per layer and head.
- Bit-widths are assigned dynamically based on variance magnitude.
