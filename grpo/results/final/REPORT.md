# Frozen validation results

Operator rows: per-run median of all measured forward+backward calls, then median across runs. Peak = median of each run's maximum allocated tensor bytes. GiB uses 2^30 bytes. No samples or slow runs removed.

| shape | mode | runs | wall ms | peak GiB |
|---|---|---:|---:|---:|
| N=512, H=2048, V=151936, frozen=False | dense | 1 | 10.456 | 1.467 |
| N=512, H=2048, V=151936, frozen=False | compiled | 1 | 7.912 | 1.325 |
| N=512, H=2048, V=151936, frozen=False | checkpoint | 1 | 49.329 | 2.340 |
| N=512, H=2048, V=151936, frozen=False | manual_torch | 1 | 14.074 | 1.369 |
| N=512, H=2048, V=151936, frozen=False | fused | 1 | 8.759 | 1.244 |
| N=512, H=2048, V=151936, frozen=False | liger | 1 | 226.655 | 2.921 |
| N=2048, H=896, V=151936, frozen=True | dense | 1 | 25.010 | 3.754 |
| N=2048, H=896, V=151936, frozen=True | compiled | 1 | 15.936 | 2.014 |
| N=2048, H=896, V=151936, frozen=True | checkpoint | 1 | 108.389 | 0.386 |
| N=2048, H=896, V=151936, frozen=True | manual_torch | 1 | 35.992 | 1.148 |
| N=2048, H=896, V=151936, frozen=True | fused | 1 | 12.322 | 0.530 |
| N=2048, H=896, V=151936, frozen=True | liger | 1 | 346.846 | 0.304 |
| N=2048, H=896, V=151936, frozen=False | fused | 3 | 15.871 | 0.784 |
| N=2048, H=896, V=151936, frozen=False | liger | 3 | 467.129 | 1.293 |
| N=2048, H=896, V=151936, frozen=False | dense | 3 | 28.396 | 3.754 |
| N=2048, H=896, V=151936, frozen=False | compiled | 3 | 18.750 | 2.014 |
| N=2048, H=896, V=151936, frozen=False | checkpoint | 3 | 117.379 | 1.038 |
| N=2048, H=896, V=151936, frozen=False | manual_torch | 3 | 39.304 | 1.284 |
| N=4096, H=896, V=151936, frozen=False | fused | 3 | 32.496 | 1.044 |
| N=4096, H=896, V=151936, frozen=False | liger | 3 | 453.416 | 1.301 |
| N=4096, H=896, V=151936, frozen=False | dense | 3 | 55.811 | 7.232 |
| N=4096, H=896, V=151936, frozen=False | compiled | 3 | 37.639 | 3.755 |
| N=4096, H=896, V=151936, frozen=False | checkpoint | 3 | 200.872 | 1.045 |
| N=4096, H=896, V=151936, frozen=False | manual_torch | 3 | 78.209 | 2.044 |
| N=512, H=896, V=151936, frozen=False | fused | 3 | 4.254 | 0.589 |
| N=512, H=896, V=151936, frozen=False | liger | 3 | 234.823 | 1.288 |
| N=512, H=896, V=151936, frozen=False | dense | 3 | 6.912 | 1.140 |
| N=512, H=896, V=151936, frozen=False | compiled | 3 | 4.537 | 0.705 |
| N=512, H=896, V=151936, frozen=False | checkpoint | 3 | 40.423 | 1.033 |
| N=512, H=896, V=151936, frozen=False | manual_torch | 3 | 10.522 | 0.714 |

## Full model training steps

Fixed pretrained initialization, real frozen trajectories, FP32 master / BF16 autocast, fused AdamW. First optimizer step excluded from steady-state medians; its data is retained. Generation is outside this timer.

| mode / microbatch | runs | wall ms | peak GiB |
|---|---:|---:|---:|
| compiled/micro8 | 3 | 255.788 | 14.724 |
| dense/micro8 | 3 | 263.077 | 16.463 |
| fused/micro8 | 3 | 259.091 | 13.758 |
| liger/micro8 | 3 | 1012.446 | 14.016 |
| dense/micro4 | 1 | 478.510 | 12.449 |
| fused/micro4 | 1 | 490.914 | 11.480 |

## Real-model numerical checks

Compare all initial parameters exactly, all first-step gradients, and actual three-step optimizer UPDATE vectors (final minus initial). Accuracy runs never supply performance numbers.
Mixed-precision UPDATE tolerance failures are retained as diagnostics, including the eager-vs-compiled control. This table does not establish equivalent BF16 training trajectories.

| mode | gradient relative L2 | gradient cosine | update relative L2 | update cosine | update within original tolerance |
|---|---:|---:|---:|---:|---|
| compiled | 0.0134253 | 0.99990988 | 0.176089 | 0.98451819 | False |
| fused | 0.0140262 | 0.99990168 | 0.135734 | 0.99083667 | False |
| liger | 0.0140907 | 0.99990075 | 0.182122 | 0.98335319 | False |
