# Compute / Communication Intensity Scaling

Run ID: `20260826_084312`

**Status:** Complete

## Design

- Model: Qwen2.5-0.5B-Instruct
- Sequence lengths: 512 and 2048
- ZeRO stages: 0 and 2
- GPUs: 1 / 2 / 4 / 8
- Global batch: 8
- Micro-batch/GPU: 1
- Three fresh-process independent replicates per configuration

## Scaling results

| Seq | ZeRO | GPUs | Step ms | tok/s | Speedup | Efficiency % | Rep CV % | Peak alloc GiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0 | 1 | 591.991 | 6994.93 | 1.000 | 100.00 | 2.63 | 10.377 |
| 512 | 0 | 2 | 400.535 | 10255.52 | 1.478 | 73.90 | 0.55 | 10.377 |
| 512 | 0 | 4 | 318.846 | 13026.16 | 1.857 | 46.42 | 4.52 | 10.377 |
| 512 | 0 | 8 | 295.542 | 14765.08 | 2.003 | 25.04 | 2.53 | 10.377 |
| 512 | 2 | 1 | 789.020 | 5135.49 | 1.000 | 100.00 | 7.42 | 12.123 |
| 512 | 2 | 2 | 713.695 | 5735.33 | 1.106 | 55.28 | 0.37 | 6.602 |
| 512 | 2 | 4 | 403.805 | 10037.78 | 1.954 | 48.85 | 2.01 | 4.831 |
| 512 | 2 | 8 | 239.816 | 17057.91 | 3.290 | 41.13 | 0.34 | 3.674 |
| 2048 | 0 | 1 | 912.178 | 17820.10 | 1.000 | 100.00 | 1.60 | 14.405 |
| 2048 | 0 | 2 | 569.484 | 28728.80 | 1.602 | 80.09 | 0.52 | 14.405 |
| 2048 | 0 | 4 | 410.255 | 41340.42 | 2.223 | 55.59 | 2.44 | 14.405 |
| 2048 | 0 | 8 | 327.689 | 53498.28 | 2.784 | 34.80 | 2.73 | 13.485 |
| 2048 | 2 | 1 | 976.952 | 16773.23 | 1.000 | 100.00 | 0.03 | 14.405 |
| 2048 | 2 | 2 | 782.679 | 20891.54 | 1.248 | 62.41 | 0.35 | 11.192 |
| 2048 | 2 | 4 | 458.428 | 35754.38 | 2.131 | 53.28 | 0.13 | 9.811 |
| 2048 | 2 | 8 | 275.114 | 59823.81 | 3.551 | 44.39 | 0.96 | 8.654 |

## Interpretation boundary

The experiment compares scaling under two compute intensities while holding model, global batch, micro-batch, optimizer and GPU mapping fixed.

It does not establish universal large-model scaling behavior.
