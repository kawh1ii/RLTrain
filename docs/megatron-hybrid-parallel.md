# Megatron Hybrid Tensor × Pipeline Parallelism

Run ID: `20260826_094804`

**Status:** Complete

| TP | PP | DP | Result |
|---:|---:|---:|:---:|
| 2 | 4 | 1 | PASS |
| 4 | 2 | 1 | PASS |

Each pipeline stage executed a real ColumnParallelLinear → GELU → RowParallelLinear tensor-parallel block.

Pipeline stages were connected through Megatron's non-interleaved 1F1B schedule.

Parameter-shard gradients and final outputs were checked against an independent serial dense reference.

No performance claim is made.
