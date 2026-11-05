# Megatron Tensor Parallel

Run ID: `20260826_092033`

**Status:** Complete

## Correctness matrix

| TP | Local weight fraction | Column | Row | Result |
|---:|---:|:---:|:---:|:---:|
| 1 | 1.000 | PASS | PASS | PASS |
| 2 | 0.500 | PASS | PASS | PASS |
| 4 | 0.250 | PASS | PASS | PASS |
| 8 | 0.125 | PASS | PASS | PASS |

## What was validated

ColumnParallelLinear physically shards the output dimension of the dense weight.

RowParallelLinear physically shards the input dimension and executes the tensor-parallel output reduction.

Forward outputs and backward gradients were compared against an unsharded torch.nn.functional.linear reference.

## Interpretation boundary

This is a tensor-parallel correctness experiment, not a throughput benchmark.

Pipeline-parallel schedule execution remains outside this milestone.
