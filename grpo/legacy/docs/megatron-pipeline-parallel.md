# Megatron Pipeline Parallel — Non-Interleaved 1F1B

Run ID: `20260826_093719`

**Status:** Complete

## Correctness matrix

| PP | Microbatches | Forward reference | Gradient reference | Schedule | Result |
|---:|---:|:---:|:---:|:---:|:---:|
| 2 | 16 | PASS | PASS | PASS | PASS |
| 4 | 16 | PASS | PASS | PASS | PASS |
| 8 | 16 | PASS | PASS | PASS | PASS |

## What was validated

Megatron Core selected and executed the non-interleaved pipeline forward/backward schedule.

Activations were transferred forward between pipeline stages through the real P2P path.

Backward gradients were transferred upstream through the real pipeline P2P path.

Every pipeline stage executed all 16 forward and backward microbatches.

Final-stage outputs, losses, and every stage's parameter gradients were checked against an independent serial dense reference.

Warmup and steady-state microbatch geometry was explicitly audited for PP=2,4,8.

## Interpretation boundary

The reported bubble fractions are analytical schedule geometry under ideal equal-stage-time assumptions, not measured throughput results.

No formal performance or scaling recommendation is made in this milestone.

Hybrid TP + PP execution remains outside this milestone and is deferred to the corresponding experiment stage.
