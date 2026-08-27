# RLTrain — Final RL Training System Study

**Status:** Complete

## Experimental system

- Hardware: 8× NVIDIA GeForce RTX 4090
- Model: Qwen2.5-0.5B-Instruct
- Algorithm: GRPO
- Train batch size: 8
- Rollout n: 3
- PPO mini-batch size: 8
- Effective actor mini-batch: 24

## Compared architectures

1. 6-trainer / 2-rollout disaggregated execution using the one-step-off-policy path and shared-memory weight synchronization.
2. 8-GPU colocated hybrid-engine execution using main_ppo_sync and TransferQueue.

## Formal normalized benchmark result

- Disaggregated median system throughput: 453.790 tokens/s
- Colocated median system throughput: 213.432 tokens/s
- Throughput ratio: 2.126×
- Disaggregated normalized-throughput advantage: 112.62%

The configuration-level workload was matched, but stochastic generation produced different realized token counts. Therefore raw step time is reported descriptively rather than as an exact realized-workload-matched formal metric.

## Direct mechanism trace

### Disaggregated

- Measured trace step: 5.082732 s
- Generation: 1.738728 s
- Actor update: 3.456629 s
- Generation ↔ actor overlap: 1.729480 s
- Fraction of generation overlapping actor update: 99.47%
- Fraction of actor update overlapping generation: 50.03%
- Overlap fraction of complete step: 34.03%

### Colocated

- Measured trace step: 7.930442 s
- Largest uncovered interval: 0.606281 s
- `sleep_replicas` overlap: 0.604300 s
- Fraction of largest gap overlapping `sleep_replicas`: 99.67%
- TransferQueue overlap inside that gap: 0.004880 s

## Mechanistic interpretation

On the frozen 8xRTX4090 configuration, the 6-trainer/2-rollout one-step-off-policy execution directly exhibits substantial generation/actor-update temporal overlap, while the colocated hybrid-engine execution directly exhibits a recurrent uncovered step interval localized almost entirely to sleep_replicas. These execution-structure observations are consistent with the independently measured token-normalized system-throughput advantage of the disaggregated configuration. They are mechanism evidence, not a causal decomposition of the performance gap.

## Scientific boundaries

- Matched configuration comparison: true
- Exact realized-token workload match: false
- Token-normalized throughput comparison: true
- Raw step-time matched-workload claim: false
- Single-variable GPU-placement causal claim: false
- Causal mechanism decomposition: false
- Statistical-significance claim: false
- Cross-hardware claim: false
- General performance recommendation: false
- Convergence claim: false
- Additional Nsight trace required: false

## Evidence freeze

- Manifest SHA256: `53db1c80c5950b52efe4e6e64a195a759bb545c70730748c98bc6a9620f75a4f`
