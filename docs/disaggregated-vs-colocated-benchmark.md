# Disaggregated vs Colocated RL Training

**Status:** Complete

## Experimental scope

This is a matched-workload comparison of two different RL training system architectures.
It is not a single-variable causal experiment in which GPU placement alone changes.

## Frozen workload

- Qwen2.5-0.5B-Instruct
- GRPO
- 8 physical RTX 4090 GPUs
- train batch size = 8
- rollout n = 3
- PPO mini-batch size = 8
- effective actor mini-batch = 24
- rollout TP = 2
- rollout DP = 1
- actor FSDP size = 2
- 7 steps / replicate
- Steps 1–2 excluded
- Steps 3–7 measured
- 3 process-level replicates per architecture

## Architectures

A: 6 Trainer + 2 Rollout disaggregated, one-step-off-policy execution, SHM weight transfer.
B: 8-GPU colocated hybrid engine, main_ppo_sync + TransferQueue.

## Primary result

- Disaggregated median step time: 5.060302 s
- Colocated median step time: 7.995051 s
- Colocated / disaggregated step-time ratio: 1.579955x
- Disaggregated median system throughput: 453.790 tokens/s
- Colocated median system throughput: 213.432 tokens/s
- Disaggregated / colocated throughput ratio: 2.126152x
- Step-time winner: `DISAGG_6PLUS2`
- System-throughput winner: `DISAGG_6PLUS2`

## Replicate stability

- Disaggregated step-time CV: 0.695%
- Colocated step-time CV: 0.810%
- Disaggregated throughput CV: 5.802%
- Colocated throughput CV: 5.190%

## Interpretation boundary

System throughput is computed as sum(total_num_tokens) / sum(step_time) over measured Steps 3–7.
veRL perf/throughput is not used as the cross-architecture primary throughput metric.
Generation is stochastic; an exact realized-token workload-match claim is made only if the recorded token counts are exactly equal.
No general cross-model, cross-hardware, statistical-significance, or convergence claim is made.
