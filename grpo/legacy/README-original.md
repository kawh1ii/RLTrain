# RLTrain

**Distributed RL Training Systems with veRL, vLLM, FSDP2, DeepSpeed, Megatron and Ray**

RLTrain is an AI infrastructure project focused on **distributed LLM training and reinforcement-learning training systems**.

The project builds a complete training stack from single-GPU execution to multi-GPU parallelism, and finally compares two real 8-GPU GRPO system architectures:

- **6 Trainer + 2 Rollout disaggregated execution**
- **8-GPU colocated hybrid-engine execution**

The emphasis is not only on making the system run, but also on:

- distributed correctness,
- GPU resource placement,
- communication and weight synchronization,
- reproducible benchmarking,
- runtime tracing,
- metric-semantics auditing,
- and evidence-based performance diagnosis.

---

## Highlights

- Built an end-to-end distributed training stack on **8× NVIDIA RTX 4090**
- Validated **DeepSpeed ZeRO-0 / 1 / 2 / 3** on 1 / 2 / 4 / 8 GPUs
- Implemented **Megatron Tensor Parallel, Pipeline Parallel, TP×PP and TP×PP×DP**
- Built real **veRL + vLLM + GRPO** training pipelines
- Implemented a **6-Trainer / 2-Rollout disaggregated architecture**
- Built an **8-GPU colocated hybrid-engine** baseline
- Diagnosed CUDA IPC, Ray GPU isolation, shared-memory weight synchronization and rank-mapping issues
- Audited framework throughput semantics before performing cross-architecture comparison
- Implemented lightweight runtime event tracing for direct phase-overlap analysis

---

## Final System Comparison

### Hardware

~~~text
8 × NVIDIA GeForce RTX 4090
~~~

### Model

~~~text
Qwen2.5-0.5B-Instruct
~~~

### Algorithm

~~~text
GRPO
~~~

### Matched high-level workload configuration

~~~text
Train batch size       = 8
Rollout n              = 3
PPO mini-batch size    = 8
Effective actor batch  = 24
Rollout TP             = 2
~~~

### Token-normalized system throughput

| Architecture | Median System Throughput |
|---|---:|
| **6 Trainer + 2 Rollout** | **453.79 tokens/s** |
| **8-GPU Colocated** | **213.43 tokens/s** |

~~~text
Throughput ratio     = 2.126×
Normalized advantage = +112.62%
~~~

The primary comparison metric is:

~~~text
System Throughput
=
sum(realized tokens)
/
sum(step wall time)
~~~

This definition is used instead of directly comparing the framework's native throughput metric across different resource placements.

---

## Why Raw Step Time Is Not the Primary Metric

The two architectures used the same high-level workload configuration, but stochastic generation produced different realized token counts.

Measured token totals across the three benchmark replicates were:

### Disaggregated

~~~text
11465
10979
12191
~~~

Mean:

~~~text
11545 tokens
~~~

### Colocated

~~~text
9295
8601
8528
~~~

Mean:

~~~text
8808 tokens
~~~

The realized-token difference was approximately:

~~~text
31.07%
~~~

Therefore raw step time is retained only as **descriptive evidence**, rather than treated as an exact matched-workload performance metric.

For reference:

~~~text
Disaggregated median step time = 5.060 s
Colocated median step time     = 7.995 s
~~~

---

## Architecture A — Disaggregated RL Training

~~~mermaid
flowchart LR
    D[Dataset / Prompts] --> R[vLLM Rollout<br/>2 GPUs]
    R --> T[Generated Trajectories]
    T --> A[Actor Trainer<br/>6 GPUs / FSDP2]
    A --> W[Shared-Memory<br/>Weight Synchronization]
    W --> R
    A -. temporal overlap .- R
~~~

Main characteristics:

~~~text
Trainer GPUs        = 6
Rollout GPUs        = 2
Trainer backend     = FSDP2
Rollout backend     = vLLM
Rollout TP          = 2
Hybrid engine       = False
Execution           = one-step-off-policy
Weight transfer     = shared memory
~~~

---

## Architecture B — Colocated Hybrid Engine

~~~mermaid
flowchart LR
    D[Dataset / Prompts] --> H[8-GPU Shared Pool]
    H --> G[vLLM Generation]
    G --> A[Actor Update]
    A --> TQ[TransferQueue<br/>Weight Update]
    TQ --> S[sleep_replicas<br/>State Transition]
    S --> G
~~~

Main characteristics:

~~~text
Shared GPU pool     = 8 GPUs
Trainer backend     = FSDP2
Rollout backend     = vLLM
Rollout TP          = 2
Hybrid engine       = True
Execution           = main_ppo_sync
Weight transfer     = TransferQueue
~~~

---

## Runtime Mechanism Trace

After the formal benchmark, lightweight event tracing was used to inspect the execution timeline directly.

### Disaggregated Execution

Measured trace:

~~~text
Mean step time                       = 5.083 s
Generation                           = 1.739 s
Actor update                         = 3.457 s
Generation ↔ actor-update overlap    = 1.729 s
~~~

Overlap ratios:

~~~text
Generation overlapped by actor update = 99.47%
Actor update overlapped by generation = 50.03%
Whole step covered by overlap         = 34.03%
~~~

The measured phase-union coverage was:

~~~text
99.94%
~~~

with only approximately:

~~~text
0.003 s
~~~

of uncovered wall time.

### Colocated Execution

Measured trace:

~~~text
Mean step time        = 7.930 s
Generation            = 2.002 s
Actor update          = 2.841 s
Uncovered wall        = 0.591 s
~~~

Largest uncovered interval:

~~~text
0.606 s
~~~

Within that interval:

~~~text
sleep_replicas overlap = 0.604 s
TransferQueue overlap  = 0.00488 s
~~~

Therefore approximately:

~~~text
99.67%
~~~

of the largest uncovered interval coincided with `sleep_replicas`.

---

## Important Interpretation Boundary

The runtime traces provide **mechanism evidence**, not a causal decomposition of the complete throughput gap.

The project supports:

~~~text
Matched-configuration system comparison = YES
Token-normalized throughput comparison  = YES
Direct temporal-overlap observation     = YES
Direct uncovered-wall observation       = YES
~~~

The project does **not** claim:

~~~text
Exact realized-token workload equality        = NO
Single-variable GPU-placement causality        = NO
Causal decomposition of the 2.126× difference = NO
Statistical significance                       = NO
Cross-hardware generalization                  = NO
Universal architecture recommendation          = NO
Training convergence improvement               = NO
~~~

---

## Distributed Training Experiments

RLTrain also contains experiments covering the lower layers required to build the final RL system.

### PyTorch Distributed / NCCL

Validated:

- rank / local-rank / world-size semantics
- barrier
- all-reduce
- all-gather
- reduce-scatter
- broadcast
- multi-GPU NCCL execution
- GPU mapping
- topology inspection

---

### DeepSpeed ZeRO

Matched experiments were performed for:

~~~text
ZeRO stages = 0, 1, 2, 3
World sizes = 1, 2, 4, 8
~~~

Global workload was kept constant using:

~~~text
micro-batch-per-GPU = 1
gradient accumulation adjusted by world size
~~~

This avoided changing the per-GPU workload while claiming a matched scaling comparison.

---

### Megatron Parallelism

Implemented and validated:

~~~text
Tensor Parallel
Pipeline Parallel
TP × PP
TP × PP × DP
~~~

Experiments include:

- `ColumnParallelLinear`
- `RowParallelLinear`
- non-interleaved 1F1B
- activation P2P
- backward-gradient P2P
- TP / PP process-group geometry
- DP gradient all-reduce
- dense-reference forward equivalence
- gradient equivalence

Example 3D configuration:

~~~text
TP = 2
PP = 2
DP = 2
World size = 8
~~~

---

## Systems Problems Diagnosed

A major goal of RLTrain is to treat failures as reproducible systems problems instead of one-off workarounds.

### CUDA IPC Capability False Positive

The host exposed no useful off-diagonal CUDA peer access, but the framework IPC capability probe treated CUDA availability as sufficient for IPC support.

The weight-transfer path was therefore redirected to the existing shared-memory fallback.

### Ray GPU Isolation Failure

An eager Python `sitecustomize` instrumentation overlay initialized CUDA too early.

Result:

~~~text
Different Ray actors received different CUDA_VISIBLE_DEVICES
but resolved to the same physical GPU.
~~~

A lazy import hook restored correct physical GPU isolation.

### Rank / ZMQ Endpoint Mapping

The rollout weight-transfer path required global rollout-rank-aware endpoint construction to avoid endpoint collisions across worker groups.

### Benchmark Metric Semantics

The native framework throughput metric included trainer-GPU-count normalization.

This made direct comparison between different trainer resource placements invalid.

A cross-architecture system-throughput metric was therefore defined explicitly.

### Instrumentation Failures

The tracing system itself exposed several bugs:

- incomplete Python loader APIs,
- `runpy -m` incompatibility,
- broken `importlib.resources` semantics,
- missing actor-update trace coverage.

These were diagnosed separately from scientific execution failures.

---

## Repository Structure

~~~text
RLTrain/
├── README.md
├── docs/
│   ├── benchmark-methodology.md
│   ├── compute-intensity-scaling.md
│   ├── disaggregated-vs-colocated-benchmark.md
│   ├── megatron-3d-parallel.md
│   ├── megatron-hybrid-parallel.md
│   ├── megatron-pipeline-parallel.md
│   ├── megatron-tensor-parallel.md
│   └── system-architecture-study.md
│
├── scripts/
│   ├── baselines/
│   ├── distributed/
│   ├── deepspeed/
│   ├── megatron/
│   ├── rl/
│   └── tracing/
│
├── results/
│   ├── final-benchmark.json
│   ├── mechanism-trace.json
│   ├── final-decision.json
│   └── evidence-manifest.json
│
├── configs/
└── figures/
~~~

---

## Core Stack

~~~text
Python
PyTorch 2.8
CUDA 12.8
NCCL
DeepSpeed
Ray
Megatron-Core
veRL
vLLM
FSDP2
TransferQueue
GRPO
~~~

---

## Project Status

~~~text
STATUS = COMPLETE

Formal benchmark       = COMPLETE
Mechanism tracing      = COMPLETE
Evidence freeze        = COMPLETE
Additional GPU runs    = NOT REQUIRED
Additional Nsight runs = NOT REQUIRED
~~~

---

## Notes

This repository is a cleaned public release of the experiment code and frozen results.

Large model checkpoints, Hugging Face caches, Ray runtime files, raw temporary logs and training checkpoints are intentionally excluded.
