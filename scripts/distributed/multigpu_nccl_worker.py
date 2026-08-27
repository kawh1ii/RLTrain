#!/usr/bin/env python3

import os
import json
import time
import socket
import statistics
from pathlib import Path
from datetime import datetime

import torch
import torch.distributed as dist


ROOT = Path(os.environ.get("RLTRAIN_WORKSPACE_ROOT", "/path/to/ai-infra"))
RESULT_DIR = ROOT / "RLTrain" / "results" / "stage_6"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID = os.environ["RLTRAIN_STAGE_6_RUN_ID"]

RANK = int(os.environ["RANK"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])

PHYSICAL_GPU_LIST = [
    int(x)
    for x in os.environ[
        "RLTRAIN_PHYSICAL_GPU_LIST"
    ].split(",")
]

PHYSICAL_GPU_ID = (
    PHYSICAL_GPU_LIST[LOCAL_RANK]
)

SIZES_MIB = [1, 8, 32, 128]

WARMUP_ITERS = 5
MEASURE_ITERS = 20


def barrier():
    dist.barrier(
        device_ids=[LOCAL_RANK]
    )


def sync():
    torch.cuda.synchronize(
        LOCAL_RANK
    )


def allclose(a, b):
    return bool(
        torch.allclose(
            a.float(),
            b.float(),
            rtol=0.0,
            atol=1e-6,
        )
    )


def percentile(values, q):
    values = sorted(values)

    if not values:
        return None

    if len(values) == 1:
        return values[0]

    position = (
        (len(values) - 1)
        * q
    )

    lower = int(position)
    upper = min(
        lower + 1,
        len(values) - 1,
    )

    fraction = (
        position - lower
    )

    return (
        values[lower]
        * (1.0 - fraction)
        +
        values[upper]
        * fraction
    )


def collective_global_pass(
    local_pass,
    device,
):
    value = torch.tensor(
        [1 if local_pass else 0],
        dtype=torch.int32,
        device=device,
    )

    dist.all_reduce(
        value,
        op=dist.ReduceOp.MIN,
    )

    return bool(
        value.item() == 1
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable."
        )

    torch.cuda.set_device(
        LOCAL_RANK
    )

    device = torch.device(
        "cuda",
        LOCAL_RANK,
    )

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        device_id=device,
    )

    result = {
        "schema_version": 1,

        "project":
            "RLTrain",

        "milestone":
            "RLTrain-stage-6",

        "run_id":
            RUN_ID,

        "timestamp":
            datetime.now()
            .astimezone()
            .isoformat(),

        "rank":
            RANK,

        "local_rank":
            LOCAL_RANK,

        "world_size":
            WORLD_SIZE,

        "physical_gpu_id":
            PHYSICAL_GPU_ID,

        "hostname":
            socket.gethostname(),

        "environment": {
            "torch":
                torch.__version__,

            "torch_cuda":
                torch.version.cuda,

            "nccl":
                torch.cuda.nccl.version(),

            "gpu":
                torch.cuda.get_device_name(
                    LOCAL_RANK
                ),

            "compute_capability":
                "%d.%d"
                % (
                    torch.cuda
                    .get_device_properties(
                        LOCAL_RANK
                    )
                    .major,

                    torch.cuda
                    .get_device_properties(
                        LOCAL_RANK
                    )
                    .minor,
                ),
        },

        "correctness": {},

        "allreduce_benchmark": [],
    }

    print(
        f"RANK={RANK} "
        f"LOCAL_RANK={LOCAL_RANK} "
        f"PHYSICAL_GPU={PHYSICAL_GPU_ID} "
        f"WORLD_SIZE={WORLD_SIZE}",
        flush=True,
    )

    print(
        "BACKEND="
        + str(dist.get_backend()),
        flush=True,
    )

    print(
        "NCCL="
        + str(
            torch.cuda.nccl.version()
        ),
        flush=True,
    )

    #
    # Barrier
    #

    barrier()
    sync()

    barrier_pass = True

    result["correctness"][
        "barrier"
    ] = barrier_pass

    #
    # AllReduce SUM
    #

    value = torch.tensor(
        [float(RANK + 1)],
        device=device,
    )

    dist.all_reduce(
        value,
        op=dist.ReduceOp.SUM,
    )

    sync()

    expected_sum = (
        WORLD_SIZE
        * (WORLD_SIZE + 1)
        / 2.0
    )

    local_pass = (
        abs(
            float(value.item())
            - expected_sum
        )
        < 1e-6
    )

    allreduce_pass = (
        collective_global_pass(
            local_pass,
            device,
        )
    )

    result["correctness"][
        "all_reduce"
    ] = allreduce_pass

    #
    # AllGather
    #

    local = torch.tensor(
        [
            float(RANK),
            float(RANK) + 0.5,
        ],
        device=device,
    )

    gathered = [
        torch.empty_like(local)
        for _ in range(WORLD_SIZE)
    ]

    dist.all_gather(
        gathered,
        local,
    )

    sync()

    expected = [
        torch.tensor(
            [
                float(r),
                float(r) + 0.5,
            ],
            device=device,
        )
        for r
        in range(WORLD_SIZE)
    ]

    local_pass = all(
        allclose(a, b)
        for a, b
        in zip(
            gathered,
            expected,
        )
    )

    allgather_pass = (
        collective_global_pass(
            local_pass,
            device,
        )
    )

    result["correctness"][
        "all_gather"
    ] = allgather_pass

    #
    # ReduceScatter SUM
    #

    rs_input = torch.full(
        (WORLD_SIZE,),
        float(RANK + 1),
        dtype=torch.float32,
        device=device,
    )

    rs_output = torch.empty(
        (1,),
        dtype=torch.float32,
        device=device,
    )

    dist.reduce_scatter_tensor(
        rs_output,
        rs_input,
        op=dist.ReduceOp.SUM,
    )

    sync()

    local_pass = (
        abs(
            float(
                rs_output.item()
            )
            - expected_sum
        )
        < 1e-6
    )

    reducescatter_pass = (
        collective_global_pass(
            local_pass,
            device,
        )
    )

    result["correctness"][
        "reduce_scatter"
    ] = reducescatter_pass

    #
    # Broadcast
    #

    if RANK == 0:
        broadcast_tensor = (
            torch.tensor(
                [
                    42.0,
                    7.0,
                    2026.0,
                ],
                device=device,
            )
        )
    else:
        broadcast_tensor = (
            torch.zeros(
                3,
                device=device,
            )
        )

    dist.broadcast(
        broadcast_tensor,
        src=0,
    )

    sync()

    expected_broadcast = (
        torch.tensor(
            [
                42.0,
                7.0,
                2026.0,
            ],
            device=device,
        )
    )

    local_pass = allclose(
        broadcast_tensor,
        expected_broadcast,
    )

    broadcast_pass = (
        collective_global_pass(
            local_pass,
            device,
        )
    )

    result["correctness"][
        "broadcast"
    ] = broadcast_pass

    all_correct = all(
        result[
            "correctness"
        ].values()
    )

    print(
        f"RANK={RANK} "
        f"CORRECTNESS="
        f"{json.dumps(result['correctness'], sort_keys=True)}",
        flush=True,
    )

    #
    # AllReduce benchmark
    #
    # Every timed trial measures exactly one
    # SUM AllReduce. Tensor reset, barrier and
    # max-rank timing reduction are all OUTSIDE
    # the timed interval.
    #

    for size_mib in SIZES_MIB:
        size_bytes = (
            size_mib
            * 1024
            * 1024
        )

        numel = (
            size_bytes // 4
        )

        tensor = torch.empty(
            numel,
            dtype=torch.float32,
            device=device,
        )

        #
        # Warmup
        #

        for _ in range(
            WARMUP_ITERS
        ):
            tensor.fill_(
                float(RANK + 1)
            )

            barrier()

            dist.all_reduce(
                tensor,
                op=dist.ReduceOp.SUM,
            )

            sync()

        latencies_ms = []

        numerical_pass = True

        for iteration in range(
            MEASURE_ITERS
        ):
            tensor.fill_(
                float(RANK + 1)
            )

            barrier()
            sync()

            start = (
                time.perf_counter()
            )

            dist.all_reduce(
                tensor,
                op=dist.ReduceOp.SUM,
            )

            sync()

            elapsed = (
                time.perf_counter()
                - start
            )

            #
            # For distributed latency,
            # use the slowest rank.
            #

            elapsed_tensor = (
                torch.tensor(
                    [elapsed],
                    dtype=torch.float64,
                    device=device,
                )
            )

            dist.all_reduce(
                elapsed_tensor,
                op=dist.ReduceOp.MAX,
            )

            global_elapsed = float(
                elapsed_tensor.item()
            )

            expected_value = (
                expected_sum
            )

            if (
                abs(
                    float(
                        tensor[0].item()
                    )
                    - expected_value
                )
                > 1e-3
            ):
                numerical_pass = False

            if RANK == 0:
                latencies_ms.append(
                    global_elapsed
                    * 1000.0
                )

        global_numerical_pass = (
            collective_global_pass(
                numerical_pass,
                device,
            )
        )

        if RANK == 0:
            median_ms = (
                statistics.median(
                    latencies_ms
                )
            )

            mean_ms = (
                statistics.mean(
                    latencies_ms
                )
            )

            p95_ms = percentile(
                latencies_ms,
                0.95,
            )

            stdev_ms = (
                statistics.stdev(
                    latencies_ms
                )
                if len(
                    latencies_ms
                ) > 1
                else 0.0
            )

            payload_gbps = (
                size_bytes
                /
                (
                    median_ms
                    / 1000.0
                )
                /
                1e9
            )

            if WORLD_SIZE > 1:
                normalized_bus_gbps = (
                    payload_gbps
                    * 2.0
                    * (
                        WORLD_SIZE - 1
                    )
                    / WORLD_SIZE
                )
            else:
                normalized_bus_gbps = None

            bench_record = {
                "size_mib":
                    size_mib,

                "size_bytes":
                    size_bytes,

                "warmup_iters":
                    WARMUP_ITERS,

                "measured_iters":
                    MEASURE_ITERS,

                "mean_latency_ms":
                    mean_ms,

                "median_latency_ms":
                    median_ms,

                "p95_latency_ms":
                    p95_ms,

                "stdev_latency_ms":
                    stdev_ms,

                "payload_gbps":
                    payload_gbps,

                "ring_normalized_bus_gbps":
                    normalized_bus_gbps,

                "numerical_pass":
                    global_numerical_pass,
            }

            result[
                "allreduce_benchmark"
            ].append(
                bench_record
            )

            print(
                "BENCH "
                f"WORLD_SIZE={WORLD_SIZE} "
                f"SIZE_MIB={size_mib} "
                f"MEDIAN_MS={median_ms:.6f} "
                f"P95_MS={p95_ms:.6f} "
                f"PAYLOAD_GBPS={payload_gbps:.6f} "
                f"BUS_GBPS={normalized_bus_gbps}",
                flush=True,
            )

        del tensor

        barrier()

    #
    # Per-rank artifact.
    #

    result[
        "all_correct"
    ] = all_correct

    result[
        "benchmark_numerical_pass"
    ] = all(
        x[
            "numerical_pass"
        ]
        for x
        in result[
            "allreduce_benchmark"
        ]
    ) if RANK == 0 else None

    rank_path = (
        RESULT_DIR
        / (
            f"RLTRAIN_STAGE_6_WS{WORLD_SIZE}_"
            f"RANK{RANK}_{RUN_ID}.json"
        )
    )

    rank_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    barrier()

    #
    # Rank-0 world-size summary.
    #

    if RANK == 0:
        summary = {
            "schema_version": 1,

            "project":
                "RLTrain",

            "milestone":
                "RLTrain-stage-6",

            "run_id":
                RUN_ID,

            "world_size":
                WORLD_SIZE,

            "physical_gpu_list":
                PHYSICAL_GPU_LIST,

            "correctness":
                result[
                    "correctness"
                ],

            "all_correct":
                all_correct,

            "allreduce_benchmark":
                result[
                    "allreduce_benchmark"
                ],

            "benchmark_numerical_pass":
                result[
                    "benchmark_numerical_pass"
                ],

            "scope": {
                "formal_training_scaling":
                    False,

                "formal_model_throughput":
                    False,

                "nccl_collective_characterization":
                    True,

                "world_size_1_interconnect_claim":
                    False,
            },
        }

        summary_path = (
            RESULT_DIR
            / (
                f"RLTRAIN_STAGE_6_WS{WORLD_SIZE}_"
                f"SUMMARY_{RUN_ID}.json"
            )
        )

        summary_path.write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            "WORLD_SIZE_SUMMARY = "
            + str(summary_path),
            flush=True,
        )

    barrier()

    dist.destroy_process_group()

    if not all_correct:
        raise RuntimeError(
            "NCCL correctness gate failed."
        )


if __name__ == "__main__":
    main()
