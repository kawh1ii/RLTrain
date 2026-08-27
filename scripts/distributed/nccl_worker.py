#!/usr/bin/env python3

import os
import json
import socket
from pathlib import Path
from datetime import datetime

import torch
import torch.distributed as dist


ROOT = Path(os.environ.get("RLTRAIN_WORKSPACE_ROOT", "/path/to/ai-infra"))
RESULT_DIR = ROOT / "RLTrain" / "results" / "stage_2"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID = os.environ["RLTRAIN_STAGE_2_RUN_ID"]


def tensor_equal(a, b):
    return bool(
        torch.allclose(
            a.float(),
            b.float(),
            rtol=0.0,
            atol=0.0,
        )
    )


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if world_size != 1:
        raise RuntimeError(
            "RLTrain-stage-2 single-GPU NCCL smoke "
            "expects WORLD_SIZE=1."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable."
        )

    torch.cuda.set_device(local_rank)

    device = torch.device(
        f"cuda:{local_rank}"
    )

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
    )

    results = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "timestamp":
            datetime.now()
            .astimezone()
            .isoformat(),

        "backend": dist.get_backend(),
        "hostname": socket.gethostname(),

        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,

        "torch_version":
            torch.__version__,

        "torch_cuda_runtime":
            torch.version.cuda,

        "nccl_version":
            torch.cuda.nccl.version(),

        "gpu":
            torch.cuda.get_device_name(
                local_rank
            ),

        "scope":
            "SINGLE_RANK_NCCL_CORRECTNESS_ONLY",

        "performance_interpretation_allowed":
            False,

        "tests": {},
    }

    print(
        f"RANK={rank} "
        f"LOCAL_RANK={local_rank} "
        f"WORLD_SIZE={world_size}"
    )

    print(
        f"BACKEND={dist.get_backend()}"
    )

    print(
        f"GPU={torch.cuda.get_device_name(local_rank)}"
    )

    print(
        f"NCCL_VERSION={torch.cuda.nccl.version()}"
    )

    # --------------------------------------------------
    # Barrier
    # --------------------------------------------------

    dist.barrier()
    torch.cuda.synchronize()

    results["tests"]["barrier"] = {
        "passed": True
    }

    print("BARRIER_CORRECT = TRUE")

    # --------------------------------------------------
    # AllReduce
    # --------------------------------------------------

    x = torch.tensor(
        [1.25, 2.50, 3.75, 5.00],
        dtype=torch.float32,
        device=device,
    )

    original = x.clone()

    dist.all_reduce(
        x,
        op=dist.ReduceOp.SUM,
    )

    torch.cuda.synchronize()

    expected = (
        original * world_size
    )

    passed = tensor_equal(
        x,
        expected,
    )

    results["tests"]["all_reduce"] = {
        "passed": passed,
        "input":
            original.cpu().tolist(),
        "output":
            x.cpu().tolist(),
        "expected":
            expected.cpu().tolist(),
    }

    print(
        f"ALL_REDUCE_CORRECT = {passed}"
    )

    # --------------------------------------------------
    # AllGather
    # --------------------------------------------------

    local = torch.tensor(
        [
            rank + 1.0,
            rank + 2.0,
        ],
        dtype=torch.float32,
        device=device,
    )

    gathered = [
        torch.empty_like(local)
        for _ in range(world_size)
    ]

    dist.all_gather(
        gathered,
        local,
    )

    torch.cuda.synchronize()

    expected_gather = [
        torch.tensor(
            [
                r + 1.0,
                r + 2.0,
            ],
            dtype=torch.float32,
            device=device,
        )
        for r in range(world_size)
    ]

    passed = all(
        tensor_equal(a, b)
        for a, b in zip(
            gathered,
            expected_gather,
        )
    )

    results["tests"]["all_gather"] = {
        "passed": passed,
        "output": [
            x.cpu().tolist()
            for x in gathered
        ],
        "expected": [
            x.cpu().tolist()
            for x in expected_gather
        ],
    }

    print(
        f"ALL_GATHER_CORRECT = {passed}"
    )

    # --------------------------------------------------
    # ReduceScatter
    # --------------------------------------------------

    chunk_numel = 8

    rs_input = torch.arange(
        chunk_numel * world_size,
        dtype=torch.float32,
        device=device,
    )

    rs_output = torch.empty(
        chunk_numel,
        dtype=torch.float32,
        device=device,
    )

    dist.reduce_scatter_tensor(
        rs_output,
        rs_input,
        op=dist.ReduceOp.SUM,
    )

    torch.cuda.synchronize()

    # For WORLD_SIZE=1, output must equal input.
    expected_rs = rs_input.clone()

    passed = tensor_equal(
        rs_output,
        expected_rs,
    )

    results["tests"]["reduce_scatter"] = {
        "passed": passed,
        "input":
            rs_input.cpu().tolist(),
        "output":
            rs_output.cpu().tolist(),
        "expected":
            expected_rs.cpu().tolist(),
    }

    print(
        f"REDUCE_SCATTER_CORRECT = {passed}"
    )

    # --------------------------------------------------
    # Broadcast
    # --------------------------------------------------

    broadcast_tensor = torch.tensor(
        [
            10.0,
            20.0,
            30.0,
        ],
        dtype=torch.float32,
        device=device,
    )

    expected_broadcast = (
        broadcast_tensor.clone()
    )

    dist.broadcast(
        broadcast_tensor,
        src=0,
    )

    torch.cuda.synchronize()

    passed = tensor_equal(
        broadcast_tensor,
        expected_broadcast,
    )

    results["tests"]["broadcast"] = {
        "passed": passed,
        "output":
            broadcast_tensor.cpu().tolist(),
        "expected":
            expected_broadcast.cpu().tolist(),
    }

    print(
        f"BROADCAST_CORRECT = {passed}"
    )

    # --------------------------------------------------
    # Final gate
    # --------------------------------------------------

    all_passed = all(
        test["passed"]
        for test in results[
            "tests"
        ].values()
    )

    results["all_tests_passed"] = (
        all_passed
    )

    if all_passed:
        results["verdict"] = (
            "NCCL_SINGLE_RANK_SMOKE_PASS"
        )
    else:
        results["verdict"] = (
            "NCCL_SINGLE_RANK_SMOKE_FAIL"
        )

    output_path = (
        RESULT_DIR
        / f"RLTRAIN_STAGE_2_NCCL_SINGLE_RANK_{RUN_ID}.json"
    )

    output_path.write_text(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"NCCL_TEST_COUNT = "
        f"{len(results['tests'])}"
    )

    print(
        f"NCCL_ALL_TESTS_PASSED = "
        f"{all_passed}"
    )

    print(
        f"NCCL_VERDICT = "
        f"{results['verdict']}"
    )

    dist.destroy_process_group()

    if not all_passed:
        raise RuntimeError(
            "NCCL correctness smoke failed."
        )


if __name__ == "__main__":
    main()
