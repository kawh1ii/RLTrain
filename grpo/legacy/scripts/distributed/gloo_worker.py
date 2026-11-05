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
    local_rank = int(
        os.environ["LOCAL_RANK"]
    )
    world_size = int(
        os.environ["WORLD_SIZE"]
    )

    if world_size != 2:
        raise RuntimeError(
            "Gloo semantic control "
            "expects WORLD_SIZE=2."
        )

    dist.init_process_group(
        backend="gloo",
        init_method="env://",
    )

    results = {
        "schema_version": 1,
        "run_id": RUN_ID,

        "timestamp":
            datetime.now()
            .astimezone()
            .isoformat(),

        "backend":
            dist.get_backend(),

        "hostname":
            socket.gethostname(),

        "rank":
            rank,

        "local_rank":
            local_rank,

        "world_size":
            world_size,

        "scope":
            "TWO_RANK_CPU_SEMANTIC_CONTROL",

        "performance_interpretation_allowed":
            False,

        "tests": {},
    }

    print(
        f"[RANK {rank}] "
        f"BACKEND={dist.get_backend()} "
        f"WORLD_SIZE={world_size}"
    )

    # --------------------------------------------------
    # Barrier
    # --------------------------------------------------

    dist.barrier()

    results["tests"]["barrier"] = {
        "passed": True
    }

    print(
        f"[RANK {rank}] "
        "BARRIER_CORRECT=TRUE"
    )

    # --------------------------------------------------
    # AllReduce
    #
    # rank0 = 1
    # rank1 = 2
    # sum = 3 on both ranks
    # --------------------------------------------------

    x = torch.tensor(
        [float(rank + 1)],
        dtype=torch.float32,
    )

    dist.all_reduce(
        x,
        op=dist.ReduceOp.SUM,
    )

    expected = torch.tensor(
        [3.0],
        dtype=torch.float32,
    )

    passed = tensor_equal(
        x,
        expected,
    )

    results["tests"]["all_reduce"] = {
        "passed": passed,
        "output":
            x.tolist(),
        "expected":
            expected.tolist(),
    }

    print(
        f"[RANK {rank}] "
        f"ALL_REDUCE_CORRECT={passed}"
    )

    # --------------------------------------------------
    # AllGather
    #
    # expected [10, 11]
    # --------------------------------------------------

    local = torch.tensor(
        [10.0 + rank],
        dtype=torch.float32,
    )

    gathered = [
        torch.empty_like(local)
        for _ in range(world_size)
    ]

    dist.all_gather(
        gathered,
        local,
    )

    expected_values = [
        torch.tensor([10.0]),
        torch.tensor([11.0]),
    ]

    passed = all(
        tensor_equal(a, b)
        for a, b in zip(
            gathered,
            expected_values,
        )
    )

    results["tests"]["all_gather"] = {
        "passed": passed,
        "output": [
            x.tolist()
            for x in gathered
        ],
        "expected": [
            x.tolist()
            for x in expected_values
        ],
    }

    print(
        f"[RANK {rank}] "
        f"ALL_GATHER_CORRECT={passed}"
    )

    # --------------------------------------------------
    # Broadcast
    #
    # rank0 has [100, 200]
    # rank1 starts with [-1, -1]
    # after broadcast both have [100, 200]
    # --------------------------------------------------

    if rank == 0:
        value = torch.tensor(
            [100.0, 200.0],
            dtype=torch.float32,
        )
    else:
        value = torch.tensor(
            [-1.0, -1.0],
            dtype=torch.float32,
        )

    dist.broadcast(
        value,
        src=0,
    )

    expected = torch.tensor(
        [100.0, 200.0],
        dtype=torch.float32,
    )

    passed = tensor_equal(
        value,
        expected,
    )

    results["tests"]["broadcast"] = {
        "passed": passed,
        "output":
            value.tolist(),
        "expected":
            expected.tolist(),
    }

    print(
        f"[RANK {rank}] "
        f"BROADCAST_CORRECT={passed}"
    )

    all_passed = all(
        test["passed"]
        for test in results[
            "tests"
        ].values()
    )

    results["all_tests_passed"] = (
        all_passed
    )

    results["verdict"] = (
        "GLOO_TWO_RANK_SEMANTIC_PASS"
        if all_passed
        else
        "GLOO_TWO_RANK_SEMANTIC_FAIL"
    )

    output_path = (
        RESULT_DIR
        / f"RLTRAIN_STAGE_2_GLOO_RANK_{rank}_{RUN_ID}.json"
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
        f"[RANK {rank}] "
        f"ALL_TESTS_PASSED="
        f"{all_passed}"
    )

    dist.destroy_process_group()

    if not all_passed:
        raise RuntimeError(
            f"Gloo correctness failed "
            f"on rank {rank}."
        )


if __name__ == "__main__":
    main()
