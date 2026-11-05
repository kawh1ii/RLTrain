#!/usr/bin/env python3

import os
import gc
import csv
import json
import math
import time
import statistics
from pathlib import Path
from datetime import datetime

import torch
import torch.distributed as dist
import deepspeed

from transformers import AutoModelForCausalLM


ROOT = Path(os.environ.get("RLTRAIN_WORKSPACE_ROOT", "/path/to/ai-infra"))
PROJECT = ROOT / "RLTrain"
RESULT_DIR = PROJECT / "results" / "stage_7"
HF_HOME = ROOT / "caches" / "huggingface"

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RUN_ID = os.environ["RLTRAIN_STAGE_7_RUN_ID"]
ZERO_STAGE = int(os.environ["RLTRAIN_STAGE_7_ZERO_STAGE"])
SEQ_LEN = int(os.environ["RLTRAIN_STAGE_7_SEQ_LEN"])
REP = int(os.environ["RLTRAIN_STAGE_7_REP"])

RANK = int(os.environ["RANK"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])

PHYSICAL_GPU_LIST = [
    int(x)
    for x
    in os.environ[
        "RLTRAIN_PHYSICAL_GPU_LIST"
    ].split(",")
]

BASE_SEED = 20260826

GLOBAL_BATCH = 8
MICRO_BATCH_PER_GPU = 1

WARMUP_OPT_STEPS = 2
MEASURED_OPT_STEPS = 8

LR = 1e-5
WEIGHT_DECAY = 0.01


if (
    GLOBAL_BATCH
    % (
        WORLD_SIZE
        * MICRO_BATCH_PER_GPU
    )
    != 0
):
    raise RuntimeError(
        "GLOBAL_BATCH divisibility failure."
    )


GRAD_ACCUM_STEPS = (
    GLOBAL_BATCH
    // (
        WORLD_SIZE
        * MICRO_BATCH_PER_GPU
    )
)

LOCAL_SAMPLES_PER_OPT_STEP = (
    MICRO_BATCH_PER_GPU
    * GRAD_ACCUM_STEPS
)

GLOBAL_TOKENS_PER_OPT_STEP = (
    GLOBAL_BATCH
    * SEQ_LEN
)


def barrier():
    dist.barrier(
        device_ids=[
            LOCAL_RANK
        ]
    )


def sync():
    torch.cuda.synchronize(
        LOCAL_RANK
    )


def gib(x):
    return x / (1024 ** 3)


def mean(xs):
    return statistics.mean(xs)


def median(xs):
    return statistics.median(xs)


def locate_snapshot():
    roots = [
        HF_HOME
        / "hub"
        / "models--Qwen--Qwen2.5-0.5B-Instruct"
        / "snapshots",

        HF_HOME
        / "transformers"
        / "models--Qwen--Qwen2.5-0.5B-Instruct"
        / "snapshots",
    ]

    for root in roots:
        if not root.exists():
            continue

        for p in sorted(
            root.iterdir()
        ):
            if (
                p.is_dir()
                and
                (p / "config.json").exists()
            ):
                return p

    raise RuntimeError(
        "Cached Qwen snapshot not found."
    )


def global_max(
    value,
    device,
):
    t = torch.tensor(
        [float(value)],
        dtype=torch.float64,
        device=device,
    )

    dist.all_reduce(
        t,
        op=dist.ReduceOp.MAX,
    )

    return float(
        t.item()
    )


def global_mean(
    value,
    device,
):
    t = torch.tensor(
        [float(value)],
        dtype=torch.float64,
        device=device,
    )

    dist.all_reduce(
        t,
        op=dist.ReduceOp.SUM,
    )

    t /= WORLD_SIZE

    return float(
        t.item()
    )


def global_and(
    value,
    device,
):
    t = torch.tensor(
        [1 if value else 0],
        dtype=torch.int32,
        device=device,
    )

    dist.all_reduce(
        t,
        op=dist.ReduceOp.MIN,
    )

    return bool(
        t.item() == 1
    )


def gather_memory(
    allocated,
    reserved,
    device,
):
    local = torch.tensor(
        [
            allocated,
            reserved,
        ],
        dtype=torch.float64,
        device=device,
    )

    gathered = [
        torch.empty_like(local)
        for _ in range(
            WORLD_SIZE
        )
    ]

    dist.all_gather(
        gathered,
        local,
    )

    return [
        {
            "rank":
                rank,

            "physical_gpu":
                PHYSICAL_GPU_LIST[
                    rank
                ],

            "peak_allocated_gib":
                float(
                    t[0].item()
                ),

            "peak_reserved_gib":
                float(
                    t[1].item()
                ),
        }

        for rank, t
        in enumerate(
            gathered
        )
    ]


def run():
    seed = (
        BASE_SEED
        + REP * 1000
    )

    torch.manual_seed(
        seed
    )

    torch.cuda.manual_seed_all(
        seed
    )

    torch.cuda.set_device(
        LOCAL_RANK
    )

    device = torch.device(
        "cuda",
        LOCAL_RANK,
    )

    snapshot = (
        locate_snapshot()
    )

    init_start = (
        time.perf_counter()
    )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            str(snapshot),

            local_files_only=True,

            dtype=torch.bfloat16,

            low_cpu_mem_usage=True,

            attn_implementation="sdpa",
        )
    )

    model.config.use_cache = False
    model.train()

    parameter_count = sum(
        p.numel()
        for p
        in model.parameters()
    )

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),

            lr=LR,

            weight_decay=
                WEIGHT_DECAY,

            foreach=False,
        )
    )

    ds_config = {
        "train_micro_batch_size_per_gpu":
            MICRO_BATCH_PER_GPU,

        "gradient_accumulation_steps":
            GRAD_ACCUM_STEPS,

        "train_batch_size":
            GLOBAL_BATCH,

        "bf16": {
            "enabled": True,
        },

        "zero_optimization": {
            "stage":
                ZERO_STAGE,
        },

        "steps_per_print":
            1000000,

        "wall_clock_breakdown":
            False,
    }

    engine, optimizer, _, _ = (
        deepspeed.initialize(
            model=model,

            optimizer=optimizer,

            config=ds_config,
        )
    )

    actual_stage = int(
        engine.zero_optimization_stage()
    )

    if actual_stage != ZERO_STAGE:
        raise RuntimeError(
            "ZeRO stage mismatch."
        )

    barrier()
    sync()

    init_seconds = (
        time.perf_counter()
        - init_start
    )

    init_seconds_max = (
        global_max(
            init_seconds,
            device,
        )
    )

    if RANK == 0:
        print(
            "=" * 120
        )

        print(
            "RLTrain-stage-7 "
            f"SEQ={SEQ_LEN} "
            f"ZERO={ZERO_STAGE} "
            f"WS={WORLD_SIZE} "
            f"REP={REP}"
        )

        print(
            "=" * 120
        )

        print(
            f"GLOBAL_BATCH="
            f"{GLOBAL_BATCH}"
        )

        print(
            f"MICRO_BATCH_PER_GPU="
            f"{MICRO_BATCH_PER_GPU}"
        )

        print(
            f"GRAD_ACCUM_STEPS="
            f"{GRAD_ACCUM_STEPS}"
        )

        print(
            f"GLOBAL_TOKENS_PER_OPT_STEP="
            f"{GLOBAL_TOKENS_PER_OPT_STEP}"
        )

        print(
            f"PARAMETERS="
            f"{parameter_count}"
        )

    total_steps = (
        WARMUP_OPT_STEPS
        + MEASURED_OPT_STEPS
    )

    generator = (
        torch.Generator(
            device="cpu"
        )
    )

    generator.manual_seed(
        seed
    )

    global_dataset = (
        torch.randint(
            low=100,

            high=int(
                engine.module
                .config
                .vocab_size
            ),

            size=(
                total_steps,
                GLOBAL_BATCH,
                SEQ_LEN,
            ),

            dtype=torch.long,

            generator=generator,

            device="cpu",
        )
    )

    local_begin = (
        RANK
        * LOCAL_SAMPLES_PER_OPT_STEP
    )

    local_end = (
        local_begin
        + LOCAL_SAMPLES_PER_OPT_STEP
    )

    def optimizer_step(
        global_batch_cpu,
        timed,
    ):
        shard = (
            global_batch_cpu[
                local_begin:
                local_end
            ]
        )

        losses = []

        if timed:
            barrier()
            sync()

            start = (
                time.perf_counter()
            )

        for micro in range(
            GRAD_ACCUM_STEPS
        ):
            begin = (
                micro
                * MICRO_BATCH_PER_GPU
            )

            end = (
                begin
                + MICRO_BATCH_PER_GPU
            )

            batch = (
                shard[
                    begin:end
                ]
                .to(
                    device,
                    non_blocking=False,
                )
            )

            outputs = engine(
                input_ids=batch,

                labels=batch,

                use_cache=False,
            )

            loss = (
                outputs.loss
            )

            engine.backward(
                loss
            )

            engine.step()

            losses.append(
                float(
                    loss.detach()
                    .float()
                    .item()
                )
            )

            del batch
            del outputs
            del loss

        if timed:
            sync()

            elapsed = (
                time.perf_counter()
                - start
            )
        else:
            elapsed = 0.0

        return (
            mean(losses),
            elapsed,
        )

    dataset_index = 0

    for _ in range(
        WARMUP_OPT_STEPS
    ):
        optimizer_step(
            global_dataset[
                dataset_index
            ],
            timed=False,
        )

        dataset_index += 1

    barrier()
    sync()

    torch.cuda.reset_peak_memory_stats(
        LOCAL_RANK
    )

    rows = []

    for step in range(
        1,
        MEASURED_OPT_STEPS + 1,
    ):
        (
            local_loss,
            local_elapsed,
        ) = optimizer_step(
            global_dataset[
                dataset_index
            ],
            timed=True,
        )

        dataset_index += 1

        wall_seconds = (
            global_max(
                local_elapsed,
                device,
            )
        )

        global_loss = (
            global_mean(
                local_loss,
                device,
            )
        )

        finite = (
            global_and(
                math.isfinite(
                    global_loss
                ),
                device,
            )
        )

        tok_s = (
            GLOBAL_TOKENS_PER_OPT_STEP
            / wall_seconds
        )

        if RANK == 0:
            row = {
                "run_id":
                    RUN_ID,

                "rep":
                    REP,

                "zero_stage":
                    ZERO_STAGE,

                "world_size":
                    WORLD_SIZE,

                "sequence_length":
                    SEQ_LEN,

                "step":
                    step,

                "global_batch":
                    GLOBAL_BATCH,

                "micro_batch_per_gpu":
                    MICRO_BATCH_PER_GPU,

                "gradient_accumulation_steps":
                    GRAD_ACCUM_STEPS,

                "global_tokens_per_optimizer_step":
                    GLOBAL_TOKENS_PER_OPT_STEP,

                "loss":
                    global_loss,

                "optimizer_step_ms":
                    wall_seconds
                    * 1000.0,

                "global_tokens_per_second":
                    tok_s,

                "loss_finite":
                    finite,
            }

            rows.append(
                row
            )

            print(
                f"STEP={step:02d} "
                f"LOSS={global_loss:.6f} "
                f"WALL_MS="
                f"{row['optimizer_step_ms']:.3f} "
                f"TOK_S="
                f"{tok_s:.2f}"
            )

    barrier()
    sync()

    memory_by_rank = (
        gather_memory(
            gib(
                torch.cuda
                .max_memory_allocated(
                    LOCAL_RANK
                )
            ),

            gib(
                torch.cuda
                .max_memory_reserved(
                    LOCAL_RANK
                )
            ),

            device,
        )
    )

    if RANK == 0:
        step_ms = [
            x[
                "optimizer_step_ms"
            ]
            for x in rows
        ]

        tok_s = [
            x[
                "global_tokens_per_second"
            ]
            for x in rows
        ]

        summary = {
            "schema_version": 1,

            "project":
                "RLTrain",

            "milestone":
                "RLTrain-stage-7",

            "run_id":
                RUN_ID,

            "timestamp":
                datetime.now()
                .astimezone()
                .isoformat(),

            "rep":
                REP,

            "zero_stage":
                ZERO_STAGE,

            "world_size":
                WORLD_SIZE,

            "physical_gpu_list":
                PHYSICAL_GPU_LIST,

            "model":
                "Qwen2.5-0.5B-Instruct",

            "model_snapshot":
                str(snapshot),

            "parameter_count":
                parameter_count,

            "configuration": {
                "sequence_length":
                    SEQ_LEN,

                "global_batch":
                    GLOBAL_BATCH,

                "micro_batch_per_gpu":
                    MICRO_BATCH_PER_GPU,

                "gradient_accumulation_steps":
                    GRAD_ACCUM_STEPS,

                "global_tokens_per_optimizer_step":
                    GLOBAL_TOKENS_PER_OPT_STEP,

                "warmup_optimizer_steps":
                    WARMUP_OPT_STEPS,

                "measured_optimizer_steps":
                    MEASURED_OPT_STEPS,

                "dtype":
                    "bfloat16",

                "attention":
                    "sdpa",
            },

            "result": {
                "mean_optimizer_step_ms":
                    mean(
                        step_ms
                    ),

                "median_optimizer_step_ms":
                    median(
                        step_ms
                    ),

                "mean_global_tokens_per_second":
                    mean(
                        tok_s
                    ),

                "all_losses_finite":
                    all(
                        x[
                            "loss_finite"
                        ]
                        for x in rows
                    ),

                "max_peak_allocated_gib":
                    max(
                        x[
                            "peak_allocated_gib"
                        ]
                        for x
                        in memory_by_rank
                    ),

                "max_peak_reserved_gib":
                    max(
                        x[
                            "peak_reserved_gib"
                        ]
                        for x
                        in memory_by_rank
                    ),
            },

            "engine_init_seconds_max":
                init_seconds_max,

            "memory_by_rank":
                memory_by_rank,
        }

        summary_path = (
            RESULT_DIR
            / (
                f"RLTRAIN_STAGE_7_SEQ{SEQ_LEN}_"
                f"ZERO{ZERO_STAGE}_"
                f"WS{WORLD_SIZE}_"
                f"REP{REP}_"
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

        csv_path = (
            RESULT_DIR
            / (
                f"RLTRAIN_STAGE_7_SEQ{SEQ_LEN}_"
                f"ZERO{ZERO_STAGE}_"
                f"WS{WORLD_SIZE}_"
                f"REP{REP}_"
                f"STEPS_{RUN_ID}.csv"
            )
        )

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = (
                csv.DictWriter(
                    f,
                    fieldnames=list(
                        rows[
                            0
                        ].keys()
                    ),
                )
            )

            writer.writeheader()
            writer.writerows(
                rows
            )

        print(
            "MEAN_OPT_STEP_MS="
            f"{summary['result']['mean_optimizer_step_ms']:.6f}"
        )

        print(
            "MEAN_TOK_S="
            f"{summary['result']['mean_global_tokens_per_second']:.6f}"
        )

        print(
            "MAX_ALLOC_GIB="
            f"{summary['result']['max_peak_allocated_gib']:.6f}"
        )

        print(
            "ALL_LOSSES_FINITE="
            f"{summary['result']['all_losses_finite']}"
        )

        print(
            "SUMMARY_JSON="
            + str(
                summary_path
            )
        )

    barrier()

    del global_dataset
    del optimizer
    del engine
    del model

    gc.collect()

    torch.cuda.empty_cache()

    barrier()
    sync()


def main():
    try:
        run()

    finally:
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


if __name__ == "__main__":
    main()
