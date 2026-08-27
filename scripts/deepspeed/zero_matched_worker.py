#!/usr/bin/env python3

import os
import gc
import csv
import json
import math
import time
import random
import statistics
from pathlib import Path
from datetime import datetime

import torch
import torch.distributed as dist
import deepspeed

from transformers import AutoModelForCausalLM


ROOT = Path(os.environ.get("RLTRAIN_WORKSPACE_ROOT", "/path/to/ai-infra"))
PROJECT = ROOT / "RLTrain"
RESULT_DIR = PROJECT / "results" / "stage_6_1_v2"
HF_HOME = ROOT / "caches" / "huggingface"

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RUN_ID = os.environ["RLTRAIN_STAGE_6_1_RUN_ID"]
ZERO_STAGE = int(
    os.environ["RLTRAIN_STAGE_6_1_ZERO_STAGE"]
)

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

SEED = 20260825

GLOBAL_BATCH = 8
MICRO_BATCH_PER_GPU = 1
SEQ_LEN = 512

TRIALS = 3
WARMUP_OPT_STEPS = 3
MEASURED_OPT_STEPS = 12

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
        "Global batch is not divisible "
        "by world_size * micro_batch."
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


def sync():
    torch.cuda.synchronize(
        LOCAL_RANK
    )


def barrier():
    dist.barrier(
        device_ids=[LOCAL_RANK]
    )


def gib(x):
    return x / (1024 ** 3)


def mean(xs):
    return statistics.mean(xs)


def median(xs):
    return statistics.median(xs)


def stdev(xs):
    if len(xs) < 2:
        return 0.0
    return statistics.stdev(xs)


def cv(xs):
    m = mean(xs)

    if m == 0:
        return 0.0

    return (
        stdev(xs)
        / m
        * 100.0
    )


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
        "Cached Qwen model not found."
    )


def global_max_vector(
    values,
    device,
):
    t = torch.tensor(
        values,
        dtype=torch.float64,
        device=device,
    )

    dist.all_reduce(
        t,
        op=dist.ReduceOp.MAX,
    )

    return [
        float(x)
        for x in t.tolist()
    ]


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

    return float(t.item())


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
        for _ in range(WORLD_SIZE)
    ]

    dist.all_gather(
        gathered,
        local,
    )

    return [
        {
            "rank": rank,

            "physical_gpu":
                PHYSICAL_GPU_LIST[rank],

            "peak_allocated_gib":
                float(t[0].item()),

            "peak_reserved_gib":
                float(t[1].item()),
        }
        for rank, t
        in enumerate(gathered)
    ]


def run():
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    torch.cuda.set_device(
        LOCAL_RANK
    )

    device = torch.device(
        "cuda",
        LOCAL_RANK,
    )

    snapshot = locate_snapshot()

    init_start = time.perf_counter()

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
        for p in model.parameters()
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        foreach=False,
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
            "stage": ZERO_STAGE,
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

    if not dist.is_initialized():
        raise RuntimeError(
            "Distributed process group "
            "was not initialized."
        )

    actual_stage = int(
        engine.zero_optimization_stage()
    )

    if actual_stage != ZERO_STAGE:
        raise RuntimeError(
            f"Requested ZeRO-{ZERO_STAGE}, "
            f"actual={actual_stage}"
        )

    barrier()
    sync()

    init_seconds = (
        time.perf_counter()
        - init_start
    )

    init_max = global_max_vector(
        [init_seconds],
        device,
    )[0]

    if RANK == 0:
        print("=" * 120)
        print(
            "RLTrain-stage-6.1 V2 MATCHED "
            f"ZERO-{ZERO_STAGE} "
            f"WORLD_SIZE={WORLD_SIZE}"
        )
        print("=" * 120)

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
            f"LOCAL_SAMPLES_PER_OPT_STEP="
            f"{LOCAL_SAMPLES_PER_OPT_STEP}"
        )

        print(
            f"SEQ_LEN={SEQ_LEN}"
        )

        print(
            "GLOBAL_TOKENS_PER_OPT_STEP="
            f"{GLOBAL_TOKENS_PER_OPT_STEP}"
        )

        print(
            f"PARAMETERS="
            f"{parameter_count}"
        )

        print(
            f"ENGINE_INIT_SECONDS_MAX="
            f"{init_max:.6f}"
        )

    total_optimizer_steps = (
        TRIALS
        * (
            WARMUP_OPT_STEPS
            + MEASURED_OPT_STEPS
        )
    )

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        SEED
    )

    #
    # Every rank constructs the exact same
    # deterministic sequence of GLOBAL batches.
    #
    global_dataset = torch.randint(
        low=100,
        high=int(
            engine.module.config.vocab_size
        ),
        size=(
            total_optimizer_steps,
            GLOBAL_BATCH,
            SEQ_LEN,
        ),
        dtype=torch.long,
        generator=generator,
        device="cpu",
    )

    local_start = (
        RANK
        * LOCAL_SAMPLES_PER_OPT_STEP
    )

    local_end = (
        local_start
        + LOCAL_SAMPLES_PER_OPT_STEP
    )

    dataset_index = 0

    trial_records = []
    step_rows = []

    def execute_optimizer_step(
        global_batch_cpu,
        timed,
    ):
        local_shard = (
            global_batch_cpu[
                local_start:
                local_end
            ]
        )

        local_losses = []

        total_forward_s = 0.0
        total_backward_s = 0.0
        total_engine_step_s = 0.0

        if timed:
            barrier()
            sync()
            optimizer_step_start = (
                time.perf_counter()
            )

        for micro_index in range(
            GRAD_ACCUM_STEPS
        ):
            begin = (
                micro_index
                * MICRO_BATCH_PER_GPU
            )

            end = (
                begin
                + MICRO_BATCH_PER_GPU
            )

            batch = (
                local_shard[
                    begin:end
                ]
                .to(
                    device,
                    non_blocking=False,
                )
            )

            if timed:
                sync()
                t0 = time.perf_counter()

            outputs = engine(
                input_ids=batch,
                labels=batch,
                use_cache=False,
            )

            loss = outputs.loss

            if timed:
                sync()
                t1 = time.perf_counter()

            engine.backward(loss)

            if timed:
                sync()
                t2 = time.perf_counter()

            engine.step()

            if timed:
                sync()
                t3 = time.perf_counter()

                total_forward_s += (
                    t1 - t0
                )

                total_backward_s += (
                    t2 - t1
                )

                total_engine_step_s += (
                    t3 - t2
                )

            local_losses.append(
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

            optimizer_step_end = (
                time.perf_counter()
            )

            local_total_s = (
                optimizer_step_end
                - optimizer_step_start
            )
        else:
            local_total_s = 0.0

        local_mean_loss = mean(
            local_losses
        )

        return (
            local_mean_loss,
            total_forward_s,
            total_backward_s,
            total_engine_step_s,
            local_total_s,
        )

    for trial in range(
        1,
        TRIALS + 1,
    ):
        if RANK == 0:
            print("")
            print(
                f"BEGIN_TRIAL={trial}"
            )

        #
        # Warmup optimizer steps.
        #
        for _ in range(
            WARMUP_OPT_STEPS
        ):
            global_batch_cpu = (
                global_dataset[
                    dataset_index
                ]
            )

            dataset_index += 1

            execute_optimizer_step(
                global_batch_cpu,
                timed=False,
            )

        barrier()
        sync()

        torch.cuda.reset_peak_memory_stats(
            LOCAL_RANK
        )

        trial_rows = []

        for step in range(
            1,
            MEASURED_OPT_STEPS + 1,
        ):
            global_batch_cpu = (
                global_dataset[
                    dataset_index
                ]
            )

            dataset_index += 1

            (
                local_loss,
                local_forward_s,
                local_backward_s,
                local_engine_step_s,
                local_total_s,
            ) = execute_optimizer_step(
                global_batch_cpu,
                timed=True,
            )

            (
                forward_s,
                backward_s,
                engine_step_s,
                total_s,
            ) = global_max_vector(
                [
                    local_forward_s,
                    local_backward_s,
                    local_engine_step_s,
                    local_total_s,
                ],
                device,
            )

            global_loss = global_mean(
                local_loss,
                device,
            )

            finite = global_and(
                math.isfinite(
                    global_loss
                ),
                device,
            )

            global_tok_s = (
                GLOBAL_TOKENS_PER_OPT_STEP
                / total_s
            )

            row = {
                "run_id":
                    RUN_ID,

                "zero_stage":
                    ZERO_STAGE,

                "world_size":
                    WORLD_SIZE,

                "trial":
                    trial,

                "optimizer_step":
                    step,

                "global_batch":
                    GLOBAL_BATCH,

                "micro_batch_per_gpu":
                    MICRO_BATCH_PER_GPU,

                "gradient_accumulation_steps":
                    GRAD_ACCUM_STEPS,

                "local_samples_per_optimizer_step":
                    LOCAL_SAMPLES_PER_OPT_STEP,

                "sequence_length":
                    SEQ_LEN,

                "global_tokens_per_optimizer_step":
                    GLOBAL_TOKENS_PER_OPT_STEP,

                "global_loss":
                    global_loss,

                "summed_forward_ms":
                    forward_s * 1000.0,

                "summed_backward_ms":
                    backward_s * 1000.0,

                "summed_engine_step_ms":
                    engine_step_s * 1000.0,

                "optimizer_step_wall_ms":
                    total_s * 1000.0,

                "global_tokens_per_second":
                    global_tok_s,

                "loss_finite":
                    finite,
            }

            if RANK == 0:
                trial_rows.append(row)
                step_rows.append(row)

                print(
                    f"TRIAL={trial} "
                    f"STEP={step:02d} "
                    f"LOSS={global_loss:.6f} "
                    f"WALL_MS="
                    f"{row['optimizer_step_wall_ms']:.3f} "
                    f"TOK_S="
                    f"{global_tok_s:.2f}"
                )

        barrier()
        sync()

        local_peak_alloc = gib(
            torch.cuda.max_memory_allocated(
                LOCAL_RANK
            )
        )

        local_peak_reserved = gib(
            torch.cuda.max_memory_reserved(
                LOCAL_RANK
            )
        )

        memory_by_rank = gather_memory(
            local_peak_alloc,
            local_peak_reserved,
            device,
        )

        if RANK == 0:
            wall_values = [
                x[
                    "optimizer_step_wall_ms"
                ]
                for x in trial_rows
            ]

            tok_values = [
                x[
                    "global_tokens_per_second"
                ]
                for x in trial_rows
            ]

            forward_values = [
                x[
                    "summed_forward_ms"
                ]
                for x in trial_rows
            ]

            backward_values = [
                x[
                    "summed_backward_ms"
                ]
                for x in trial_rows
            ]

            engine_values = [
                x[
                    "summed_engine_step_ms"
                ]
                for x in trial_rows
            ]

            losses = [
                x[
                    "global_loss"
                ]
                for x in trial_rows
            ]

            trial_record = {
                "trial":
                    trial,

                "mean_optimizer_step_ms":
                    mean(
                        wall_values
                    ),

                "median_optimizer_step_ms":
                    median(
                        wall_values
                    ),

                "step_cv_percent":
                    cv(
                        wall_values
                    ),

                "mean_global_tokens_per_second":
                    mean(
                        tok_values
                    ),

                "mean_summed_forward_ms":
                    mean(
                        forward_values
                    ),

                "mean_summed_backward_ms":
                    mean(
                        backward_values
                    ),

                "mean_summed_engine_step_ms":
                    mean(
                        engine_values
                    ),

                "mean_loss":
                    mean(
                        losses
                    ),

                "all_losses_finite":
                    all(
                        x[
                            "loss_finite"
                        ]
                        for x
                        in trial_rows
                    ),

                "memory_by_rank":
                    memory_by_rank,

                "max_peak_allocated_gib":
                    max(
                        x[
                            "peak_allocated_gib"
                        ]
                        for x
                        in memory_by_rank
                    ),

                "mean_peak_allocated_gib":
                    mean(
                        [
                            x[
                                "peak_allocated_gib"
                            ]
                            for x
                            in memory_by_rank
                        ]
                    ),

                "max_peak_reserved_gib":
                    max(
                        x[
                            "peak_reserved_gib"
                        ]
                        for x
                        in memory_by_rank
                    ),
            }

            trial_records.append(
                trial_record
            )

            print(
                f"TRIAL_{trial}_"
                f"MEAN_OPT_STEP_MS="
                f"{trial_record['mean_optimizer_step_ms']:.6f}"
            )

            print(
                f"TRIAL_{trial}_"
                f"MEAN_TOK_S="
                f"{trial_record['mean_global_tokens_per_second']:.6f}"
            )

            print(
                f"TRIAL_{trial}_"
                f"MAX_ALLOC_GIB="
                f"{trial_record['max_peak_allocated_gib']:.6f}"
            )

        barrier()

    if RANK == 0:
        trial_step_means = [
            x[
                "mean_optimizer_step_ms"
            ]
            for x in trial_records
        ]

        trial_tok_means = [
            x[
                "mean_global_tokens_per_second"
            ]
            for x in trial_records
        ]

        summary = {
            "schema_version": 2,

            "project":
                "RLTrain",

            "milestone":
                "RLTrain-stage-6.1",

            "run_id":
                RUN_ID,

            "timestamp":
                datetime.now()
                .astimezone()
                .isoformat(),

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
                "global_batch":
                    GLOBAL_BATCH,

                "micro_batch_per_gpu":
                    MICRO_BATCH_PER_GPU,

                "gradient_accumulation_steps":
                    GRAD_ACCUM_STEPS,

                "local_samples_per_optimizer_step":
                    LOCAL_SAMPLES_PER_OPT_STEP,

                "sequence_length":
                    SEQ_LEN,

                "global_tokens_per_optimizer_step":
                    GLOBAL_TOKENS_PER_OPT_STEP,

                "dtype":
                    "bfloat16",

                "attention":
                    "sdpa",

                "trials":
                    TRIALS,

                "warmup_optimizer_steps":
                    WARMUP_OPT_STEPS,

                "measured_optimizer_steps":
                    MEASURED_OPT_STEPS,
            },

            "engine_init_seconds_max":
                init_max,

            "trials":
                trial_records,

            "overall": {
                "mean_trial_optimizer_step_ms":
                    mean(
                        trial_step_means
                    ),

                "median_trial_optimizer_step_ms":
                    median(
                        trial_step_means
                    ),

                "trial_optimizer_step_cv_percent":
                    cv(
                        trial_step_means
                    ),

                "mean_trial_global_tokens_per_second":
                    mean(
                        trial_tok_means
                    ),

                "max_peak_allocated_gib":
                    max(
                        x[
                            "max_peak_allocated_gib"
                        ]
                        for x
                        in trial_records
                    ),

                "mean_peak_allocated_gib":
                    mean(
                        [
                            x[
                                "mean_peak_allocated_gib"
                            ]
                            for x
                            in trial_records
                        ]
                    ),

                "max_peak_reserved_gib":
                    max(
                        x[
                            "max_peak_reserved_gib"
                        ]
                        for x
                        in trial_records
                    ),

                "all_losses_finite":
                    all(
                        x[
                            "all_losses_finite"
                        ]
                        for x
                        in trial_records
                    ),
            },

            "scope": {
                "matched_global_batch":
                    True,

                "matched_micro_batch_per_gpu":
                    True,

                "matched_global_tokens_per_optimizer_step":
                    True,

                "gradient_accumulation_adjusted_by_world_size":
                    True,

                "formal_zero_memory_comparison":
                    True,

                "formal_zero_optimizer_step_time_comparison":
                    True,

                "general_performance_recommendation":
                    False,
            },
        }

        summary_path = (
            RESULT_DIR
            / (
                f"RLTRAIN_STAGE_6_1_V2_ZERO{ZERO_STAGE}_"
                f"WS{WORLD_SIZE}_"
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
                f"RLTRAIN_STAGE_6_1_V2_ZERO{ZERO_STAGE}_"
                f"WS{WORLD_SIZE}_"
                f"STEPS_{RUN_ID}.csv"
            )
        )

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    step_rows[0].keys()
                ),
            )

            writer.writeheader()
            writer.writerows(
                step_rows
            )

        print("")
        print(
            f"ZERO_STAGE={ZERO_STAGE}"
        )

        print(
            f"WORLD_SIZE={WORLD_SIZE}"
        )

        print(
            "MICRO_BATCH_PER_GPU="
            f"{MICRO_BATCH_PER_GPU}"
        )

        print(
            "GRAD_ACCUM_STEPS="
            f"{GRAD_ACCUM_STEPS}"
        )

        print(
            "MEDIAN_TRIAL_OPT_STEP_MS="
            f"{summary['overall']['median_trial_optimizer_step_ms']:.6f}"
        )

        print(
            "MEAN_GLOBAL_TOKENS_PER_SECOND="
            f"{summary['overall']['mean_trial_global_tokens_per_second']:.6f}"
        )

        print(
            "MAX_PEAK_ALLOCATED_GIB="
            f"{summary['overall']['max_peak_allocated_gib']:.6f}"
        )

        print(
            "MAX_PEAK_RESERVED_GIB="
            f"{summary['overall']['max_peak_reserved_gib']:.6f}"
        )

        print(
            "TRIAL_CV_PERCENT="
            f"{summary['overall']['trial_optimizer_step_cv_percent']:.6f}"
        )

        print(
            "ALL_LOSSES_FINITE="
            f"{summary['overall']['all_losses_finite']}"
        )

        print(
            "SUMMARY_JSON="
            + str(summary_path)
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
                dist.barrier(
                    device_ids=[
                        LOCAL_RANK
                    ]
                )
            except Exception:
                pass

            try:
                dist.destroy_process_group()
            except Exception:
                pass


if __name__ == "__main__":
    main()
