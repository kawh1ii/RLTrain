#!/usr/bin/env python3

import os
import gc
import csv
import json
import time
import math
import random
import statistics
from pathlib import Path
from datetime import datetime

import torch
import deepspeed
from transformers import AutoModelForCausalLM


ROOT = Path(os.environ.get("RLTRAIN_WORKSPACE_ROOT", "/path/to/ai-infra"))
PROJECT = ROOT / "RLTrain"
RESULT_DIR = PROJECT / "results" / "stage_3"
HF_HOME = ROOT / "caches" / "huggingface"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID = os.environ["RLTRAIN_STAGE_3_RUN_ID"]
ZERO_STAGE = int(os.environ["RLTRAIN_STAGE_3_ZERO_STAGE"])

SEED = 20260825
SEQ_LEN = 512
MICRO_BATCH = 1
WARMUP_STEPS = 3
MEASURED_STEPS = 10
LR = 1e-5
WEIGHT_DECAY = 0.01


def sync():
    torch.cuda.synchronize()


def gib(x):
    return x / (1024 ** 3)


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

        for p in sorted(root.iterdir()):
            if (
                p.is_dir()
                and (p / "config.json").exists()
            ):
                return p

    raise RuntimeError(
        "Local Qwen2.5-0.5B-Instruct snapshot not found."
    )


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

    return stdev(xs) / m * 100.0


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable."
        )

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "RLTrain-stage-3 single-GPU smoke requires exactly one visible GPU."
        )

    local_rank = int(
        os.environ.get(
            "LOCAL_RANK",
            "0",
        )
    )

    torch.cuda.set_device(local_rank)

    device = torch.device(
        f"cuda:{local_rank}"
    )

    snapshot = locate_snapshot()

    print("=" * 120)
    print(
        f"RLTrain-stage-3 DEEPSPEED ZERO-{ZERO_STAGE} SINGLE-GPU SMOKE"
    )
    print("=" * 120)

    print(f"RUN_ID = {RUN_ID}")
    print(f"ZERO_STAGE = {ZERO_STAGE}")
    print(f"DEEPSPEED = {deepspeed.__version__}")
    print(f"TORCH = {torch.__version__}")
    print(f"TORCH_CUDA = {torch.version.cuda}")
    print(
        f"GPU = {torch.cuda.get_device_name(0)}"
    )
    print(f"MODEL = {snapshot}")
    print(f"SEQ_LEN = {SEQ_LEN}")
    print(f"MICRO_BATCH = {MICRO_BATCH}")
    print(f"WARMUP_STEPS = {WARMUP_STEPS}")
    print(f"MEASURED_STEPS = {MEASURED_STEPS}")
    print("WORLD_SIZE = 1")
    print(
        "FORMAL_ZERO_MEMORY_CLAIM = FALSE"
    )
    print(
        "FORMAL_ZERO_PERFORMANCE_CLAIM = FALSE"
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot),
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )

    model.config.use_cache = False
    model.train()

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"PARAMETERS = {parameter_count}"
    )

    print(
        f"TRAINABLE_PARAMETERS = {trainable_count}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        foreach=True,
    )

    ds_config = {
        "train_micro_batch_size_per_gpu":
            MICRO_BATCH,

        "gradient_accumulation_steps":
            1,

        "train_batch_size":
            MICRO_BATCH,

        "bf16": {
            "enabled": True
        },

        "zero_optimization": {
            "stage": ZERO_STAGE
        },

        "steps_per_print":
            1000000,

        "wall_clock_breakdown":
            False,
    }

    print(
        "DEEPSPEED_CONFIG = "
        + json.dumps(
            ds_config,
            sort_keys=True,
        )
    )

    engine, optimizer, _, _ = (
        deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            config=ds_config,
        )
    )

    print(
        f"ENGINE_DEVICE = {engine.device}"
    )

    try:
        actual_stage = (
            engine.zero_optimization_stage()
        )
    except Exception:
        actual_stage = None

    print(
        f"ENGINE_ZERO_STAGE = {actual_stage}"
    )

    if (
        actual_stage is not None
        and int(actual_stage) != ZERO_STAGE
    ):
        raise RuntimeError(
            f"Requested ZeRO-{ZERO_STAGE}, "
            f"engine reports {actual_stage}"
        )

    vocab_size = int(
        engine.module.config.vocab_size
    )

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        SEED + ZERO_STAGE
    )

    total_steps = (
        WARMUP_STEPS
        + MEASURED_STEPS
    )

    cpu_batches = torch.randint(
        low=100,
        high=vocab_size,
        size=(
            total_steps,
            MICRO_BATCH,
            SEQ_LEN,
        ),
        dtype=torch.long,
        generator=generator,
        device="cpu",
    )

    print("")
    print("WARMUP_BEGIN")

    for step in range(WARMUP_STEPS):
        batch = cpu_batches[step].to(
            device,
            non_blocking=False,
        )

        outputs = engine(
            input_ids=batch,
            labels=batch,
            use_cache=False,
        )

        loss = outputs.loss

        engine.backward(loss)
        engine.step()

        print(
            f"WARMUP_{step + 1} "
            f"LOSS={float(loss.detach().float()):.6f}"
        )

        del batch
        del outputs
        del loss

    sync()

    torch.cuda.reset_peak_memory_stats(
        device
    )

    rows = []

    print("")
    print("MEASURE_BEGIN")

    for i in range(MEASURED_STEPS):
        batch = cpu_batches[
            WARMUP_STEPS + i
        ].to(
            device,
            non_blocking=False,
        )

        sync()
        total_start = time.perf_counter()

        sync()
        fwd_start = time.perf_counter()

        outputs = engine(
            input_ids=batch,
            labels=batch,
            use_cache=False,
        )

        loss = outputs.loss

        sync()
        fwd_end = time.perf_counter()

        bwd_start = time.perf_counter()

        engine.backward(loss)

        sync()
        bwd_end = time.perf_counter()

        step_start = time.perf_counter()

        engine.step()

        sync()
        step_end = time.perf_counter()

        total_end = time.perf_counter()

        forward_ms = (
            fwd_end - fwd_start
        ) * 1000.0

        backward_ms = (
            bwd_end - bwd_start
        ) * 1000.0

        engine_step_ms = (
            step_end - step_start
        ) * 1000.0

        total_ms = (
            total_end - total_start
        ) * 1000.0

        token_count = (
            MICRO_BATCH * SEQ_LEN
        )

        tokens_per_second = (
            token_count
            / (total_ms / 1000.0)
        )

        loss_value = float(
            loss.detach()
            .float()
            .item()
        )

        allocated = gib(
            torch.cuda.memory_allocated(
                device
            )
        )

        reserved = gib(
            torch.cuda.memory_reserved(
                device
            )
        )

        row = {
            "run_id":
                RUN_ID,

            "zero_stage":
                ZERO_STAGE,

            "step":
                i + 1,

            "loss":
                loss_value,

            "forward_ms":
                forward_ms,

            "backward_ms":
                backward_ms,

            "engine_step_ms":
                engine_step_ms,

            "total_step_ms":
                total_ms,

            "tokens_per_second":
                tokens_per_second,

            "allocated_vram_gib":
                allocated,

            "reserved_vram_gib":
                reserved,
        }

        rows.append(row)

        print(
            f"MEASURE_{i + 1:02d} "
            f"LOSS={loss_value:.6f} "
            f"FWD_MS={forward_ms:.3f} "
            f"BWD_MS={backward_ms:.3f} "
            f"ENGINE_STEP_MS={engine_step_ms:.3f} "
            f"TOTAL_MS={total_ms:.3f} "
            f"TOK_S={tokens_per_second:.2f}"
        )

        del batch
        del outputs
        del loss

    sync()

    peak_allocated = gib(
        torch.cuda.max_memory_allocated(
            device
        )
    )

    peak_reserved = gib(
        torch.cuda.max_memory_reserved(
            device
        )
    )

    losses = [
        x["loss"]
        for x in rows
    ]

    total_times = [
        x["total_step_ms"]
        for x in rows
    ]

    forwards = [
        x["forward_ms"]
        for x in rows
    ]

    backwards = [
        x["backward_ms"]
        for x in rows
    ]

    engine_steps = [
        x["engine_step_ms"]
        for x in rows
    ]

    throughputs = [
        x["tokens_per_second"]
        for x in rows
    ]

    all_finite = all(
        math.isfinite(x)
        for x in losses
    )

    summary = {
        "schema_version": 1,
        "project": "RLTrain",
        "milestone": "RLTrain-stage-3",

        "run_id":
            RUN_ID,

        "timestamp":
            datetime.now()
            .astimezone()
            .isoformat(),

        "zero_stage":
            ZERO_STAGE,

        "world_size":
            1,

        "scope":
            "SINGLE_GPU_DEEPSPEED_ZERO_CODE_PATH_SMOKE",

        "formal_zero_memory_claim":
            False,

        "formal_zero_performance_claim":
            False,

        "environment": {
            "torch":
                torch.__version__,

            "deepspeed":
                deepspeed.__version__,

            "torch_cuda":
                torch.version.cuda,

            "gpu":
                torch.cuda.get_device_name(
                    0
                ),

            "model_snapshot":
                str(snapshot),
        },

        "configuration":
            ds_config,

        "results": {
            "measured_steps":
                MEASURED_STEPS,

            "mean_forward_ms":
                mean(forwards),

            "mean_backward_ms":
                mean(backwards),

            "mean_engine_step_ms":
                mean(engine_steps),

            "mean_total_step_ms":
                mean(total_times),

            "median_total_step_ms":
                median(total_times),

            "stdev_total_step_ms":
                stdev(total_times),

            "cv_total_step_percent":
                cv(total_times),

            "mean_tokens_per_second":
                mean(throughputs),

            "median_tokens_per_second":
                median(throughputs),

            "peak_allocated_vram_gib":
                peak_allocated,

            "peak_reserved_vram_gib":
                peak_reserved,

            "all_losses_finite":
                all_finite,

            "first_measured_loss":
                losses[0],

            "final_measured_loss":
                losses[-1],
        },
    }

    if all_finite:
        verdict = (
            f"DEEPSPEED_ZERO_{ZERO_STAGE}_"
            f"SINGLE_GPU_SMOKE_PASS"
        )
    else:
        verdict = (
            f"DEEPSPEED_ZERO_{ZERO_STAGE}_"
            f"SINGLE_GPU_SMOKE_FAIL"
        )

    summary["verdict"] = verdict

    json_path = (
        RESULT_DIR
        / f"RLTRAIN_STAGE_3_ZERO_{ZERO_STAGE}_{RUN_ID}.json"
    )

    csv_path = (
        RESULT_DIR
        / f"RLTRAIN_STAGE_3_ZERO_{ZERO_STAGE}_{RUN_ID}.csv"
    )

    json_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    print("")
    print("=" * 120)
    print(
        f"ZERO-{ZERO_STAGE} SUMMARY"
    )
    print("=" * 120)

    print(
        f"MEAN_FORWARD_MS = "
        f"{summary['results']['mean_forward_ms']:.6f}"
    )

    print(
        f"MEAN_BACKWARD_MS = "
        f"{summary['results']['mean_backward_ms']:.6f}"
    )

    print(
        f"MEAN_ENGINE_STEP_MS = "
        f"{summary['results']['mean_engine_step_ms']:.6f}"
    )

    print(
        f"MEAN_TOTAL_STEP_MS = "
        f"{summary['results']['mean_total_step_ms']:.6f}"
    )

    print(
        f"MEAN_TOKENS_PER_SECOND = "
        f"{summary['results']['mean_tokens_per_second']:.6f}"
    )

    print(
        f"CV_TOTAL_STEP_PERCENT = "
        f"{summary['results']['cv_total_step_percent']:.6f}"
    )

    print(
        f"PEAK_ALLOCATED_VRAM_GIB = "
        f"{peak_allocated:.6f}"
    )

    print(
        f"PEAK_RESERVED_VRAM_GIB = "
        f"{peak_reserved:.6f}"
    )

    print(
        f"ALL_LOSSES_FINITE = "
        f"{all_finite}"
    )

    print(
        f"FORMAL_ZERO_MEMORY_CLAIM = FALSE"
    )

    print(
        f"FORMAL_ZERO_PERFORMANCE_CLAIM = FALSE"
    )

    print(
        f"ZERO_STAGE_VERDICT = "
        f"{verdict}"
    )

    print(
        f"JSON = {json_path}"
    )

    print(
        f"CSV = {csv_path}"
    )

    del cpu_batches
    del engine
    del optimizer
    del model

    gc.collect()
    torch.cuda.empty_cache()

    if not all_finite:
        raise RuntimeError(
            f"ZeRO-{ZERO_STAGE} "
            f"produced non-finite loss."
        )


if __name__ == "__main__":
    main()
