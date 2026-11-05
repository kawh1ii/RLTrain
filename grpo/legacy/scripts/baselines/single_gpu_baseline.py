#!/usr/bin/env python3

import os
import gc
import csv
import json
import math
import time
import traceback
import statistics
from pathlib import Path
from datetime import datetime

import torch
import transformers
from transformers import AutoModelForCausalLM


ROOT = Path(os.environ.get("RLTRAIN_WORKSPACE_ROOT", "/path/to/ai-infra"))
PROJECT = ROOT / "RLTrain"
RESULT_DIR = PROJECT / "results" / "stage_1"
DOC_DIR = PROJECT / "docs"

HF_HOME = ROOT / "caches" / "huggingface"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
DOC_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID = os.environ.get(
    "RLTRAIN_STAGE_1_RUN_ID",
    datetime.now().strftime("%Y%m%d_%H%M%S"),
)

STEP_CSV = RESULT_DIR / f"RLTRAIN_STAGE_1_STEP_RESULTS_{RUN_ID}.csv"
TRIAL_CSV = RESULT_DIR / f"RLTRAIN_STAGE_1_TRIAL_RESULTS_{RUN_ID}.csv"
SUMMARY_JSON = RESULT_DIR / f"RLTRAIN_STAGE_1_SUMMARY_{RUN_ID}.json"
DECISION_JSON = RESULT_DIR / "RLTRAIN_STAGE_1_DECISION.json"
REPORT_MD = DOC_DIR / "RLTRAIN_STAGE_1_SINGLE_GPU_BASELINE.md"

SEED = 20260825

DTYPE = torch.bfloat16

WORKLOADS = [
    {
        "name": "SHORT_SEQ512",
        "sequence_length": 512,
        "micro_batch_size": 1,
    },
    {
        "name": "LONG_SEQ2048",
        "sequence_length": 2048,
        "micro_batch_size": 1,
    },
]

TRIALS = 3
WARMUP_STEPS = 5
MEASURED_STEPS = 30

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01


def section(title):
    print("")
    print("=" * 120)
    print(title)
    print("=" * 120, flush=True)


def locate_model_snapshot():
    candidates = [
        HF_HOME
        / "hub"
        / "models--Qwen--Qwen2.5-0.5B-Instruct"
        / "snapshots",

        HF_HOME
        / "transformers"
        / "models--Qwen--Qwen2.5-0.5B-Instruct"
        / "snapshots",
    ]

    for parent in candidates:
        if not parent.exists():
            continue

        snapshots = sorted(
            [x for x in parent.iterdir() if x.is_dir()]
        )

        for snapshot in snapshots:
            if (snapshot / "config.json").exists():
                return snapshot

    raise RuntimeError(
        "Cached Qwen/Qwen2.5-0.5B-Instruct snapshot not found."
    )


def cuda_sync():
    torch.cuda.synchronize()


def bytes_to_gib(value):
    return value / (1024 ** 3)


def mean(values):
    return statistics.mean(values)


def median(values):
    return statistics.median(values)


def stdev(values):
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def cv_percent(values):
    m = mean(values)

    if m == 0:
        return 0.0

    return stdev(values) / m * 100.0


def write_csv(path, rows):
    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def create_dataset(
    vocab_size,
    batch_size,
    seq_len,
    number_of_steps,
    seed,
):
    generator = torch.Generator(
        device="cpu"
    )
    generator.manual_seed(seed)

    low = 100

    if vocab_size <= low + 1:
        low = 0

    data = torch.randint(
        low=low,
        high=vocab_size,
        size=(
            number_of_steps,
            batch_size,
            seq_len,
        ),
        dtype=torch.long,
        generator=generator,
        device="cpu",
    )

    return data


def load_model(snapshot):
    load_mode = None

    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(snapshot),
            local_files_only=True,
            torch_dtype=DTYPE,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        load_mode = "SDPA"
    except Exception as first_error:
        print(
            "SDPA explicit load failed; "
            "retrying with model default attention."
        )
        print(
            f"SDPA_LOAD_ERROR = "
            f"{type(first_error).__name__}: "
            f"{first_error}"
        )

        model = AutoModelForCausalLM.from_pretrained(
            str(snapshot),
            local_files_only=True,
            torch_dtype=DTYPE,
            low_cpu_mem_usage=True,
        )
        load_mode = "MODEL_DEFAULT"

    model.config.use_cache = False
    model.train()
    model.to("cuda:0")

    return model, load_mode


def run_training_step(
    model,
    optimizer,
    input_ids,
    measure,
):
    if measure:
        cuda_sync()
        total_start = time.perf_counter()

        zero_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        cuda_sync()
        zero_end = time.perf_counter()

        forward_start = time.perf_counter()

        outputs = model(
            input_ids=input_ids,
            labels=input_ids,
            use_cache=False,
        )

        loss = outputs.loss

        cuda_sync()
        forward_end = time.perf_counter()

        backward_start = time.perf_counter()
        loss.backward()
        cuda_sync()
        backward_end = time.perf_counter()

        optimizer_start = time.perf_counter()
        optimizer.step()
        cuda_sync()
        optimizer_end = time.perf_counter()

        total_end = time.perf_counter()

        return {
            "loss": float(loss.detach().float().item()),

            "zero_grad_ms": (
                zero_end - zero_start
            ) * 1000.0,

            "forward_ms": (
                forward_end - forward_start
            ) * 1000.0,

            "backward_ms": (
                backward_end - backward_start
            ) * 1000.0,

            "optimizer_ms": (
                optimizer_end - optimizer_start
            ) * 1000.0,

            "total_step_ms": (
                total_end - total_start
            ) * 1000.0,
        }

    optimizer.zero_grad(set_to_none=True)

    outputs = model(
        input_ids=input_ids,
        labels=input_ids,
        use_cache=False,
    )

    loss = outputs.loss
    loss.backward()
    optimizer.step()

    return {
        "loss": float(
            loss.detach().float().item()
        )
    }


def run_workload(
    workload,
    model_snapshot,
    all_step_rows,
    all_trial_rows,
):
    workload_name = workload["name"]
    seq_len = workload["sequence_length"]
    batch_size = workload["micro_batch_size"]

    section(
        f"WORKLOAD — {workload_name}"
    )

    print(f"sequence_length = {seq_len}")
    print(f"micro_batch_size = {batch_size}")
    print(f"trials = {TRIALS}")
    print(f"warmup_steps = {WARMUP_STEPS}")
    print(f"measured_steps = {MEASURED_STEPS}")

    section(
        f"{workload_name} — MODEL LOAD"
    )

    model, attention_mode = load_model(
        model_snapshot
    )

    number_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    vocab_size = int(
        model.config.vocab_size
    )

    print(
        f"attention_mode = {attention_mode}"
    )
    print(
        f"number_parameters = "
        f"{number_parameters}"
    )
    print(
        f"trainable_parameters = "
        f"{trainable_parameters}"
    )
    print(
        f"vocab_size = {vocab_size}"
    )
    print(
        f"model_dtype = "
        f"{next(model.parameters()).dtype}"
    )
    print(
        "gradient_checkpointing = FALSE"
    )
    print(
        "torch_compile = FALSE"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        foreach=True,
    )

    total_dataset_steps = (
        TRIALS
        * (
            WARMUP_STEPS
            + MEASURED_STEPS
        )
    )

    dataset = create_dataset(
        vocab_size=vocab_size,
        batch_size=batch_size,
        seq_len=seq_len,
        number_of_steps=total_dataset_steps,
        seed=SEED + seq_len,
    )

    dataset_index = 0

    workload_trial_summaries = []

    for trial in range(1, TRIALS + 1):
        section(
            f"{workload_name} — TRIAL {trial}"
        )

        print(
            f"warmup starting: "
            f"{WARMUP_STEPS} steps"
        )

        for warmup_step in range(
            1,
            WARMUP_STEPS + 1,
        ):
            cpu_batch = dataset[
                dataset_index
            ]
            dataset_index += 1

            input_ids = cpu_batch.to(
                "cuda:0",
                non_blocking=False,
            )

            result = run_training_step(
                model=model,
                optimizer=optimizer,
                input_ids=input_ids,
                measure=False,
            )

            del input_ids

            print(
                f"WARMUP "
                f"{warmup_step}/{WARMUP_STEPS} "
                f"loss={result['loss']:.6f}"
            )

        cuda_sync()

        torch.cuda.reset_peak_memory_stats(
            0
        )

        trial_step_rows = []

        for step in range(
            1,
            MEASURED_STEPS + 1,
        ):
            cpu_batch = dataset[
                dataset_index
            ]
            dataset_index += 1

            input_ids = cpu_batch.to(
                "cuda:0",
                non_blocking=False,
            )

            result = run_training_step(
                model=model,
                optimizer=optimizer,
                input_ids=input_ids,
                measure=True,
            )

            token_count = (
                batch_size * seq_len
            )

            tokens_per_second = (
                token_count
                / (
                    result[
                        "total_step_ms"
                    ]
                    / 1000.0
                )
            )

            samples_per_second = (
                batch_size
                / (
                    result[
                        "total_step_ms"
                    ]
                    / 1000.0
                )
            )

            allocated_gib = bytes_to_gib(
                torch.cuda.memory_allocated(
                    0
                )
            )

            reserved_gib = bytes_to_gib(
                torch.cuda.memory_reserved(
                    0
                )
            )

            row = {
                "run_id": RUN_ID,
                "workload": workload_name,
                "trial": trial,
                "step": step,
                "sequence_length": seq_len,
                "micro_batch_size": batch_size,
                "tokens_per_step": token_count,

                "loss": result["loss"],

                "zero_grad_ms":
                    result["zero_grad_ms"],

                "forward_ms":
                    result["forward_ms"],

                "backward_ms":
                    result["backward_ms"],

                "optimizer_ms":
                    result["optimizer_ms"],

                "total_step_ms":
                    result["total_step_ms"],

                "tokens_per_second":
                    tokens_per_second,

                "samples_per_second":
                    samples_per_second,

                "allocated_vram_gib":
                    allocated_gib,

                "reserved_vram_gib":
                    reserved_gib,
            }

            trial_step_rows.append(row)
            all_step_rows.append(row)

            print(
                f"MEASURE "
                f"{step:02d}/"
                f"{MEASURED_STEPS} "
                f"loss="
                f"{result['loss']:.6f} "
                f"forward="
                f"{result['forward_ms']:.3f}ms "
                f"backward="
                f"{result['backward_ms']:.3f}ms "
                f"optimizer="
                f"{result['optimizer_ms']:.3f}ms "
                f"total="
                f"{result['total_step_ms']:.3f}ms "
                f"tok/s="
                f"{tokens_per_second:.2f}"
            )

            del input_ids

        cuda_sync()

        peak_allocated_gib = bytes_to_gib(
            torch.cuda.max_memory_allocated(
                0
            )
        )

        peak_reserved_gib = bytes_to_gib(
            torch.cuda.max_memory_reserved(
                0
            )
        )

        total_ms_values = [
            r["total_step_ms"]
            for r in trial_step_rows
        ]

        forward_values = [
            r["forward_ms"]
            for r in trial_step_rows
        ]

        backward_values = [
            r["backward_ms"]
            for r in trial_step_rows
        ]

        optimizer_values = [
            r["optimizer_ms"]
            for r in trial_step_rows
        ]

        token_rate_values = [
            r["tokens_per_second"]
            for r in trial_step_rows
        ]

        sample_rate_values = [
            r["samples_per_second"]
            for r in trial_step_rows
        ]

        loss_values = [
            r["loss"]
            for r in trial_step_rows
        ]

        trial_summary = {
            "run_id": RUN_ID,
            "workload": workload_name,
            "trial": trial,

            "sequence_length": seq_len,
            "micro_batch_size":
                batch_size,

            "measured_steps":
                MEASURED_STEPS,

            "mean_forward_ms":
                mean(forward_values),

            "median_forward_ms":
                median(forward_values),

            "mean_backward_ms":
                mean(backward_values),

            "median_backward_ms":
                median(backward_values),

            "mean_optimizer_ms":
                mean(optimizer_values),

            "median_optimizer_ms":
                median(optimizer_values),

            "mean_total_step_ms":
                mean(total_ms_values),

            "median_total_step_ms":
                median(total_ms_values),

            "stdev_total_step_ms":
                stdev(total_ms_values),

            "cv_total_step_percent":
                cv_percent(
                    total_ms_values
                ),

            "mean_tokens_per_second":
                mean(token_rate_values),

            "median_tokens_per_second":
                median(token_rate_values),

            "mean_samples_per_second":
                mean(sample_rate_values),

            "mean_loss":
                mean(loss_values),

            "final_loss":
                loss_values[-1],

            "peak_allocated_vram_gib":
                peak_allocated_gib,

            "peak_reserved_vram_gib":
                peak_reserved_gib,

            "all_losses_finite":
                all(
                    math.isfinite(x)
                    for x in loss_values
                ),
        }

        workload_trial_summaries.append(
            trial_summary
        )
        all_trial_rows.append(
            trial_summary
        )

        print("")
        print(
            f"TRIAL_{trial}_MEAN_STEP_MS = "
            f"{trial_summary['mean_total_step_ms']:.6f}"
        )

        print(
            f"TRIAL_{trial}_MEAN_TOKENS_PER_SECOND = "
            f"{trial_summary['mean_tokens_per_second']:.6f}"
        )

        print(
            f"TRIAL_{trial}_PEAK_ALLOCATED_VRAM_GIB = "
            f"{peak_allocated_gib:.6f}"
        )

        print(
            f"TRIAL_{trial}_PEAK_RESERVED_VRAM_GIB = "
            f"{peak_reserved_gib:.6f}"
        )

        print(
            f"TRIAL_{trial}_ALL_LOSSES_FINITE = "
            f"{trial_summary['all_losses_finite']}"
        )

    trial_mean_step = [
        x["mean_total_step_ms"]
        for x in workload_trial_summaries
    ]

    trial_mean_tok = [
        x["mean_tokens_per_second"]
        for x in workload_trial_summaries
    ]

    overall = {
        "workload": workload_name,
        "sequence_length": seq_len,
        "micro_batch_size": batch_size,
        "attention_mode":
            attention_mode,

        "trial_count": TRIALS,

        "mean_trial_step_ms":
            mean(trial_mean_step),

        "median_trial_step_ms":
            median(trial_mean_step),

        "stdev_trial_step_ms":
            stdev(trial_mean_step),

        "trial_step_cv_percent":
            cv_percent(trial_mean_step),

        "mean_trial_tokens_per_second":
            mean(trial_mean_tok),

        "median_trial_tokens_per_second":
            median(trial_mean_tok),

        "tokens_per_second_cv_percent":
            cv_percent(trial_mean_tok),

        "max_peak_allocated_vram_gib":
            max(
                x[
                    "peak_allocated_vram_gib"
                ]
                for x
                in workload_trial_summaries
            ),

        "max_peak_reserved_vram_gib":
            max(
                x[
                    "peak_reserved_vram_gib"
                ]
                for x
                in workload_trial_summaries
            ),

        "all_trials_finite":
            all(
                x["all_losses_finite"]
                for x
                in workload_trial_summaries
            ),
    }

    del dataset
    del optimizer
    del model

    gc.collect()
    torch.cuda.empty_cache()
    cuda_sync()

    return overall


def main():
    section(
        "RLTrain — "
        "RLTrain-stage-1 SINGLE-GPU TRAINING BASELINE"
    )

    print(
        "SCIENTIFIC_TRAINING_EXECUTION = TRUE"
    )
    print(
        "PROJECT1_MODIFICATION = FALSE"
    )
    print(
        "MULTI_GPU_EXECUTION = FALSE"
    )
    print(
        "MODEL_DOWNLOAD = FALSE"
    )
    print(
        "SYNTHETIC_TOKENIZED_DATASET = TRUE"
    )

    section("1. ENVIRONMENT")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "RLTrain-stage-1 requires exactly one "
            "visible GPU."
        )

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "BF16 is not supported on "
            "the visible GPU."
        )

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    torch.set_float32_matmul_precision(
        "high"
    )

    device_name = torch.cuda.get_device_name(
        0
    )

    properties = (
        torch.cuda.get_device_properties(
            0
        )
    )

    snapshot = locate_model_snapshot()

    print(
        f"python = "
        f"{os.sys.executable}"
    )
    print(
        f"torch = "
        f"{torch.__version__}"
    )
    print(
        f"transformers = "
        f"{transformers.__version__}"
    )
    print(
        f"torch_cuda_runtime = "
        f"{torch.version.cuda}"
    )
    print(
        f"cuda_available = "
        f"{torch.cuda.is_available()}"
    )
    print(
        f"gpu = {device_name}"
    )
    print(
        f"compute_capability = "
        f"{properties.major}."
        f"{properties.minor}"
    )
    print(
        f"vram_gib = "
        f"{bytes_to_gib(properties.total_memory):.6f}"
    )
    print(
        f"model_snapshot = "
        f"{snapshot}"
    )
    print(
        f"dtype = {DTYPE}"
    )
    print(
        f"learning_rate = "
        f"{LEARNING_RATE}"
    )
    print(
        f"weight_decay = "
        f"{WEIGHT_DECAY}"
    )
    print(
        "optimizer = "
        "torch.optim.AdamW(foreach=True)"
    )
    print(
        "input_transfer_inside_step_timer = FALSE"
    )
    print(
        "phase_timing_cuda_synchronize = TRUE"
    )

    all_step_rows = []
    all_trial_rows = []
    workload_summaries = []
    errors = []

    for workload in WORKLOADS:
        try:
            summary = run_workload(
                workload=workload,
                model_snapshot=snapshot,
                all_step_rows=all_step_rows,
                all_trial_rows=all_trial_rows,
            )

            workload_summaries.append(
                summary
            )

        except Exception as e:
            error_record = {
                "workload":
                    workload["name"],
                "error_type":
                    type(e).__name__,
                "error":
                    str(e),
                "traceback":
                    traceback.format_exc(),
            }

            errors.append(error_record)

            section(
                f"ERROR — "
                f"{workload['name']}"
            )

            print(
                error_record["traceback"]
            )

            gc.collect()
            torch.cuda.empty_cache()

    write_csv(
        STEP_CSV,
        all_step_rows,
    )

    write_csv(
        TRIAL_CSV,
        all_trial_rows,
    )

    expected_trials = (
        len(WORKLOADS) * TRIALS
    )

    completed_trials = len(
        all_trial_rows
    )

    finite_trials = all(
        row["all_losses_finite"]
        for row in all_trial_rows
    ) if all_trial_rows else False

    if (
        len(errors) == 0
        and completed_trials
        == expected_trials
        and finite_trials
    ):
        verdict = (
            "RLTRAIN_STAGE_1_SINGLE_GPU_BASELINE_COMPLETE_"
            "READY_FOR_RLTRAIN_STAGE_2"
        )
    else:
        verdict = (
            "RLTRAIN_STAGE_1_INCOMPLETE_REVIEW_REQUIRED"
        )

    summary = {
        "schema_version": 1,
        "project": "RLTrain",
        "milestone": "RLTrain-stage-1",

        "run_id": RUN_ID,

        "timestamp":
            datetime.now()
            .astimezone()
            .isoformat(),

        "environment": {
            "python":
                os.sys.executable,

            "torch":
                torch.__version__,

            "transformers":
                transformers.__version__,

            "torch_cuda_runtime":
                torch.version.cuda,

            "gpu":
                device_name,

            "compute_capability":
                f"{properties.major}."
                f"{properties.minor}",

            "vram_gib":
                bytes_to_gib(
                    properties.total_memory
                ),

            "model_snapshot":
                str(snapshot),
        },

        "configuration": {
            "dtype":
                str(DTYPE),

            "trials":
                TRIALS,

            "warmup_steps":
                WARMUP_STEPS,

            "measured_steps":
                MEASURED_STEPS,

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "optimizer":
                "AdamW_foreach",

            "gradient_checkpointing":
                False,

            "torch_compile":
                False,

            "input_transfer_in_timer":
                False,

            "synchronized_phase_timing":
                True,

            "workloads":
                WORKLOADS,
        },

        "workload_summaries":
            workload_summaries,

        "completed_trials":
            completed_trials,

        "expected_trials":
            expected_trials,

        "all_losses_finite":
            finite_trials,

        "errors":
            errors,

        "verdict":
            verdict,
    }

    SUMMARY_JSON.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    decision = {
        "milestone": "RLTrain-stage-1",
        "run_id": RUN_ID,
        "verdict": verdict,
        "completed_trials":
            completed_trials,
        "expected_trials":
            expected_trials,
        "all_losses_finite":
            finite_trials,
        "error_count":
            len(errors),
        "next_milestone": (
            "RLTrain-stage-2 torch.distributed / "
            "NCCL Primitives"
            if verdict.startswith(
                "RLTRAIN_STAGE_1_SINGLE"
            )
            else None
        ),
    }

    DECISION_JSON.write_text(
        json.dumps(
            decision,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    report_lines = []

    report_lines.append(
        "# RLTrain-stage-1 Single-GPU Training Baseline"
    )
    report_lines.append("")
    report_lines.append(
        f"Run ID: `{RUN_ID}`"
    )
    report_lines.append("")
    report_lines.append(
        f"Verdict: `{verdict}`"
    )
    report_lines.append("")
    report_lines.append(
        "## Environment"
    )
    report_lines.append("")
    report_lines.append(
        f"- GPU: {device_name}"
    )
    report_lines.append(
        f"- PyTorch: {torch.__version__}"
    )
    report_lines.append(
        f"- Torch CUDA runtime: "
        f"{torch.version.cuda}"
    )
    report_lines.append(
        f"- Transformers: "
        f"{transformers.__version__}"
    )
    report_lines.append(
        f"- Model: "
        f"Qwen/Qwen2.5-0.5B-Instruct"
    )
    report_lines.append(
        f"- Dtype: BF16"
    )
    report_lines.append("")
    report_lines.append(
        "## Experimental Protocol"
    )
    report_lines.append("")
    report_lines.append(
        f"- Trials per workload: {TRIALS}"
    )
    report_lines.append(
        f"- Warmup steps per trial: "
        f"{WARMUP_STEPS}"
    )
    report_lines.append(
        f"- Measured steps per trial: "
        f"{MEASURED_STEPS}"
    )
    report_lines.append(
        "- Synthetic deterministic "
        "tokenized input"
    )
    report_lines.append(
        "- Input H2D transfer excluded "
        "from timed training step"
    )
    report_lines.append(
        "- CUDA synchronized phase timing"
    )
    report_lines.append(
        "- torch.compile disabled"
    )
    report_lines.append(
        "- Gradient checkpointing disabled"
    )
    report_lines.append("")
    report_lines.append(
        "## Workload Summary"
    )
    report_lines.append("")
    report_lines.append(
        "| Workload | Seq | Batch | "
        "Mean Step ms | Mean tok/s | "
        "Trial CV % | Peak Alloc GiB | "
        "Peak Reserved GiB |"
    )
    report_lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    )

    for item in workload_summaries:
        report_lines.append(
            f"| {item['workload']} "
            f"| {item['sequence_length']} "
            f"| {item['micro_batch_size']} "
            f"| {item['mean_trial_step_ms']:.3f} "
            f"| {item['mean_trial_tokens_per_second']:.2f} "
            f"| {item['trial_step_cv_percent']:.3f} "
            f"| {item['max_peak_allocated_vram_gib']:.3f} "
            f"| {item['max_peak_reserved_vram_gib']:.3f} |"
        )

    report_lines.append("")
    report_lines.append(
        "## Evidence"
    )
    report_lines.append("")
    report_lines.append(
        f"- `{STEP_CSV}`"
    )
    report_lines.append(
        f"- `{TRIAL_CSV}`"
    )
    report_lines.append(
        f"- `{SUMMARY_JSON}`"
    )
    report_lines.append(
        f"- `{DECISION_JSON}`"
    )
    report_lines.append("")
    report_lines.append(
        "## Notes"
    )
    report_lines.append("")
    report_lines.append(
        "This milestone establishes the "
        "single-GPU training reference "
        "before distributed execution."
    )
    report_lines.append("")
    report_lines.append(
        "The workload deliberately uses "
        "fixed synthetic tokenized data "
        "so later distributed comparisons "
        "can control sequence length and "
        "tokens per step."
    )
    report_lines.append("")

    if errors:
        report_lines.append(
            "## Errors"
        )
        report_lines.append("")

        for error in errors:
            report_lines.append(
                f"- {error['workload']}: "
                f"{error['error_type']}: "
                f"{error['error']}"
            )

        report_lines.append("")

    REPORT_MD.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    section(
        "RLTrain-stage-1 FINAL SUMMARY"
    )

    for item in workload_summaries:
        print("")
        print(
            f"WORKLOAD = "
            f"{item['workload']}"
        )
        print(
            f"SEQ = "
            f"{item['sequence_length']}"
        )
        print(
            f"MEAN_STEP_MS = "
            f"{item['mean_trial_step_ms']:.6f}"
        )
        print(
            f"MEAN_TOKENS_PER_SECOND = "
            f"{item['mean_trial_tokens_per_second']:.6f}"
        )
        print(
            f"TRIAL_STEP_CV_PERCENT = "
            f"{item['trial_step_cv_percent']:.6f}"
        )
        print(
            f"MAX_PEAK_ALLOCATED_VRAM_GIB = "
            f"{item['max_peak_allocated_vram_gib']:.6f}"
        )
        print(
            f"MAX_PEAK_RESERVED_VRAM_GIB = "
            f"{item['max_peak_reserved_vram_gib']:.6f}"
        )

    print("")
    print(
        f"COMPLETED_TRIALS = "
        f"{completed_trials}"
    )
    print(
        f"EXPECTED_TRIALS = "
        f"{expected_trials}"
    )
    print(
        f"ALL_LOSSES_FINITE = "
        f"{finite_trials}"
    )
    print(
        f"ERROR_COUNT = "
        f"{len(errors)}"
    )
    print(
        f"FINAL_VERDICT = "
        f"{verdict}"
    )

    section(
        "RLTrain-stage-1 ARTIFACTS"
    )

    print(
        f"STEP_CSV = {STEP_CSV}"
    )
    print(
        f"TRIAL_CSV = {TRIAL_CSV}"
    )
    print(
        f"SUMMARY_JSON = "
        f"{SUMMARY_JSON}"
    )
    print(
        f"DECISION_JSON = "
        f"{DECISION_JSON}"
    )
    print(
        f"REPORT_MD = {REPORT_MD}"
    )

    if verdict != (
        "RLTRAIN_STAGE_1_SINGLE_GPU_BASELINE_COMPLETE_"
        "READY_FOR_RLTRAIN_STAGE_2"
    ):
        raise RuntimeError(
            f"RLTrain-stage-1 gate failed: {verdict}"
        )


if __name__ == "__main__":
    main()
