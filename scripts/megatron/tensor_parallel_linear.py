#!/usr/bin/env python3

import os
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)


ROOT = Path(os.environ.get("RLTRAIN_WORKSPACE_ROOT", "/path/to/ai-infra"))
PROJECT = ROOT / "RLTrain"

RESULT_DIR = (
    PROJECT
    / "results"
    / "stage_8_1"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RUN_ID = os.environ[
    "RLTRAIN_STAGE_8_1_RUN_ID"
]

TP_SIZE = int(
    os.environ[
        "RLTRAIN_STAGE_8_1_TP"
    ]
)

RANK = int(
    os.environ["RANK"]
)

LOCAL_RANK = int(
    os.environ["LOCAL_RANK"]
)

WORLD_SIZE = int(
    os.environ["WORLD_SIZE"]
)


if WORLD_SIZE != TP_SIZE:
    raise RuntimeError(
        "RLTrain-stage-8.1 requires "
        "WORLD_SIZE == TP_SIZE."
    )


#
# All dimensions deliberately divisible by 8
# so the exact same dense workload can be used
# for TP=1/2/4/8.
#

SEQ = 5
BATCH = 3

INPUT_SIZE = 512
OUTPUT_SIZE = 768

ABS_TOL = 2e-4
REL_TOL = 2e-4

DTYPE = torch.float32

SEED = 20260826


def barrier():
    dist.barrier(
        device_ids=[
            LOCAL_RANK
        ]
    )


def max_abs_error(a, b):
    return float(
        (
            a.detach().float()
            -
            b.detach().float()
        )
        .abs()
        .max()
        .item()
    )


def max_rel_error(a, b):
    a = a.detach().float()
    b = b.detach().float()

    denom = torch.maximum(
        b.abs(),
        torch.tensor(
            1e-6,
            device=b.device,
            dtype=b.dtype,
        ),
    )

    return float(
        (
            (a - b).abs()
            / denom
        )
        .max()
        .item()
    )


def close_pass(a, b):
    return bool(
        torch.allclose(
            a.detach().float(),
            b.detach().float(),
            rtol=REL_TOL,
            atol=ABS_TOL,
        )
    )


def global_and(value, device):
    t = torch.tensor(
        [
            1
            if value
            else 0
        ],
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


def deterministic_tensors(device):
    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        SEED
    )

    x = torch.randn(
        SEQ,
        BATCH,
        INPUT_SIZE,
        generator=generator,
        dtype=DTYPE,
    ).to(device)

    col_weight = (
        torch.randn(
            OUTPUT_SIZE,
            INPUT_SIZE,
            generator=generator,
            dtype=DTYPE,
        )
        * 0.02
    ).to(device)

    col_bias = (
        torch.randn(
            OUTPUT_SIZE,
            generator=generator,
            dtype=DTYPE,
        )
        * 0.02
    ).to(device)

    col_probe = (
        torch.randn(
            SEQ,
            BATCH,
            OUTPUT_SIZE,
            generator=generator,
            dtype=DTYPE,
        )
        * 0.02
    ).to(device)

    row_weight = (
        torch.randn(
            OUTPUT_SIZE,
            INPUT_SIZE,
            generator=generator,
            dtype=DTYPE,
        )
        * 0.02
    ).to(device)

    row_bias = (
        torch.randn(
            OUTPUT_SIZE,
            generator=generator,
            dtype=DTYPE,
        )
        * 0.02
    ).to(device)

    row_probe = (
        torch.randn(
            SEQ,
            BATCH,
            OUTPUT_SIZE,
            generator=generator,
            dtype=DTYPE,
        )
        * 0.02
    ).to(device)

    return {
        "x":
            x,

        "col_weight":
            col_weight,

        "col_bias":
            col_bias,

        "col_probe":
            col_probe,

        "row_weight":
            row_weight,

        "row_bias":
            row_bias,

        "row_probe":
            row_probe,
    }


def test_column_parallel(
    config,
    tp_group,
    tp_rank,
    tensors,
    device,
):
    output_partition = (
        OUTPUT_SIZE
        // TP_SIZE
    )

    out_begin = (
        tp_rank
        * output_partition
    )

    out_end = (
        out_begin
        + output_partition
    )

    layer = ColumnParallelLinear(
        INPUT_SIZE,
        OUTPUT_SIZE,

        config=config,

        init_method=
            lambda tensor: None,

        bias=True,

        gather_output=True,

        skip_bias_add=False,

        skip_weight_param_allocation=False,

        tp_group=tp_group,
    ).to(device)

    expected_weight_shape = (
        output_partition,
        INPUT_SIZE,
    )

    expected_bias_shape = (
        output_partition,
    )

    shape_pass = (
        tuple(
            layer.weight.shape
        )
        == expected_weight_shape

        and

        tuple(
            layer.bias.shape
        )
        == expected_bias_shape
    )

    with torch.no_grad():
        layer.weight.copy_(
            tensors[
                "col_weight"
            ][
                out_begin:
                out_end,
                :
            ]
        )

        layer.bias.copy_(
            tensors[
                "col_bias"
            ][
                out_begin:
                out_end
            ]
        )

    parameter_value_pass = (
        close_pass(
            layer.weight,
            tensors[
                "col_weight"
            ][
                out_begin:
                out_end,
                :
            ],
        )

        and

        close_pass(
            layer.bias,
            tensors[
                "col_bias"
            ][
                out_begin:
                out_end
            ],
        )
    )

    tp_input = (
        tensors[
            "x"
        ]
        .clone()
        .detach()
        .requires_grad_(
            True
        )
    )

    dense_input = (
        tensors[
            "x"
        ]
        .clone()
        .detach()
        .requires_grad_(
            True
        )
    )

    dense_weight = (
        tensors[
            "col_weight"
        ]
        .clone()
        .detach()
        .requires_grad_(
            True
        )
    )

    dense_bias = (
        tensors[
            "col_bias"
        ]
        .clone()
        .detach()
        .requires_grad_(
            True
        )
    )

    tp_output, tp_extra_bias = (
        layer(
            tp_input
        )
    )

    dense_output = F.linear(
        dense_input,
        dense_weight,
        dense_bias,
    )

    forward_pass = (
        close_pass(
            tp_output,
            dense_output,
        )

        and

        tp_extra_bias is None
    )

    forward_abs_error = (
        max_abs_error(
            tp_output,
            dense_output,
        )
    )

    forward_rel_error = (
        max_rel_error(
            tp_output,
            dense_output,
        )
    )

    tp_loss = (
        tp_output
        * tensors[
            "col_probe"
        ]
    ).sum()

    dense_loss = (
        dense_output
        * tensors[
            "col_probe"
        ]
    ).sum()

    tp_loss.backward()
    dense_loss.backward()

    input_grad_pass = (
        close_pass(
            tp_input.grad,
            dense_input.grad,
        )
    )

    weight_grad_pass = (
        close_pass(
            layer.weight.grad,
            dense_weight.grad[
                out_begin:
                out_end,
                :
            ],
        )
    )

    bias_grad_pass = (
        close_pass(
            layer.bias.grad,
            dense_bias.grad[
                out_begin:
                out_end
            ],
        )
    )

    result = {
        "expected_weight_shape":
            list(
                expected_weight_shape
            ),

        "observed_weight_shape":
            list(
                layer.weight.shape
            ),

        "expected_bias_shape":
            list(
                expected_bias_shape
            ),

        "observed_bias_shape":
            list(
                layer.bias.shape
            ),

        "weight_fraction_vs_dense":
            (
                layer.weight.numel()
                /
                dense_weight.numel()
            ),

        "shape_pass":
            shape_pass,

        "parameter_value_pass":
            parameter_value_pass,

        "forward_pass":
            forward_pass,

        "forward_max_abs_error":
            forward_abs_error,

        "forward_max_rel_error":
            forward_rel_error,

        "input_grad_pass":
            input_grad_pass,

        "input_grad_max_abs_error":
            max_abs_error(
                tp_input.grad,
                dense_input.grad,
            ),

        "weight_grad_pass":
            weight_grad_pass,

        "weight_grad_max_abs_error":
            max_abs_error(
                layer.weight.grad,
                dense_weight.grad[
                    out_begin:
                    out_end,
                    :
                ],
            ),

        "bias_grad_pass":
            bias_grad_pass,

        "bias_grad_max_abs_error":
            max_abs_error(
                layer.bias.grad,
                dense_bias.grad[
                    out_begin:
                    out_end
                ],
            ),
    }

    result["pass"] = all(
        [
            result[
                "shape_pass"
            ],

            result[
                "parameter_value_pass"
            ],

            result[
                "forward_pass"
            ],

            result[
                "input_grad_pass"
            ],

            result[
                "weight_grad_pass"
            ],

            result[
                "bias_grad_pass"
            ],
        ]
    )

    del layer
    del tp_input
    del dense_input
    del dense_weight
    del dense_bias
    del tp_output
    del dense_output

    return result


def test_row_parallel(
    config,
    tp_group,
    tp_rank,
    tensors,
    device,
):
    input_partition = (
        INPUT_SIZE
        // TP_SIZE
    )

    in_begin = (
        tp_rank
        * input_partition
    )

    in_end = (
        in_begin
        + input_partition
    )

    layer = RowParallelLinear(
        INPUT_SIZE,
        OUTPUT_SIZE,

        config=config,

        init_method=
            lambda tensor: None,

        bias=True,

        input_is_parallel=True,

        skip_bias_add=False,

        tp_group=tp_group,
    ).to(device)

    expected_weight_shape = (
        OUTPUT_SIZE,
        input_partition,
    )

    expected_bias_shape = (
        OUTPUT_SIZE,
    )

    shape_pass = (
        tuple(
            layer.weight.shape
        )
        == expected_weight_shape

        and

        tuple(
            layer.bias.shape
        )
        == expected_bias_shape
    )

    with torch.no_grad():
        layer.weight.copy_(
            tensors[
                "row_weight"
            ][
                :,
                in_begin:
                in_end
            ]
        )

        layer.bias.copy_(
            tensors[
                "row_bias"
            ]
        )

    parameter_value_pass = (
        close_pass(
            layer.weight,
            tensors[
                "row_weight"
            ][
                :,
                in_begin:
                in_end
            ],
        )

        and

        close_pass(
            layer.bias,
            tensors[
                "row_bias"
            ],
        )
    )

    tp_input_local = (
        tensors[
            "x"
        ][
            :,
            :,
            in_begin:
            in_end
        ]
        .clone()
        .detach()
        .requires_grad_(
            True
        )
    )

    dense_input = (
        tensors[
            "x"
        ]
        .clone()
        .detach()
        .requires_grad_(
            True
        )
    )

    dense_weight = (
        tensors[
            "row_weight"
        ]
        .clone()
        .detach()
        .requires_grad_(
            True
        )
    )

    dense_bias = (
        tensors[
            "row_bias"
        ]
        .clone()
        .detach()
        .requires_grad_(
            True
        )
    )

    tp_output, tp_extra_bias = (
        layer(
            tp_input_local
        )
    )

    dense_output = F.linear(
        dense_input,
        dense_weight,
        dense_bias,
    )

    forward_pass = (
        close_pass(
            tp_output,
            dense_output,
        )

        and

        tp_extra_bias is None
    )

    forward_abs_error = (
        max_abs_error(
            tp_output,
            dense_output,
        )
    )

    forward_rel_error = (
        max_rel_error(
            tp_output,
            dense_output,
        )
    )

    tp_loss = (
        tp_output
        * tensors[
            "row_probe"
        ]
    ).sum()

    dense_loss = (
        dense_output
        * tensors[
            "row_probe"
        ]
    ).sum()

    tp_loss.backward()
    dense_loss.backward()

    input_grad_slice_pass = (
        close_pass(
            tp_input_local.grad,
            dense_input.grad[
                :,
                :,
                in_begin:
                in_end
            ],
        )
    )

    gathered_input_grads = [
        torch.empty_like(
            tp_input_local.grad
        )

        for _ in range(
            TP_SIZE
        )
    ]

    dist.all_gather(
        gathered_input_grads,
        tp_input_local.grad,
        group=tp_group,
    )

    assembled_input_grad = (
        torch.cat(
            gathered_input_grads,
            dim=-1,
        )
    )

    assembled_input_grad_pass = (
        close_pass(
            assembled_input_grad,
            dense_input.grad,
        )
    )

    weight_grad_pass = (
        close_pass(
            layer.weight.grad,
            dense_weight.grad[
                :,
                in_begin:
                in_end
            ],
        )
    )

    bias_grad_pass = (
        close_pass(
            layer.bias.grad,
            dense_bias.grad,
        )
    )

    result = {
        "expected_weight_shape":
            list(
                expected_weight_shape
            ),

        "observed_weight_shape":
            list(
                layer.weight.shape
            ),

        "expected_bias_shape":
            list(
                expected_bias_shape
            ),

        "observed_bias_shape":
            list(
                layer.bias.shape
            ),

        "weight_fraction_vs_dense":
            (
                layer.weight.numel()
                /
                dense_weight.numel()
            ),

        "shape_pass":
            shape_pass,

        "parameter_value_pass":
            parameter_value_pass,

        "forward_pass":
            forward_pass,

        "forward_max_abs_error":
            forward_abs_error,

        "forward_max_rel_error":
            forward_rel_error,

        "input_grad_slice_pass":
            input_grad_slice_pass,

        "input_grad_slice_max_abs_error":
            max_abs_error(
                tp_input_local.grad,
                dense_input.grad[
                    :,
                    :,
                    in_begin:
                    in_end
                ],
            ),

        "assembled_input_grad_pass":
            assembled_input_grad_pass,

        "assembled_input_grad_max_abs_error":
            max_abs_error(
                assembled_input_grad,
                dense_input.grad,
            ),

        "weight_grad_pass":
            weight_grad_pass,

        "weight_grad_max_abs_error":
            max_abs_error(
                layer.weight.grad,
                dense_weight.grad[
                    :,
                    in_begin:
                    in_end
                ],
            ),

        "bias_grad_pass":
            bias_grad_pass,

        "bias_grad_max_abs_error":
            max_abs_error(
                layer.bias.grad,
                dense_bias.grad,
            ),
    }

    result["pass"] = all(
        [
            result[
                "shape_pass"
            ],

            result[
                "parameter_value_pass"
            ],

            result[
                "forward_pass"
            ],

            result[
                "input_grad_slice_pass"
            ],

            result[
                "assembled_input_grad_pass"
            ],

            result[
                "weight_grad_pass"
            ],

            result[
                "bias_grad_pass"
            ],
        ]
    )

    del layer
    del tp_input_local
    del dense_input
    del dense_weight
    del dense_bias
    del tp_output
    del dense_output
    del assembled_input_grad

    return result


def main():
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

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=
            TP_SIZE,

        pipeline_model_parallel_size=1,

        context_parallel_size=1,

        expert_model_parallel_size=1,

        create_gloo_process_groups=False,
    )

    barrier()

    tp_group = (
        parallel_state
        .get_tensor_model_parallel_group()
    )

    tp_rank = (
        parallel_state
        .get_tensor_model_parallel_rank()
    )

    tp_world = (
        parallel_state
        .get_tensor_model_parallel_world_size()
    )

    if tp_world != TP_SIZE:
        raise RuntimeError(
            "Megatron TP world size mismatch."
        )

    config = ModelParallelConfig(
        tensor_model_parallel_size=
            TP_SIZE,

        pipeline_model_parallel_size=1,

        sequence_parallel=False,

        perform_initialization=False,

        use_cpu_initialization=False,

        params_dtype=DTYPE,

        gradient_accumulation_fusion=False,
    )

    tensors = (
        deterministic_tensors(
            device
        )
    )

    column_result = (
        test_column_parallel(
            config,
            tp_group,
            tp_rank,
            tensors,
            device,
        )
    )

    barrier()

    row_result = (
        test_row_parallel(
            config,
            tp_group,
            tp_rank,
            tensors,
            device,
        )
    )

    local_pass = (
        column_result[
            "pass"
        ]

        and

        row_result[
            "pass"
        ]
    )

    all_ranks_pass = (
        global_and(
            local_pass,
            device,
        )
    )

    record = {
        "schema_version":
            1,

        "project":
            "RLTrain",

        "milestone":
            "RLTrain-stage-8.1",

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

        "tp_rank":
            tp_rank,

        "tp_size":
            TP_SIZE,

        "configuration": {
            "sequence":
                SEQ,

            "batch":
                BATCH,

            "input_size":
                INPUT_SIZE,

            "output_size":
                OUTPUT_SIZE,

            "dtype":
                str(DTYPE),

            "abs_tolerance":
                ABS_TOL,

            "rel_tolerance":
                REL_TOL,

            "gather_column_output":
                True,

            "row_input_is_parallel":
                True,
        },

        "column_parallel":
            column_result,

        "row_parallel":
            row_result,

        "local_pass":
            local_pass,

        "all_ranks_pass":
            all_ranks_pass,
    }

    rank_path = (
        RESULT_DIR
        / (
            f"RLTRAIN_STAGE_8_1_TP{TP_SIZE}_"
            f"RANK{RANK}_"
            f"{RUN_ID}.json"
        )
    )

    rank_path.write_text(
        json.dumps(
            record,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"RANK={RANK} "
        f"TP_RANK={tp_rank} "
        f"COLUMN_PASS="
        f"{column_result['pass']} "
        f"ROW_PASS="
        f"{row_result['pass']} "
        f"LOCAL_PASS={local_pass}",
        flush=True,
    )

    print(
        f"RANK={RANK} "
        f"COLUMN_FWD_MAX_ABS="
        f"{column_result['forward_max_abs_error']:.9e} "
        f"ROW_FWD_MAX_ABS="
        f"{row_result['forward_max_abs_error']:.9e}",
        flush=True,
    )

    barrier()

    if RANK == 0:
        rank_records = []

        for rank in range(
            TP_SIZE
        ):
            path = (
                RESULT_DIR
                / (
                    f"RLTRAIN_STAGE_8_1_TP{TP_SIZE}_"
                    f"RANK{rank}_"
                    f"{RUN_ID}.json"
                )
            )

            rank_records.append(
                json.loads(
                    path.read_text(
                        encoding="utf-8"
                    )
                )
            )

        expected_fraction = (
            1.0
            / TP_SIZE
        )

        shard_fraction_pass = all(
            abs(
                x[
                    "column_parallel"
                ][
                    "weight_fraction_vs_dense"
                ]
                - expected_fraction
            )
            < 1e-12

            and

            abs(
                x[
                    "row_parallel"
                ][
                    "weight_fraction_vs_dense"
                ]
                - expected_fraction
            )
            < 1e-12

            for x in rank_records
        )

        maximum_errors = {
            "column_forward_abs":
                max(
                    x[
                        "column_parallel"
                    ][
                        "forward_max_abs_error"
                    ]
                    for x
                    in rank_records
                ),

            "column_input_grad_abs":
                max(
                    x[
                        "column_parallel"
                    ][
                        "input_grad_max_abs_error"
                    ]
                    for x
                    in rank_records
                ),

            "column_weight_grad_abs":
                max(
                    x[
                        "column_parallel"
                    ][
                        "weight_grad_max_abs_error"
                    ]
                    for x
                    in rank_records
                ),

            "column_bias_grad_abs":
                max(
                    x[
                        "column_parallel"
                    ][
                        "bias_grad_max_abs_error"
                    ]
                    for x
                    in rank_records
                ),

            "row_forward_abs":
                max(
                    x[
                        "row_parallel"
                    ][
                        "forward_max_abs_error"
                    ]
                    for x
                    in rank_records
                ),

            "row_input_grad_abs":
                max(
                    x[
                        "row_parallel"
                    ][
                        "assembled_input_grad_max_abs_error"
                    ]
                    for x
                    in rank_records
                ),

            "row_weight_grad_abs":
                max(
                    x[
                        "row_parallel"
                    ][
                        "weight_grad_max_abs_error"
                    ]
                    for x
                    in rank_records
                ),

            "row_bias_grad_abs":
                max(
                    x[
                        "row_parallel"
                    ][
                        "bias_grad_max_abs_error"
                    ]
                    for x
                    in rank_records
                ),
        }

        case_pass = (
            all(
                x[
                    "local_pass"
                ]
                for x
                in rank_records
            )

            and

            shard_fraction_pass

            and

            all_ranks_pass
        )

        summary = {
            "schema_version":
                1,

            "project":
                "RLTrain",

            "milestone":
                "RLTrain-stage-8.1",

            "run_id":
                RUN_ID,

            "tp_size":
                TP_SIZE,

            "expected_local_weight_fraction":
                expected_fraction,

            "shard_fraction_pass":
                shard_fraction_pass,

            "maximum_errors":
                maximum_errors,

            "all_ranks_pass":
                all_ranks_pass,

            "case_pass":
                case_pass,
        }

        summary_path = (
            RESULT_DIR
            / (
                f"RLTRAIN_STAGE_8_1_TP{TP_SIZE}_"
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

        print("")
        print(
            f"TP_SIZE={TP_SIZE}"
        )

        print(
            "EXPECTED_LOCAL_WEIGHT_FRACTION="
            f"{expected_fraction:.6f}"
        )

        print(
            "SHARD_FRACTION_PASS="
            f"{shard_fraction_pass}"
        )

        print(
            "COLUMN_FORWARD_MAX_ABS="
            f"{maximum_errors['column_forward_abs']:.9e}"
        )

        print(
            "ROW_FORWARD_MAX_ABS="
            f"{maximum_errors['row_forward_abs']:.9e}"
        )

        print(
            "ALL_RANKS_PASS="
            f"{all_ranks_pass}"
        )

        print(
            "CASE_PASS="
            f"{case_pass}"
        )

        print(
            "SUMMARY_JSON="
            + str(
                summary_path
            )
        )

    barrier()

    if not all_ranks_pass:
        raise RuntimeError(
            "RLTrain-stage-8.1 TP correctness failed."
        )

    try:
        parallel_state.destroy_model_parallel()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
