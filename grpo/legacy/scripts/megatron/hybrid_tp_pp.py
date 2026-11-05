#!/usr/bin/env python3

import os
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)


ROOT = Path(os.environ.get("RLTRAIN_WORKSPACE_ROOT", "/path/to/ai-infra"))
PROJECT = ROOT / "RLTrain"

RESULT_DIR = (
    PROJECT
    / "results"
    / "stage_8_3"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RUN_ID = os.environ[
    "RLTRAIN_STAGE_8_3_RUN_ID"
]

TP_SIZE = int(
    os.environ[
        "RLTRAIN_STAGE_8_3_TP"
    ]
)

PP_SIZE = int(
    os.environ[
        "RLTRAIN_STAGE_8_3_PP"
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


if (
    TP_SIZE
    * PP_SIZE
    != WORLD_SIZE
):
    raise RuntimeError(
        "RLTrain-stage-8.3 requires "
        "TP * PP == WORLD_SIZE."
    )


SEQ_LEN = 4
MICRO_BATCH_SIZE = 2
NUM_MICROBATCHES = 16

HIDDEN_SIZE = 64
FFN_HIDDEN_SIZE = 128

DTYPE = torch.float32

DATA_SEED = 2026082602
WEIGHT_SEED_BASE = 2026082800

ABS_TOL = 3e-4
REL_TOL = 3e-4


def barrier():
    dist.barrier(
        device_ids=[
            LOCAL_RANK
        ]
    )


def close_pass(a, b):
    return bool(
        torch.allclose(
            a.detach().float(),
            b.detach().float(),
            atol=ABS_TOL,
            rtol=REL_TOL,
        )
    )


def max_abs_error(a, b):
    return float(
        (
            a.detach().float()
            - b.detach().float()
        )
        .abs()
        .max()
        .item()
    )


def global_and(
    value,
    device,
):
    x = torch.tensor(
        [
            1
            if value
            else 0
        ],
        dtype=torch.int32,
        device=device,
    )

    dist.all_reduce(
        x,
        op=dist.ReduceOp.MIN,
    )

    return bool(
        x.item() == 1
    )


def group_members(
    group,
    device,
):
    size = dist.get_world_size(
        group=group
    )

    rank_tensor = torch.tensor(
        [RANK],
        dtype=torch.int64,
        device=device,
    )

    gathered = [
        torch.empty_like(
            rank_tensor
        )
        for _ in range(size)
    ]

    dist.all_gather(
        gathered,
        rank_tensor,
        group=group,
    )

    return [
        int(x.item())
        for x in gathered
    ]


def make_stage_parameters(
    stage_id,
    device,
):
    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        WEIGHT_SEED_BASE
        + stage_id
    )

    w1 = (
        torch.randn(
            FFN_HIDDEN_SIZE,
            HIDDEN_SIZE,
            generator=generator,
            dtype=DTYPE,
        )
        * 0.025
    ).to(device)

    b1 = (
        torch.randn(
            FFN_HIDDEN_SIZE,
            generator=generator,
            dtype=DTYPE,
        )
        * 0.01
    ).to(device)

    w2 = (
        torch.randn(
            HIDDEN_SIZE,
            FFN_HIDDEN_SIZE,
            generator=generator,
            dtype=DTYPE,
        )
        * 0.025
    ).to(device)

    b2 = (
        torch.randn(
            HIDDEN_SIZE,
            generator=generator,
            dtype=DTYPE,
        )
        * 0.01
    ).to(device)

    return (
        w1,
        b1,
        w2,
        b2,
    )


def make_microbatches(
    device,
):
    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        DATA_SEED
    )

    result = []

    for microbatch_id in range(
        NUM_MICROBATCHES
    ):
        x = torch.randn(
            SEQ_LEN,
            MICRO_BATCH_SIZE,
            HIDDEN_SIZE,
            generator=generator,
            dtype=DTYPE,
        ).to(device)

        target = torch.randn(
            SEQ_LEN,
            MICRO_BATCH_SIZE,
            HIDDEN_SIZE,
            generator=generator,
            dtype=DTYPE,
        ).to(device)

        result.append(
            {
                "microbatch_id":
                    microbatch_id,

                "x":
                    x,

                "target":
                    target,
            }
        )

    return result


class HybridStage(nn.Module):
    def __init__(
        self,
        stage_id,
        tp_rank,
        config,
        device,
    ):
        super().__init__()

        self.stage_id = (
            stage_id
        )

        self.tp_rank = (
            tp_rank
        )

        self.config = config

        self.model_type = (
            ModelType
            .encoder_or_decoder
        )

        self.input_tensor = None

        self.forward_count = 0

        self.col = (
            ColumnParallelLinear(
                HIDDEN_SIZE,
                FFN_HIDDEN_SIZE,

                config=config,

                init_method=
                    lambda tensor: None,

                bias=True,

                gather_output=False,

                skip_bias_add=False,

                skip_weight_param_allocation=False,
            )
            .to(device)
        )

        self.row = (
            RowParallelLinear(
                FFN_HIDDEN_SIZE,
                HIDDEN_SIZE,

                config=config,

                init_method=
                    lambda tensor: None,

                bias=True,

                input_is_parallel=True,

                skip_bias_add=False,
            )
            .to(device)
        )

        (
            w1,
            b1,
            w2,
            b2,
        ) = make_stage_parameters(
            stage_id,
            device,
        )

        partition = (
            FFN_HIDDEN_SIZE
            // TP_SIZE
        )

        begin = (
            tp_rank
            * partition
        )

        end = (
            begin
            + partition
        )

        with torch.no_grad():
            self.col.weight.copy_(
                w1[
                    begin:end,
                    :
                ]
            )

            self.col.bias.copy_(
                b1[
                    begin:end
                ]
            )

            self.row.weight.copy_(
                w2[
                    :,
                    begin:end
                ]
            )

            self.row.bias.copy_(
                b2
            )

        self.expected_col_weight = (
            w1[
                begin:end,
                :
            ]
            .clone()
            .detach()
        )

        self.expected_col_bias = (
            b1[
                begin:end
            ]
            .clone()
            .detach()
        )

        self.expected_row_weight = (
            w2[
                :,
                begin:end
            ]
            .clone()
            .detach()
        )

        self.expected_row_bias = (
            b2
            .clone()
            .detach()
        )

        self.col_weight_grad_calls = 0
        self.col_bias_grad_calls = 0
        self.row_weight_grad_calls = 0
        self.row_bias_grad_calls = 0

        self.col.weight.register_hook(
            self._col_weight_hook
        )

        self.col.bias.register_hook(
            self._col_bias_hook
        )

        self.row.weight.register_hook(
            self._row_weight_hook
        )

        self.row.bias.register_hook(
            self._row_bias_hook
        )

    def _col_weight_hook(
        self,
        grad,
    ):
        self.col_weight_grad_calls += 1
        return grad

    def _col_bias_hook(
        self,
        grad,
    ):
        self.col_bias_grad_calls += 1
        return grad

    def _row_weight_hook(
        self,
        grad,
    ):
        self.row_weight_grad_calls += 1
        return grad

    def _row_bias_hook(
        self,
        grad,
    ):
        self.row_bias_grad_calls += 1
        return grad

    def set_input_tensor(
        self,
        input_tensor,
    ):
        if isinstance(
            input_tensor,
            list,
        ):
            self.input_tensor = (
                input_tensor[0]
                if len(input_tensor)
                else None
            )
        else:
            self.input_tensor = (
                input_tensor
            )

    def forward(
        self,
        data_input=None,
    ):
        self.forward_count += 1

        if self.stage_id == 0:
            x = data_input
        else:
            x = self.input_tensor

        if x is None:
            raise RuntimeError(
                f"PP stage "
                f"{self.stage_id} "
                "received no activation."
            )

        y, extra_bias = (
            self.col(x)
        )

        if extra_bias is not None:
            raise RuntimeError(
                "Unexpected ColumnParallel "
                "extra bias."
            )

        y = F.gelu(
            y,
            approximate="none",
        )

        y, extra_bias = (
            self.row(y)
        )

        if extra_bias is not None:
            raise RuntimeError(
                "Unexpected RowParallel "
                "extra bias."
            )

        return y


def build_dense_reference(
    microbatches,
    device,
):
    stages = []

    for stage_id in range(
        PP_SIZE
    ):
        (
            w1,
            b1,
            w2,
            b2,
        ) = make_stage_parameters(
            stage_id,
            device,
        )

        stages.append(
            {
                "w1":
                    w1
                    .clone()
                    .detach()
                    .requires_grad_(True),

                "b1":
                    b1
                    .clone()
                    .detach()
                    .requires_grad_(True),

                "w2":
                    w2
                    .clone()
                    .detach()
                    .requires_grad_(True),

                "b2":
                    b2
                    .clone()
                    .detach()
                    .requires_grad_(True),
            }
        )

    outputs = []
    losses = []

    total_loss = torch.zeros(
        (),
        dtype=DTYPE,
        device=device,
    )

    for item in microbatches:
        y = (
            item["x"]
            .clone()
            .detach()
        )

        for stage in stages:
            y = F.linear(
                y,
                stage["w1"],
                stage["b1"],
            )

            y = F.gelu(
                y,
                approximate="none",
            )

            y = F.linear(
                y,
                stage["w2"],
                stage["b2"],
            )

        outputs.append(
            y.detach().clone()
        )

        loss = F.mse_loss(
            y,
            item["target"],
            reduction="mean",
        )

        losses.append(
            float(
                loss
                .detach()
                .item()
            )
        )

        total_loss = (
            total_loss
            + loss
            / NUM_MICROBATCHES
        )

    total_loss.backward()

    return {
        "stages":
            stages,

        "outputs":
            outputs,

        "losses":
            losses,

        "mean_loss":
            float(
                total_loss
                .detach()
                .item()
            ),
    }


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

        pipeline_model_parallel_size=
            PP_SIZE,

        context_parallel_size=1,

        expert_model_parallel_size=1,

        create_gloo_process_groups=False,
    )

    barrier()

    tp_rank = (
        parallel_state
        .get_tensor_model_parallel_rank()
    )

    pp_rank = (
        parallel_state
        .get_pipeline_model_parallel_rank()
    )

    dp_rank = (
        parallel_state
        .get_data_parallel_rank()
    )

    tp_world = (
        parallel_state
        .get_tensor_model_parallel_world_size()
    )

    pp_world = (
        parallel_state
        .get_pipeline_model_parallel_world_size()
    )

    dp_world = (
        parallel_state
        .get_data_parallel_world_size()
    )

    if not (
        tp_world == TP_SIZE
        and
        pp_world == PP_SIZE
        and
        dp_world == 1
    ):
        raise RuntimeError(
            "Hybrid TP/PP geometry mismatch."
        )

    tp_group = (
        parallel_state
        .get_tensor_model_parallel_group()
    )

    pp_group = (
        parallel_state
        .get_pipeline_model_parallel_group()
    )

    tp_members = (
        group_members(
            tp_group,
            device,
        )
    )

    pp_members = (
        group_members(
            pp_group,
            device,
        )
    )

    config = ModelParallelConfig(
        tensor_model_parallel_size=
            TP_SIZE,

        pipeline_model_parallel_size=
            PP_SIZE,

        sequence_parallel=False,

        pipeline_dtype=DTYPE,

        overlap_p2p_comm=False,

        batch_p2p_comm=True,

        deallocate_pipeline_outputs=False,

        params_dtype=DTYPE,

        perform_initialization=False,

        use_cpu_initialization=False,

        gradient_accumulation_fusion=False,
    )

    config.hidden_size = (
        HIDDEN_SIZE
    )

    model = HybridStage(
        pp_rank,
        tp_rank,
        config,
        device,
    )

    model.train()

    microbatches = (
        make_microbatches(
            device
        )
    )

    pipeline_data = []

    for item in microbatches:
        pipeline_data.append(
            {
                "microbatch_id":
                    item[
                        "microbatch_id"
                    ],

                "x":
                    item[
                        "x"
                    ]
                    .clone()
                    .detach(),

                "target":
                    item[
                        "target"
                    ]
                    .clone()
                    .detach(),
            }
        )

    observed_outputs = []
    observed_ids = []

    def forward_step_func(
        data_iterator,
        stage_model,
    ):
        item = next(
            data_iterator
        )

        microbatch_id = int(
            item[
                "microbatch_id"
            ]
        )

        if (
            stage_model.stage_id
            == 0
        ):
            data_input = (
                item["x"]
            )
        else:
            data_input = None

        output = stage_model(
            data_input
        )

        if (
            stage_model.stage_id
            ==
            PP_SIZE - 1
        ):
            target = (
                item["target"]
            )

            def loss_func(
                output_tensor
            ):
                observed_ids.append(
                    microbatch_id
                )

                observed_outputs.append(
                    output_tensor
                    .detach()
                    .clone()
                )

                loss = F.mse_loss(
                    output_tensor,
                    target,
                    reduction="mean",
                )

                return (
                    loss,

                    {
                        "mse":
                            loss
                            .detach()
                            .clone(),

                        "microbatch_id":
                            torch.tensor(
                                microbatch_id,
                                dtype=torch.int64,
                                device=device,
                            ),
                    },
                )

            return (
                output,
                loss_func,
            )

        return (
            output,
            None,
        )

    schedule = (
        get_forward_backward_func()
    )

    schedule_name = (
        schedule.__name__
    )

    expected_schedule = (
        "forward_backward_pipelining_"
        "without_interleaving"
    )

    schedule_pass = (
        schedule_name
        ==
        expected_schedule
    )

    torch.cuda.synchronize(
        device
    )

    losses_reduced = schedule(
        forward_step_func=
            forward_step_func,

        data_iterator=
            iter(
                pipeline_data
            ),

        model=[
            model
        ],

        num_microbatches=
            NUM_MICROBATCHES,

        seq_length=
            SEQ_LEN,

        micro_batch_size=
            MICRO_BATCH_SIZE,

        forward_only=False,

        collect_non_loss_data=False,
    )

    torch.cuda.synchronize(
        device
    )

    reference = (
        build_dense_reference(
            microbatches,
            device,
        )
    )

    ref_stage = (
        reference[
            "stages"
        ][
            pp_rank
        ]
    )

    partition = (
        FFN_HIDDEN_SIZE
        // TP_SIZE
    )

    begin = (
        tp_rank
        * partition
    )

    end = (
        begin
        + partition
    )

    shard_shape_pass = (
        tuple(
            model
            .col
            .weight
            .shape
        )
        ==
        (
            partition,
            HIDDEN_SIZE,
        )

        and

        tuple(
            model
            .row
            .weight
            .shape
        )
        ==
        (
            HIDDEN_SIZE,
            partition,
        )
    )

    parameter_init_pass = all(
        [
            close_pass(
                model.col.weight,
                model.expected_col_weight,
            ),

            close_pass(
                model.col.bias,
                model.expected_col_bias,
            ),

            close_pass(
                model.row.weight,
                model.expected_row_weight,
            ),

            close_pass(
                model.row.bias,
                model.expected_row_bias,
            ),
        ]
    )

    col_weight_grad_pass = (
        model.col.weight.grad
        is not None

        and

        close_pass(
            model.col.weight.grad,

            ref_stage[
                "w1"
            ].grad[
                begin:end,
                :
            ],
        )
    )

    col_bias_grad_pass = (
        model.col.bias.grad
        is not None

        and

        close_pass(
            model.col.bias.grad,

            ref_stage[
                "b1"
            ].grad[
                begin:end
            ],
        )
    )

    row_weight_grad_pass = (
        model.row.weight.grad
        is not None

        and

        close_pass(
            model.row.weight.grad,

            ref_stage[
                "w2"
            ].grad[
                :,
                begin:end
            ],
        )
    )

    row_bias_grad_pass = (
        model.row.bias.grad
        is not None

        and

        close_pass(
            model.row.bias.grad,

            ref_stage[
                "b2"
            ].grad,
        )
    )

    execution_count_pass = all(
        [
            model.forward_count
                == NUM_MICROBATCHES,

            model.col_weight_grad_calls
                == NUM_MICROBATCHES,

            model.col_bias_grad_calls
                == NUM_MICROBATCHES,

            model.row_weight_grad_calls
                == NUM_MICROBATCHES,

            model.row_bias_grad_calls
                == NUM_MICROBATCHES,
        ]
    )

    if (
        pp_rank
        ==
        PP_SIZE - 1
    ):
        observed_loss_values = [
            float(
                item["mse"]
                .detach()
                .item()
            )
            for item
            in losses_reduced
        ]

        observed_loss_ids = [
            int(
                item[
                    "microbatch_id"
                ]
                .detach()
                .item()
            )
            for item
            in losses_reduced
        ]

        order_pass = (
            observed_ids
            ==
            list(
                range(
                    NUM_MICROBATCHES
                )
            )

            and

            observed_loss_ids
            ==
            list(
                range(
                    NUM_MICROBATCHES
                )
            )
        )

        output_pass = (
            len(
                observed_outputs
            )
            ==
            NUM_MICROBATCHES

            and

            all(
                close_pass(
                    observed_outputs[i],

                    reference[
                        "outputs"
                    ][i],
                )
                for i
                in range(
                    NUM_MICROBATCHES
                )
            )
        )

        loss_pass = (
            len(
                observed_loss_values
            )
            ==
            NUM_MICROBATCHES

            and

            all(
                abs(a - b)
                <= (
                    ABS_TOL
                    +
                    REL_TOL
                    * abs(b)
                )

                for a, b
                in zip(
                    observed_loss_values,

                    reference[
                        "losses"
                    ],
                )
            )
        )

        if output_pass:
            output_max_abs_error = max(
                max_abs_error(
                    observed_outputs[i],

                    reference[
                        "outputs"
                    ][i],
                )
                for i
                in range(
                    NUM_MICROBATCHES
                )
            )
        else:
            output_max_abs_error = (
                float("inf")
            )

    else:
        order_pass = True

        output_pass = (
            len(
                observed_outputs
            )
            == 0
        )

        loss_pass = (
            len(
                losses_reduced
            )
            == 0
        )

        output_max_abs_error = 0.0

        observed_loss_values = []
        observed_loss_ids = []

    local_pass = all(
        [
            schedule_pass,

            shard_shape_pass,

            parameter_init_pass,

            col_weight_grad_pass,

            col_bias_grad_pass,

            row_weight_grad_pass,

            row_bias_grad_pass,

            execution_count_pass,

            order_pass,

            output_pass,

            loss_pass,
        ]
    )

    all_ranks_pass = (
        global_and(
            local_pass,
            device,
        )
    )

    def grad_error(
        actual,
        expected,
    ):
        if actual is None:
            return float("inf")

        return max_abs_error(
            actual,
            expected,
        )

    record = {
        "schema_version":
            1,

        "project":
            "RLTrain",

        "milestone":
            "RLTrain-stage-8.3",

        "run_id":
            RUN_ID,

        "timestamp":
            datetime.now()
            .astimezone()
            .isoformat(),

        "rank":
            RANK,

        "tp_rank":
            tp_rank,

        "pp_rank":
            pp_rank,

        "dp_rank":
            dp_rank,

        "configuration": {
            "tp":
                TP_SIZE,

            "pp":
                PP_SIZE,

            "dp":
                1,

            "world_size":
                WORLD_SIZE,

            "num_microbatches":
                NUM_MICROBATCHES,

            "seq_len":
                SEQ_LEN,

            "micro_batch_size":
                MICRO_BATCH_SIZE,

            "hidden_size":
                HIDDEN_SIZE,

            "ffn_hidden_size":
                FFN_HIDDEN_SIZE,

            "schedule":
                schedule_name,
        },

        "groups": {
            "tp_members":
                tp_members,

            "pp_members":
                pp_members,
        },

        "checks": {
            "schedule_pass":
                schedule_pass,

            "shard_shape_pass":
                shard_shape_pass,

            "parameter_init_pass":
                parameter_init_pass,

            "col_weight_grad_pass":
                col_weight_grad_pass,

            "col_bias_grad_pass":
                col_bias_grad_pass,

            "row_weight_grad_pass":
                row_weight_grad_pass,

            "row_bias_grad_pass":
                row_bias_grad_pass,

            "execution_count_pass":
                execution_count_pass,

            "microbatch_order_pass":
                order_pass,

            "last_stage_output_pass":
                output_pass,

            "last_stage_loss_pass":
                loss_pass,
        },

        "errors": {
            "col_weight_grad_abs":
                grad_error(
                    model.col.weight.grad,

                    ref_stage[
                        "w1"
                    ].grad[
                        begin:end,
                        :
                    ],
                ),

            "col_bias_grad_abs":
                grad_error(
                    model.col.bias.grad,

                    ref_stage[
                        "b1"
                    ].grad[
                        begin:end
                    ],
                ),

            "row_weight_grad_abs":
                grad_error(
                    model.row.weight.grad,

                    ref_stage[
                        "w2"
                    ].grad[
                        :,
                        begin:end
                    ],
                ),

            "row_bias_grad_abs":
                grad_error(
                    model.row.bias.grad,

                    ref_stage[
                        "b2"
                    ].grad,
                ),

            "final_output_abs":
                output_max_abs_error,
        },

        "local_pass":
            local_pass,

        "all_ranks_pass":
            all_ranks_pass,
    }

    rank_path = (
        RESULT_DIR
        / (
            f"RLTRAIN_STAGE_8_3_TP{TP_SIZE}_"
            f"PP{PP_SIZE}_"
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
        f"PP_RANK={pp_rank} "
        f"TP_GROUP={tp_members} "
        f"PP_GROUP={pp_members} "
        f"LOCAL_PASS={local_pass}",
        flush=True,
    )

    barrier()

    if RANK == 0:
        records = []

        for rank in range(
            WORLD_SIZE
        ):
            path = (
                RESULT_DIR
                / (
                    f"RLTRAIN_STAGE_8_3_TP{TP_SIZE}_"
                    f"PP{PP_SIZE}_"
                    f"RANK{rank}_"
                    f"{RUN_ID}.json"
                )
            )

            if not path.exists():
                raise RuntimeError(
                    "Missing rank artifact: "
                    + str(path)
                )

            records.append(
                json.loads(
                    path.read_text(
                        encoding="utf-8"
                    )
                )
            )

        all_local_pass = all(
            x["local_pass"]
            for x in records
        )

        max_errors = {
            key:
                max(
                    x[
                        "errors"
                    ][key]
                    for x
                    in records
                )

            for key
            in [
                "col_weight_grad_abs",
                "col_bias_grad_abs",
                "row_weight_grad_abs",
                "row_bias_grad_abs",
                "final_output_abs",
            ]
        }

        unique_tp_groups = sorted(
            {
                tuple(
                    x[
                        "groups"
                    ][
                        "tp_members"
                    ]
                )
                for x
                in records
            }
        )

        unique_pp_groups = sorted(
            {
                tuple(
                    x[
                        "groups"
                    ][
                        "pp_members"
                    ]
                )
                for x
                in records
            }
        )

        group_geometry_pass = (
            len(
                unique_tp_groups
            )
            ==
            WORLD_SIZE
            // TP_SIZE

            and

            len(
                unique_pp_groups
            )
            ==
            WORLD_SIZE
            // PP_SIZE

            and

            all(
                len(x)
                == TP_SIZE
                for x
                in unique_tp_groups
            )

            and

            all(
                len(x)
                == PP_SIZE
                for x
                in unique_pp_groups
            )
        )

        case_pass = all(
            [
                all_local_pass,

                all_ranks_pass,

                group_geometry_pass,
            ]
        )

        summary = {
            "schema_version":
                1,

            "project":
                "RLTrain",

            "milestone":
                "RLTrain-stage-8.3",

            "run_id":
                RUN_ID,

            "configuration": {
                "tp":
                    TP_SIZE,

                "pp":
                    PP_SIZE,

                "dp":
                    1,

                "world_size":
                    WORLD_SIZE,
            },

            "unique_tp_groups":
                [
                    list(x)
                    for x
                    in unique_tp_groups
                ],

            "unique_pp_groups":
                [
                    list(x)
                    for x
                    in unique_pp_groups
                ],

            "group_geometry_pass":
                group_geometry_pass,

            "maximum_errors":
                max_errors,

            "all_ranks_pass":
                all_ranks_pass,

            "case_pass":
                case_pass,
        }

        summary_path = (
            RESULT_DIR
            / (
                f"RLTRAIN_STAGE_8_3_TP{TP_SIZE}_"
                f"PP{PP_SIZE}_"
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
            f"TP={TP_SIZE} "
            f"PP={PP_SIZE} "
            "DP=1"
        )

        print(
            "TP_GROUPS="
            + str(
                unique_tp_groups
            )
        )

        print(
            "PP_GROUPS="
            + str(
                unique_pp_groups
            )
        )

        print(
            "GROUP_GEOMETRY_PASS="
            + str(
                group_geometry_pass
            )
        )

        print(
            "MAX_COL_WEIGHT_GRAD_ABS="
            f"{max_errors['col_weight_grad_abs']:.9e}"
        )

        print(
            "MAX_ROW_WEIGHT_GRAD_ABS="
            f"{max_errors['row_weight_grad_abs']:.9e}"
        )

        print(
            "FINAL_OUTPUT_MAX_ABS="
            f"{max_errors['final_output_abs']:.9e}"
        )

        print(
            "CASE_PASS="
            + str(
                case_pass
            )
        )

        print(
            "SUMMARY_JSON="
            + str(
                summary_path
            )
        )

        if not case_pass:
            raise RuntimeError(
                "RLTrain-stage-8.3 hybrid "
                "TP/PP gate failed."
            )

    barrier()

    try:
        parallel_state.destroy_model_parallel()

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
