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
RESULT_DIR = PROJECT / "results" / "stage_8_4"

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RUN_ID = os.environ["RLTRAIN_STAGE_8_4_RUN_ID"]

RANK = int(os.environ["RANK"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])


TP_SIZE = 2
PP_SIZE = 2
DP_SIZE = 2

if TP_SIZE * PP_SIZE * DP_SIZE != WORLD_SIZE:
    raise RuntimeError(
        "Expected TP2 * PP2 * DP2 == WORLD_SIZE=8."
    )


SEQ_LEN = 4
MICRO_BATCH_SIZE = 2
NUM_MICROBATCHES = 8

HIDDEN_SIZE = 64
FFN_HIDDEN_SIZE = 128

DTYPE = torch.float32

DATA_SEED_BASE = 2026082900
WEIGHT_SEED_BASE = 2026083000

ABS_TOL = 3e-4
REL_TOL = 3e-4


def barrier():
    dist.barrier(
        device_ids=[LOCAL_RANK]
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


def global_and(value, device):
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


def group_members(group, device):
    group_size = dist.get_world_size(
        group=group
    )

    local = torch.tensor(
        [RANK],
        dtype=torch.int64,
        device=device,
    )

    gathered = [
        torch.empty_like(local)
        for _ in range(group_size)
    ]

    dist.all_gather(
        gathered,
        local,
        group=group,
    )

    return [
        int(x.item())
        for x in gathered
    ]


def group_coordinates(
    group,
    tp_rank,
    pp_rank,
    dp_rank,
    device,
):
    group_size = dist.get_world_size(
        group=group
    )

    local = torch.tensor(
        [
            tp_rank,
            pp_rank,
            dp_rank,
        ],
        dtype=torch.int64,
        device=device,
    )

    gathered = [
        torch.empty_like(local)
        for _ in range(group_size)
    ]

    dist.all_gather(
        gathered,
        local,
        group=group,
    )

    return [
        [
            int(t[0].item()),
            int(t[1].item()),
            int(t[2].item()),
        ]
        for t in gathered
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

    return w1, b1, w2, b2


def make_microbatches(
    replica_id,
    device,
):
    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        DATA_SEED_BASE
        + replica_id
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

        self.stage_id = stage_id
        self.tp_rank = tp_rank

        self.config = config

        self.model_type = (
            ModelType.encoder_or_decoder
        )

        self.input_tensor = None
        self.forward_count = 0

        self.col = ColumnParallelLinear(
            HIDDEN_SIZE,
            FFN_HIDDEN_SIZE,
            config=config,
            init_method=lambda tensor: None,
            bias=True,
            gather_output=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
        ).to(device)

        self.row = RowParallelLinear(
            FFN_HIDDEN_SIZE,
            HIDDEN_SIZE,
            config=config,
            init_method=lambda tensor: None,
            bias=True,
            input_is_parallel=True,
            skip_bias_add=False,
        ).to(device)

        w1, b1, w2, b2 = (
            make_stage_parameters(
                stage_id,
                device,
            )
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
                if input_tensor
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
                "Pipeline stage received "
                "no activation."
            )

        y, extra_bias = self.col(x)

        if extra_bias is not None:
            raise RuntimeError(
                "Unexpected ColumnParallelLinear "
                "extra bias."
            )

        y = F.gelu(
            y,
            approximate="none",
        )

        y, extra_bias = self.row(y)

        if extra_bias is not None:
            raise RuntimeError(
                "Unexpected RowParallelLinear "
                "extra bias."
            )

        return y


def build_dense_reference(
    replica_id,
    device,
):
    microbatches = (
        make_microbatches(
            replica_id,
            device,
        )
    )

    stages = []

    for stage_id in range(
        PP_SIZE
    ):
        w1, b1, w2, b2 = (
            make_stage_parameters(
                stage_id,
                device,
            )
        )

        stages.append(
            {
                "w1":
                    w1.clone()
                    .detach()
                    .requires_grad_(True),

                "b1":
                    b1.clone()
                    .detach()
                    .requires_grad_(True),

                "w2":
                    w2.clone()
                    .detach()
                    .requires_grad_(True),

                "b2":
                    b2.clone()
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
                loss.detach().item()
            )
        )

        total_loss = (
            total_loss
            + loss
            / NUM_MICROBATCHES
        )

    total_loss.backward()

    return {
        "microbatches":
            microbatches,

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

    geometry_size_pass = (
        tp_world == TP_SIZE
        and
        pp_world == PP_SIZE
        and
        dp_world == DP_SIZE
        and
        tp_world * pp_world * dp_world
        == WORLD_SIZE
    )

    tp_group = (
        parallel_state
        .get_tensor_model_parallel_group()
    )

    pp_group = (
        parallel_state
        .get_pipeline_model_parallel_group()
    )

    dp_group = (
        parallel_state
        .get_data_parallel_group()
    )

    tp_members = group_members(
        tp_group,
        device,
    )

    pp_members = group_members(
        pp_group,
        device,
    )

    dp_members = group_members(
        dp_group,
        device,
    )

    tp_coords = group_coordinates(
        tp_group,
        tp_rank,
        pp_rank,
        dp_rank,
        device,
    )

    pp_coords = group_coordinates(
        pp_group,
        tp_rank,
        pp_rank,
        dp_rank,
        device,
    )

    dp_coords = group_coordinates(
        dp_group,
        tp_rank,
        pp_rank,
        dp_rank,
        device,
    )

    tp_orthogonal_pass = (
        sorted(
            x[0]
            for x in tp_coords
        )
        == [0, 1]

        and

        len(
            {
                x[1]
                for x in tp_coords
            }
        )
        == 1

        and

        len(
            {
                x[2]
                for x in tp_coords
            }
        )
        == 1
    )

    pp_orthogonal_pass = (
        sorted(
            x[1]
            for x in pp_coords
        )
        == [0, 1]

        and

        len(
            {
                x[0]
                for x in pp_coords
            }
        )
        == 1

        and

        len(
            {
                x[2]
                for x in pp_coords
            }
        )
        == 1
    )

    dp_orthogonal_pass = (
        sorted(
            x[2]
            for x in dp_coords
        )
        == [0, 1]

        and

        len(
            {
                x[0]
                for x in dp_coords
            }
        )
        == 1

        and

        len(
            {
                x[1]
                for x in dp_coords
            }
        )
        == 1
    )

    coordinate_orthogonality_pass = (
        tp_orthogonal_pass
        and
        pp_orthogonal_pass
        and
        dp_orthogonal_pass
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

    local_microbatches = (
        make_microbatches(
            dp_rank,
            device,
        )
    )

    pipeline_data = []

    for item in local_microbatches:
        pipeline_data.append(
            {
                "microbatch_id":
                    item[
                        "microbatch_id"
                    ],

                "x":
                    item["x"]
                    .clone()
                    .detach(),

                "target":
                    item["target"]
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

        data_input = (
            item["x"]
            if stage_model.stage_id == 0
            else None
        )

        output = stage_model(
            data_input
        )

        if (
            stage_model.stage_id
            == PP_SIZE - 1
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
        == expected_schedule
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

    #
    # Independent dense references for
    # BOTH data-parallel replicas.
    #

    references = {
        replica:
            build_dense_reference(
                replica,
                device,
            )

        for replica
        in range(
            DP_SIZE
        )
    }

    local_reference = (
        references[
            dp_rank
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

    local_ref_stage = (
        local_reference[
            "stages"
        ][
            pp_rank
        ]
    )

    local_expected = {
        "col_weight":
            local_ref_stage[
                "w1"
            ].grad[
                begin:end,
                :
            ],

        "col_bias":
            local_ref_stage[
                "b1"
            ].grad[
                begin:end
            ],

        "row_weight":
            local_ref_stage[
                "w2"
            ].grad[
                :,
                begin:end
            ],

        "row_bias":
            local_ref_stage[
                "b2"
            ].grad,
    }

    params = {
        "col_weight":
            model.col.weight,

        "col_bias":
            model.col.bias,

        "row_weight":
            model.row.weight,

        "row_bias":
            model.row.bias,
    }

    pre_sync_grads = {
        name:
            param.grad
            .detach()
            .clone()

        for name, param
        in params.items()
    }

    local_gradient_pass = all(
        close_pass(
            pre_sync_grads[
                name
            ],
            local_expected[
                name
            ],
        )

        for name
        in params
    )

    local_gradient_errors = {
        name:
            max_abs_error(
                pre_sync_grads[
                    name
                ],
                local_expected[
                    name
                ],
            )

        for name
        in params
    }

    #
    # Build exact expected DP-average gradient.
    #

    average_expected = {}

    ref0 = (
        references[0][
            "stages"
        ][
            pp_rank
        ]
    )

    ref1 = (
        references[1][
            "stages"
        ][
            pp_rank
        ]
    )

    average_expected[
        "col_weight"
    ] = (
        ref0["w1"].grad[
            begin:end,
            :
        ]
        +
        ref1["w1"].grad[
            begin:end,
            :
        ]
    ) / DP_SIZE

    average_expected[
        "col_bias"
    ] = (
        ref0["b1"].grad[
            begin:end
        ]
        +
        ref1["b1"].grad[
            begin:end
        ]
    ) / DP_SIZE

    average_expected[
        "row_weight"
    ] = (
        ref0["w2"].grad[
            :,
            begin:end
        ]
        +
        ref1["w2"].grad[
            :,
            begin:end
        ]
    ) / DP_SIZE

    average_expected[
        "row_bias"
    ] = (
        ref0["b2"].grad
        +
        ref1["b2"].grad
    ) / DP_SIZE

    #
    # Verify the two data replicas are actually
    # scientifically distinct before reduction.
    #

    replica_gradient_difference = max(
        max_abs_error(
            ref0["w1"].grad[
                begin:end,
                :
            ],
            ref1["w1"].grad[
                begin:end,
                :
            ],
        ),

        max_abs_error(
            ref0["w2"].grad[
                :,
                begin:end
            ],
            ref1["w2"].grad[
                :,
                begin:end
            ],
        ),
    )

    replica_gradient_distinct_pass = (
        replica_gradient_difference
        > 1e-8
    )

    #
    # REAL DATA-PARALLEL GRADIENT SYNCHRONIZATION.
    #
    # Every DP group contains the same TP shard
    # at the same pipeline stage, but from
    # different data replicas.
    #

    dp_allreduce_count = 0

    for param in params.values():
        if param.grad is None:
            raise RuntimeError(
                "Missing gradient before DP sync."
            )

        dist.all_reduce(
            param.grad,
            op=dist.ReduceOp.SUM,
            group=dp_group,
        )

        param.grad.div_(
            float(
                DP_SIZE
            )
        )

        dp_allreduce_count += 1

    torch.cuda.synchronize(
        device
    )

    post_sync_grads = {
        name:
            param.grad
            .detach()
            .clone()

        for name, param
        in params.items()
    }

    dp_gradient_average_pass = all(
        close_pass(
            post_sync_grads[
                name
            ],
            average_expected[
                name
            ],
        )

        for name
        in params
    )

    dp_gradient_errors = {
        name:
            max_abs_error(
                post_sync_grads[
                    name
                ],
                average_expected[
                    name
                ],
            )

        for name
        in params
    }

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

    if pp_rank == PP_SIZE - 1:
        observed_loss_values = [
            float(
                x["mse"]
                .detach()
                .item()
            )
            for x
            in losses_reduced
        ]

        observed_loss_ids = [
            int(
                x[
                    "microbatch_id"
                ]
                .detach()
                .item()
            )
            for x
            in losses_reduced
        ]

        microbatch_order_pass = (
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

        forward_output_pass = (
            len(
                observed_outputs
            )
            ==
            NUM_MICROBATCHES

            and

            all(
                close_pass(
                    observed_outputs[i],
                    local_reference[
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
                    local_reference[
                        "losses"
                    ],
                )
            )
        )

        if forward_output_pass:
            final_output_max_abs = max(
                max_abs_error(
                    observed_outputs[i],
                    local_reference[
                        "outputs"
                    ][i],
                )
                for i
                in range(
                    NUM_MICROBATCHES
                )
            )
        else:
            final_output_max_abs = (
                float("inf")
            )

    else:
        microbatch_order_pass = True

        forward_output_pass = (
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

        final_output_max_abs = 0.0

    local_pass = all(
        [
            geometry_size_pass,

            coordinate_orthogonality_pass,

            schedule_pass,

            execution_count_pass,

            local_gradient_pass,

            replica_gradient_distinct_pass,

            dp_allreduce_count == 4,

            dp_gradient_average_pass,

            microbatch_order_pass,

            forward_output_pass,

            loss_pass,
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
            "RLTrain-stage-8.4",

        "run_id":
            RUN_ID,

        "timestamp":
            datetime.now()
            .astimezone()
            .isoformat(),

        "rank":
            RANK,

        "coordinates": {
            "tp_rank":
                tp_rank,

            "pp_rank":
                pp_rank,

            "dp_rank":
                dp_rank,
        },

        "configuration": {
            "tp":
                TP_SIZE,

            "pp":
                PP_SIZE,

            "dp":
                DP_SIZE,

            "world_size":
                WORLD_SIZE,

            "num_microbatches":
                NUM_MICROBATCHES,

            "micro_batch_size":
                MICRO_BATCH_SIZE,

            "sequence_length":
                SEQ_LEN,

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

            "dp_members":
                dp_members,

            "tp_coordinates":
                tp_coords,

            "pp_coordinates":
                pp_coords,

            "dp_coordinates":
                dp_coords,
        },

        "checks": {
            "geometry_size_pass":
                geometry_size_pass,

            "coordinate_orthogonality_pass":
                coordinate_orthogonality_pass,

            "schedule_pass":
                schedule_pass,

            "execution_count_pass":
                execution_count_pass,

            "local_gradient_dense_reference_pass":
                local_gradient_pass,

            "replica_gradient_distinct_pass":
                replica_gradient_distinct_pass,

            "dp_gradient_average_pass":
                dp_gradient_average_pass,

            "microbatch_order_pass":
                microbatch_order_pass,

            "forward_output_pass":
                forward_output_pass,

            "loss_pass":
                loss_pass,
        },

        "evidence": {
            "dp_allreduce_count":
                dp_allreduce_count,

            "replica_gradient_difference":
                replica_gradient_difference,

            "local_gradient_errors":
                local_gradient_errors,

            "dp_average_gradient_errors":
                dp_gradient_errors,

            "final_output_max_abs":
                final_output_max_abs,
        },

        "local_pass":
            local_pass,

        "all_ranks_pass":
            all_ranks_pass,
    }

    rank_path = (
        RESULT_DIR
        / (
            f"RLTRAIN_STAGE_8_4_TRLTRAIN_PRLTRAIN_DRLTRAIN_"
            f"RANK{RANK}_{RUN_ID}.json"
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
        f"DP_RANK={dp_rank} "
        f"TP_GROUP={tp_members} "
        f"PP_GROUP={pp_members} "
        f"DP_GROUP={dp_members} "
        f"LOCAL_GRAD_PASS={local_gradient_pass} "
        f"DP_AVG_PASS={dp_gradient_average_pass} "
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
                    "RLTRAIN_STAGE_8_4_TRLTRAIN_PRLTRAIN_DRLTRAIN_"
                    f"RANK{rank}_{RUN_ID}.json"
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

        unique_tp_groups = sorted(
            {
                tuple(
                    x[
                        "groups"
                    ][
                        "tp_members"
                    ]
                )
                for x in records
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
                for x in records
            }
        )

        unique_dp_groups = sorted(
            {
                tuple(
                    x[
                        "groups"
                    ][
                        "dp_members"
                    ]
                )
                for x in records
            }
        )

        group_geometry_pass = all(
            [
                len(
                    unique_tp_groups
                )
                == 4,

                len(
                    unique_pp_groups
                )
                == 4,

                len(
                    unique_dp_groups
                )
                == 4,

                all(
                    len(x)
                    == 2
                    for x
                    in unique_tp_groups
                ),

                all(
                    len(x)
                    == 2
                    for x
                    in unique_pp_groups
                ),

                all(
                    len(x)
                    == 2
                    for x
                    in unique_dp_groups
                ),
            ]
        )

        local_gradient_pass_all = all(
            x[
                "checks"
            ][
                "local_gradient_dense_reference_pass"
            ]
            for x
            in records
        )

        dp_average_pass_all = all(
            x[
                "checks"
            ][
                "dp_gradient_average_pass"
            ]
            for x
            in records
        )

        distinct_replica_pass_all = all(
            x[
                "checks"
            ][
                "replica_gradient_distinct_pass"
            ]
            for x
            in records
        )

        execution_pass_all = all(
            x[
                "checks"
            ][
                "execution_count_pass"
            ]
            for x
            in records
        )

        output_pass_all = all(
            x[
                "checks"
            ][
                "forward_output_pass"
            ]
            for x
            in records
        )

        loss_pass_all = all(
            x[
                "checks"
            ][
                "loss_pass"
            ]
            for x
            in records
        )

        max_local_gradient_error = max(
            max(
                x[
                    "evidence"
                ][
                    "local_gradient_errors"
                ].values()
            )
            for x
            in records
        )

        max_dp_average_error = max(
            max(
                x[
                    "evidence"
                ][
                    "dp_average_gradient_errors"
                ].values()
            )
            for x
            in records
        )

        max_final_output_error = max(
            x[
                "evidence"
            ][
                "final_output_max_abs"
            ]
            for x
            in records
        )

        case_pass = all(
            [
                all(
                    x[
                        "local_pass"
                    ]
                    for x
                    in records
                ),

                all_ranks_pass,

                group_geometry_pass,

                local_gradient_pass_all,

                dp_average_pass_all,

                distinct_replica_pass_all,

                execution_pass_all,

                output_pass_all,

                loss_pass_all,
            ]
        )

        summary = {
            "schema_version":
                1,

            "project":
                "RLTrain",

            "milestone":
                "RLTrain-stage-8.4",

            "run_id":
                RUN_ID,

            "configuration": {
                "tp":
                    TP_SIZE,

                "pp":
                    PP_SIZE,

                "dp":
                    DP_SIZE,

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

            "unique_dp_groups":
                [
                    list(x)
                    for x
                    in unique_dp_groups
                ],

            "checks": {
                "group_geometry_pass":
                    group_geometry_pass,

                "local_gradient_dense_reference_pass":
                    local_gradient_pass_all,

                "dp_average_gradient_reference_pass":
                    dp_average_pass_all,

                "different_dp_data_pass":
                    distinct_replica_pass_all,

                "execution_counts_pass":
                    execution_pass_all,

                "forward_output_pass":
                    output_pass_all,

                "loss_pass":
                    loss_pass_all,
            },

            "maximum_errors": {
                "local_gradient_abs":
                    max_local_gradient_error,

                "dp_average_gradient_abs":
                    max_dp_average_error,

                "final_output_abs":
                    max_final_output_error,
            },

            "case_pass":
                case_pass,
        }

        summary_path = (
            RESULT_DIR
            / (
                "RLTRAIN_STAGE_8_4_TRLTRAIN_PRLTRAIN_DRLTRAIN_"
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
            "TP=2 PP=2 DP=2 WORLD_SIZE=8"
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
            "DP_GROUPS="
            + str(
                unique_dp_groups
            )
        )

        print(
            "GROUP_GEOMETRY_PASS="
            + str(
                group_geometry_pass
            )
        )

        print(
            "LOCAL_GRADIENT_DENSE_REFERENCE_PASS="
            + str(
                local_gradient_pass_all
            )
        )

        print(
            "DIFFERENT_DP_DATA_PASS="
            + str(
                distinct_replica_pass_all
            )
        )

        print(
            "DP_AVERAGE_GRADIENT_REFERENCE_PASS="
            + str(
                dp_average_pass_all
            )
        )

        print(
            "MAX_LOCAL_GRADIENT_ABS_ERROR="
            f"{max_local_gradient_error:.9e}"
        )

        print(
            "MAX_DP_AVERAGE_GRADIENT_ABS_ERROR="
            f"{max_dp_average_error:.9e}"
        )

        print(
            "FINAL_OUTPUT_MAX_ABS_ERROR="
            f"{max_final_output_error:.9e}"
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
                "RLTrain-stage-8.4 3D parallel "
                "correctness gate failed."
            )

    barrier()

    try:
        parallel_state.destroy_model_parallel()

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
