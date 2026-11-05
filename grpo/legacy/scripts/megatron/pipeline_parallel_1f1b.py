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
from megatron.core.pipeline_parallel.schedules import get_pp_rank_microbatches


ROOT = Path(os.environ.get("RLTRAIN_WORKSPACE_ROOT", "/path/to/ai-infra"))
PROJECT = ROOT / "RLTrain"

RESULT_DIR = (
    PROJECT
    / "results"
    / "stage_8_2"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RUN_ID = os.environ["RLTRAIN_STAGE_8_2_RUN_ID"]

PP_SIZE = int(
    os.environ["RLTRAIN_STAGE_8_2_PP"]
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


if WORLD_SIZE != PP_SIZE:
    raise RuntimeError(
        "RLTrain-stage-8.2 requires "
        "WORLD_SIZE == PP_SIZE."
    )


SEQ_LEN = 4
MICRO_BATCH_SIZE = 2
HIDDEN_SIZE = 64

NUM_MICROBATCHES = 16

DTYPE = torch.float32

DATA_SEED = 2026082601
WEIGHT_SEED_BASE = 2026082700

ABS_TOL = 2e-4
REL_TOL = 2e-4


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
    x = torch.tensor(
        [1 if value else 0],
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

    weight = (
        torch.randn(
            HIDDEN_SIZE,
            HIDDEN_SIZE,
            generator=generator,
            dtype=DTYPE,
        )
        * 0.025
    ).to(device)

    bias = (
        torch.randn(
            HIDDEN_SIZE,
            generator=generator,
            dtype=DTYPE,
        )
        * 0.01
    ).to(device)

    return weight, bias


def make_microbatches(device):
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


class PipelineStageModel(nn.Module):
    def __init__(
        self,
        stage_id,
        config,
        device,
    ):
        super().__init__()

        self.stage_id = stage_id

        self.config = config

        self.model_type = (
            ModelType.encoder_or_decoder
        )

        self.linear = nn.Linear(
            HIDDEN_SIZE,
            HIDDEN_SIZE,
            bias=True,
            device=device,
            dtype=DTYPE,
        )

        weight, bias = (
            make_stage_parameters(
                stage_id,
                device,
            )
        )

        with torch.no_grad():
            self.linear.weight.copy_(
                weight
            )

            self.linear.bias.copy_(
                bias
            )

        self.input_tensor = None

        self.forward_count = 0

        self.weight_grad_hook_count = 0

        self.bias_grad_hook_count = 0

        self.linear.weight.register_hook(
            self._weight_grad_hook
        )

        self.linear.bias.register_hook(
            self._bias_grad_hook
        )

    def _weight_grad_hook(
        self,
        grad,
    ):
        self.weight_grad_hook_count += 1

        return grad

    def _bias_grad_hook(
        self,
        grad,
    ):
        self.bias_grad_hook_count += 1

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
                if len(input_tensor) > 0
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
                f"Stage {self.stage_id} "
                "received no input tensor."
            )

        y = self.linear(x)

        y = torch.tanh(y)

        return y


def build_serial_reference(
    microbatches,
    device,
):
    weights = []

    biases = []

    for stage_id in range(
        PP_SIZE
    ):
        weight, bias = (
            make_stage_parameters(
                stage_id,
                device,
            )
        )

        weight = (
            weight
            .clone()
            .detach()
            .requires_grad_(True)
        )

        bias = (
            bias
            .clone()
            .detach()
            .requires_grad_(True)
        )

        weights.append(
            weight
        )

        biases.append(
            bias
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

        for stage_id in range(
            PP_SIZE
        ):
            y = F.linear(
                y,
                weights[stage_id],
                biases[stage_id],
            )

            y = torch.tanh(y)

        output_snapshot = (
            y.detach().clone()
        )

        loss = F.mse_loss(
            y,
            item["target"],
            reduction="mean",
        )

        outputs.append(
            output_snapshot
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
        "weights":
            weights,

        "biases":
            biases,

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
        tensor_model_parallel_size=1,

        pipeline_model_parallel_size=
            PP_SIZE,

        context_parallel_size=1,

        expert_model_parallel_size=1,

        create_gloo_process_groups=False,
    )

    barrier()

    pp_rank = (
        parallel_state
        .get_pipeline_model_parallel_rank()
    )

    pp_world = (
        parallel_state
        .get_pipeline_model_parallel_world_size()
    )

    if pp_world != PP_SIZE:
        raise RuntimeError(
            "Megatron PP world size mismatch."
        )

    if pp_rank != RANK:
        raise RuntimeError(
            "Expected PP rank to equal "
            "global rank in pure PP case."
        )

    config = ModelParallelConfig(
        tensor_model_parallel_size=1,

        pipeline_model_parallel_size=
            PP_SIZE,

        sequence_parallel=False,

        pipeline_dtype=DTYPE,

        overlap_p2p_comm=False,

        batch_p2p_comm=True,

        deallocate_pipeline_outputs=False,

        params_dtype=DTYPE,
    )

    #
    # schedules.py uses this field
    # to construct activation tensor shape.
    #

    config.hidden_size = (
        HIDDEN_SIZE
    )

    model = PipelineStageModel(
        pp_rank,
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

    observed_microbatch_ids = []

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
            data_input = item["x"]
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

            target = item[
                "target"
            ]

            def loss_func(
                output_tensor
            ):
                observed_microbatch_ids.append(
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
                            .detach().clone(),

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

    forward_backward_func = (
        get_forward_backward_func()
    )

    schedule_name = (
        forward_backward_func
        .__name__
    )

    expected_schedule_name = (
        "forward_backward_pipelining_"
        "without_interleaving"
    )

    schedule_selector_pass = (
        schedule_name
        ==
        expected_schedule_name
    )

    (
        total_microbatches,
        all_warmup,
        warmup_microbatches,
        remaining_microbatches,
    ) = get_pp_rank_microbatches(
        NUM_MICROBATCHES,
        1,
        config
        .microbatch_group_size_per_vp_stage,
        False,
    )

    expected_warmup = (
        PP_SIZE
        - pp_rank
        - 1
    )

    expected_warmup = min(
        expected_warmup,
        NUM_MICROBATCHES,
    )

    schedule_geometry_pass = all(
        [
            total_microbatches
                == NUM_MICROBATCHES,

            warmup_microbatches
                == expected_warmup,

            remaining_microbatches
                ==
                NUM_MICROBATCHES
                - expected_warmup,

            remaining_microbatches
                > 0,

            all_warmup is False,
        ]
    )

    torch.cuda.synchronize(
        device
    )

    losses_reduced = (
        forward_backward_func(
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

            collect_non_loss_data=
                False,
        )
    )

    torch.cuda.synchronize(
        device
    )

    #
    # Build a complete serial dense reference
    # independently on every rank.
    #

    reference = (
        build_serial_reference(
            microbatches,
            device,
        )
    )

    local_reference_weight_grad = (
        reference[
            "weights"
        ][
            pp_rank
        ].grad
    )

    local_reference_bias_grad = (
        reference[
            "biases"
        ][
            pp_rank
        ].grad
    )

    weight_grad_pass = (
        model.linear.weight.grad
        is not None

        and

        close_pass(
            model
            .linear
            .weight
            .grad,

            local_reference_weight_grad,
        )
    )

    bias_grad_pass = (
        model.linear.bias.grad
        is not None

        and

        close_pass(
            model
            .linear
            .bias
            .grad,

            local_reference_bias_grad,
        )
    )

    weight_grad_max_abs_error = (
        max_abs_error(
            model
            .linear
            .weight
            .grad,

            local_reference_weight_grad,
        )
        if (
            model.linear.weight.grad
            is not None
        )
        else float("inf")
    )

    bias_grad_max_abs_error = (
        max_abs_error(
            model
            .linear
            .bias
            .grad,

            local_reference_bias_grad,
        )
        if (
            model.linear.bias.grad
            is not None
        )
        else float("inf")
    )

    execution_count_pass = (
        model.forward_count
        ==
        NUM_MICROBATCHES

        and

        model.weight_grad_hook_count
        ==
        NUM_MICROBATCHES

        and

        model.bias_grad_hook_count
        ==
        NUM_MICROBATCHES
    )

    if (
        pp_rank
        ==
        PP_SIZE - 1
    ):
        loss_count_pass = (
            len(
                losses_reduced
            )
            ==
            NUM_MICROBATCHES
        )

        observed_loss_values = []

        observed_loss_ids = []

        for item in losses_reduced:
            observed_loss_values.append(
                float(
                    item[
                        "mse"
                    ]
                    .detach()
                    .item()
                )
            )

            observed_loss_ids.append(
                int(
                    item[
                        "microbatch_id"
                    ]
                    .detach()
                    .item()
                )
            )

        microbatch_order_pass = (
            observed_loss_ids
            ==
            list(
                range(
                    NUM_MICROBATCHES
                )
            )

            and

            observed_microbatch_ids
            ==
            list(
                range(
                    NUM_MICROBATCHES
                )
            )
        )

        loss_value_pass = (
            len(
                observed_loss_values
            )
            ==
            len(
                reference[
                    "losses"
                ]
            )

            and

            all(
                abs(
                    a - b
                )
                <=
                (
                    ABS_TOL
                    +
                    REL_TOL
                    *
                    abs(b)
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

        output_count_pass = (
            len(
                observed_outputs
            )
            ==
            NUM_MICROBATCHES
        )

        if output_count_pass:
            output_errors = [
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
            ]

            output_equivalence_pass = all(
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

            output_max_abs_error = max(
                output_errors
            )
        else:
            output_equivalence_pass = False
            output_max_abs_error = (
                float("inf")
            )

    else:
        loss_count_pass = (
            len(
                losses_reduced
            )
            == 0
        )

        microbatch_order_pass = True

        loss_value_pass = True

        output_count_pass = (
            len(
                observed_outputs
            )
            == 0
        )

        output_equivalence_pass = True

        output_max_abs_error = 0.0

        observed_loss_values = []

        observed_loss_ids = []

    local_pass = all(
        [
            schedule_selector_pass,

            schedule_geometry_pass,

            execution_count_pass,

            weight_grad_pass,

            bias_grad_pass,

            loss_count_pass,

            microbatch_order_pass,

            loss_value_pass,

            output_count_pass,

            output_equivalence_pass,
        ]
    )

    all_ranks_pass = (
        global_and(
            local_pass,
            device,
        )
    )

    result = {
        "schema_version":
            1,

        "project":
            "RLTrain",

        "milestone":
            "RLTrain-stage-8.2",

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

        "pp_rank":
            pp_rank,

        "pp_size":
            PP_SIZE,

        "configuration": {
            "seq_len":
                SEQ_LEN,

            "micro_batch_size":
                MICRO_BATCH_SIZE,

            "num_microbatches":
                NUM_MICROBATCHES,

            "hidden_size":
                HIDDEN_SIZE,

            "dtype":
                str(DTYPE),

            "overlap_p2p_comm":
                False,

            "batch_p2p_comm":
                True,

            "deallocate_pipeline_outputs":
                False,

            "abs_tolerance":
                ABS_TOL,

            "rel_tolerance":
                REL_TOL,
        },

        "schedule": {
            "observed_name":
                schedule_name,

            "expected_name":
                expected_schedule_name,

            "selector_pass":
                schedule_selector_pass,

            "total_microbatches":
                total_microbatches,

            "warmup_microbatches":
                warmup_microbatches,

            "expected_warmup_microbatches":
                expected_warmup,

            "remaining_microbatches":
                remaining_microbatches,

            "all_microbatches_in_warmup":
                all_warmup,

            "geometry_pass":
                schedule_geometry_pass,
        },

        "execution": {
            "forward_count":
                model.forward_count,

            "weight_grad_hook_count":
                model
                .weight_grad_hook_count,

            "bias_grad_hook_count":
                model
                .bias_grad_hook_count,

            "execution_count_pass":
                execution_count_pass,
        },

        "gradient_equivalence": {
            "weight_grad_pass":
                weight_grad_pass,

            "weight_grad_max_abs_error":
                weight_grad_max_abs_error,

            "bias_grad_pass":
                bias_grad_pass,

            "bias_grad_max_abs_error":
                bias_grad_max_abs_error,
        },

        "last_stage_equivalence": {
            "is_last_stage":
                (
                    pp_rank
                    ==
                    PP_SIZE - 1
                ),

            "loss_count_pass":
                loss_count_pass,

            "microbatch_order_pass":
                microbatch_order_pass,

            "loss_value_pass":
                loss_value_pass,

            "output_count_pass":
                output_count_pass,

            "output_equivalence_pass":
                output_equivalence_pass,

            "output_max_abs_error":
                output_max_abs_error,

            "observed_loss_values":
                observed_loss_values,

            "observed_loss_ids":
                observed_loss_ids,

            "serial_mean_loss":
                reference[
                    "mean_loss"
                ],
        },

        "local_pass":
            local_pass,

        "all_ranks_pass":
            all_ranks_pass,
    }

    rank_path = (
        RESULT_DIR
        / (
            f"RLTRAIN_STAGE_8_2_PP{PP_SIZE}_"
            f"RANK{RANK}_"
            f"{RUN_ID}.json"
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

    print(
        f"RANK={RANK} "
        f"PP_RANK={pp_rank} "
        f"WARMUP={warmup_microbatches} "
        f"STEADY_REMAINING={remaining_microbatches} "
        f"FORWARD_COUNT={model.forward_count} "
        f"BACKWARD_COUNT={model.weight_grad_hook_count} "
        f"WEIGHT_GRAD_PASS={weight_grad_pass} "
        f"BIAS_GRAD_PASS={bias_grad_pass} "
        f"LOCAL_PASS={local_pass}",
        flush=True,
    )

    barrier()

    if RANK == 0:
        records = []

        for rank in range(
            PP_SIZE
        ):
            path = (
                RESULT_DIR
                / (
                    f"RLTRAIN_STAGE_8_2_PP{PP_SIZE}_"
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

        records = sorted(
            records,
            key=lambda x:
                x["pp_rank"],
        )

        warmup_vector = [
            x[
                "schedule"
            ][
                "warmup_microbatches"
            ]
            for x
            in records
        ]

        expected_warmup_vector = [
            PP_SIZE
            - rank
            - 1
            for rank
            in range(
                PP_SIZE
            )
        ]

        warmup_vector_pass = (
            warmup_vector
            ==
            expected_warmup_vector
        )

        gradient_equivalence_pass = all(
            x[
                "gradient_equivalence"
            ][
                "weight_grad_pass"
            ]

            and

            x[
                "gradient_equivalence"
            ][
                "bias_grad_pass"
            ]

            for x
            in records
        )

        execution_counts_pass = all(
            x[
                "execution"
            ][
                "forward_count"
            ]
            ==
            NUM_MICROBATCHES

            and

            x[
                "execution"
            ][
                "weight_grad_hook_count"
            ]
            ==
            NUM_MICROBATCHES

            for x
            in records
        )

        schedule_selector_all_pass = all(
            x[
                "schedule"
            ][
                "selector_pass"
            ]
            for x
            in records
        )

        schedule_geometry_all_pass = all(
            x[
                "schedule"
            ][
                "geometry_pass"
            ]
            for x
            in records
        )

        last_stage_record = (
            records[-1]
        )

        forward_equivalence_pass = (
            last_stage_record[
                "last_stage_equivalence"
            ][
                "output_equivalence_pass"
            ]

            and

            last_stage_record[
                "last_stage_equivalence"
            ][
                "loss_value_pass"
            ]

            and

            last_stage_record[
                "last_stage_equivalence"
            ][
                "microbatch_order_pass"
            ]
        )

        max_weight_grad_error = max(
            x[
                "gradient_equivalence"
            ][
                "weight_grad_max_abs_error"
            ]
            for x
            in records
        )

        max_bias_grad_error = max(
            x[
                "gradient_equivalence"
            ][
                "bias_grad_max_abs_error"
            ]
            for x
            in records
        )

        final_output_error = (
            last_stage_record[
                "last_stage_equivalence"
            ][
                "output_max_abs_error"
            ]
        )

        #
        # Analytical schedule-slot bubble fraction
        # for ideal equal-time non-interleaved 1F1B:
        #
        # bubble = (P - 1) / (M + P - 1)
        #
        # This is NOT a measured performance claim.
        #

        theoretical_bubble_fraction = (
            (PP_SIZE - 1)
            /
            (
                NUM_MICROBATCHES
                + PP_SIZE
                - 1
            )
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

                warmup_vector_pass,

                gradient_equivalence_pass,

                execution_counts_pass,

                schedule_selector_all_pass,

                schedule_geometry_all_pass,

                forward_equivalence_pass,

                all_ranks_pass,
            ]
        )

        summary = {
            "schema_version":
                1,

            "project":
                "RLTrain",

            "milestone":
                "RLTrain-stage-8.2",

            "run_id":
                RUN_ID,

            "pp_size":
                PP_SIZE,

            "num_microbatches":
                NUM_MICROBATCHES,

            "warmup_vector":
                warmup_vector,

            "expected_warmup_vector":
                expected_warmup_vector,

            "warmup_vector_pass":
                warmup_vector_pass,

            "schedule_selector_all_pass":
                schedule_selector_all_pass,

            "schedule_geometry_all_pass":
                schedule_geometry_all_pass,

            "execution_counts_pass":
                execution_counts_pass,

            "gradient_equivalence_pass":
                gradient_equivalence_pass,

            "forward_equivalence_pass":
                forward_equivalence_pass,

            "maximum_errors": {
                "weight_gradient_abs":
                    max_weight_grad_error,

                "bias_gradient_abs":
                    max_bias_grad_error,

                "final_output_abs":
                    final_output_error,
            },

            "analytical_schedule_geometry": {
                "theoretical_ideal_equal_stage_"
                "bubble_fraction":
                    theoretical_bubble_fraction,

                "measured_performance_claim":
                    False,
            },

            "case_pass":
                case_pass,
        }

        summary_path = (
            RESULT_DIR
            / (
                f"RLTRAIN_STAGE_8_2_PP{PP_SIZE}_"
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
            f"PP_SIZE={PP_SIZE}"
        )

        print(
            "SCHEDULE="
            + expected_schedule_name
        )

        print(
            "WARMUP_VECTOR="
            + str(
                warmup_vector
            )
        )

        print(
            "WARMUP_VECTOR_PASS="
            + str(
                warmup_vector_pass
            )
        )

        print(
            "FORWARD_EQUIVALENCE_PASS="
            + str(
                forward_equivalence_pass
            )
        )

        print(
            "GRADIENT_EQUIVALENCE_PASS="
            + str(
                gradient_equivalence_pass
            )
        )

        print(
            "EXECUTION_COUNTS_PASS="
            + str(
                execution_counts_pass
            )
        )

        print(
            "MAX_WEIGHT_GRAD_ABS_ERROR="
            f"{max_weight_grad_error:.9e}"
        )

        print(
            "MAX_BIAS_GRAD_ABS_ERROR="
            f"{max_bias_grad_error:.9e}"
        )

        print(
            "FINAL_OUTPUT_MAX_ABS_ERROR="
            f"{final_output_error:.9e}"
        )

        print(
            "THEORETICAL_IDEAL_EQUAL_STAGE_"
            "BUBBLE_FRACTION="
            f"{theoretical_bubble_fraction:.9f}"
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
                "RLTrain-stage-8.2 pipeline "
                "correctness gate failed."
            )

    barrier()

    try:
        parallel_state.destroy_model_parallel()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
