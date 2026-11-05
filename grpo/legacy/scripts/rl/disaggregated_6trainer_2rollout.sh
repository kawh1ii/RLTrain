#!/usr/bin/env bash

set -u

ROOT="${RLTRAIN_WORKSPACE_ROOT:-/path/to/ai-infra}"
PROJECT="$ROOT/RLTrain"

VERL_ENV="$ROOT/envs/project2-verl"

SCRIPT_DIR="$PROJECT/scripts"
RESULT_DIR="$PROJECT/results/stage_9_2"
ARTIFACT_DIR="$ROOT/artifacts"

OVERLAY_DIR="$ROOT/overlays/stage_9_dependencies"

MODEL_ROOT="$ROOT/caches/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots"

SOURCE_TRAIN="$PROJECT/data/stage_5/train.parquet"
SOURCE_VAL="$PROJECT/data/stage_5/test.parquet"

DATA_DIR="$PROJECT/data/stage_9_2"

mkdir -p "$RESULT_DIR"
mkdir -p "$ARTIFACT_DIR"
mkdir -p "$OVERLAY_DIR"
mkdir -p "$DATA_DIR"

RUN_ID="$(date +%Y%m%d_%H%M%S)"

TXT="$ARTIFACT_DIR/RLTRAIN_STAGE_9_2_DISAGGREGATED_6TRAIN_2ROLLOUT_${RUN_ID}.txt"

RAW_LOG="$RESULT_DIR/RLTRAIN_STAGE_9_2_RAW_${RUN_ID}.log"

GPU_CSV="$RESULT_DIR/RLTRAIN_STAGE_9_2_GPU_TIMELINE_${RUN_ID}.csv"

DECISION_JSON="$RESULT_DIR/RLTRAIN_STAGE_9_2_DECISION.json"

(
    set -euo pipefail

    echo "========================================================================================================================"
    echo "RLTrain — RLTrain-stage-9.2 REAL VERL DISAGGREGATED TRAINER / ROLLOUT BRING-UP"
    echo "========================================================================================================================"
    echo

    echo "TRAINER_GPU_COUNT = 6"
    echo "ROLLOUT_GPU_COUNT = 2"
    echo "TOTAL_GPU_COUNT = 8"
    echo

    echo "ALGORITHM = GRPO"
    echo "TRAINER_BACKEND = FSDP2"
    echo "ROLLOUT_BACKEND = VLLM"
    echo "HYBRID_ENGINE = FALSE"
    echo "CHECKPOINT_ENGINE = NCCL"
    echo

    echo "SCIENTIFIC_RLHF_EXECUTION = TRUE"
    echo "REAL_MODEL_EXECUTION = TRUE"
    echo "FORMAL_PERFORMANCE_CLAIM = FALSE"
    echo "FORMAL_SCALING_CLAIM = FALSE"

    echo
    echo "========================================================================================================================"
    echo "1. RLTrain-stage-9.1 GATE"
    echo "========================================================================================================================"

    "$VERL_ENV/bin/python" - <<'PY'
import json
from pathlib import Path

path = Path(
    "/path/to/ai-infra/"
    "RLTrain/results/"
    "stage_9_1/RLTRAIN_STAGE_9_1_DECISION.json"
)

if not path.exists():
    raise SystemExit(
        "RLTrain-stage-9.1 decision missing."
    )

data = json.loads(
    path.read_text(
        encoding="utf-8"
    )
)

expected = (
    "RLTRAIN_STAGE_9_1_EXACT_VERL_PLACEMENT_"
    "CONFIG_FREEZE_COMPLETE_"
    "READY_FOR_RLTRAIN_STAGE_9_2"
)

print(
    "RLTRAIN_STAGE_9_1_VERDICT = "
    + str(data.get("verdict"))
)

if data.get("verdict") != expected:
    raise SystemExit(
        "RLTrain-stage-9.1 gate not satisfied."
    )
PY

    echo
    echo "========================================================================================================================"
    echo "2. HOST / ENVIRONMENT GATE"
    echo "========================================================================================================================"

    GPU_COUNT="$(
        nvidia-smi \
            --query-gpu=index \
            --format=csv,noheader \
        | wc -l \
        | tr -d ' '
    )"

    echo "PHYSICAL_GPU_COUNT = $GPU_COUNT"

    if [[ "$GPU_COUNT" -ne 8 ]]; then
        echo "ERROR = expected exactly 8 GPUs"
        exit 10
    fi

    "$VERL_ENV/bin/python" - <<'PY'
import importlib.metadata as md
import torch
import ray

print("torch=" + torch.__version__)
print("torch_cuda=" + str(torch.version.cuda))
print("ray=" + ray.__version__)
print("verl=" + md.version("verl"))
print("vllm=" + md.version("vllm"))
print("numpy=" + md.version("numpy"))

assert torch.__version__ == "2.8.0+cu128"
assert torch.version.cuda == "12.8"
assert ray.__version__ == "2.57.0"
assert md.version("verl") == "0.8.0"
assert md.version("vllm") == "0.10.2"

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 8

print(
    "RLTRAIN_STAGE_9_2_ENVIRONMENT_GATE = PASS"
)
PY

    echo
    echo "========================================================================================================================"
    echo "3. ONE-STEP-OFF-POLICY ENTRYPOINT / DEPENDENCY PREFLIGHT"
    echo "========================================================================================================================"

    #
    # Do NOT mutate project2-verl.
    # TransferQueue, if absent, is placed in a standalone PYTHONPATH overlay.
    #

    export PYTHONPATH="$OVERLAY_DIR:${PYTHONPATH:-}"

    set +e

    "$VERL_ENV/bin/python" - <<'PY'
import importlib

modules = [
    "verl.experimental.one_step_off_policy.main_ppo",
    "verl.experimental.one_step_off_policy.ray_trainer",
]

for name in modules:
    try:
        importlib.import_module(name)

        print(
            name
            + " = IMPORT_PASS"
        )

    except Exception as exc:
        print(
            name
            + " = IMPORT_FAIL"
        )

        print(
            repr(exc)
        )

        raise
PY

    PREFLIGHT_STATUS=$?

    set -e

    if [[ "$PREFLIGHT_STATUS" -ne 0 ]]; then

        echo
        echo "ONE_STEP_OFF_PREFLIGHT = INITIAL_FAIL"
        echo "CHECKING_TRANSFER_QUEUE = TRUE"

        set +e

        "$VERL_ENV/bin/python" - <<'PY'
try:
    import transfer_queue
    print("TRANSFER_QUEUE_IMPORT = PASS")
except Exception as exc:
    print("TRANSFER_QUEUE_IMPORT = FAIL")
    print(repr(exc))
    raise
PY

        TQ_STATUS=$?

        set -e

        if [[ "$TQ_STATUS" -ne 0 ]]; then

            echo
            echo "TRANSFER_QUEUE_OVERLAY_INSTALL_REQUIRED = TRUE"

            #
            # Network selection.
            #

            if (
                echo > /dev/tcp/127.0.0.1/7890
            ) >/dev/null 2>&1; then

                export HTTP_PROXY="http://127.0.0.1:7890"
                export HTTPS_PROXY="http://127.0.0.1:7890"
                export http_proxy="$HTTP_PROXY"
                export https_proxy="$HTTPS_PROXY"

                echo "NETWORK_MODE = LOCAL_PROXY_7890"

            else

                unset HTTP_PROXY HTTPS_PROXY || true
                unset http_proxy https_proxy || true

                echo "NETWORK_MODE = DIRECT"
            fi

            "$VERL_ENV/bin/python" \
                -m pip install \
                --index-url https://pypi.org/simple \
                --no-deps \
                --target "$OVERLAY_DIR" \
                "TransferQueue==0.1.8"

        else

            echo "TRANSFER_QUEUE_OVERLAY_INSTALL_REQUIRED = FALSE"

        fi

        echo
        echo "RETRY_ONE_STEP_OFF_IMPORT = TRUE"

        "$VERL_ENV/bin/python" - <<'PY'
import importlib

importlib.import_module(
    "verl.experimental.one_step_off_policy.main_ppo"
)

importlib.import_module(
    "verl.experimental.one_step_off_policy.ray_trainer"
)

print(
    "ONE_STEP_OFF_POLICY_IMPORT = PASS"
)
PY

    else

        echo "ONE_STEP_OFF_POLICY_IMPORT = PASS"
        echo "TRANSFER_QUEUE_OVERLAY_INSTALL_REQUIRED = FALSE"

    fi

    echo
    echo "RLTRAIN_VERL_MODIFICATION = FALSE"

    echo
    echo "========================================================================================================================"
    echo "4. MODEL PATH"
    echo "========================================================================================================================"

    MODEL_PATH="$(
        find "$MODEL_ROOT" \
            -mindepth 1 \
            -maxdepth 1 \
            -type d \
        | head -n 1
    )"

    if [[ -z "$MODEL_PATH" ]]; then
        echo "ERROR = local Qwen2.5-0.5B-Instruct snapshot missing"
        exit 20
    fi

    echo "MODEL_PATH = $MODEL_PATH"
    echo "MODEL_DOWNLOAD = FALSE"

    if [[ ! -f "$MODEL_PATH/config.json" ]]; then
        echo "ERROR = model config.json missing"
        exit 21
    fi

    echo
    echo "========================================================================================================================"
    echo "5. MATCHED RLTrain-stage-9 DATASET"
    echo "========================================================================================================================"

    if [[ ! -f "$SOURCE_TRAIN" ]]; then
        echo "ERROR = source RLTrain-stage-5 train parquet missing"
        exit 30
    fi

    if [[ ! -f "$SOURCE_VAL" ]]; then
        echo "ERROR = source RLTrain-stage-5 val parquet missing"
        exit 31
    fi

    TRAIN_FILE="$DATA_DIR/train_32.parquet"
    VAL_FILE="$DATA_DIR/test.parquet"

    RLTRAIN_SOURCE_TRAIN="$SOURCE_TRAIN" \
    RLTRAIN_SOURCE_VAL="$SOURCE_VAL" \
    RLTRAIN_TARGET_TRAIN="$TRAIN_FILE" \
    RLTRAIN_TARGET_VAL="$VAL_FILE" \
    "$VERL_ENV/bin/python" - <<'PY'
import os
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


source_train = Path(
    os.environ["RLTRAIN_SOURCE_TRAIN"]
)

source_val = Path(
    os.environ["RLTRAIN_SOURCE_VAL"]
)

target_train = Path(
    os.environ["RLTRAIN_TARGET_TRAIN"]
)

target_val = Path(
    os.environ["RLTRAIN_TARGET_VAL"]
)


train = pq.read_table(
    source_train
)

val = pq.read_table(
    source_val
)


if train.num_rows <= 0:
    raise RuntimeError(
        "Source train dataset is empty."
    )


copies = []

while sum(
    table.num_rows
    for table in copies
) < 32:
    copies.append(train)


expanded = pa.concat_tables(
    copies
).slice(
    0,
    32,
)


pq.write_table(
    expanded,
    target_train,
)

pq.write_table(
    val,
    target_val,
)


manifest = {
    "source_train":
        str(source_train),

    "source_train_rows":
        train.num_rows,

    "target_train":
        str(target_train),

    "target_train_rows":
        expanded.num_rows,

    "source_val":
        str(source_val),

    "val_rows":
        val.num_rows,

    "schema":
        str(train.schema),

    "semantic_change":
        False,

    "operation":
        (
            "deterministic row repetition "
            "for systems bring-up only"
        ),
}


manifest_path = (
    target_train.parent
    / "manifest.json"
)

manifest_path.write_text(
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


print(
    "SOURCE_TRAIN_ROWS="
    + str(train.num_rows)
)

print(
    "TARGET_TRAIN_ROWS="
    + str(expanded.num_rows)
)

print(
    "VAL_ROWS="
    + str(val.num_rows)
)

print(
    "TRAIN_FILE="
    + str(target_train)
)

print(
    "VAL_FILE="
    + str(target_val)
)

print(
    "DATASET_REPLICATION_ONLY = TRUE"
)

print(
    "CONVERGENCE_CLAIM = FALSE"
)
PY

    echo
    echo "========================================================================================================================"
    echo "6. CLEAR STALE RAY"
    echo "========================================================================================================================"

    "$VERL_ENV/bin/ray" stop --force || true

    sleep 2

    echo
    echo "========================================================================================================================"
    echo "7. START GPU TELEMETRY"
    echo "========================================================================================================================"

    echo "timestamp,index,utilization_gpu_percent,memory_used_mib,memory_total_mib" \
        > "$GPU_CSV"

    (
        while true; do

            nvidia-smi \
                --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total \
                --format=csv,noheader,nounits \
            >> "$GPU_CSV" 2>/dev/null || true

            sleep 1

        done
    ) &

    MONITOR_PID=$!

    echo "GPU_MONITOR_PID = $MONITOR_PID"
    echo "GPU_TIMELINE = $GPU_CSV"

    cleanup_monitor() {
        kill "$MONITOR_PID" \
            >/dev/null 2>&1 || true

        wait "$MONITOR_PID" \
            >/dev/null 2>&1 || true
    }

    trap cleanup_monitor EXIT

    echo
    echo "========================================================================================================================"
    echo "8. REAL ONE-STEP-OFF GRPO — TRAINER 6 GPU / ROLLOUT 2 GPU"
    echo "========================================================================================================================"

    echo "TRAIN_BATCH_SIZE = 8"
    echo "ROLLOUT_N = 3"
    echo "GENERATED_RESPONSES_PER_BATCH = 24"
    echo "PPO_MINI_BATCH_SIZE = 4"
    echo "TOTAL_TRAINING_STEPS = 2"
    echo
    echo "TRAINER_N_GPUS_PER_NODE = 6"
    echo "ROLLOUT_N_GPUS_PER_NODE = 2"
    echo "ROLLOUT_TENSOR_PARALLEL_SIZE = 2"
    echo "ACTOR_FSDP_SIZE = 2"

    EXP_NAME="stage_9_2_disaggregated_6_2_${RUN_ID}"

    CHECKPOINT_DIR="$RESULT_DIR/checkpoints_${RUN_ID}"

    set +e

    "$VERL_ENV/bin/python" \
        -m verl.experimental.one_step_off_policy.main_ppo \
        data.train_files="$TRAIN_FILE" \
        data.val_files="$VAL_FILE" \
        data.train_batch_size=8 \
        data.max_prompt_length=128 \
        data.max_response_length=64 \
        data.filter_overlong_prompts=True \
        data.truncation=error \
        data.shuffle=False \
        algorithm.adv_estimator=grpo \
        algorithm.use_kl_in_reward=False \
        actor_rollout_ref.hybrid_engine=False \
        actor_rollout_ref.model.path="$MODEL_PATH" \
        actor_rollout_ref.model.use_remove_padding=False \
        actor_rollout_ref.model.enable_gradient_checkpointing=False \
        actor_rollout_ref.model.use_fused_kernels=False \
        actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
        actor_rollout_ref.actor.fsdp_config.fsdp_size=2 \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=4 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.ppo_epochs=1 \
        actor_rollout_ref.actor.use_dynamic_bsz=False \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.use_torch_compile=False \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.n=3 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.data_parallel_size=1 \
        actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.30 \
        actor_rollout_ref.rollout.enforce_eager=True \
        actor_rollout_ref.rollout.free_cache_engine=True \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.rollout.enable_prefix_caching=False \
        actor_rollout_ref.rollout.max_model_len=256 \
        actor_rollout_ref.rollout.max_num_batched_tokens=256 \
        actor_rollout_ref.rollout.max_num_seqs=32 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
        actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
        actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=256 \
        reward.reward_manager.name=naive \
        reward_model.reward_manager=naive \
        critic.strategy=fsdp2 \
        trainer.logger='["console"]' \
        trainer.project_name=p2_rltrain \
        trainer.experiment_name="$EXP_NAME" \
        trainer.val_before_train=False \
        trainer.balance_batch=False \
        trainer.critic_warmup=0 \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=6 \
        trainer.save_freq=-1 \
        trainer.test_freq=-1 \
        trainer.total_epochs=2 \
        trainer.total_training_steps=2 \
        trainer.resume_mode=disable \
        trainer.default_hdfs_dir=null \
        trainer.default_local_dir="$CHECKPOINT_DIR" \
        rollout.nnodes=1 \
        rollout.n_gpus_per_node=2 \
        2>&1 | tee "$RAW_LOG"

    TRAIN_STATUS=${PIPESTATUS[0]}

    set -e

    cleanup_monitor

    trap - EXIT

    echo
    echo "TRAINER_EXIT_CODE = $TRAIN_STATUS"

    echo
    echo "========================================================================================================================"
    echo "9. SCIENTIFIC EXECUTION ADJUDICATION"
    echo "========================================================================================================================"

    RLTRAIN_RAW_LOG="$RAW_LOG" \
    RLTRAIN_DECISION_JSON="$DECISION_JSON" \
    RLTRAIN_TRAIN_STATUS="$TRAIN_STATUS" \
    RLTRAIN_GPU_CSV="$GPU_CSV" \
    "$VERL_ENV/bin/python" - <<'PY'
import os
import re
import csv
import json
from pathlib import Path
from datetime import datetime


raw_path = Path(
    os.environ["RLTRAIN_RAW_LOG"]
)

decision_path = Path(
    os.environ["RLTRAIN_DECISION_JSON"]
)

gpu_csv_path = Path(
    os.environ["RLTRAIN_GPU_CSV"]
)

exit_code = int(
    os.environ["RLTRAIN_TRAIN_STATUS"]
)


text = raw_path.read_text(
    encoding="utf-8",
    errors="replace",
)


def contains_any(patterns):
    lower = text.lower()

    return any(
        pattern.lower()
        in lower

        for pattern
        in patterns
    )


progress_100 = bool(
    re.search(
        r"Training Progress:.*100%",
        text,
        flags=re.IGNORECASE,
    )
)


global_step_signal = (
    contains_any(
        [
            "global_step",
            "global steps",
            "step:2",
            "step=2",
        ]
    )
    or
    progress_100
)


rollout_signal = contains_any(
    [
        "generate_sequences",
        "rollout/",
        "gen/",
        "response_length",
    ]
)


actor_update_signal = contains_any(
    [
        "update_actor",
        "actor/",
        "pg_loss",
        "policy_loss",
    ]
)


reward_signal = contains_any(
    [
        "reward/",
        "reward_score",
        "rewards/",
        "reward",
    ]
)


wait_prev_gen_signal = contains_any(
    [
        "wait_prev_gen",
    ]
)


weight_sync_signal = contains_any(
    [
        "update_weights",
        "checkpoint_engine",
        "checkpoint engine",
    ]
)


vllm_signal = contains_any(
    [
        "vllm",
        "llmserver",
        "llm server",
    ]
)


one_step_off_signal = contains_any(
    [
        "one_step_off",
        "one-step",
        "separaterayppotrainer",
    ]
)


traceback_seen = (
    "Traceback (most recent call last)"
    in text
)


trainer_exit_pass = (
    exit_code == 0
)


scientific_execution_pass = all(
    [
        trainer_exit_pass,
        progress_100,
        global_step_signal,
        rollout_signal,
        actor_update_signal,
        reward_signal,
        wait_prev_gen_signal,
        weight_sync_signal,
        vllm_signal,
    ]
)


#
# GPU telemetry summary.
#

gpu_stats = {}


if gpu_csv_path.exists():

    with gpu_csv_path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:

        reader = csv.reader(f)

        header = next(
            reader,
            None,
        )

        for row in reader:

            if len(row) < 5:
                continue

            try:
                index = int(
                    row[1].strip()
                )

                util = float(
                    row[2].strip()
                )

                mem = float(
                    row[3].strip()
                )

            except Exception:
                continue

            item = gpu_stats.setdefault(
                index,
                {
                    "samples":
                        0,

                    "util_sum":
                        0.0,

                    "util_max":
                        0.0,

                    "mem_max_mib":
                        0.0,
                },
            )

            item[
                "samples"
            ] += 1

            item[
                "util_sum"
            ] += util

            item[
                "util_max"
            ] = max(
                item[
                    "util_max"
                ],
                util,
            )

            item[
                "mem_max_mib"
            ] = max(
                item[
                    "mem_max_mib"
                ],
                mem,
            )


for index, item in gpu_stats.items():

    if item[
        "samples"
    ] > 0:

        item[
            "util_mean"
        ] = (
            item[
                "util_sum"
            ]
            /
            item[
                "samples"
            ]
        )

    else:

        item[
            "util_mean"
        ] = 0.0

    del item[
        "util_sum"
    ]


if scientific_execution_pass:

    verdict = (
        "RLTRAIN_STAGE_9_2_REAL_VERL_"
        "ONE_STEP_OFF_POLICY_"
        "DISAGGREGATED_6TRAIN_2ROLLOUT_"
        "BRINGUP_COMPLETE_"
        "READY_FOR_RLTRAIN_STAGE_9_3"
    )

else:

    verdict = (
        "RLTRAIN_STAGE_9_2_INCOMPLETE_"
        "REVIEW_REQUIRED"
    )


decision = {
    "schema_version":
        1,

    "project":
        "RLTrain",

    "milestone":
        "RLTrain-stage-9.2",

    "timestamp":
        datetime.now()
        .astimezone()
        .isoformat(),

    "configuration": {
        "physical_gpus":
            8,

        "trainer_gpus":
            6,

        "rollout_gpus":
            2,

        "hybrid_engine":
            False,

        "algorithm":
            "grpo",

        "trainer_backend":
            "fsdp2",

        "rollout_backend":
            "vllm",

        "rollout_tensor_parallel_size":
            2,

        "train_batch_size":
            8,

        "rollout_n":
            3,

        "total_training_steps":
            2,
    },

    "checks": {
        "trainer_exit_pass":
            trainer_exit_pass,

        "training_progress_100":
            progress_100,

        "global_step_signal":
            global_step_signal,

        "rollout_signal":
            rollout_signal,

        "actor_update_signal":
            actor_update_signal,

        "reward_signal":
            reward_signal,

        "wait_prev_gen_signal":
            wait_prev_gen_signal,

        "weight_sync_signal":
            weight_sync_signal,

        "vllm_signal":
            vllm_signal,

        "one_step_off_signal":
            one_step_off_signal,

        "traceback_seen":
            traceback_seen,

        "scientific_execution_pass":
            scientific_execution_pass,
    },

    "gpu_telemetry":
        gpu_stats,

    "scope": {
        "real_rlhf_execution":
            True,

        "real_disaggregated_trainer_rollout":
            True,

        "real_nccl_weight_sync":
            weight_sync_signal,

        "formal_performance_claim":
            False,

        "formal_scaling_claim":
            False,

        "formal_convergence_claim":
            False,
    },

    "raw_log":
        str(raw_path),

    "gpu_timeline":
        str(gpu_csv_path),

    "verdict":
        verdict,

    "next_milestone": (
        (
            "RLTrain-stage-9.3 matched placement "
            "allocation benchmark"
        )
        if scientific_execution_pass
        else None
    ),
}


decision_path.write_text(
    json.dumps(
        decision,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


print(
    "TRAINER_EXIT_PASS = "
    + str(
        trainer_exit_pass
    )
)

print(
    "TRAINING_PROGRESS_100 = "
    + str(
        progress_100
    )
)

print(
    "GLOBAL_STEP_SIGNAL = "
    + str(
        global_step_signal
    )
)

print(
    "ROLLOUT_SIGNAL = "
    + str(
        rollout_signal
    )
)

print(
    "ACTOR_UPDATE_SIGNAL = "
    + str(
        actor_update_signal
    )
)

print(
    "REWARD_SIGNAL = "
    + str(
        reward_signal
    )
)

print(
    "WAIT_PREV_GEN_SIGNAL = "
    + str(
        wait_prev_gen_signal
    )
)

print(
    "WEIGHT_SYNC_SIGNAL = "
    + str(
        weight_sync_signal
    )
)

print(
    "VLLM_SIGNAL = "
    + str(
        vllm_signal
    )
)

print(
    "TRACEBACK_SEEN = "
    + str(
        traceback_seen
    )
)

print(
    "SCIENTIFIC_EXECUTION_PASS = "
    + str(
        scientific_execution_pass
    )
)


print()

print(
    "GPU_TELEMETRY_SUMMARY"
)


for index in sorted(
    gpu_stats
):

    item = gpu_stats[
        index
    ]

    print(
        f"GPU_{index}_"
        f"UTIL_MEAN_PERCENT="
        f"{item['util_mean']:.3f}"
    )

    print(
        f"GPU_{index}_"
        f"UTIL_MAX_PERCENT="
        f"{item['util_max']:.3f}"
    )

    print(
        f"GPU_{index}_"
        f"MEMORY_MAX_MIB="
        f"{item['mem_max_mib']:.1f}"
    )


print()

print(
    "REAL_RLHF_EXECUTION = TRUE"
)

print(
    "REAL_DISAGGREGATED_TRAINER_ROLLOUT = TRUE"
)

print(
    "FORMAL_PERFORMANCE_CLAIM = FALSE"
)

print(
    "FORMAL_SCALING_CLAIM = FALSE"
)

print(
    "FORMAL_CONVERGENCE_CLAIM = FALSE"
)

print(
    "FINAL_VERDICT = "
    + verdict
)

print(
    "DECISION_JSON = "
    + str(
        decision_path
    )
)


if not scientific_execution_pass:
    raise SystemExit(
        "RLTrain-stage-9.2 gate failed."
    )
PY

    echo
    echo "========================================================================================================================"
    echo "10. POST-RUN ENVIRONMENT FREEZE"
    echo "========================================================================================================================"

    "$VERL_ENV/bin/python" - <<'PY'
import importlib.metadata as md
import torch
import ray

assert torch.__version__ == "2.8.0+cu128"
assert torch.version.cuda == "12.8"
assert ray.__version__ == "2.57.0"
assert md.version("verl") == "0.8.0"
assert md.version("vllm") == "0.10.2"

print(
    "RLTRAIN_VERL_POST_M9_2_FROZEN = TRUE"
)
PY

    echo
    echo "========================================================================================================================"
    echo "11. FINAL GPU STATE"
    echo "========================================================================================================================"

    "$VERL_ENV/bin/ray" stop --force || true

    sleep 2

    nvidia-smi \
        --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader

    echo
    echo "========================================================================================================================"
    echo "END OF RLTrain-stage-9.2"
    echo "========================================================================================================================"

) 2>&1 | tee "$TXT"

STATUS=${PIPESTATUS[0]}

printf '%s\n' "$TXT"

exit "$STATUS"
