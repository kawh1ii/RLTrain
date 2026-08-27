#!/usr/bin/env bash

set -u

ROOT="${RLTRAIN_WORKSPACE_ROOT:-/path/to/ai-infra}"
PROJECT="$ROOT/RLTrain"
ENV="$ROOT/envs/project2-verl"
VERL_SITE="$ENV/lib/python3.12/site-packages"

TQ_OVERLAY="$PROJECT/overlays/stage_9_transferqueue_0_1_7"

RESULT_DIR="$PROJECT/results/stage_9_4_1e"
DATA_DIR="$PROJECT/data/stage_9_4_1e"
ARTIFACT_DIR="$ROOT/artifacts"

MODEL_ROOT="$ROOT/caches/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots"

SOURCE_TRAIN="$PROJECT/data/stage_5/train.parquet"
SOURCE_VAL="$PROJECT/data/stage_5/test.parquet"

mkdir -p "$RESULT_DIR"
mkdir -p "$DATA_DIR"
mkdir -p "$ARTIFACT_DIR"

RUN_ID="$(date +%Y%m%d_%H%M%S)"

TXT="$ARTIFACT_DIR/RLTRAIN_STAGE_9_4_1E_COLOCATED_8GPU_BRINGUP_${RUN_ID}.txt"
RAW_LOG="$RESULT_DIR/RLTRAIN_STAGE_9_4_1E_RAW_${RUN_ID}.log"
GPU_CSV="$RESULT_DIR/RLTRAIN_STAGE_9_4_1E_GPU_${RUN_ID}.csv"
DECISION_JSON="$RESULT_DIR/RLTRAIN_STAGE_9_4_1E_DECISION_${RUN_ID}.json"

(
    set -euo pipefail

    echo "========================================================================================================================"
    echo "RLTrain — RLTrain-stage-9.4.1E REAL 8-GPU COLOCATED HYBRID-ENGINE BRING-UP"
    echo "========================================================================================================================"
    echo

    echo "ENTRYPOINT = verl.trainer.main_ppo_sync"
    echo "ALGORITHM = GRPO"
    echo "HYBRID_ENGINE = TRUE"
    echo "COLOCATED_GPU_COUNT = 8"
    echo "TRAINER_GPU_COUNT = 8"
    echo "SEPARATE_ROLLOUT_RESOURCE_POOL = FALSE"
    echo "ROLLOUT_BACKEND = VLLM"
    echo "ROLLOUT_TP = 2"
    echo "ROLLOUT_DP = 1"
    echo "ROLLOUT_TOPOLOGY = NATIVE_CONFIG_TRLTRAIN_DP1"
    echo "TRANSFERQUEUE = 0.1.7_ISOLATED_OVERLAY"
    echo "TRAINER_BACKEND = FSDP2"
    echo "ACTOR_FSDP_SIZE = 2"
    echo
    echo "M9_3_SHM_OVERLAY = FALSE"
    echo "M9_3_ZMQ_RANKFIX = FALSE"
    echo "M9_3_BUCKET384_OVERRIDE = FALSE"
    echo
    echo "SCIENTIFIC_RLHF_EXECUTION = TRUE"
    echo "REAL_MODEL_EXECUTION = TRUE"
    echo "FORMAL_PERFORMANCE_CLAIM = FALSE"
    echo "FORMAL_SCALING_CLAIM = FALSE"
    echo "FORMAL_CONVERGENCE_CLAIM = FALSE"

    echo
    echo "========================================================================================================================"
    echo "1. RLTrain-stage-9.3.13 GATE"
    echo "========================================================================================================================"

    "$ENV/bin/python" - <<'PY'
import json
from pathlib import Path

path = Path(
    "/path/to/ai-infra/"
    "RLTrain/results/"
    "stage_9_3_12/RLTRAIN_STAGE_9_3_13_DECISION.json"
)

if not path.exists():
    raise SystemExit(
        "RLTrain-stage-9.3.13 decision missing."
    )

data = json.loads(
    path.read_text(
        encoding="utf-8"
    )
)

expected = (
    "RLTRAIN_STAGE_9_3_13_"
    "THROUGHPUT_SEMANTICS_CORRECTED_"
    "6TRAIN2ROLLOUT_WINNER_CONFIRMED_"
    "READY_FOR_RLTRAIN_STAGE_9_4"
)

print(
    "RLTRAIN_STAGE_9_3_13_VERDICT = "
    + str(
        data.get("verdict")
    )
)

if data.get("verdict") != expected:
    raise SystemExit(
        "RLTrain-stage-9.3.13 gate failed."
    )

print(
    "RLTRAIN_STAGE_9_3_13_GATE = PASS"
)
PY

    echo
    echo "========================================================================================================================"
    echo "2. ENVIRONMENT / HOST GATE"
    echo "========================================================================================================================"

    if [[ ! -f "$TQ_OVERLAY/transfer_queue/__init__.py" ]]; then
        echo "TRANSFERQUEUE_OVERLAY_MISSING"
        exit 21
    fi

    export PYTHONPATH="$TQ_OVERLAY"

    echo "TRANSFERQUEUE_OVERLAY = $TQ_OVERLAY"

    PYTHONPATH="$TQ_OVERLAY" \
    "$ENV/bin/python" - <<'PYTQ'
import importlib
import importlib.metadata as md
import torch
import ray

tq = importlib.import_module(
    "transfer_queue"
)

version = md.version(
    "TransferQueue"
)

required = [
    "init",
    "close",
    "kv_list",
    "kv_batch_get",
    "kv_batch_put",
    "kv_clear",
    "async_kv_put",
    "async_kv_batch_put",
]

missing = [
    name
    for name in required
    if not hasattr(
        tq,
        name,
    )
]

print(
    "TRANSFERQUEUE_VERSION = "
    + version
)

print(
    "TRANSFERQUEUE_MODULE = "
    + str(
        tq.__file__
    )
)

print(
    "TRANSFERQUEUE_MISSING_API = "
    + str(
        missing
    )
)

print(
    "torch="
    + torch.__version__
)

print(
    "ray="
    + ray.__version__
)

print(
    "numpy="
    + md.version(
        "numpy"
    )
)

assert version == "0.1.7"
assert not missing
assert torch.__version__ == "2.8.0+cu128"
assert ray.__version__ == "2.57.0"
assert md.version("numpy") == "1.26.4"

print(
    "TRANSFERQUEUE_RUNTIME_GATE = PASS"
)
PYTQ
    unset RLTRAIN_FORCE_VERL_SHM_WEIGHT_TRANSFER || true
    unset RLTRAIN_SHM_OVERLAY_VERBOSE || true
    unset RLTRAIN_ROLLOUT_PHYSICAL_GPUS || true
    unset CUDA_VISIBLE_DEVICES || true
    unset RAY_ADDRESS || true

    "$ENV/bin/python" - <<'PY'
import importlib.metadata as md
import torch
import ray

print("torch=" + torch.__version__)
print("torch_cuda=" + str(torch.version.cuda))
print("ray=" + ray.__version__)
print("verl=" + md.version("verl"))
print("vllm=" + md.version("vllm"))
print("numpy=" + md.version("numpy"))
print("cuda_available=" + str(torch.cuda.is_available()))
print("gpu_count=" + str(torch.cuda.device_count()))

assert torch.__version__ == "2.8.0+cu128"
assert torch.version.cuda == "12.8"
assert ray.__version__ == "2.57.0"
assert md.version("verl") == "0.8.0"
assert md.version("vllm") == "0.10.2"
assert md.version("numpy") == "1.26.4"
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 8

print("RLTRAIN_STAGE_9_4_1E_ENVIRONMENT_GATE = PASS")
PY

    echo
    echo "========================================================================================================================"
    echo "3. STANDARD HYBRID-ENGINE CONFIG GATE"
    echo "========================================================================================================================"

    CFG="$RESULT_DIR/RLTRAIN_STAGE_9_4_1E_CONFIG_${RUN_ID}.txt"

    (
        cd "$VERL_SITE"

        "$ENV/bin/python" \
            -m verl.trainer.main_ppo_sync \
            --cfg job
    ) > "$CFG" 2>&1

    RLTRAIN_CFG="$CFG" \
    "$ENV/bin/python" - <<'PY'
import os
from pathlib import Path

path = Path(
    os.environ["RLTRAIN_CFG"]
)

text = path.read_text(
    encoding="utf-8",
    errors="replace",
)

assert "hybrid_engine: true" in text

print(
    "STANDARD_MAIN_PPO_HYBRID_ENGINE = TRUE"
)

print(
    "STANDARD_MAIN_PPO_CONFIG_GATE = PASS"
)
PY

    echo
    echo "========================================================================================================================"
    echo "4. MODEL / DATA"
    echo "========================================================================================================================"

    MODEL_PATH="$(
        find "$MODEL_ROOT" \
            -mindepth 1 \
            -maxdepth 1 \
            -type d \
        | head -n 1
    )"

    if [[ -z "$MODEL_PATH" ]]; then
        echo "MODEL_NOT_FOUND"
        exit 20
    fi

    echo "MODEL_PATH = $MODEL_PATH"
    echo "MODEL_DOWNLOAD = FALSE"

    TRAIN_FILE="$DATA_DIR/train_32.parquet"
    VAL_FILE="$DATA_DIR/test.parquet"

    RLTRAIN_SOURCE_TRAIN="$SOURCE_TRAIN" \
    RLTRAIN_SOURCE_VAL="$SOURCE_VAL" \
    RLTRAIN_TARGET_TRAIN="$TRAIN_FILE" \
    RLTRAIN_TARGET_VAL="$VAL_FILE" \
    "$ENV/bin/python" - <<'PY'
import os
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

copies = []
rows = 0

while rows < 32:
    copies.append(train)
    rows += train.num_rows

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

print(
    "SOURCE_TRAIN_ROWS = "
    + str(train.num_rows)
)

print(
    "TARGET_TRAIN_ROWS = "
    + str(expanded.num_rows)
)

print(
    "VAL_ROWS = "
    + str(val.num_rows)
)

print(
    "DATASET_REPLICATION_ONLY = TRUE"
)
PY

    echo
    echo "========================================================================================================================"
    echo "5. CLEAR STALE RAY"
    echo "========================================================================================================================"

    "$ENV/bin/ray" stop --force \
        >/dev/null 2>&1 \
        || true

    sleep 3

    echo "STALE_RAY_CLEANUP = COMPLETE"

    echo
    echo "========================================================================================================================"
    echo "6. GPU TELEMETRY"
    echo "========================================================================================================================"

    echo \
        "timestamp,index,utilization_gpu_percent,memory_used_mib,memory_total_mib" \
        > "$GPU_CSV"

    (
        while true; do

            nvidia-smi \
                --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total \
                --format=csv,noheader,nounits \
                >> "$GPU_CSV" \
                2>/dev/null \
                || true

            sleep 1

        done
    ) &

    MONITOR_PID=$!

    echo "GPU_MONITOR_PID = $MONITOR_PID"
    echo "GPU_TIMELINE = $GPU_CSV"

    echo
    echo "========================================================================================================================"
    echo "7. REAL STANDARD COLOCATED GRPO"
    echo "========================================================================================================================"

    export CUDA_HOME="/usr/local/cuda-12.8"
    export PATH="$CUDA_HOME/bin:$ENV/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

    export PYTHONUNBUFFERED=1
    export TOKENIZERS_PARALLELISM=false
    export TRANSFORMERS_OFFLINE=1
    export HF_HUB_OFFLINE=1
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    export NCCL_DEBUG=WARN

    EXP_NAME="stage_9_4_1e_colocated_${RUN_ID}"

    CHECKPOINT_DIR="$RESULT_DIR/checkpoints_${RUN_ID}"

    echo "TRAIN_BATCH_SIZE = 8"
    echo "ROLLOUT_N = 3"
    echo "ACTOR_PPO_MINI_BATCH_SIZE = 8"
    echo "EFFECTIVE_ACTOR_MINI_BATCH = 24"
    echo "ACTOR_DATA_PARALLEL_SIZE = 8"
    echo "MINI_BATCH_DIVISIBILITY = 24_MOD_8_EQ_0"
    echo "TOTAL_TRAINING_STEPS = 2"
    echo "TRAINER_N_GPUS_PER_NODE = 8"
    echo "ROLLOUT_TP = 2"

    echo
    echo "========================================================================================================================"
    echo "ACTOR MINI-BATCH DIVISIBILITY PREFLIGHT"
    echo "========================================================================================================================"

    "$ENV/bin/python" - <<'PYBATCH'
train_batch_size = 8
rollout_n = 3
ppo_mini_batch_size = 8
actor_data_parallel_size = 8

generated_samples_per_step = (
    train_batch_size
    * rollout_n
)

effective_mini_batch = (
    ppo_mini_batch_size
    * rollout_n
)

divisibility_pass = (
    effective_mini_batch
    %
    actor_data_parallel_size
    ==
    0
)

print(
    "TRAIN_BATCH_SIZE = "
    + str(train_batch_size)
)

print(
    "ROLLOUT_N = "
    + str(rollout_n)
)

print(
    "GENERATED_SAMPLES_PER_STEP = "
    + str(generated_samples_per_step)
)

print(
    "PPO_MINI_BATCH_SIZE = "
    + str(ppo_mini_batch_size)
)

print(
    "EFFECTIVE_ACTOR_MINI_BATCH = "
    + str(effective_mini_batch)
)

print(
    "ACTOR_DATA_PARALLEL_SIZE = "
    + str(actor_data_parallel_size)
)

print(
    "DIVISIBILITY_REMAINDER = "
    + str(
        effective_mini_batch
        %
        actor_data_parallel_size
    )
)

print(
    "ACTOR_MINI_BATCH_DIVISIBILITY_PASS = "
    + str(divisibility_pass)
)

assert generated_samples_per_step == 24
assert effective_mini_batch == 24
assert divisibility_pass
PYBATCH

    set +e

    (
        cd "$VERL_SITE"

        "$ENV/bin/python" \
            -m verl.trainer.main_ppo_sync \
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
            actor_rollout_ref.hybrid_engine=True \
            actor_rollout_ref.model.path="$MODEL_PATH" \
            actor_rollout_ref.model.use_remove_padding=False \
            actor_rollout_ref.model.enable_gradient_checkpointing=False \
            actor_rollout_ref.model.use_fused_kernels=False \
            actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
            actor_rollout_ref.actor.fsdp_config.fsdp_size=2 \
            actor_rollout_ref.actor.fsdp_config.param_offload=False \
            actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
            actor_rollout_ref.actor.optim.lr=1e-6 \
            actor_rollout_ref.actor.ppo_mini_batch_size=8 \
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
            trainer.n_gpus_per_node=8 \
            trainer.save_freq=-1 \
            trainer.test_freq=-1 \
            trainer.total_epochs=1 \
            trainer.total_training_steps=2 \
            trainer.resume_mode=disable \
            trainer.default_hdfs_dir=null \
            trainer.default_local_dir="$CHECKPOINT_DIR"

    ) 2>&1 | tee "$RAW_LOG"

    TRAINER_STATUS=${PIPESTATUS[0]}

    set -e

    kill "$MONITOR_PID" \
        >/dev/null 2>&1 \
        || true

    wait "$MONITOR_PID" \
        >/dev/null 2>&1 \
        || true

    "$ENV/bin/ray" stop --force \
        >/dev/null 2>&1 \
        || true

    sleep 3

    echo "TRAINER_EXIT_CODE = $TRAINER_STATUS"

    echo
    echo "========================================================================================================================"
    echo "8. SCIENTIFIC EXECUTION ADJUDICATION"
    echo "========================================================================================================================"

    RLTRAIN_RAW_LOG="$RAW_LOG" \
    RLTRAIN_DECISION_JSON="$DECISION_JSON" \
    RLTRAIN_TRAINER_STATUS="$TRAINER_STATUS" \
    "$ENV/bin/python" - <<'PY'
import json
import os
import re
from pathlib import Path
from datetime import datetime


raw_path = Path(
    os.environ["RLTRAIN_RAW_LOG"]
)

decision_path = Path(
    os.environ["RLTRAIN_DECISION_JSON"]
)

exit_code = int(
    os.environ["RLTRAIN_TRAINER_STATUS"]
)

text = raw_path.read_text(
    encoding="utf-8",
    errors="replace",
)

ansi = re.compile(
    r"\x1b\[[0-9;]*m"
)

text = ansi.sub(
    "",
    text,
)


exit_pass = (
    exit_code == 0
)

step1_seen = bool(
    re.search(
        r"step:1\s+-\s+training/global_step:1",
        text,
    )
)

step2_seen = bool(
    re.search(
        r"step:2\s+-\s+training/global_step:2",
        text,
    )
)

training_100 = bool(
    re.search(
        r"Training Progress:\s*100%",
        text,
    )
)

vllm_seen = (
    "vllm"
    in text.lower()
)

actor_update_seen = (
    "timing_s/update_actor:"
    in text
)

reward_seen = (
    "timing_s/reward:"
    in text
)

generation_seen = (
    (
        "timing_s/gen:"
        in text
    )
    or
    (
        "timing_s/generate"
        in text
    )
)


transferqueue_real_seen = (
    (
        "TransferQueue"
        in text
    )
    or
    (
        "transfer_queue"
        in text
    )
)

transferqueue_missing_seen = (
    "transfer_queue is not installed"
    in text
)

hybrid_true_seen = (
    (
        "'hybrid_engine': True"
        in text
    )
    or
    (
        "hybrid_engine=True"
        in text
    )
)


forbidden_disaggregated_signals = {
    "shm_fallback":
        (
            "Falling back to shared memory for weight transfer"
            in text
        ),

    "p2_rankfix":
        (
            "[RLTrain-stage-9-ZMQ-RANKFIX]"
            in text
        ),

    "one_step_task_runner":
        (
            "OneStepTaskRunner"
            in text
        ),
}


fatal_patterns = {
    "ray_task_error":
        "RayTaskError"
        in text,

    "assertion_error":
        "AssertionError"
        in text,

    "cuda_oom":
        "CUDA out of memory"
        in text,

    "duplicate_gpu":
        "Duplicate GPU detected"
        in text,

    "peer_access":
        (
            "peer access is not supported"
            in text.lower()
        ),

    "nccl_fatal":
        (
            "ncclInvalidUsage"
            in text
            or
            "NCCL error"
            in text
        ),
}


fatal_seen = any(
    fatal_patterns.values()
)


traceback_count = text.count(
    "Traceback (most recent call last)"
)


usage_reporting_traceback = (
    traceback_count > 0
    and
    "vllm/usage/usage_lib.py"
    in text
    and
    "cpuinfo/cpuinfo.py"
    in text
    and
    "JSONDecodeError"
    in text
)


if traceback_count == 0:

    traceback_classification = (
        "NONE"
    )

elif (
    usage_reporting_traceback
    and
    not fatal_seen
):

    traceback_classification = (
        "BACKGROUND_USAGE_REPORTING_NONBLOCKING"
    )

else:

    traceback_classification = (
        "REVIEW_REQUIRED"
    )


scientific_execution_pass = all(
    [
        exit_pass,
        step1_seen,
        step2_seen,
        training_100,
        vllm_seen,
        actor_update_seen,
        reward_seen,
        generation_seen,
        hybrid_true_seen,
        transferqueue_real_seen,
        not transferqueue_missing_seen,
        not any(
            forbidden_disaggregated_signals.values()
        ),
        not fatal_seen,
        traceback_classification
        in {
            "NONE",
            "BACKGROUND_USAGE_REPORTING_NONBLOCKING",
        },
    ]
)


if scientific_execution_pass:

    verdict = (
        "RLTRAIN_STAGE_9_4_1E_REAL_8GPU_"
        "COLOCATED_HYBRID_ENGINE_"
        "TWO_STEP_BRINGUP_COMPLETE_"
        "READY_FOR_RLTRAIN_STAGE_9_4_2"
    )

else:

    verdict = (
        "RLTRAIN_STAGE_9_4_1E_INCOMPLETE_"
        "REVIEW_REQUIRED"
    )


decision = {
    "schema_version":
        1,

    "project":
        "RLTrain",

    "milestone":
        "RLTrain-stage-9.4.1E",

    "timestamp":
        datetime.now()
        .astimezone()
        .isoformat(),

    "configuration": {
        "entrypoint":
            "verl.trainer.main_ppo_sync",

        "hybrid_engine":
            True,

        "trainer_gpus":
            8,

        "separate_rollout_pool":
            False,

        "rollout_tp":
            2,

        "rollout_dp":
            4,

        "algorithm":
            "GRPO",

        "trainer_backend":
            "FSDP2",

        "rollout_backend":
            "vLLM",

        "total_training_steps":
            2,
    },

    "checks": {
        "trainer_exit_pass":
            exit_pass,

        "step1_seen":
            step1_seen,

        "step2_seen":
            step2_seen,

        "training_100":
            training_100,

        "vllm_seen":
            vllm_seen,

        "actor_update_seen":
            actor_update_seen,

        "reward_seen":
            reward_seen,

        "generation_seen":
            generation_seen,

        "hybrid_engine_true_seen":
            hybrid_true_seen,

        "transferqueue_real_seen":
            transferqueue_real_seen,

        "transferqueue_missing_absent":
            not transferqueue_missing_seen,

        "forbidden_disaggregated_signals":
            forbidden_disaggregated_signals,

        "fatal_patterns":
            fatal_patterns,

        "traceback_count":
            traceback_count,

        "traceback_classification":
            traceback_classification,

        "scientific_execution_pass":
            scientific_execution_pass,
    },

    "scope": {
        "formal_performance_claim":
            False,

        "formal_scaling_claim":
            False,

        "formal_convergence_claim":
            False,

        "m9_3_transport_overlay_used":
            False,
    },

    "raw_log":
        str(
            raw_path
        ),

    "verdict":
        verdict,

    "next_milestone": (
        "RLTrain-stage-9.4.2 matched colocated vs disaggregated benchmark"
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
    + str(exit_pass)
)

print(
    "GLOBAL_STEP_1_SEEN = "
    + str(step1_seen)
)

print(
    "GLOBAL_STEP_2_SEEN = "
    + str(step2_seen)
)

print(
    "TRAINING_PROGRESS_100 = "
    + str(training_100)
)

print()

print(
    "HYBRID_ENGINE_TRUE_SEEN = "
    + str(hybrid_true_seen)
)

print(
    "TRANSFERQUEUE_REAL_SIGNAL = "
    + str(transferqueue_real_seen)
)

print(
    "TRANSFERQUEUE_MISSING_ABSENT = "
    + str(not transferqueue_missing_seen)
)

print(
    "VLLM_SIGNAL = "
    + str(vllm_seen)
)

print(
    "GENERATION_SIGNAL = "
    + str(generation_seen)
)

print(
    "ACTOR_UPDATE_SIGNAL = "
    + str(actor_update_seen)
)

print(
    "REWARD_SIGNAL = "
    + str(reward_seen)
)

print()

print(
    "SHM_FALLBACK_ABSENT = "
    + str(
        not forbidden_disaggregated_signals[
            "shm_fallback"
        ]
    )
)

print(
    "M9_3_RANKFIX_ABSENT = "
    + str(
        not forbidden_disaggregated_signals[
            "p2_rankfix"
        ]
    )
)

print(
    "ONE_STEP_TASK_RUNNER_ABSENT = "
    + str(
        not forbidden_disaggregated_signals[
            "one_step_task_runner"
        ]
    )
)

print()

print(
    "TRACEBACK_COUNT = "
    + str(traceback_count)
)

print(
    "TRACEBACK_CLASSIFICATION = "
    + traceback_classification
)

print(
    "FATAL_RUNTIME_ERROR_SEEN = "
    + str(fatal_seen)
)

print()

print(
    "SCIENTIFIC_EXECUTION_PASS = "
    + str(
        scientific_execution_pass
    )
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

print()

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
        "RLTrain-stage-9.4.1E bring-up failed."
    )
PY

    echo
    echo "========================================================================================================================"
    echo "9. FINAL GPU STATE"
    echo "========================================================================================================================"

    nvidia-smi \
        --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader

    echo
    echo "========================================================================================================================"
    echo "END OF RLTrain-stage-9.4.1E"
    echo "========================================================================================================================"

) 2>&1 | tee "$TXT"

STATUS=${PIPESTATUS[0]}

printf '%s\n' "$TXT"

exit "$STATUS"
