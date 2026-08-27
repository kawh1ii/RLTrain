#!/usr/bin/env bash

set -u

ROOT="${RLTRAIN_WORKSPACE_ROOT:-/path/to/ai-infra}"
PROJECT="$ROOT/RLTrain"

ENV="$ROOT/envs/project2-verl"
VERL_SITE="$ENV/lib/python3.12/site-packages"

SHM_OVERLAY="$PROJECT/overlays/stage_9_shm_rankfix"
TQ_OVERLAY="$PROJECT/overlays/stage_9_transferqueue_0_1_7"

SCRIPT_DIR="$PROJECT/scripts"
RESULT_DIR="$PROJECT/results/stage_9_4_2"
DATA_DIR="$PROJECT/data/stage_9_4_2"
ARTIFACT_DIR="$ROOT/artifacts"

MODEL_ROOT="$ROOT/caches/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots"

SOURCE_TRAIN="$PROJECT/data/stage_5/train.parquet"
SOURCE_VAL="$PROJECT/data/stage_5/test.parquet"

mkdir -p "$RESULT_DIR"
mkdir -p "$DATA_DIR"
mkdir -p "$ARTIFACT_DIR"

RUN_ID="$(date +%Y%m%d_%H%M%S)"

TXT="$ARTIFACT_DIR/RLTRAIN_STAGE_9_4_2_FORMAL_DISAGG_VS_COLOCATED_${RUN_ID}.txt"


(
    set -euo pipefail

    echo "========================================================================================================================"
    echo "RLTrain — RLTrain-stage-9.4.2 FORMAL DISAGGREGATED VS COLOCATED SYSTEM BENCHMARK"
    echo "========================================================================================================================"
    echo

    echo "CONFIG_A = DISAGG_6PLUS2"
    echo "CONFIG_B = COLOCATED_8GPU"
    echo
    echo "PHYSICAL_GPU_COUNT = 8"
    echo "GPU_MODEL = NVIDIA_GEFORCE_RTX_4090"
    echo
    echo "TRAIN_BATCH_SIZE = 8"
    echo "ROLLOUT_N = 3"
    echo "PPO_MINI_BATCH_SIZE = 8"
    echo "EFFECTIVE_ACTOR_MINI_BATCH = 24"
    echo "ROLLOUT_TP = 2"
    echo "ROLLOUT_DP = 1"
    echo "ACTOR_FSDP_SIZE = 2"
    echo
    echo "REPLICATES_PER_ARCHITECTURE = 3"
    echo "STEPS_PER_REPLICATE = 7"
    echo "EXCLUDED_STEPS = 1,2"
    echo "MEASURED_STEPS = 3,4,5,6,7"
    echo
    echo "PRIMARY_METRIC_1 = END_TO_END_STEP_WALL_TIME"
    echo "PRIMARY_METRIC_2 = SYSTEM_TOKENS_PER_SECOND"
    echo "SYSTEM_THROUGHPUT_FORMULA = SUM_TOTAL_TOKENS_DIV_SUM_STEP_TIME"
    echo
    echo "FORMAL_MATCHED_WORKLOAD_SYSTEM_COMPARISON = TRUE"
    echo "FORMAL_SINGLE_VARIABLE_PLACEMENT_CAUSAL_CLAIM = FALSE"
    echo "FORMAL_GENERAL_PERFORMANCE_RECOMMENDATION = FALSE"
    echo "FORMAL_STATISTICAL_SIGNIFICANCE_CLAIM = FALSE"

    echo
    echo "========================================================================================================================"
    echo "1. PREVIOUS CORRECTNESS GATES"
    echo "========================================================================================================================"

    "$ENV/bin/python" - <<'PY'
import json
from pathlib import Path


#
# Colocated gate.
#

colocated_dir = Path(
    "/path/to/ai-infra/"
    "RLTrain/results/"
    "stage_9_4_1f"
)

colocated = sorted(
    colocated_dir.glob(
        "RLTRAIN_STAGE_9_4_1F_DECISION_*.json"
    ),
    key=lambda p:
        p.stat().st_mtime,
    reverse=True,
)

if not colocated:
    raise SystemExit(
        "RLTrain-stage-9.4.1F decision missing."
    )

colocated_data = json.loads(
    colocated[0].read_text(
        encoding="utf-8"
    )
)

expected_colocated = (
    "RLTRAIN_STAGE_9_4_1F_REAL_8GPU_"
    "COLOCATED_MAIN_PPO_SYNC_"
    "TRANSFERQUEUE_TWO_STEP_"
    "BRINGUP_COMPLETE_"
    "READY_FOR_RLTRAIN_STAGE_9_4_2"
)

print(
    "COLOCATED_GATE_VERDICT = "
    + str(
        colocated_data.get(
            "verdict"
        )
    )
)

assert (
    colocated_data.get(
        "verdict"
    )
    ==
    expected_colocated
)


#
# Disaggregated matched-minibatch gate.
#

disagg_dir = Path(
    "/path/to/ai-infra/"
    "RLTrain/results/"
    "stage_9_4_2a"
)

disagg = sorted(
    disagg_dir.glob(
        "RLTRAIN_STAGE_9_4_2A_DECISION_*.json"
    ),
    key=lambda p:
        p.stat().st_mtime,
    reverse=True,
)

if not disagg:
    raise SystemExit(
        "RLTrain-stage-9.4.2A decision missing."
    )

disagg_data = json.loads(
    disagg[0].read_text(
        encoding="utf-8"
    )
)

expected_disagg = (
    "RLTRAIN_STAGE_9_4_2A_6TRAIN2ROLLOUT_"
    "PPO_MINIBATCH8_TWO_STEP_"
    "MATCHED_BRINGUP_COMPLETE_"
    "READY_FOR_RLTRAIN_STAGE_9_4_2"
)

print(
    "DISAGG_GATE_VERDICT = "
    + str(
        disagg_data.get(
            "verdict"
        )
    )
)

assert (
    disagg_data.get(
        "verdict"
    )
    ==
    expected_disagg
)

print(
    "CORRECTNESS_GATES = PASS"
)
PY

    echo
    echo "========================================================================================================================"
    echo "2. ENVIRONMENT FREEZE"
    echo "========================================================================================================================"

    unset PYTHONPATH || true
    unset RLTRAIN_FORCE_VERL_SHM_WEIGHT_TRANSFER || true
    unset RLTRAIN_SHM_OVERLAY_VERBOSE || true
    unset RLTRAIN_ROLLOUT_PHYSICAL_GPUS || true
    unset CUDA_VISIBLE_DEVICES || true
    unset RAY_ADDRESS || true

    "$ENV/bin/python" - <<'PY'
import importlib.metadata as md
import torch
import ray

print(
    "torch="
    + torch.__version__
)

print(
    "torch_cuda="
    + str(
        torch.version.cuda
    )
)

print(
    "ray="
    + ray.__version__
)

print(
    "verl="
    + md.version(
        "verl"
    )
)

print(
    "vllm="
    + md.version(
        "vllm"
    )
)

print(
    "numpy="
    + md.version(
        "numpy"
    )
)

print(
    "gpu_count="
    + str(
        torch.cuda.device_count()
    )
)

assert torch.__version__ == "2.8.0+cu128"
assert torch.version.cuda == "12.8"
assert ray.__version__ == "2.57.0"
assert md.version("verl") == "0.8.0"
assert md.version("vllm") == "0.10.2"
assert md.version("numpy") == "1.26.4"
assert torch.cuda.device_count() == 8

print(
    "ENVIRONMENT_GATE = PASS"
)
PY

    if [[ ! -f "$SHM_OVERLAY/sitecustomize.py" ]]; then
        echo "SHM_OVERLAY_MISSING"
        exit 31
    fi

    if [[ ! -f "$TQ_OVERLAY/transfer_queue/__init__.py" ]]; then
        echo "TRANSFERQUEUE_OVERLAY_MISSING"
        exit 32
    fi

    echo "SHM_OVERLAY_GATE = PASS"
    echo "TRANSFERQUEUE_OVERLAY_GATE = PASS"

    echo
    echo "========================================================================================================================"
    echo "3. MODEL / DATA FREEZE"
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
        exit 33
    fi

    echo "MODEL_PATH = $MODEL_PATH"
    echo "MODEL_DOWNLOAD = FALSE"

    TRAIN_FILE="$DATA_DIR/train_96.parquet"
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
    os.environ[
        "RLTRAIN_SOURCE_TRAIN"
    ]
)

source_val = Path(
    os.environ[
        "RLTRAIN_SOURCE_VAL"
    ]
)

target_train = Path(
    os.environ[
        "RLTRAIN_TARGET_TRAIN"
    ]
)

target_val = Path(
    os.environ[
        "RLTRAIN_TARGET_VAL"
    ]
)


train = pq.read_table(
    source_train
)

val = pq.read_table(
    source_val
)


copies = []
rows = 0

while rows < 96:

    copies.append(
        train
    )

    rows += train.num_rows


expanded = pa.concat_tables(
    copies
).slice(
    0,
    96,
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
    + str(
        train.num_rows
    )
)

print(
    "TARGET_TRAIN_ROWS = "
    + str(
        expanded.num_rows
    )
)

print(
    "VAL_ROWS = "
    + str(
        val.num_rows
    )
)

print(
    "DATASET_REPLICATION_ONLY = TRUE"
)
PY

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

    export RLTRAIN_RUN_ID="$RUN_ID"

    echo
    echo "========================================================================================================================"
    echo "4. SIX FORMAL REPLICATES"
    echo "========================================================================================================================"

    echo "RUN_ORDER = A1,B1,B2,A2,A3,B3"

    run_case() {

        local CONFIG="$1"
        local REP="$2"

        echo
        echo "========================================================================================================================"
        echo "BEGIN CONFIG=$CONFIG REP=$REP"
        echo "========================================================================================================================"

        "$ENV/bin/ray" stop --force \
            >/dev/null 2>&1 \
            || true

        sleep 3

        unset PYTHONPATH || true
        unset RLTRAIN_FORCE_VERL_SHM_WEIGHT_TRANSFER || true
        unset RLTRAIN_SHM_OVERLAY_VERBOSE || true
        unset RLTRAIN_ROLLOUT_PHYSICAL_GPUS || true
        unset CUDA_VISIBLE_DEVICES || true
        unset RAY_ADDRESS || true

        RAW_LOG="$RESULT_DIR/RLTRAIN_STAGE_9_4_2_${CONFIG}_REP${REP}_${RUN_ID}.log"

        GPU_CSV="$RESULT_DIR/RLTRAIN_STAGE_9_4_2_${CONFIG}_REP${REP}_GPU_${RUN_ID}.csv"

        SUMMARY_JSON="$RESULT_DIR/RLTRAIN_STAGE_9_4_2_${CONFIG}_REP${REP}_${RUN_ID}.json"

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

        set +e


        if [[ "$CONFIG" == "DISAGG_6PLUS2" ]]; then

            echo "ENTRYPOINT = verl.experimental.one_step_off_policy.main_ppo"
            echo "HYBRID_ENGINE = FALSE"
            echo "TRAINER_GPUS = 6"
            echo "ROLLOUT_GPUS = 2"
            echo "WEIGHT_TRANSFER = SHARED_MEMORY"
            echo "ROLLOUT_PHYSICAL_GPUS = 6,7"

            export PYTHONPATH="$SHM_OVERLAY"

            export RLTRAIN_FORCE_VERL_SHM_WEIGHT_TRANSFER=1
            export RLTRAIN_SHM_OVERLAY_VERBOSE=1
            export RLTRAIN_ROLLOUT_PHYSICAL_GPUS="6,7"

            (
                cd "$VERL_SITE"

                "$ENV/bin/python" \
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
                    actor_rollout_ref.rollout.free_cache_engine=False \
                    actor_rollout_ref.rollout.enable_chunked_prefill=False \
                    actor_rollout_ref.rollout.enable_prefix_caching=False \
                    actor_rollout_ref.rollout.max_model_len=256 \
                    actor_rollout_ref.rollout.max_num_batched_tokens=256 \
                    actor_rollout_ref.rollout.max_num_seqs=32 \
                    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
                    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
                    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
                    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=384 \
                    reward.reward_manager.name=naive \
                    reward_model.reward_manager=naive \
                    critic.strategy=fsdp2 \
                    trainer.logger='["console"]' \
                    trainer.project_name=p2_rltrain \
                    trainer.experiment_name="stage_9_4_2_disagg_rep${REP}_${RUN_ID}" \
                    trainer.val_before_train=False \
                    trainer.balance_batch=False \
                    trainer.critic_warmup=0 \
                    trainer.nnodes=1 \
                    trainer.n_gpus_per_node=6 \
                    trainer.save_freq=-1 \
                    trainer.test_freq=-1 \
                    trainer.total_epochs=1 \
                    trainer.total_training_steps=7 \
                    trainer.resume_mode=disable \
                    trainer.default_hdfs_dir=null \
                    trainer.default_local_dir="$RESULT_DIR/checkpoints_disagg_rep${REP}_${RUN_ID}" \
                    rollout.nnodes=1 \
                    rollout.n_gpus_per_node=2

            ) 2>&1 | tee "$RAW_LOG"

            STATUS=${PIPESTATUS[0]}


        elif [[ "$CONFIG" == "COLOCATED_8GPU" ]]; then

            echo "ENTRYPOINT = verl.trainer.main_ppo_sync"
            echo "HYBRID_ENGINE = TRUE"
            echo "SHARED_GPU_POOL = 8"
            echo "TRANSFERQUEUE_VERSION = 0.1.7"

            export PYTHONPATH="$TQ_OVERLAY"

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
                    trainer.experiment_name="stage_9_4_2_colocated_rep${REP}_${RUN_ID}" \
                    trainer.val_before_train=False \
                    trainer.balance_batch=False \
                    trainer.critic_warmup=0 \
                    trainer.nnodes=1 \
                    trainer.n_gpus_per_node=8 \
                    trainer.save_freq=-1 \
                    trainer.test_freq=-1 \
                    trainer.total_epochs=1 \
                    trainer.total_training_steps=7 \
                    trainer.resume_mode=disable \
                    trainer.default_hdfs_dir=null \
                    trainer.default_local_dir="$RESULT_DIR/checkpoints_colocated_rep${REP}_${RUN_ID}"

            ) 2>&1 | tee "$RAW_LOG"

            STATUS=${PIPESTATUS[0]}


        else

            echo "UNKNOWN_CONFIG = $CONFIG"

            STATUS=90

        fi


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

        echo "TRAINER_EXIT_CODE = $STATUS"

        RLTRAIN_CONFIG="$CONFIG" \
        RLTRAIN_REP="$REP" \
        RLTRAIN_RAW_LOG="$RAW_LOG" \
        RLTRAIN_GPU_CSV="$GPU_CSV" \
        RLTRAIN_SUMMARY_JSON="$SUMMARY_JSON" \
        RLTRAIN_EXIT_CODE="$STATUS" \
        "$ENV/bin/python" \
            "$SCRIPT_DIR/stage_9_4_2_parse_run.py"

        PARSER_STATUS=$?

        if [[ "$STATUS" -ne 0 ]]; then
            exit "$STATUS"
        fi

        if [[ "$PARSER_STATUS" -ne 0 ]]; then
            exit "$PARSER_STATUS"
        fi

        echo "RUN_VERDICT = PASS"

        sleep 5
    }


    run_case \
        "DISAGG_6PLUS2" \
        1

    run_case \
        "COLOCATED_8GPU" \
        1

    run_case \
        "COLOCATED_8GPU" \
        2

    run_case \
        "DISAGG_6PLUS2" \
        2

    run_case \
        "DISAGG_6PLUS2" \
        3

    run_case \
        "COLOCATED_8GPU" \
        3


    echo
    echo "========================================================================================================================"
    echo "5. FINAL MATCHED-WORKLOAD SYSTEM COMPARISON"
    echo "========================================================================================================================"

    "$ENV/bin/python" \
        "$SCRIPT_DIR/stage_9_4_2_finalize.py"

    echo
    echo "========================================================================================================================"
    echo "6. FINAL ENVIRONMENT FREEZE"
    echo "========================================================================================================================"

    unset PYTHONPATH || true
    unset RLTRAIN_FORCE_VERL_SHM_WEIGHT_TRANSFER || true
    unset RLTRAIN_SHM_OVERLAY_VERBOSE || true
    unset RLTRAIN_ROLLOUT_PHYSICAL_GPUS || true

    "$ENV/bin/python" - <<'PY'
import importlib.metadata as md
import torch
import ray

assert torch.__version__ == "2.8.0+cu128"
assert torch.version.cuda == "12.8"
assert ray.__version__ == "2.57.0"
assert md.version("verl") == "0.8.0"
assert md.version("vllm") == "0.10.2"
assert md.version("numpy") == "1.26.4"

print(
    "RLTRAIN_VERL_POST_BENCHMARK_FROZEN = TRUE"
)
PY

    "$ENV/bin/ray" stop --force \
        >/dev/null 2>&1 \
        || true

    sleep 2

    echo
    echo "========================================================================================================================"
    echo "END OF RLTrain-stage-9.4.2"
    echo "========================================================================================================================"

) 2>&1 | tee "$TXT"

STATUS=${PIPESTATUS[0]}

printf '%s\n' "$TXT"

exit "$STATUS"
