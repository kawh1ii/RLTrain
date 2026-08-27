#!/usr/bin/env bash

set -u

ROOT="${RLTRAIN_WORKSPACE_ROOT:-/path/to/ai-infra}"
PROJECT="$ROOT/RLTrain"

ENV="$ROOT/envs/project2-verl"
VERL_SITE="$ENV/lib/python3.12/site-packages"

TRACE_V2="$PROJECT/overlays/stage_9_phase_trace_v2"
TRACE_V3="$PROJECT/overlays/stage_9_phase_trace_v3"

SHM_OVERLAY="$PROJECT/overlays/stage_9_shm_rankfix"

SCRIPT_DIR="$PROJECT/scripts"

TRACE_RESULT_DIR="$PROJECT/results/stage_9_5_2"
AUDIT_DIR="$PROJECT/results/stage_9_5_3a"
RESULT_DIR="$PROJECT/results/stage_9_5_3b"

DATA_DIR="$PROJECT/data/stage_9_4_2"
DOC_DIR="$PROJECT/docs"
ARTIFACT_DIR="$ROOT/artifacts"

MODEL_ROOT="$ROOT/caches/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots"

mkdir -p "$RESULT_DIR"
mkdir -p "$DOC_DIR"
mkdir -p "$ARTIFACT_DIR"


RUN_ID="$(date +%Y%m%d_%H%M%S)"

TXT="$ARTIFACT_DIR/RLTRAIN_STAGE_9_5_3B_FINAL_DISAGG_TRACE_REPAIR_${RUN_ID}.txt"

DISAGG_TRACE="$TRACE_RESULT_DIR/RLTRAIN_STAGE_9_5_2_DISAGG_TRACE_${RUN_ID}.jsonl"

DISAGG_RAW="$RESULT_DIR/RLTRAIN_STAGE_9_5_3B_DISAGG_RAW_${RUN_ID}.log"

COLOCATED_TRACE_ALIAS="$TRACE_RESULT_DIR/RLTRAIN_STAGE_9_5_2_COLOCATED_TRACE_${RUN_ID}.jsonl"

OUTPUT_JSON="$RESULT_DIR/RLTRAIN_STAGE_9_5_3B_TRACE_ANALYSIS_${RUN_ID}.json"

ADJUDICATION_JSON="$RESULT_DIR/RLTRAIN_STAGE_9_5_3B_DECISION_${RUN_ID}.json"

REPORT="$DOC_DIR/RLTRAIN_STAGE_9_5_3B_FINAL_DISAGG_TRACE_REPAIR.md"

ANALYZER="$SCRIPT_DIR/stage_9_5_2_analyze_trace.py"


(
set -euo pipefail


echo "========================================================================================================================"
echo "RLTrain — RLTrain-stage-9.5.3B FINAL DISAGGREGATED TRACE REPAIR"
echo "========================================================================================================================"
echo

echo "SCIENTIFIC_RERUN_DISAGG = TRUE"
echo "SCIENTIFIC_RERUN_COLOCATED = FALSE"
echo "REAL_MODEL_EXECUTION = TRUE"

echo "PACKAGE_INSTALLATION = FALSE"
echo "ENVIRONMENT_MODIFICATION = FALSE"
echo "RLTRAIN_VERL_MODIFICATION = FALSE"
echo "RLTRAIN_VERL_SOURCE_MODIFICATION = FALSE"

echo
echo "DISAGG_ONLY_RERUN = TRUE"
echo "COLOCATED_TRACE_REUSED = TRUE"

echo
echo "TOTAL_TRAINING_STEPS = 4"
echo "WARMUP_STEPS = 1,2"
echo "MEASURED_STEPS = 3,4"

echo
echo "FORMAL_PERFORMANCE_CLAIM = FALSE"
echo "FORMAL_STATISTICAL_CLAIM = FALSE"
echo "CAUSAL_MECHANISM_CLAIM = FALSE"


echo
echo "========================================================================================================================"
echo "1. RLTrain-stage-9.5.3A GATE"
echo "========================================================================================================================"


RLTRAIN_AUDIT_DIR="$AUDIT_DIR" \
"$ENV/bin/python" - <<'PY'
import os
import json
from pathlib import Path


root = Path(
    os.environ[
        "RLTRAIN_AUDIT_DIR"
    ]
)


candidates = sorted(
    root.glob(
        "RLTRAIN_STAGE_9_5_3A_TRACE_COVERAGE_*.json"
    ),
    key=lambda p:
        p.stat().st_mtime,
    reverse=True,
)


if not candidates:

    raise SystemExit(
        "RLTrain-stage-9.5.3A decision missing."
    )


path = candidates[0]


data = json.loads(
    path.read_text(
        encoding="utf-8"
    )
)


expected = (
    "RLTRAIN_STAGE_9_5_3A_TRACE_COVERAGE_AUDIT_COMPLETE_"
    "READY_FOR_FINAL_DISAGG_TRACE_REPAIR"
)


print(
    "GATE_JSON = "
    + str(
        path
    )
)

print(
    "RLTRAIN_STAGE_9_5_3A_VERDICT = "
    + str(
        data.get(
            "verdict"
        )
    )
)


assert (
    data.get(
        "verdict"
    )
    ==
    expected
)


assert (
    data[
        "colocated"
    ][
        "sleep_replicas_gap_mechanism_candidate"
    ]
    is True
)


assert (
    data[
        "disagg"
    ][
        "actor_trace_missing"
    ]
    is True
)


print(
    "COLOCATED_SLEEP_REPLICAS_LOCALIZATION = PASS"
)

print(
    "DISAGG_ACTOR_TRACE_REPAIR_REQUIRED = TRUE"
)

print(
    "RLTRAIN_STAGE_9_5_3A_GATE = PASS"
)
PY


echo
echo "========================================================================================================================"
echo "2. LOCATE / FREEZE SUCCESSFUL COLOCATED TRACE"
echo "========================================================================================================================"


COLOCATED_TRACE_SOURCE="$(
    find "$TRACE_RESULT_DIR" \
        -maxdepth 1 \
        -type f \
        -name 'RLTRAIN_STAGE_9_5_2_COLOCATED_TRACE_*.jsonl' \
        -size +0c \
        -printf '%T@ %p\n' \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
)"


if [[ -z "$COLOCATED_TRACE_SOURCE" ]]; then

    echo "COLOCATED_TRACE_NOT_FOUND"

    exit 20

fi


echo "COLOCATED_TRACE_SOURCE = $COLOCATED_TRACE_SOURCE"


COLOCATED_LINES="$(
    wc -l \
        < "$COLOCATED_TRACE_SOURCE"
)"


echo "COLOCATED_TRACE_EVENT_LINES = $COLOCATED_LINES"


if [[ "$COLOCATED_LINES" -lt 100 ]]; then

    echo "COLOCATED_TRACE_TOO_SMALL"

    exit 21

fi


cp \
    "$COLOCATED_TRACE_SOURCE" \
    "$COLOCATED_TRACE_ALIAS"


echo "COLOCATED_TRACE_ALIAS = $COLOCATED_TRACE_ALIAS"
echo "COLOCATED_SCIENTIFIC_RERUN = FALSE"
echo "COLOCATED_TRACE_REUSE_GATE = PASS"


echo
echo "========================================================================================================================"
echo "3. BUILD TRACE OVERLAY V3 — ADD SEPARATION TRAINER"
echo "========================================================================================================================"


if [[ ! -f "$TRACE_V2/sitecustomize.py" ]]; then

    echo "TRACE_V2_MISSING"

    exit 22

fi


rm -rf "$TRACE_V3"

mkdir -p "$TRACE_V3"


RLTRAIN_SOURCE="$TRACE_V2/sitecustomize.py" \
RLTRAIN_TARGET="$TRACE_V3/sitecustomize.py" \
"$ENV/bin/python" - <<'PY'
import os
from pathlib import Path


src = Path(
    os.environ[
        "RLTRAIN_SOURCE"
    ]
)

dst = Path(
    os.environ[
        "RLTRAIN_TARGET"
    ]
)


text = src.read_text(
    encoding="utf-8"
)


target = (
    '"verl.experimental.separation.ray_trainer"'
)


if target not in text:

    anchor = (
        '"verl.experimental.one_step_off_policy.ray_trainer",'
    )


    count = text.count(
        anchor
    )


    print(
        "ONE_STEP_TARGET_ANCHOR_COUNT = "
        + str(
            count
        )
    )


    if count != 1:

        raise RuntimeError(
            "Expected exactly one one-step target anchor."
        )


    text = text.replace(
        anchor,
        (
            anchor
            +
            "\n    "
            +
            target
            +
            ","
        ),
        1,
    )


dst.write_text(
    text,
    encoding="utf-8",
)


verify = dst.read_text(
    encoding="utf-8"
)


assert (
    '"verl.experimental.one_step_off_policy.ray_trainer"'
    in verify
)

assert (
    '"verl.experimental.separation.ray_trainer"'
    in verify
)

assert (
    '"verl.trainer.ppo.ray_trainer"'
    in verify
)


print(
    "ONE_STEP_OFF_POLICY_TRACE_TARGET = TRUE"
)

print(
    "SEPARATION_TRACE_TARGET = TRUE"
)

print(
    "BASE_PPO_TRACE_TARGET = TRUE"
)

print(
    "METRIC_UTILS_TRACE_TARGET = FALSE"
)

print(
    "TRACE_V3_BUILD = PASS"
)
PY


echo "TRACE_V3 = $TRACE_V3"


echo
echo "========================================================================================================================"
echo "4. ZERO-GPU TRACE TARGET PREFLIGHT"
echo "========================================================================================================================"


PROBE="$RESULT_DIR/RLTRAIN_STAGE_9_5_3B_TRACE_TARGET_PROBE_${RUN_ID}.jsonl"

rm -f "$PROBE"


PYTHONPATH="$TRACE_V3" \
RLTRAIN_TRACE_FILE="$PROBE" \
RLTRAIN_TRACE_ARCH="DISAGG_TRACE_PREFLIGHT" \
"$ENV/bin/python" - <<'PY'
import inspect
import importlib


one = importlib.import_module(
    "verl.experimental.one_step_off_policy.ray_trainer"
)

sep = importlib.import_module(
    "verl.experimental.separation.ray_trainer"
)

base = importlib.import_module(
    "verl.trainer.ppo.ray_trainer"
)


print(
    "ONE_STEP_MODULE = "
    + str(
        one.__file__
    )
)

print(
    "SEPARATION_MODULE = "
    + str(
        sep.__file__
    )
)

print(
    "BASE_PPO_MODULE = "
    + str(
        base.__file__
    )
)


sep_timer = getattr(
    sep,
    "marked_timer",
    None,
)

base_timer = getattr(
    base,
    "marked_timer",
    None,
)

one_timer = getattr(
    one,
    "marked_timer",
    None,
)


sep_wrapped = bool(
    getattr(
        sep_timer,
        "_p2_trace_wrapped",
        False,
    )
)

base_wrapped = bool(
    getattr(
        base_timer,
        "_p2_trace_wrapped",
        False,
    )
)

one_wrapped = bool(
    getattr(
        one_timer,
        "_p2_trace_wrapped",
        False,
    )
)


print(
    "SEPARATION_MARKED_TIMER_WRAPPED = "
    + str(
        sep_wrapped
    )
)

print(
    "BASE_PPO_MARKED_TIMER_WRAPPED = "
    + str(
        base_wrapped
    )
)

print(
    "ONE_STEP_MARKED_TIMER_WRAPPED = "
    + str(
        one_wrapped
    )
)


source = inspect.getsource(
    sep
)


update_actor_site_count = (
    source.count(
        'marked_timer("update_actor"'
    )
    +
    source.count(
        "marked_timer('update_actor'"
    )
)


print(
    "SEPARATION_UPDATE_ACTOR_TIMER_SITE_COUNT = "
    + str(
        update_actor_site_count
    )
)


print()
print(
    "ONE_STEP_TRAINER_CLASS_MRO"
)


for name, obj in sorted(
    vars(
        one
    ).items()
):

    if not inspect.isclass(
        obj
    ):
        continue

    if "trainer" not in name.lower():
        continue

    try:

        mro = [
            (
                cls.__module__
                +
                "."
                +
                cls.__qualname__
            )
            for cls
            in inspect.getmro(
                obj
            )
        ]

    except Exception:

        continue


    print(
        name
        + " = "
        + str(
            mro
        )
    )


if not sep_wrapped:

    raise SystemExit(
        "Separation marked_timer was not wrapped."
    )


if update_actor_site_count < 1:

    raise SystemExit(
        "No separation update_actor timer site."
    )


print(
    "FINAL_DISAGG_TRACE_TARGET_PREFLIGHT = PASS"
)
PY


if [[ ! -s "$PROBE" ]]; then

    echo "TRACE_PREFLIGHT_PROBE_EMPTY"

    exit 23

fi


rm -f "$PROBE"


echo
echo "========================================================================================================================"
echo "5. ENVIRONMENT / DATA GATE"
echo "========================================================================================================================"


MODEL_PATH="$(
    find "$MODEL_ROOT" \
        -mindepth 1 \
        -maxdepth 1 \
        -type d \
    | head -n 1
)"


TRAIN_FILE="$DATA_DIR/train_96.parquet"

VAL_FILE="$DATA_DIR/test.parquet"


if [[ -z "$MODEL_PATH" ]]; then

    echo "MODEL_NOT_FOUND"

    exit 24

fi


if [[ ! -f "$TRAIN_FILE" ]]; then

    echo "TRAIN_FILE_MISSING"

    exit 25

fi


if [[ ! -f "$VAL_FILE" ]]; then

    echo "VAL_FILE_MISSING"

    exit 26

fi


if [[ ! -f "$SHM_OVERLAY/sitecustomize.py" ]]; then

    echo "SHM_OVERLAY_MISSING"

    exit 27

fi


unset PYTHONPATH || true
unset CUDA_VISIBLE_DEVICES || true
unset RAY_ADDRESS || true


"$ENV/bin/python" - <<'PY'
import importlib.metadata as md
import torch
import ray


assert torch.__version__ == "2.8.0+cu128"
assert torch.version.cuda == "12.8"

assert ray.__version__ == "2.57.0"

assert md.version(
    "verl"
) == "0.8.0"

assert md.version(
    "vllm"
) == "0.10.2"

assert md.version(
    "numpy"
) == "1.26.4"

assert torch.cuda.device_count() == 8


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
    "gpu_count="
    + str(
        torch.cuda.device_count()
    )
)

print(
    "ENVIRONMENT_GATE = PASS"
)
PY


echo
echo "========================================================================================================================"
echo "6. FINAL DISAGGREGATED 4-STEP TRACE"
echo "========================================================================================================================"


"$ENV/bin/ray" stop --force \
    >/dev/null 2>&1 \
    || true


sleep 3


rm -f "$DISAGG_TRACE"
rm -f "$DISAGG_RAW"


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


export PYTHONPATH="$TRACE_V3:$SHM_OVERLAY"


export RLTRAIN_TRACE_FILE="$DISAGG_TRACE"

export RLTRAIN_TRACE_ARCH="DISAGG_6PLUS2"


export RLTRAIN_TRACE_LOAD_SHM_OVERLAY=1

export RLTRAIN_TRACE_SHM_SITECUSTOMIZE="$SHM_OVERLAY/sitecustomize.py"


export RLTRAIN_FORCE_VERL_SHM_WEIGHT_TRANSFER=1

export RLTRAIN_SHM_OVERLAY_VERBOSE=1

export RLTRAIN_ROLLOUT_PHYSICAL_GPUS="6,7"


echo "ENTRYPOINT = verl.experimental.one_step_off_policy.main_ppo"

echo "HYBRID_ENGINE = FALSE"

echo "TRAINER_GPUS = 6"

echo "ROLLOUT_GPUS = 2"

echo "ROLLOUT_TP = 2"

echo "ROLLOUT_DP = 1"

echo "ACTOR_FSDP_SIZE = 2"

echo "WEIGHT_TRANSFER = SHARED_MEMORY"

echo "WEIGHT_TRANSFER_BUCKET_MIB = 384"

echo
echo "TRAIN_BATCH_SIZE = 8"

echo "ROLLOUT_N = 3"

echo "PPO_MINI_BATCH_SIZE = 8"

echo "TOTAL_TRAINING_STEPS = 4"


set +e


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
        trainer.experiment_name="stage_9_5_3b_disagg_${RUN_ID}" \
        trainer.val_before_train=False \
        trainer.balance_batch=False \
        trainer.critic_warmup=0 \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=6 \
        trainer.save_freq=-1 \
        trainer.test_freq=-1 \
        trainer.total_epochs=1 \
        trainer.total_training_steps=4 \
        trainer.resume_mode=disable \
        trainer.default_hdfs_dir=null \
        trainer.default_local_dir="$RESULT_DIR/checkpoints_disagg_${RUN_ID}" \
        rollout.nnodes=1 \
        rollout.n_gpus_per_node=2


) 2>&1 | tee "$DISAGG_RAW"


TRAINER_STATUS=${PIPESTATUS[0]}


set -e


"$ENV/bin/ray" stop --force \
    >/dev/null 2>&1 \
    || true


sleep 3


echo "TRAINER_EXIT_CODE = $TRAINER_STATUS"


if [[ "$TRAINER_STATUS" -ne 0 ]]; then

    echo "FINAL_DISAGG_TRACE_EXECUTION = FAIL"

    exit "$TRAINER_STATUS"

fi


if [[ ! -s "$DISAGG_TRACE" ]]; then

    echo "DISAGG_TRACE_EMPTY"

    exit 28

fi


TRACE_LINES="$(
    wc -l \
        < "$DISAGG_TRACE"
)"


echo "DISAGG_TRACE_EVENT_LINES = $TRACE_LINES"


if [[ "$TRACE_LINES" -lt 20 ]]; then

    echo "DISAGG_TRACE_TOO_SMALL"

    exit 29

fi


if ! grep -q \
    'step:4 - training/global_step:4' \
    "$DISAGG_RAW"
then

    echo "DISAGG_STEP4_MISSING"

    exit 30

fi


if ! grep -q \
    'Training Progress: 100%' \
    "$DISAGG_RAW"
then

    echo "DISAGG_TRAINING_100_MISSING"

    exit 31

fi


echo
echo "========================================================================================================================"
echo "7. UPDATE-ACTOR TRACE STRUCTURE GATE"
echo "========================================================================================================================"


RLTRAIN_TRACE="$DISAGG_TRACE" \
"$ENV/bin/python" - <<'PY'
import os
import json
from pathlib import Path
from collections import Counter


path = Path(
    os.environ[
        "RLTRAIN_TRACE"
    ]
)


events = []


with path.open(
    "r",
    encoding="utf-8",
    errors="replace",
) as f:

    for line in f:

        try:

            events.append(
                json.loads(
                    line
                )
            )

        except Exception:

            pass


timer_begin = [
    event
    for event in events
    if event.get(
        "event"
    )
    ==
    "TIMER_BEGIN"
]


timer_end = [
    event
    for event in events
    if event.get(
        "event"
    )
    ==
    "TIMER_END"
]


names_begin = Counter(
    str(
        event.get(
            "name"
        )
    )
    for event
    in timer_begin
)


names_end = Counter(
    str(
        event.get(
            "name"
        )
    )
    for event
    in timer_end
)


step_begin = names_begin[
    "step"
]

actor_begin = names_begin[
    "update_actor"
]

actor_end = names_end[
    "update_actor"
]

generate_begin = names_begin[
    "generate_async"
]

sync_begin = names_begin[
    "sync_rollout_weights"
]


print(
    "PARSED_TRACE_EVENT_COUNT = "
    + str(
        len(
            events
        )
    )
)

print(
    "STEP_TIMER_BEGIN_COUNT = "
    + str(
        step_begin
    )
)

print(
    "UPDATE_ACTOR_TIMER_BEGIN_COUNT = "
    + str(
        actor_begin
    )
)

print(
    "UPDATE_ACTOR_TIMER_END_COUNT = "
    + str(
        actor_end
    )
)

print(
    "GENERATE_ASYNC_TIMER_BEGIN_COUNT = "
    + str(
        generate_begin
    )
)

print(
    "SYNC_ROLLOUT_WEIGHTS_TIMER_BEGIN_COUNT = "
    + str(
        sync_begin
    )
)


if step_begin < 4:

    raise SystemExit(
        "Expected four step timers."
    )


if actor_begin < 2:

    raise SystemExit(
        "update_actor trace still missing."
    )


if actor_begin != actor_end:

    raise SystemExit(
        "update_actor BEGIN/END imbalance."
    )


if generate_begin < 2:

    raise SystemExit(
        "generate_async trace missing."
    )


print(
    "DISAGG_ACTOR_TRACE_RECOVERY_GATE = PASS"
)
PY


echo
echo "========================================================================================================================"
echo "8. DIRECT INTERVAL ANALYSIS — NEW DISAGG + FROZEN COLOCATED"
echo "========================================================================================================================"


if [[ ! -f "$ANALYZER" ]]; then

    echo "ANALYZER_MISSING"

    exit 32

fi


RLTRAIN_RESULT_DIR="$TRACE_RESULT_DIR" \
RLTRAIN_RUN_ID="$RUN_ID" \
RLTRAIN_OUTPUT_JSON="$OUTPUT_JSON" \
RLTRAIN_REPORT_MD="$REPORT" \
"$ENV/bin/python" \
    "$ANALYZER"


echo
echo "========================================================================================================================"
echo "9. FINAL MECHANISM TRACE ADJUDICATION"
echo "========================================================================================================================"


RLTRAIN_ANALYSIS="$OUTPUT_JSON" \
RLTRAIN_DECISION="$ADJUDICATION_JSON" \
RLTRAIN_RUN_ID="$RUN_ID" \
RLTRAIN_COLOCATED_SOURCE="$COLOCATED_TRACE_SOURCE" \
"$ENV/bin/python" - <<'PY'
import os
import json
from pathlib import Path
from datetime import datetime


analysis_path = Path(
    os.environ[
        "RLTRAIN_ANALYSIS"
    ]
)

decision_path = Path(
    os.environ[
        "RLTRAIN_DECISION"
    ]
)

run_id = os.environ[
    "RLTRAIN_RUN_ID"
]

colocated_source = os.environ[
    "RLTRAIN_COLOCATED_SOURCE"
]


data = json.loads(
    analysis_path.read_text(
        encoding="utf-8"
    )
)


disagg = data[
    "results"
][
    "DISAGG_6PLUS2"
]


colocated = data[
    "results"
][
    "COLOCATED_8GPU"
]


actor_s = float(
    disagg[
        "means"
    ][
        "actor_update_union_s"
    ]
)

generation_s = float(
    disagg[
        "means"
    ][
        "generation_union_s"
    ]
)

overlap_s = float(
    disagg[
        "means"
    ][
        "generation_actor_overlap_s"
    ]
)

step_s = float(
    disagg[
        "means"
    ][
        "step_duration_s"
    ]
)


overlap_fraction_step = (
    overlap_s
    /
    step_s
    if step_s > 0
    else None
)


overlap_fraction_actor = (
    overlap_s
    /
    actor_s
    if actor_s > 0
    else None
)


overlap_fraction_generation = (
    overlap_s
    /
    generation_s
    if generation_s > 0
    else None
)


largest_gap = data[
    "summary"
][
    "largest_colocated_uncovered_gap"
]


gap_s = float(
    largest_gap[
        "duration_s"
    ]
)

sleep_s = float(
    largest_gap[
        "sleep_replicas_overlap_s"
    ]
)

tq_s = float(
    largest_gap[
        "tq_overlap_s"
    ]
)


sleep_fraction = (
    sleep_s
    /
    gap_s
    if gap_s > 0
    else None
)


actor_trace_recovered = (
    actor_s > 0.10
)


direct_overlap_adjudicable = (
    actor_trace_recovered
    and
    generation_s > 0
)


direct_overlap_observed = (
    direct_overlap_adjudicable
    and
    overlap_s > 0.01
)


sleep_localization_confirmed = (
    sleep_fraction is not None
    and
    sleep_fraction >= 0.95
)


if (
    actor_trace_recovered
    and
    sleep_localization_confirmed
):

    verdict = (
        "RLTRAIN_STAGE_9_5_3B_FINAL_DISAGG_TRACE_REPAIR_"
        "AND_CROSS_ARCH_MECHANISM_TRACE_COMPLETE_"
        "READY_FOR_RLTRAIN_STAGE_9_5_FINAL_ADJUDICATION"
    )

else:

    verdict = (
        "RLTRAIN_STAGE_9_5_3B_TRACE_REPAIR_REVIEW_REQUIRED"
    )


decision = {
    "schema_version":
        1,

    "project":
        "RLTrain",

    "milestone":
        "RLTrain-stage-9.5.3B",

    "timestamp":
        datetime.now()
        .astimezone()
        .isoformat(),

    "run_id":
        run_id,

    "scientific_execution": {
        "disagg_rerun":
            True,

        "colocated_rerun":
            False,

        "colocated_trace_source":
            colocated_source,
    },

    "disaggregated": {
        "step_mean_s":
            step_s,

        "generation_union_mean_s":
            generation_s,

        "actor_update_union_mean_s":
            actor_s,

        "generation_actor_overlap_mean_s":
            overlap_s,

        "generation_actor_overlap_fraction_of_step":
            overlap_fraction_step,

        "generation_actor_overlap_fraction_of_actor":
            overlap_fraction_actor,

        "generation_actor_overlap_fraction_of_generation":
            overlap_fraction_generation,

        "actor_trace_recovered":
            actor_trace_recovered,

        "direct_overlap_adjudicable":
            direct_overlap_adjudicable,

        "direct_overlap_observed":
            direct_overlap_observed,
    },

    "colocated": {
        "step_mean_s":
            colocated[
                "means"
            ][
                "step_duration_s"
            ],

        "largest_uncovered_gap_s":
            gap_s,

        "sleep_replicas_overlap_s":
            sleep_s,

        "sleep_replicas_fraction_of_largest_gap":
            sleep_fraction,

        "transferqueue_overlap_s":
            tq_s,

        "sleep_replicas_localization_confirmed":
            sleep_localization_confirmed,
    },

    "scope": {
        "formal_performance_claim":
            False,

        "formal_statistical_claim":
            False,

        "causal_mechanism_claim":
            False,

        "direct_interval_observation":
            True,

        "nsight_required_now":
            False,
    },

    "verdict":
        verdict,

    "next_action":
        (
            "RLTRAIN_STAGE_9_5_FINAL_MECHANISM_ADJUDICATION_"
            "AND_EVIDENCE_FREEZE"
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
    "DISAGG_ACTOR_TRACE_RECOVERED = "
    + str(
        actor_trace_recovered
    )
)

print(
    "DISAGG_GENERATION_MEAN_S = "
    + str(
        generation_s
    )
)

print(
    "DISAGG_ACTOR_UPDATE_MEAN_S = "
    + str(
        actor_s
    )
)

print(
    "DISAGG_GENERATION_ACTOR_OVERLAP_MEAN_S = "
    + str(
        overlap_s
    )
)

print(
    "DISAGG_GENERATION_ACTOR_OVERLAP_FRACTION_OF_STEP = "
    + str(
        overlap_fraction_step
    )
)

print(
    "DISAGG_GENERATION_ACTOR_OVERLAP_FRACTION_OF_ACTOR = "
    + str(
        overlap_fraction_actor
    )
)

print(
    "DISAGG_GENERATION_ACTOR_OVERLAP_FRACTION_OF_GENERATION = "
    + str(
        overlap_fraction_generation
    )
)

print(
    "DISAGG_DIRECT_OVERLAP_ADJUDICABLE = "
    + str(
        direct_overlap_adjudicable
    )
)

print(
    "DISAGG_DIRECT_OVERLAP_OBSERVED = "
    + str(
        direct_overlap_observed
    )
)

print()

print(
    "COLOCATED_LARGEST_UNCOVERED_GAP_S = "
    + str(
        gap_s
    )
)

print(
    "COLOCATED_SLEEP_REPLICAS_OVERLAP_S = "
    + str(
        sleep_s
    )
)

print(
    "COLOCATED_SLEEP_REPLICAS_FRACTION_OF_LARGEST_GAP = "
    + str(
        sleep_fraction
    )
)

print(
    "COLOCATED_SLEEP_REPLICAS_LOCALIZATION_CONFIRMED = "
    + str(
        sleep_localization_confirmed
    )
)

print()

print(
    "NSIGHT_REQUIRED_NOW = FALSE"
)

print(
    "CAUSAL_MECHANISM_CLAIM = FALSE"
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
PY


echo
echo "========================================================================================================================"
echo "10. FINAL ENVIRONMENT FREEZE"
echo "========================================================================================================================"


unset PYTHONPATH || true

unset RLTRAIN_TRACE_FILE || true
unset RLTRAIN_TRACE_ARCH || true

unset RLTRAIN_TRACE_LOAD_SHM_OVERLAY || true
unset RLTRAIN_TRACE_SHM_SITECUSTOMIZE || true

unset RLTRAIN_FORCE_VERL_SHM_WEIGHT_TRANSFER || true
unset RLTRAIN_SHM_OVERLAY_VERBOSE || true
unset RLTRAIN_ROLLOUT_PHYSICAL_GPUS || true

unset CUDA_VISIBLE_DEVICES || true
unset RAY_ADDRESS || true


"$ENV/bin/python" - <<'PY'
import importlib.metadata as md
import torch
import ray


assert torch.__version__ == "2.8.0+cu128"
assert torch.version.cuda == "12.8"

assert ray.__version__ == "2.57.0"

assert md.version(
    "verl"
) == "0.8.0"

assert md.version(
    "vllm"
) == "0.10.2"

assert md.version(
    "numpy"
) == "1.26.4"


print(
    "RLTRAIN_VERL_POST_TRACE_FROZEN = TRUE"
)
PY


"$ENV/bin/ray" stop --force \
    >/dev/null 2>&1 \
    || true


echo
echo "========================================================================================================================"
echo "END OF RLTrain-stage-9.5.3B"
echo "========================================================================================================================"

) 2>&1 | tee "$TXT"

STATUS=${PIPESTATUS[0]}

printf '%s\n' "$TXT"

exit "$STATUS"
